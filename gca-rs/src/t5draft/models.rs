//! The model files, published apart from gca: `models.json` at the root of
//! the repository lists every one with its kind, version, input layout,
//! download address and checksum. `gca model install` takes the newest one
//! of each kind this gca can read and `gca model update` replaces it when a
//! newer one comes out, so a new model needs no new gca. A model made for
//! another input layout is left out: an older gca keeps the newest model it
//! can read.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::runtime::{INPUT_FORMAT, TYPE_INPUT_FORMAT};
use super::{MODEL_KEY, TYPE_MODEL_KEY};

/// Where the list of models is, unless GCA_MODELS_URL names another (a
/// mirror, or a file:// address).
pub const MANIFEST_URL: &str =
    "https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/models.json";
pub const MANIFEST_ENV: &str = "GCA_MODELS_URL";
/// The folder models are installed in, unless GCA_MODELS_DIR names another.
pub const DIR_ENV: &str = "GCA_MODELS_DIR";
/// The version of models.json's layout this gca reads.
const SCHEMA: u32 = 1;
/// What gca installed, in the models folder.
const INSTALLED: &str = "installed.json";

/// The two kinds of model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Writes subject drafts.
    Draft,
    /// Ranks the types.
    Type,
}

impl Kind {
    pub const ALL: [Kind; 2] = [Kind::Draft, Kind::Type];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Draft => "draft",
            Kind::Type => "type",
        }
    }

    /// What people call it.
    pub fn label(self) -> &'static str {
        match self {
            Kind::Draft => "subject model",
            Kind::Type => "type model",
        }
    }

    /// The git config setting that names its file.
    pub fn config_key(self) -> &'static str {
        match self {
            Kind::Draft => MODEL_KEY,
            Kind::Type => TYPE_MODEL_KEY,
        }
    }

    /// The input layout this gca gives it.
    pub fn input_format(self) -> u32 {
        match self {
            Kind::Draft => INPUT_FORMAT,
            Kind::Type => TYPE_INPUT_FORMAT,
        }
    }
}

/// models.json.
#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub models: Vec<Entry>,
}

/// One published model file.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Entry {
    pub kind: String,
    /// Dotted numbers, newer ones larger: 2026.10.04.
    pub version: String,
    /// The input layout it was trained on (`gca.draft.input_format` or
    /// `gca.type.input_format` in the file).
    pub input_format: u32,
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
    /// One line on what changed, for `gca model list`.
    #[serde(default)]
    pub notes: Option<String>,
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Manifest> {
        let m: Manifest = serde_json::from_str(text).context("the model list is not valid")?;
        if m.schema > SCHEMA {
            bail!("the model list is newer than this gca reads; update gca");
        }
        Ok(m)
    }

    /// The newest model of this kind made for the input layout this gca
    /// gives it.
    pub fn newest(&self, kind: Kind) -> Option<&Entry> {
        self.models
            .iter()
            .filter(|e| e.kind == kind.name() && e.input_format == kind.input_format())
            .max_by(|a, b| compare_versions(&a.version, &b.version))
    }
}

/// Dotted versions compared number by number (2026.10.4 < 2026.10.12);
/// parts that are not numbers compare as text.
pub fn compare_versions(a: &str, b: &str) -> std::cmp::Ordering {
    let parts = |s: &str| -> Vec<(u64, String)> {
        s.split(['.', '-'])
            .map(|p| (p.parse::<u64>().unwrap_or(0), p.to_string()))
            .collect()
    };
    parts(a).cmp(&parts(b))
}

/// What gca installed: for each kind, its version and file.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Installed {
    #[serde(flatten)]
    pub models: BTreeMap<String, InstalledModel>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct InstalledModel {
    pub version: String,
    pub file: PathBuf,
    pub sha256: String,
}

impl Installed {
    pub fn read(dir: &Path) -> Installed {
        std::fs::read_to_string(dir.join(INSTALLED))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn write(&self, dir: &Path) -> Result<()> {
        let path = dir.join(INSTALLED);
        std::fs::write(&path, serde_json::to_string_pretty(self)? + "\n")
            .with_context(|| format!("could not write {}", path.display()))
    }

    pub fn get(&self, kind: Kind) -> Option<&InstalledModel> {
        self.models.get(kind.name())
    }
}

/// The folder models are installed in: GCA_MODELS_DIR, else
/// %LOCALAPPDATA%\gca\models on Windows, ~/Library/Application Support/gca/models
/// on macOS, and $XDG_DATA_HOME/gca/models or ~/.local/share/gca/models
/// elsewhere.
pub fn dir() -> Result<PathBuf> {
    let var = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if let Some(d) = var(DIR_ENV) {
        return Ok(d);
    }
    let base = if cfg!(windows) {
        var("LOCALAPPDATA").context("LOCALAPPDATA is not set")?
    } else if cfg!(target_os = "macos") {
        var("HOME")
            .context("HOME is not set")?
            .join("Library/Application Support")
    } else {
        match var("XDG_DATA_HOME") {
            Some(d) => d,
            None => var("HOME").context("HOME is not set")?.join(".local/share"),
        }
    };
    Ok(base.join("gca").join("models"))
}

/// The list of models: GCA_MODELS_URL, else [`MANIFEST_URL`].
pub fn manifest_url() -> String {
    std::env::var(MANIFEST_ENV)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| MANIFEST_URL.to_string())
}

/// Downloads and reads the list of models.
pub fn fetch_manifest() -> Result<Manifest> {
    let url = manifest_url();
    let out = curl()
        .args(["-fsSL", "--retry", "2", "--connect-timeout", "20", &url])
        .stdin(Stdio::null())
        .output()
        .map_err(no_curl)?;
    if !out.status.success() {
        bail!(
            "could not download the model list from {url}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Manifest::parse(&String::from_utf8_lossy(&out.stdout))
}

/// The file a model of this kind and version is installed as.
pub fn file_name(entry: &Entry) -> String {
    let version: String = entry
        .version
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{}-{version}.gguf", entry.kind)
}

/// Downloads a model into `dir` unless it is already there, and checks its
/// size and checksum. Returns its path.
pub fn download(entry: &Entry, dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("could not create {}", dir.display()))?;
    let path = dir.join(file_name(entry));
    let wanted = |sum: &str| sum.eq_ignore_ascii_case(entry.sha256.trim());
    if path.is_file() && sha256(&path).is_ok_and(|s| wanted(&s)) {
        return Ok(path);
    }
    let part = path.with_extension("gguf.part");
    let progress = if std::io::stderr().is_terminal() {
        "--progress-bar"
    } else {
        "-sS"
    };
    let status = curl()
        .args([
            "-fL",
            progress,
            "--retry",
            "2",
            "--connect-timeout",
            "20",
            "-o",
        ])
        .arg(&part)
        .arg(&entry.url)
        .stdin(Stdio::null())
        .status()
        .map_err(no_curl)?;
    if !status.success() {
        let _ = std::fs::remove_file(&part);
        bail!("could not download {}", entry.url);
    }
    let size = std::fs::metadata(&part)?.len();
    let sum = sha256(&part)?;
    if size != entry.bytes || !wanted(&sum) {
        let _ = std::fs::remove_file(&part);
        bail!(
            "{} is not the file the model list describes ({size} bytes, sha256 {sum})",
            entry.url
        );
    }
    std::fs::rename(&part, &path)
        .with_context(|| format!("could not move the model to {}", path.display()))?;
    Ok(path)
}

/// The SHA-256 of a file, in lowercase hex.
pub fn sha256(path: &Path) -> Result<String> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn curl() -> Command {
    Command::new("curl")
}

fn no_curl(e: std::io::Error) -> anyhow::Error {
    anyhow!(
        "could not run curl ({e}); install curl, or download the file from the address in {} \
         and set it with `gca config draft-model <FILE>` or `gca config type-model <FILE>`",
        manifest_url()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    fn entry(kind: &str, version: &str, format: u32) -> Entry {
        Entry {
            kind: kind.into(),
            version: version.into(),
            input_format: format,
            url: format!("https://example.com/{kind}-{version}.gguf"),
            sha256: "0".repeat(64),
            bytes: 1,
            notes: None,
        }
    }

    #[test]
    fn versions_compare_number_by_number() {
        assert_eq!(compare_versions("2026.10.4", "2026.10.12"), Ordering::Less);
        assert_eq!(
            compare_versions("2026.11.01", "2026.10.30"),
            Ordering::Greater
        );
        assert_eq!(compare_versions("1.2", "1.2"), Ordering::Equal);
        assert_eq!(compare_versions("1.2", "1.2.1"), Ordering::Less);
    }

    #[test]
    fn picks_the_newest_model_this_gca_can_read() {
        let m = Manifest {
            schema: 1,
            models: vec![
                entry("draft", "2026.10.01", 1),
                entry("draft", "2026.12.01", 2), // another input layout
                entry("draft", "2026.11.15", 1),
                entry("type", "2026.10.04", 1),
                entry("other", "2027.01.01", 1),
            ],
        };
        assert_eq!(m.newest(Kind::Draft).unwrap().version, "2026.11.15");
        assert_eq!(m.newest(Kind::Type).unwrap().version, "2026.10.04");
        let none = Manifest {
            schema: 1,
            models: vec![entry("type", "2027.01.01", 9)],
        };
        assert!(none.newest(Kind::Type).is_none());
    }

    #[test]
    fn reads_the_list_and_refuses_a_newer_layout() {
        let text = r#"{"schema": 1, "models": [{"kind": "type", "version": "2026.10.04",
            "input_format": 1, "url": "https://example.com/t.gguf", "sha256": "ab", "bytes": 3,
            "notes": "first", "future_field": true}]}"#;
        let m = Manifest::parse(text).unwrap();
        assert_eq!(m.models[0].notes.as_deref(), Some("first"));
        assert!(Manifest::parse(r#"{"schema": 2, "models": []}"#).is_err());
        assert!(Manifest::parse("not json").is_err());
    }

    #[test]
    fn names_files_by_kind_and_version() {
        assert_eq!(
            file_name(&entry("type", "2026.10.04", 1)),
            "type-2026.10.04.gguf"
        );
        assert_eq!(file_name(&entry("draft", "1/../x", 1)), "draft-1_.._x.gguf");
    }

    #[test]
    fn hashes_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
