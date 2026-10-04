//! Subject drafts for mechanical commits: dependency changes (manifests,
//! lockfiles, GitHub Actions), releases, renames, removals, one-word typo
//! fixes, and new test, doc or workflow files.
//!
//! A draft only pre-fills the subject prompt; the user accepts it or types
//! over it. A change that is not one of these patterns gets no draft rather
//! than a generic one such as "update README": checked against 166,417
//! held-out commits, people almost never kept those.

use regex::Regex;
use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::git::FileChange;
use crate::heuristics;

/// Longest subject a draft may have; longer ones fall back to a shorter form.
const MAX_SUBJECT: usize = 72;

#[derive(Debug, PartialEq)]
pub struct Draft {
    pub subject: String,
    /// The type such commits usually get, when the pattern implies one.
    pub kind: Option<&'static str>,
}

/// A draft for the staged change, if it follows a mechanical pattern.
pub fn draft(files: &[FileChange], diff: &str) -> Option<Draft> {
    if files.is_empty() {
        return None;
    }
    let sections = parse_diff(diff);
    release(files, &sections)
        .or_else(|| dependencies(files, &sections))
        .or_else(|| renames(files))
        .or_else(|| removals(files))
        .or_else(|| tests_only(files))
        .or_else(|| docs_only(files, &sections))
        .or_else(|| ci_only(files))
        .filter(|d| !d.subject.is_empty() && d.subject.len() <= MAX_SUBJECT)
}

// ---------------------------------------------------------------- the diff

/// One file's part of a unified diff.
#[derive(Debug, Default)]
struct Section {
    path: String,
    hunks: Vec<Hunk>,
}

#[derive(Debug, Default)]
struct Hunk {
    /// First line of the hunk in the old file.
    old_start: usize,
    removed: Vec<String>,
    added: Vec<String>,
}

static HUNK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^@@ -(\d+)(?:,\d+)? \+\d+").unwrap());

fn parse_diff(diff: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut in_hunk = false;
    for line in diff.lines() {
        if line.starts_with("diff --git ") {
            sections.push(Section::default());
            in_hunk = false;
            continue;
        }
        let Some(section) = sections.last_mut() else {
            continue;
        };
        if !in_hunk {
            if let Some(p) = line.strip_prefix("+++ ") {
                if p != "/dev/null" {
                    section.path = strip_side(p);
                }
            } else if let Some(p) = line.strip_prefix("--- ") {
                if p != "/dev/null" && section.path.is_empty() {
                    section.path = strip_side(p);
                }
            }
        }
        if let Some(c) = HUNK.captures(line) {
            in_hunk = true;
            section.hunks.push(Hunk {
                old_start: c[1].parse().unwrap_or(0),
                ..Hunk::default()
            });
            continue;
        }
        if !in_hunk {
            continue;
        }
        let Some(hunk) = section.hunks.last_mut() else {
            continue;
        };
        if let Some(l) = line.strip_prefix('-') {
            hunk.removed.push(l.to_string());
        } else if let Some(l) = line.strip_prefix('+') {
            hunk.added.push(l.to_string());
        }
    }
    sections
}

/// `b/src/x.rs` or `"b/src/\303\251.rs"` to `src/x.rs`; quoted names stay escaped.
fn strip_side(p: &str) -> String {
    let p = p.trim_end_matches('\t');
    let unquoted = p
        .strip_prefix('"')
        .and_then(|q| q.strip_suffix('"'))
        .unwrap_or(p);
    unquoted
        .strip_prefix("a/")
        .or_else(|| unquoted.strip_prefix("b/"))
        .unwrap_or(unquoted)
        .to_string()
}

fn section<'a>(sections: &'a [Section], path: &str) -> Option<&'a Section> {
    sections.iter().find(|s| s.path == path)
}

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn dirname(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(d, _)| d)
}

fn stem(name: &str) -> &str {
    name.split_once('.').map_or(name, |(s, _)| s)
}

// ------------------------------------------------------- manifests and locks

#[derive(Clone, Copy, PartialEq, Debug)]
enum Manifest {
    Npm,
    Cargo,
    Pyproject,
    Requirements,
    GoMod,
    /// `uses: owner/action@version` in a GitHub Actions workflow.
    Actions,
}

fn manifest(path: &str) -> Option<Manifest> {
    let name = basename(path);
    match name {
        "package.json" => Some(Manifest::Npm),
        "Cargo.toml" => Some(Manifest::Cargo),
        "pyproject.toml" => Some(Manifest::Pyproject),
        "go.mod" => Some(Manifest::GoMod),
        _ if path.starts_with(".github/workflows/")
            && (name.ends_with(".yml") || name.ends_with(".yaml")) =>
        {
            Some(Manifest::Actions)
        }
        _ if name.starts_with("requirements") && name.ends_with(".txt") => {
            Some(Manifest::Requirements)
        }
        _ if name.ends_with(".txt") && dirname(path).ends_with("requirements") => {
            Some(Manifest::Requirements)
        }
        _ => None,
    }
}

fn is_lockfile(path: &str) -> bool {
    matches!(
        basename(path),
        "package-lock.json"
            | "npm-shrinkwrap.json"
            | "yarn.lock"
            | "pnpm-lock.yaml"
            | "bun.lock"
            | "bun.lockb"
            | "Cargo.lock"
            | "poetry.lock"
            | "uv.lock"
            | "Pipfile.lock"
            | "pdm.lock"
            | "go.sum"
            | "Gemfile.lock"
            | "composer.lock"
            | "flake.lock"
            | "mix.lock"
            | "pubspec.lock"
            | "Podfile.lock"
            | "packages.lock.json"
    )
}

fn is_changelog(path: &str) -> bool {
    let upper = basename(path).to_ascii_uppercase();
    ["CHANGELOG", "CHANGES", "HISTORY", "RELEASE", "NEWS"]
        .iter()
        .any(|n| upper.starts_with(n))
}

/// Files a release tool leaves behind: changesets it consumed, a VERSION file.
fn is_release_leftover(f: &FileChange) -> bool {
    (f.status == 'D' && f.path.starts_with(".changeset/") && f.path.ends_with(".md"))
        || matches!(basename(&f.path), "VERSION" | "version.txt")
}

// A version as manifests write it: an optional range operator, then a digit.
const VER: &str = r#"[~^=<>!*v\s]*\d[\w.+\-*]*"#;

static NPM_DEP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r#"^\s*"((?:@[\w.\-]+/)?[\w.\-]+)"\s*:\s*"({VER})"\s*,?\s*$"#
    ))
    .unwrap()
});
static TOML_DEP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r#"^\s*([A-Za-z0-9_\-]+)\s*=\s*"({VER})"\s*$"#)).unwrap());
static TOML_TABLE_DEP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^\s*([A-Za-z0-9_\-]+)\s*=\s*\{[^}]*\bversion\s*=\s*"([^"]+)""#).unwrap()
});
static PEP508_DEP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^\s*"?([A-Za-z0-9][\w.\-]*)(?:\[[^\]]*\])?\s*((?:===|==|>=|<=|~=|!=|>|<)\s*[\w.*+\-]+(?:\s*,\s*(?:===|==|>=|<=|~=|!=|>|<)\s*[\w.*+\-]+)*)\s*(?:;[^"]*)?"?\s*,?\s*(?:#.*)?$"#).unwrap()
});
// `uses: actions/checkout@v4`, or a pinned commit with the version in a comment.
static ACTION_DEP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^\s*(?:-\s*)?uses:\s*['"]?([\w.\-]+/[\w.\-/]+)@([\w.\-]+)['"]?\s*(?:#\s*(v?\d[\w.\-]*))?\s*$"#).unwrap()
});
static GO_DEP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(?:require\s+)?([\w.\-]+\.[\w.\-]+(?:/[\w.\-]+)*)\s+(v\d[\w.+\-]*)\s*(?://\s*indirect)?\s*$").unwrap()
});

/// Keys in TOML manifests that look like `name = "1.2"` but are not dependencies.
const TOML_NOT_DEPS: [&str; 6] = [
    "version",
    "edition",
    "rust-version",
    "resolver",
    "name",
    "license",
];

/// (name, version) when the line declares a dependency.
fn dependency(kind: Manifest, line: &str) -> Option<(String, String)> {
    let pick = |re: &Regex| {
        re.captures(line)
            .map(|c| (c[1].to_string(), c[2].trim().to_string()))
    };
    let dep = match kind {
        Manifest::Npm => pick(&NPM_DEP).filter(|(n, _)| n != "version"),
        Manifest::Cargo => pick(&TOML_TABLE_DEP).or_else(|| pick(&TOML_DEP)),
        Manifest::Pyproject => pick(&TOML_DEP).or_else(|| pick(&PEP508_DEP)),
        Manifest::Requirements => pick(&PEP508_DEP),
        Manifest::GoMod => pick(&GO_DEP),
        Manifest::Actions => ACTION_DEP.captures(line).map(|c| {
            let pinned = c[2].len() == 40 && c[2].bytes().all(|b| b.is_ascii_hexdigit());
            let version = match c.get(3) {
                Some(comment) if pinned => comment.as_str().to_string(),
                _ => c[2].to_string(),
            };
            (c[1].to_string(), version)
        }),
    }?;
    if matches!(kind, Manifest::Cargo | Manifest::Pyproject)
        && TOML_NOT_DEPS.contains(&dep.0.as_str())
    {
        return None;
    }
    Some(dep)
}

static NPM_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"^\s*"version"\s*:\s*"([^"]+)"\s*,?\s*$"#).unwrap());
static TOML_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"^\s*version\s*=\s*"([^"]+)"\s*$"#).unwrap());

/// The package's own version on this line of a manifest.
fn own_version(kind: Manifest, line: &str, hunk: &Hunk) -> Option<String> {
    match kind {
        Manifest::Npm => NPM_VERSION.captures(line).map(|c| c[1].to_string()),
        // `version = ...` near the top is [package] / [project]; further down it
        // may belong to a `[dependencies.x]` table.
        Manifest::Cargo | Manifest::Pyproject if hunk.old_start <= 25 => {
            TOML_VERSION.captures(line).map(|c| c[1].to_string())
        }
        _ => None,
    }
}

/// Lines that only move punctuation around, as when an entry gains a comma.
fn is_noise(line: &str) -> bool {
    matches!(
        line.trim(),
        "" | "{" | "}" | "}," | "[" | "]" | "]," | "(" | ")"
    )
}

fn is_comment(kind: Manifest, line: &str) -> bool {
    let t = line.trim_start();
    match kind {
        Manifest::Npm => false,
        Manifest::GoMod => t.starts_with("//"),
        Manifest::Cargo | Manifest::Pyproject | Manifest::Requirements | Manifest::Actions => {
            t.starts_with('#')
        }
    }
}

/// What a manifest's diff changes.
#[derive(Default, Debug)]
struct ManifestChange {
    /// The package's own version: (old, new).
    version: Option<(String, String)>,
    removed: BTreeMap<String, String>,
    added: BTreeMap<String, String>,
    /// Changed lines that are neither versions nor dependencies.
    other: usize,
}

fn read_manifest(kind: Manifest, s: &Section) -> ManifestChange {
    let mut m = ManifestChange::default();
    let mut old_version = None;
    let mut new_version = None;
    for hunk in &s.hunks {
        for (lines, is_added) in [(&hunk.removed, false), (&hunk.added, true)] {
            for line in lines {
                if is_noise(line) || is_comment(kind, line) {
                    continue;
                }
                if let Some(v) = own_version(kind, line, hunk) {
                    if is_added {
                        new_version = Some(v);
                    } else {
                        old_version = Some(v);
                    }
                } else if let Some((name, ver)) = dependency(kind, line) {
                    let side = if is_added {
                        &mut m.added
                    } else {
                        &mut m.removed
                    };
                    side.insert(name, ver);
                } else {
                    m.other += 1;
                }
            }
        }
    }
    match (old_version, new_version) {
        (Some(a), Some(b)) if a != b => m.version = Some((a, b)),
        (None, None) => {}
        // A version line added or removed on its own is some other edit.
        (Some(_), Some(_)) => {}
        _ => m.other += 1,
    }
    // An entry that only gained or lost a trailing comma is unchanged.
    let same: Vec<String> = m
        .removed
        .iter()
        .filter(|(n, v)| m.added.get(*n) == Some(*v))
        .map(|(n, _)| n.clone())
        .collect();
    for n in same {
        m.removed.remove(&n);
        m.added.remove(&n);
    }
    m
}

fn bare(version: &str) -> &str {
    version.trim_start_matches(|c: char| "~^=<>!* ".contains(c))
}

/// `1.2.3` → `v1.2.3`, the way release commits usually name versions.
fn with_v(version: &str) -> String {
    if version.starts_with(|c: char| c.is_ascii_digit()) {
        format!("v{version}")
    } else {
        version.to_string()
    }
}

/// Whether `new` is an older version than `old` (numeric parts only).
fn is_downgrade(old: &str, new: &str) -> bool {
    let parts = |v: &str| -> Vec<u64> {
        bare(v)
            .trim_start_matches('v')
            .split(|c: char| !c.is_ascii_digit())
            .take_while(|p| !p.is_empty())
            .filter_map(|p| p.parse().ok())
            .collect()
    };
    let (a, b) = (parts(old), parts(new));
    !a.is_empty() && !b.is_empty() && b < a
}

// ---------------------------------------------------------------- patterns

/// The package's own version changed, and nothing but release bookkeeping
/// came with it: manifests, lockfiles, changelogs, consumed changesets, and
/// files where the old version only gave way to the new one, such as an
/// install line in a README or a version constant.
fn release(files: &[FileChange], sections: &[Section]) -> Option<Draft> {
    let mut changes = Vec::new();
    let mut others = Vec::new();
    for f in files {
        if let Some(kind) = manifest(&f.path) {
            let m = read_manifest(kind, section(sections, &f.path)?);
            if m.other > 0 {
                return None;
            }
            if let Some(change) = m.version {
                changes.push(change);
            }
        } else if !(is_lockfile(&f.path) || is_changelog(&f.path) || is_release_leftover(f)) {
            others.push(f);
        }
    }
    for f in others {
        if f.status != 'M' || !only_swaps_versions(section(sections, &f.path)?, &changes) {
            return None;
        }
    }
    let mut versions: Vec<&str> = changes.iter().map(|(_, new)| new.as_str()).collect();
    versions.sort();
    versions.dedup();
    let subject = match versions.as_slice() {
        [] => return None,
        [one] => format!("release {}", with_v(bare(one))),
        many => format!("release {} packages", many.len()),
    };
    Some(Draft {
        subject,
        kind: Some("chore"),
    })
}

/// Whether each changed line of this file is the line it replaces with an
/// old version of the package swapped for the new one, line for line.
fn only_swaps_versions(s: &Section, changes: &[(String, String)]) -> bool {
    let swaps: Vec<(&str, &str)> = changes
        .iter()
        .map(|(old, new)| (bare(old), bare(new)))
        .filter(|(old, new)| !old.is_empty() && old != new)
        .collect();
    !swaps.is_empty()
        && !s.hunks.is_empty()
        && s.hunks.iter().all(|h| {
            !h.removed.is_empty()
                && h.removed.len() == h.added.len()
                && h.removed.iter().zip(&h.added).all(|(old_line, new_line)| {
                    let swapped = swaps.iter().fold(old_line.clone(), |line, (old, new)| {
                        swap_version(&line, old, new)
                    });
                    old_line != new_line && swapped == *new_line
                })
        })
}

/// The line with each mention of version `old` replaced by `new`, except
/// where it is part of a longer version (`10.5.1` or `0.5.12` for `0.5.1`).
fn swap_version(line: &str, old: &str, new: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut last = 0;
    for (i, _) in line.match_indices(old) {
        let before = line[..i].chars().next_back();
        let mut after = line[i + old.len()..].chars();
        let longer = before.is_some_and(|c| c.is_ascii_digit() || c == '.')
            || match after.next() {
                Some(c) if c.is_ascii_digit() => true,
                Some('.') => after.next().is_some_and(|c| c.is_ascii_digit()),
                _ => false,
            };
        if !longer {
            out.push_str(&line[last..i]);
            out.push_str(new);
            last = i + old.len();
        }
    }
    out.push_str(&line[last..]);
    out
}

/// Only manifests' dependency lines and lockfiles changed.
fn dependencies(files: &[FileChange], sections: &[Section]) -> Option<Draft> {
    let mut bumped: BTreeMap<String, (String, String)> = BTreeMap::new();
    let mut added: BTreeMap<String, String> = BTreeMap::new();
    let mut removed: BTreeMap<String, String> = BTreeMap::new();
    let mut locks = Vec::new();
    for f in files {
        if is_lockfile(&f.path) {
            locks.push(f.path.as_str());
            continue;
        }
        let kind = manifest(&f.path)?;
        if f.status != 'M' {
            return None;
        }
        let m = read_manifest(kind, section(sections, &f.path)?);
        if m.other > 0 || m.version.is_some() {
            return None;
        }
        for (name, old) in &m.removed {
            match m.added.get(name) {
                Some(new) => {
                    bumped.insert(name.clone(), (old.clone(), new.clone()));
                }
                None => {
                    removed.insert(name.clone(), old.clone());
                }
            }
        }
        for (name, new) in &m.added {
            if !m.removed.contains_key(name) {
                added.insert(name.clone(), new.clone());
            }
        }
    }
    // "zod dependency", "vite and vitest dependencies", "3 dependencies"
    let names = |deps: &BTreeMap<String, String>| match deps.keys().collect::<Vec<_>>()[..] {
        [one] => format!("{one} dependency"),
        [a, b] => format!("{a} and {b} dependencies"),
        _ => format!("{} dependencies", deps.len()),
    };
    let subject = match (bumped.len(), added.len(), removed.len()) {
        (0, 0, 0) => match locks.as_slice() {
            [] => return None,
            [one] => format!("update {}", basename(one)),
            _ => "update lockfiles".to_string(),
        },
        (1, 0, 0) => {
            let (name, (old, new)) = bumped.iter().next()?;
            let verb = if is_downgrade(old, new) {
                "downgrade"
            } else {
                "bump"
            };
            let full = format!("{verb} {name} from {} to {}", bare(old), bare(new));
            if full.len() <= MAX_SUBJECT {
                full
            } else {
                format!("{verb} {name}")
            }
        }
        (2, 0, 0) => {
            let n: Vec<&String> = bumped.keys().collect();
            format!("bump {} and {}", n[0], n[1])
        }
        (_, 0, 0) => format!("bump {} dependencies", bumped.len()),
        (0, _, 0) => format!("add {}", names(&added)),
        (0, 0, _) => format!("remove {}", names(&removed)),
        _ => "update dependencies".to_string(),
    };
    Some(Draft {
        subject: shorten(subject, "update dependencies"),
        kind: None,
    })
}

/// Every file moved without being edited.
fn renames(files: &[FileChange]) -> Option<Draft> {
    let moves: Vec<(&str, &str)> = files
        .iter()
        .map(|f| match (&f.old_path, f.status, f.score) {
            (Some(old), 'R', Some(100)) => Some((old.as_str(), f.path.as_str())),
            _ => None,
        })
        .collect::<Option<_>>()?;
    let subject = match moves.as_slice() {
        [(old, new)] => {
            if dirname(old) == dirname(new) {
                format!("rename {} to {}", basename(old), basename(new))
            } else if basename(old) == basename(new) {
                format!("move {} to {}", basename(new), dir_label(dirname(new)))
            } else {
                shorten(
                    format!("move {old} to {new}"),
                    &format!("move {} to {}", basename(old), basename(new)),
                )
            }
        }
        many => {
            // A renamed directory: every file keeps its path below the prefix.
            let common_old = common_dir(many.iter().map(|m| m.0));
            let common_new = common_dir(many.iter().map(|m| m.1));
            let same_tail = many.iter().all(|(o, n)| {
                o.strip_prefix(common_old.as_str()) == n.strip_prefix(common_new.as_str())
            });
            if same_tail
                && !common_old.is_empty()
                && !common_new.is_empty()
                && common_old != common_new
            {
                format!(
                    "rename {} to {}",
                    dir_label(common_old.trim_end_matches('/')),
                    dir_label(common_new.trim_end_matches('/'))
                )
            } else if many.iter().all(|(_, n)| dirname(n) == dirname(many[0].1)) {
                format!(
                    "move {} files to {}",
                    many.len(),
                    dir_label(dirname(many[0].1))
                )
            } else {
                return None;
            }
        }
    };
    // People file pure moves under chore, fix and refactor about equally, so
    // the model picks the type.
    Some(Draft {
        subject,
        kind: None,
    })
}

/// Every file was deleted.
fn removals(files: &[FileChange]) -> Option<Draft> {
    if !files.iter().all(|f| f.status == 'D') {
        return None;
    }
    let subject = match files {
        [one] => shorten(
            format!("remove {}", one.path),
            &format!("remove {}", basename(&one.path)),
        ),
        many => {
            let common = common_dir(many.iter().map(|f| f.path.as_str()));
            if common.is_empty() {
                format!("remove {} files", many.len())
            } else {
                format!(
                    "remove {} files from {}",
                    many.len(),
                    dir_label(common.trim_end_matches('/'))
                )
            }
        }
    };
    Some(Draft {
        subject: shorten(subject, &format!("remove {} files", files.len())),
        kind: None,
    })
}

/// What a test file tests: `proxy.test.ts`, `test_proxy.py`, `proxy_test.go` → `proxy`.
fn tested(path: &str) -> String {
    let name = basename(path);
    let s = stem(name);
    let s = s.strip_prefix("test_").unwrap_or(s);
    let s = s
        .strip_suffix("_test")
        .or_else(|| s.strip_suffix("_spec"))
        .or_else(|| s.strip_suffix("Tests"))
        .or_else(|| s.strip_suffix("Test"))
        .unwrap_or(s);
    s.to_string()
}

fn tests_only(files: &[FileChange]) -> Option<Draft> {
    let paths: Vec<String> = files.iter().map(|f| f.path.clone()).collect();
    if heuristics::classify(&paths) != Some("test") {
        return None;
    }
    let mut subjects: Vec<String> = files.iter().map(|f| tested(&f.path)).collect();
    subjects.sort();
    subjects.dedup();
    let verb = if files.iter().all(|f| f.status == 'A') {
        "add"
    } else if files.iter().all(|f| f.status == 'D') {
        "remove"
    } else {
        return None;
    };
    let subject = match subjects.as_slice() {
        [one]
            if one.len() >= 2
                && !matches!(one.as_str(), "test" | "tests" | "mod" | "index" | "main") =>
        {
            format!("{verb} tests for {one}")
        }
        _ => return None,
    };
    Some(Draft {
        subject,
        kind: Some("test"),
    })
}

/// How a documentation file is named in a subject.
fn doc_label(path: &str) -> String {
    let name = basename(path);
    let upper = stem(name).to_ascii_uppercase();
    match upper.as_str() {
        "README" => "README".into(),
        "CHANGELOG" | "CHANGES" => "changelog".into(),
        "CONTRIBUTING" => "contributing guide".into(),
        "LICENSE" | "LICENCE" => "license".into(),
        "CODE_OF_CONDUCT" => "code of conduct".into(),
        "SECURITY" => "security policy".into(),
        _ if dirname(path).split('/').any(|d| d == "docs" || d == "doc") => {
            format!("{} docs", stem(name))
        }
        _ => name.to_string(),
    }
}

fn docs_only(files: &[FileChange], sections: &[Section]) -> Option<Draft> {
    let paths: Vec<String> = files.iter().map(|f| f.path.clone()).collect();
    if heuristics::classify(&paths) != Some("docs") {
        return None;
    }
    let typos = typo_count(files, sections);
    let subject = match (files, typos) {
        ([one], Some(1)) => format!("fix typo in {}", doc_label(&one.path)),
        ([one], Some(_)) => format!("fix typos in {}", doc_label(&one.path)),
        (_, Some(_)) => "fix typos in docs".to_string(),
        ([one], None) => match one.status {
            'A' => format!("add {}", doc_label(&one.path)),
            'D' => format!("remove {}", doc_label(&one.path)),
            _ => return None,
        },
        _ => return None,
    };
    Some(Draft {
        subject,
        kind: Some("docs"),
    })
}

/// How many one-word fixes the change consists of, if that is all it is.
fn typo_count(files: &[FileChange], sections: &[Section]) -> Option<usize> {
    let mut count = 0;
    for f in files {
        if f.status != 'M' {
            return None;
        }
        for hunk in &section(sections, &f.path)?.hunks {
            if hunk.removed.len() != hunk.added.len() || hunk.removed.is_empty() {
                return None;
            }
            for (old, new) in hunk.removed.iter().zip(&hunk.added) {
                if !one_word_fix(old, new) {
                    return None;
                }
                count += 1;
            }
        }
    }
    (1..=5).contains(&count).then_some(count)
}

/// The lines differ in exactly one word, and only by a letter or two.
fn one_word_fix(old: &str, new: &str) -> bool {
    let a: Vec<&str> = old.split_whitespace().collect();
    let b: Vec<&str> = new.split_whitespace().collect();
    if a.len() != b.len() {
        return false;
    }
    let diffs: Vec<(&str, &str)> = a.into_iter().zip(b).filter(|(x, y)| x != y).collect();
    match diffs.as_slice() {
        [(x, y)] => {
            // Version numbers, and markup such as added backticks, are not typos.
            let letters = |w: &str| -> String { w.chars().filter(|c| c.is_alphabetic()).collect() };
            if x.chars().chain(y.chars()).any(|c| c.is_ascii_digit()) || letters(x) == letters(y) {
                return false;
            }
            let d = edit_distance(x, y);
            (1..=2).contains(&d) && x.chars().count().max(y.chars().count()) >= 3
        }
        _ => false,
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    // Optimal string alignment: substitutions, insertions, deletions, and a
    // swap of two neighbouring letters, each counting as one edit.
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[n][m]
}

fn ci_only(files: &[FileChange]) -> Option<Draft> {
    let paths: Vec<String> = files.iter().map(|f| f.path.clone()).collect();
    if heuristics::classify(&paths) != Some("ci") {
        return None;
    }
    let label = |path: &str| {
        if path.starts_with(".github/workflows/") {
            format!("{} workflow", stem(basename(path)))
        } else {
            basename(path).to_string()
        }
    };
    let subject = match files {
        [one] if one.status == 'A' => format!("add {}", label(&one.path)),
        [one] if one.status == 'D' => format!("remove {}", label(&one.path)),
        _ => return None,
    };
    Some(Draft {
        subject,
        kind: Some("ci"),
    })
}

// ---------------------------------------------------------------- helpers

/// The longest directory prefix (ending in `/`) the paths share, or "".
fn common_dir<'a>(mut paths: impl Iterator<Item = &'a str>) -> String {
    let Some(first) = paths.next() else {
        return String::new();
    };
    let mut prefix: Vec<&str> = first.split('/').collect();
    prefix.pop(); // the file name
    for p in paths {
        let dirs: Vec<&str> = p.split('/').collect();
        let dirs = &dirs[..dirs.len() - 1];
        let n = prefix.iter().zip(dirs).take_while(|(a, b)| a == b).count();
        prefix.truncate(n);
    }
    if prefix.is_empty() {
        String::new()
    } else {
        format!("{}/", prefix.join("/"))
    }
}

fn dir_label(dir: &str) -> String {
    if dir.is_empty() {
        "the repository root".to_string()
    } else {
        format!("{dir}/")
    }
}

fn shorten(subject: String, fallback: &str) -> String {
    if subject.len() <= MAX_SUBJECT {
        subject
    } else {
        fallback.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(status: char, path: &str) -> FileChange {
        FileChange {
            status,
            path: path.to_string(),
            old_path: None,
            score: None,
        }
    }

    fn moved(old: &str, new: &str, score: u8) -> FileChange {
        FileChange {
            status: 'R',
            path: new.to_string(),
            old_path: Some(old.to_string()),
            score: Some(score),
        }
    }

    /// A one-hunk diff for `path` with these removed and added lines.
    fn hunk(path: &str, start: usize, removed: &[&str], added: &[&str]) -> String {
        let mut d = format!(
            "diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n@@ -{start},3 +{start},3 @@ ctx\n"
        );
        for l in removed {
            d.push_str(&format!("-{l}\n"));
        }
        for l in added {
            d.push_str(&format!("+{l}\n"));
        }
        d
    }

    fn subject(files: &[FileChange], diff: &str) -> Option<String> {
        draft(files, diff).map(|d| d.subject)
    }

    #[test]
    fn npm_bump() {
        let diff = hunk(
            "package.json",
            12,
            &[r#"    "react": "^18.2.0","#],
            &[r#"    "react": "^18.3.1","#],
        );
        let files = [file('M', "package.json"), file('M', "package-lock.json")];
        assert_eq!(
            subject(&files, &diff).as_deref(),
            Some("bump react from 18.2.0 to 18.3.1")
        );
    }

    #[test]
    fn scoped_npm_bump_and_trailing_comma() {
        let diff = hunk(
            "packages/ui/package.json",
            20,
            &[r#"    "@types/node": "20.1.0""#, r#"    "zod": "^3.22.0""#],
            &[r#"    "@types/node": "20.1.0","#, r#"    "zod": "^3.23.8""#],
        );
        let files = [
            file('M', "packages/ui/package.json"),
            file('M', "pnpm-lock.yaml"),
        ];
        assert_eq!(
            subject(&files, &diff).as_deref(),
            Some("bump zod from 3.22.0 to 3.23.8")
        );
    }

    #[test]
    fn cargo_bumps_and_additions() {
        let two = hunk(
            "Cargo.toml",
            30,
            &[
                r#"serde = { version = "1.0.190", features = ["derive"] }"#,
                r#"regex = "1.9""#,
            ],
            &[
                r#"serde = { version = "1.0.200", features = ["derive"] }"#,
                r#"regex = "1.10""#,
            ],
        );
        let files = [file('M', "Cargo.toml"), file('M', "Cargo.lock")];
        assert_eq!(
            subject(&files, &two).as_deref(),
            Some("bump regex and serde")
        );

        let add = hunk("Cargo.toml", 30, &[], &[r#"tempfile = "3""#]);
        assert_eq!(
            subject(&files, &add).as_deref(),
            Some("add tempfile dependency")
        );
        let add_two = hunk(
            "Cargo.toml",
            30,
            &[],
            &[r#"tempfile = "3""#, r#"insta = "1""#],
        );
        assert_eq!(
            subject(&files, &add_two).as_deref(),
            Some("add insta and tempfile dependencies")
        );
        let remove_three = hunk(
            "Cargo.toml",
            30,
            &[r#"a = "1""#, r#"b = "1""#, r#"c = "1""#],
            &[],
        );
        assert_eq!(
            subject(&files, &remove_three).as_deref(),
            Some("remove 3 dependencies")
        );
    }

    #[test]
    fn edition_is_not_a_dependency() {
        let diff = hunk(
            "Cargo.toml",
            4,
            &[r#"edition = "2018""#],
            &[r#"edition = "2021""#],
        );
        assert_eq!(subject(&[file('M', "Cargo.toml")], &diff), None);
    }

    #[test]
    fn python_and_go() {
        let req = hunk(
            "requirements.txt",
            1,
            &["requests==2.31.0"],
            &["requests==2.32.3"],
        );
        assert_eq!(
            subject(&[file('M', "requirements.txt")], &req).as_deref(),
            Some("bump requests from 2.31.0 to 2.32.3")
        );
        let py = hunk(
            "pyproject.toml",
            40,
            &[r#"    "httpx>=0.25","#],
            &[r#"    "httpx>=0.27","#],
        );
        assert_eq!(
            subject(&[file('M', "pyproject.toml"), file('M', "uv.lock")], &py).as_deref(),
            Some("bump httpx from 0.25 to 0.27")
        );
        let go = hunk(
            "go.mod",
            5,
            &["\tgithub.com/spf13/cobra v1.7.0"],
            &["\tgithub.com/spf13/cobra v1.8.0"],
        );
        assert_eq!(
            subject(&[file('M', "go.mod"), file('M', "go.sum")], &go).as_deref(),
            Some("bump github.com/spf13/cobra from v1.7.0 to v1.8.0")
        );
    }

    #[test]
    fn lockfile_only() {
        let files = [file('M', "Cargo.lock")];
        assert_eq!(
            subject(&files, &hunk("Cargo.lock", 100, &["x"], &["y"])).as_deref(),
            Some("update Cargo.lock")
        );
    }

    #[test]
    fn a_code_change_next_to_a_bump_gets_no_draft() {
        let diff = hunk(
            "package.json",
            12,
            &[r#"    "react": "^18.2.0","#],
            &[r#"    "react": "^18.3.1","#],
        );
        let files = [file('M', "package.json"), file('M', "src/app.tsx")];
        assert_eq!(subject(&files, &diff), None);
        let scripts = hunk(
            "package.json",
            5,
            &[r#"    "build": "tsc","#],
            &[r#"    "build": "tsc -p .","#],
        );
        assert_eq!(subject(&[file('M', "package.json")], &scripts), None);
    }

    #[test]
    fn release() {
        let mut diff = hunk(
            "package.json",
            2,
            &[r#"  "version": "1.4.0","#],
            &[r#"  "version": "1.5.0","#],
        );
        diff.push_str(&hunk("CHANGELOG.md", 1, &[], &["## 1.5.0"]));
        let files = [
            file('M', "package.json"),
            file('M', "CHANGELOG.md"),
            file('D', ".changeset/brave-owls-sing.md"),
        ];
        let d = draft(&files, &diff).unwrap();
        assert_eq!(d.subject, "release v1.5.0");
        assert_eq!(d.kind, Some("chore"));

        let cargo = hunk(
            "Cargo.toml",
            1,
            &[r#"version = "0.2.0""#],
            &[r#"version = "0.3.0""#],
        );
        assert_eq!(
            subject(&[file('M', "Cargo.toml"), file('M', "Cargo.lock")], &cargo).as_deref(),
            Some("release v0.3.0")
        );
    }

    #[test]
    fn a_release_may_swap_the_version_elsewhere() {
        // gca's own 0.6.0: the README's install line and the install
        // scripts' comments name the new version too
        let mut diff = hunk(
            "gca-rs/Cargo.toml",
            1,
            &[r#"version = "0.5.1""#],
            &[r#"version = "0.6.0""#],
        );
        diff.push_str(&hunk(
            "install.sh",
            10,
            &["#   GCA_VERSION=v0.5.1      install this release instead of the latest"],
            &["#   GCA_VERSION=v0.6.0      install this release instead of the latest"],
        ));
        diff.push_str(&hunk(
            "src/version.py",
            1,
            &[r#"__version__ = "0.5.1"  # not 10.5.1"#],
            &[r#"__version__ = "0.6.0"  # not 10.5.1"#],
        ));
        diff.push_str(&hunk(
            "README.md",
            70,
            &["`$env:GCA_UNINSTALL = 1` before the same command. `GCA_VERSION=v0.5.1`"],
            &["`$env:GCA_UNINSTALL = 1` before the same command. `GCA_VERSION=v0.6.0`"],
        ));
        let mut files = vec![
            file('M', "gca-rs/Cargo.toml"),
            file('M', "gca-rs/Cargo.lock"),
            file('M', "CHANGELOG.md"),
            file('M', "README.md"),
            file('M', "install.sh"),
            file('M', "src/version.py"),
        ];
        let d = draft(&files, &diff).unwrap();
        assert_eq!(d.subject, "release v0.6.0");
        assert_eq!(d.kind, Some("chore"));

        // anything else in such a file is no longer only a release: here a
        // second hunk of the README
        let other = "@@ -90,3 +90,3 @@ ctx\n-old words\n+new words\n";
        assert_eq!(subject(&files, &format!("{diff}{other}")), None);
        let bumped_elsewhere = hunk(
            "docs/install.md",
            3,
            &["npm i other@0.5.1"],
            &["npm i other@0.5.2"],
        );
        files.push(file('M', "docs/install.md"));
        assert_eq!(subject(&files, &format!("{diff}{bumped_elsewhere}")), None);
        // a new file is not a version swap
        files.pop();
        files.push(file('A', "docs/0.6.0.md"));
        let added = hunk("docs/0.6.0.md", 0, &[], &["# 0.6.0"]);
        assert_eq!(subject(&files, &format!("{diff}{added}")), None);
    }

    #[test]
    fn versions_are_swapped_only_where_whole() {
        assert_eq!(
            swap_version("v0.5.1 and 0.5.1.", "0.5.1", "0.6.0"),
            "v0.6.0 and 0.6.0."
        );
        assert_eq!(
            swap_version("10.5.1, 0.5.12, 0.5.1.2", "0.5.1", "0.6.0"),
            "10.5.1, 0.5.12, 0.5.1.2"
        );
        assert_eq!(
            swap_version("pkg@0.5.1-beta", "0.5.1", "0.6.0"),
            "pkg@0.6.0-beta"
        );
    }

    #[test]
    fn renames_and_moves() {
        let d = draft(&[moved("src/util.rs", "src/helpers.rs", 100)], "").unwrap();
        assert_eq!(d.subject, "rename util.rs to helpers.rs");
        assert_eq!(d.kind, None);
        assert_eq!(
            subject(&[moved("src/util.rs", "src/core/util.rs", 100)], "").as_deref(),
            Some("move util.rs to src/core/")
        );
        let dir = [
            moved("lib/a/x.rs", "lib/b/x.rs", 100),
            moved("lib/a/y/z.rs", "lib/b/y/z.rs", 100),
        ];
        assert_eq!(
            subject(&dir, "").as_deref(),
            Some("rename lib/a/ to lib/b/")
        );
        // Edited while moving: not a plain rename.
        assert_eq!(subject(&[moved("a.rs", "b.rs", 87)], ""), None);
        // Scattered moves have nothing better than "rename 2 files" to say.
        let scattered = [
            moved("a/x.rs", "b/x.rs", 100),
            moved("c/y.rs", "d/z.rs", 100),
        ];
        assert_eq!(subject(&scattered, ""), None);
    }

    #[test]
    fn removals() {
        assert_eq!(
            subject(&[file('D', "scripts/old.sh")], "").as_deref(),
            Some("remove scripts/old.sh")
        );
        let many = [file('D', "legacy/a.py"), file('D', "legacy/b/c.py")];
        assert_eq!(
            subject(&many, "").as_deref(),
            Some("remove 2 files from legacy/")
        );
    }

    #[test]
    fn tests_docs_and_ci() {
        assert_eq!(
            subject(
                &[file('A', "packages/shadcn/src/registry/proxy.test.ts")],
                ""
            )
            .as_deref(),
            Some("add tests for proxy")
        );
        assert_eq!(
            subject(&[file('A', "tests/test_parser.py")], "").as_deref(),
            Some("add tests for parser")
        );
        assert_eq!(
            subject(&[file('A', "docs/install.md")], "").as_deref(),
            Some("add install docs")
        );
        assert_eq!(
            subject(&[file('A', ".github/workflows/release.yml")], "").as_deref(),
            Some("add release workflow")
        );
        let typo = hunk(
            "README.md",
            8,
            &["Run the instalation script."],
            &["Run the installation script."],
        );
        let d = draft(&[file('M', "README.md")], &typo).unwrap();
        assert_eq!(d.subject, "fix typo in README");
        assert_eq!(d.kind, Some("docs"));
    }

    #[test]
    fn generic_updates_get_no_draft() {
        // "update tests", "update README", "update release workflow": people
        // describe these in their own words.
        assert_eq!(subject(&[file('M', "tests/test_parser.py")], ""), None);
        assert_eq!(
            subject(
                &[file('M', "README.md")],
                &hunk("README.md", 3, &["a"], &["b c"])
            ),
            None
        );
        assert_eq!(
            subject(&[file('M', ".github/workflows/release.yml")], ""),
            None
        );
    }

    #[test]
    fn versions_and_markup_are_not_typos() {
        let version = hunk(
            "docs/schedule.md",
            3,
            &["Chromium v120 ships next."],
            &["Chromium v122 ships next."],
        );
        assert_eq!(subject(&[file('M', "docs/schedule.md")], &version), None);
        let ticks = hunk(
            "README.md",
            3,
            &["Run gca --help first."],
            &["Run `gca` --help first."],
        );
        assert_eq!(subject(&[file('M', "README.md")], &ticks), None);
    }

    #[test]
    fn github_actions_and_downgrades() {
        let action = hunk(
            ".github/workflows/ci.yml",
            12,
            &["      - uses: actions/checkout@v3"],
            &["      - uses: actions/checkout@v4"],
        );
        assert_eq!(
            subject(&[file('M', ".github/workflows/ci.yml")], &action).as_deref(),
            Some("bump actions/checkout from v3 to v4")
        );
        let pinned = hunk(
            ".github/workflows/ci.yml",
            12,
            &["      - uses: actions/setup-node@1a4442cacd436585916779262731d5b162bc6ec7 # v3.8.2"],
            &["      - uses: actions/setup-node@39370e3970a6d050c480ffad4ff0ed4d3fdee5af # v4.1.0"],
        );
        assert_eq!(
            subject(&[file('M', ".github/workflows/ci.yml")], &pinned).as_deref(),
            Some("bump actions/setup-node from v3.8.2 to v4.1.0")
        );
        let down = hunk(
            "package.json",
            20,
            &[r#"    "medusa-telemetry": "0.0.18","#],
            &[r#"    "medusa-telemetry": "0.0.17","#],
        );
        assert_eq!(
            subject(&[file('M', "package.json")], &down).as_deref(),
            Some("downgrade medusa-telemetry from 0.0.18 to 0.0.17")
        );
    }

    #[test]
    fn ordinary_code_gets_no_draft() {
        let diff = hunk("src/lib.rs", 10, &["    1"], &["    2"]);
        assert_eq!(subject(&[file('M', "src/lib.rs")], &diff), None);
        assert_eq!(subject(&[], ""), None);
    }

    #[test]
    fn edit_distance_counts_swaps_once() {
        assert_eq!(edit_distance("teh", "the"), 1);
        assert_eq!(edit_distance("recieve", "receive"), 1);
        assert_eq!(edit_distance("abc", "abd"), 1);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }
}
