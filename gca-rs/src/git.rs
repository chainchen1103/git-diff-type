use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub struct Stats {
    pub files_changed: u64,
    pub additions: u64,
    pub deletions: u64,
}

fn run(args: &[&str]) -> Result<String> {
    run_with_index(args, None)
}

fn run_with_index(args: &[&str], index: Option<&Path>) -> Result<String> {
    let mut command = Command::new("git");
    command.args(args);
    if let Some(index) = index {
        command.env("GIT_INDEX_FILE", index);
    }
    let out = command
        .output()
        .map_err(|e| anyhow!("failed to run git: {e}"))?;
    if !out.status.success() {
        return Err(anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn run_inherit(args: &[&str]) -> Result<()> {
    let status = Command::new("git")
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| anyhow!("failed to run git: {e}"))?;
    if !status.success() {
        return Err(anyhow!("git {} exited with {}", args.join(" "), status));
    }
    Ok(())
}

fn with_paths<'a>(mut args: Vec<&'a str>, paths: &'a [String]) -> Vec<&'a str> {
    if !paths.is_empty() {
        args.push("--");
        args.extend(paths.iter().map(String::as_str));
    }
    args
}

pub fn staged_diff(paths: &[String]) -> Result<String> {
    diff_with_index(paths, None)
}

fn diff_with_index(paths: &[String], index: Option<&Path>) -> Result<String> {
    let args = with_paths(
        vec![
            "diff",
            "--cached",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--no-relative",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "--unified=3",
            "--output-indicator-new=+",
            "--output-indicator-old=-",
            "--output-indicator-context= ",
        ],
        paths,
    );
    Ok(run_with_index(&args, index)?.trim().to_string())
}

/// Staged file names and line counts from a single `git diff --numstat -z`.
pub fn staged_summary(paths: &[String]) -> Result<(Vec<String>, Stats)> {
    summary_with_index(paths, None)
}

fn summary_with_index(paths: &[String], index: Option<&Path>) -> Result<(Vec<String>, Stats)> {
    let args = with_paths(
        vec![
            "diff",
            "--cached",
            "--numstat",
            "-z",
            "--no-relative",
            "--no-ext-diff",
            "--no-textconv",
        ],
        paths,
    );
    Ok(parse_numstat_z(&run_with_index(&args, index)?))
}

pub fn preview(paths: &[String]) -> Result<(String, Vec<String>, Stats)> {
    let index_path = PathBuf::from(run(&["rev-parse", "--git-path", "index"])?.trim());
    let temp_dir = tempfile::tempdir().context("failed to create preview directory")?;
    let preview_index = temp_dir.path().join("index");
    if index_path.exists() {
        std::fs::copy(&index_path, &preview_index).context("failed to copy Git index")?;
    }
    let index = Some(preview_index.as_path());
    if !paths.is_empty() {
        run_with_index(&with_paths(vec!["add"], paths), index)?;
    }
    let mut diff = diff_with_index(paths, index)?;
    if diff.is_empty() && paths.is_empty() {
        run_with_index(&["add", "-A"], index)?;
        diff = diff_with_index(paths, index)?;
    }
    let (files, stats) = summary_with_index(paths, index)?;
    Ok((diff, files, stats))
}

// Records are `adds\tdels\tpath\0`, or `adds\tdels\t\0old\0new\0` for a
// rename. Binary files report `-` for both counts.
fn parse_numstat_z(out: &str) -> (Vec<String>, Stats) {
    let mut files = Vec::new();
    let mut stats = Stats {
        files_changed: 0,
        additions: 0,
        deletions: 0,
    };
    let mut fields = out.split('\0');
    while let Some(record) = fields.next() {
        let mut parts = record.splitn(3, '\t');
        let (Some(adds), Some(dels), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let path = if path.is_empty() {
            let (Some(old), Some(new)) = (fields.next(), fields.next()) else {
                break;
            };
            if old.is_empty() || new.is_empty() {
                continue;
            }
            new
        } else {
            path
        };
        stats.files_changed += 1;
        stats.additions = stats
            .additions
            .saturating_add(adds.parse::<u64>().unwrap_or(0));
        stats.deletions = stats
            .deletions
            .saturating_add(dels.parse::<u64>().unwrap_or(0));
        files.push(path.to_string());
    }
    (files, stats)
}

pub fn add_all() -> Result<()> {
    run_inherit(&["add", "-A"])
}

pub fn add_paths(paths: &[String]) -> Result<()> {
    run_inherit(&with_paths(vec!["add"], paths))
}

pub fn add_dry_run(paths: &[String]) -> Result<String> {
    if paths.is_empty() {
        run(&["add", "--dry-run", "-A"])
    } else {
        run(&with_paths(vec!["add", "--dry-run"], paths))
    }
}

/// With paths, only those paths are committed and other staged changes stay staged.
pub fn commit(message: &str, paths: &[String]) -> Result<()> {
    run_inherit(&with_paths(vec!["commit", "-m", message], paths))
}

pub fn push(remote: Option<&str>) -> Result<()> {
    match remote {
        Some(r) => run_inherit(&["push", "--", r]),
        None => run_inherit(&["push"]),
    }
}

pub fn get_config(key: &str) -> Option<String> {
    let out = Command::new("git")
        .args(["config", "--get", key])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

pub fn set_config_global(key: &str, value: &str) -> Result<()> {
    run_inherit(&["config", "--global", key, value])
}

#[cfg(test)]
mod tests {
    use super::parse_numstat_z;

    #[test]
    fn numstat_z_plain_binary_and_rename() {
        let out = "3\t1\tsrc/a.rs\0-\t-\tlogo.png\x000\t2\t\0old/b.rs\0new/b.rs\0";
        let (files, stats) = parse_numstat_z(out);
        assert_eq!(files, ["src/a.rs", "logo.png", "new/b.rs"]);
        assert_eq!(stats.files_changed, 3);
        assert_eq!(stats.additions, 3);
        assert_eq!(stats.deletions, 3);
    }

    #[test]
    fn numstat_z_empty() {
        let (files, stats) = parse_numstat_z("");
        assert!(files.is_empty());
        assert_eq!(stats.files_changed, 0);
    }

    #[test]
    fn numstat_z_preserves_tabs_and_newlines_in_paths() {
        let (files, stats) = parse_numstat_z("1\t2\tsrc/tab\tline\n.rs\0");
        assert_eq!(files, ["src/tab\tline\n.rs"]);
        assert_eq!(stats.additions, 1);
        assert_eq!(stats.deletions, 2);
    }

    #[test]
    fn numstat_z_ignores_incomplete_rename() {
        let (files, stats) = parse_numstat_z("1\t2\t\0old.rs\0");
        assert!(files.is_empty());
        assert_eq!(stats.files_changed, 0);
    }

    #[test]
    fn numstat_z_large_totals() {
        let (_, stats) = parse_numstat_z("4294967295\t0\ta\x001\t0\tb\0");
        assert_eq!(stats.additions, 4294967296);
    }
}
