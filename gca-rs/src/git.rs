//! Everything gca asks of git. Reads use machine-readable output; commit and
//! push inherit the terminal so hooks, editors and credential prompts work.

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::history::LogEntry;

pub struct Stats {
    pub files_changed: u64,
    pub additions: u64,
    pub deletions: u64,
}

/// A staged file: git's status letter (A, M, D, R, ...) and its path.
pub struct FileChange {
    pub status: char,
    pub path: String,
    /// For a rename or copy: where it came from, and how similar it is (0-100).
    pub old_path: Option<String>,
    pub score: Option<u8>,
}

/// Settings that change what `git diff` prints. The model was trained on
/// git's defaults, so these are pinned whatever the user has configured.
/// Keep in sync with DIFF_CONFIG in miner.py; staged_diff() matches its DIFF_FLAGS.
const DIFF_CONFIG: [&str; 10] = [
    "-c",
    "diff.noprefix=false",
    "-c",
    "diff.mnemonicPrefix=false",
    "-c",
    "diff.relative=false",
    "-c",
    "core.quotePath=true",
    // A moved file is one rename, not a deletion plus an addition.
    "-c",
    "diff.renames=true",
];

fn git(index: Option<&Path>) -> Command {
    let mut cmd = Command::new("git");
    if let Some(index) = index {
        cmd.env("GIT_INDEX_FILE", index);
    }
    cmd
}

fn run(index: Option<&Path>, args: &[&str]) -> Result<String> {
    let out = git(index)
        .args(args)
        .output()
        .map_err(|e| anyhow!("could not run git: {e}"))?;
    if !out.status.success() {
        bail!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
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
        .map_err(|e| anyhow!("could not run git: {e}"))?;
    if !status.success() {
        bail!("git {} exited with {status}", args.first().unwrap_or(&""));
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

pub fn ensure_work_tree() -> Result<()> {
    match run(None, &["rev-parse", "--is-inside-work-tree"]) {
        Ok(out) if out.trim() == "true" => Ok(()),
        _ => bail!("not inside a git working tree"),
    }
}

/// Absolute path of a file inside the git directory, such as `index`.
fn git_path(name: &str) -> Result<PathBuf> {
    let out = run(None, &["rev-parse", "--git-path", name])?;
    let path = PathBuf::from(out.trim());
    Ok(if path.is_absolute() {
        path
    } else {
        std::env::current_dir()?.join(path)
    })
}

/// A merge, cherry-pick or revert that `git commit` should conclude, since
/// git has already prepared its message.
pub fn operation_in_progress() -> Result<Option<&'static str>> {
    for (file, name) in [
        ("MERGE_HEAD", "merge"),
        ("CHERRY_PICK_HEAD", "cherry-pick"),
        ("REVERT_HEAD", "revert"),
    ] {
        if git_path(file)?.exists() {
            return Ok(Some(name));
        }
    }
    Ok(None)
}

/// Whether git is replaying commits (a merge, cherry-pick, revert or rebase),
/// which keep their original messages.
pub fn replaying() -> Result<bool> {
    if operation_in_progress()?.is_some() {
        return Ok(true);
    }
    for dir in ["rebase-merge", "rebase-apply", "sequencer"] {
        if git_path(dir)?.exists() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The top directory of the work tree.
pub fn toplevel() -> Result<PathBuf> {
    Ok(PathBuf::from(
        run(None, &["rev-parse", "--show-toplevel"])?.trim(),
    ))
}

/// Where git looks for a hook: `.git/hooks`, or core.hooksPath, which a
/// relative path means from the top of the work tree.
pub fn hook_path(name: &str) -> Result<PathBuf> {
    let top = toplevel()?;
    let top_arg = top.to_string_lossy();
    let rel = format!("hooks/{name}");
    let out = run(None, &["-C", &top_arg, "rev-parse", "--git-path", &rel])?;
    let path = PathBuf::from(out.trim());
    Ok(if path.is_absolute() {
        path
    } else {
        top.join(path)
    })
}

/// A throwaway copy of the index. Previewing `-a` or a list of paths stages
/// into this copy, so the user's own index is never touched before the
/// commit is confirmed. Removed when dropped.
pub struct ScratchIndex {
    path: PathBuf,
}

impl ScratchIndex {
    pub fn new() -> Result<Self> {
        // Honour a custom index the user pointed git at.
        let real = match std::env::var_os("GIT_INDEX_FILE").filter(|v| !v.is_empty()) {
            Some(custom) => std::env::current_dir()?.join(custom),
            None => git_path("index")?,
        };
        let path = real.with_file_name(format!("gca-index-{}", std::process::id()));
        if real.exists() {
            copy_index(&real, &path)?;
        }
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Copies the index and keeps its modification time. Git trusts an entry
/// whose file looks older than the index, so a copy that looks newer would
/// hide a change made as soon after `git add` as the file system can tell.
fn copy_index(from: &Path, to: &Path) -> Result<()> {
    std::fs::copy(from, to)
        .with_context(|| format!("could not copy the index to {}", to.display()))?;
    let time = std::fs::metadata(from).and_then(|m| m.modified());
    let file = std::fs::File::options().write(true).open(to);
    if let (Ok(time), Ok(file)) = (time, file) {
        let _ = file.set_modified(time);
    }
    Ok(())
}

impl Drop for ScratchIndex {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// `git add -u`: modified and deleted tracked files, as `git commit -a` would.
pub fn add_tracked(index: &Path) -> Result<()> {
    run(Some(index), &["add", "-u"]).map(|_| ())
}

pub fn add_paths(index: Option<&Path>, paths: &[String]) -> Result<()> {
    run(index, &with_paths(vec!["add"], paths)).map(|_| ())
}

/// The staged diff, formatted the way the training data was.
pub fn staged_diff(index: Option<&Path>, paths: &[String]) -> Result<String> {
    let mut args = DIFF_CONFIG.to_vec();
    args.extend([
        "diff",
        "--cached",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--unified=3",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--output-indicator-new=+",
        "--output-indicator-old=-",
        "--output-indicator-context= ",
    ]);
    Ok(run(index, &with_paths(args, paths))?.trim().to_string())
}

pub fn staged_files(index: Option<&Path>, paths: &[String]) -> Result<Vec<FileChange>> {
    let mut args = DIFF_CONFIG.to_vec();
    args.extend(["diff", "--cached", "--name-status", "-z"]);
    Ok(parse_name_status_z(&run(index, &with_paths(args, paths))?))
}

pub fn staged_stats(index: Option<&Path>, paths: &[String]) -> Result<Stats> {
    let mut args = DIFF_CONFIG.to_vec();
    args.extend(["diff", "--cached", "--numstat", "-z"]);
    Ok(parse_numstat_z(&run(index, &with_paths(args, paths))?))
}

// Records are `X\0path\0`, or `R100\0old\0new\0` for renames and copies.
fn parse_name_status_z(out: &str) -> Vec<FileChange> {
    let mut files = Vec::new();
    let mut fields = out.split('\0');
    while let Some(status) = fields.next() {
        let Some(letter) = status.chars().next() else {
            continue;
        };
        let first = fields.next().unwrap_or_default();
        let (path, old_path, score) = if letter == 'R' || letter == 'C' {
            let score = status[1..].parse::<u8>().ok();
            (
                fields.next().unwrap_or_default(),
                Some(first.to_string()),
                score,
            )
        } else {
            (first, None, None)
        };
        files.push(FileChange {
            status: letter,
            path: path.to_string(),
            old_path,
            score,
        });
    }
    files
}

// Records are `adds\tdels\tpath\0`, or `adds\tdels\t\0old\0new\0` for a
// rename. Binary files report `-` for both counts.
fn parse_numstat_z(out: &str) -> Stats {
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
        if path.is_empty() {
            // Rename or copy: the old and new paths follow as two fields.
            let (Some(old), Some(new)) = (fields.next(), fields.next()) else {
                break;
            };
            if old.is_empty() || new.is_empty() {
                continue;
            }
        }
        stats.files_changed += 1;
        stats.additions = stats
            .additions
            .saturating_add(adds.parse::<u64>().unwrap_or(0));
        stats.deletions = stats
            .deletions
            .saturating_add(dels.parse::<u64>().unwrap_or(0));
    }
    stats
}

/// `git status --short`, for explaining why there is nothing to commit.
/// `--no-optional-locks` keeps it from rewriting the index.
pub fn status_short() -> Result<String> {
    run(
        None,
        &[
            "--no-optional-locks",
            "-c",
            "color.status=never",
            "status",
            "--short",
        ],
    )
}

/// Subjects and changed files of the most recent non-merge commits. Empty
/// before the first commit.
///
/// Only files moved unchanged are paired as renames: pairing the edited ones
/// compares the contents of every added file with every deleted one, which
/// took 0.37 s of 0.42 s for the last 500 commits of shadcn-ui. A file moved
/// and edited is listed under both paths.
pub fn recent_log(depth: usize) -> Vec<LogEntry> {
    let n = depth.to_string();
    let args = [
        "-c",
        "core.quotePath=false",
        "-c",
        "diff.relative=false",
        "-c",
        "log.showSignature=false",
        "log",
        "--no-merges",
        "-n",
        &n,
        "--format=%x1e%H%x1f%an <%ae>%x1f%s",
        "--name-only",
        "--find-renames=100%",
    ];
    run(None, &args)
        .map(|out| parse_log(&out))
        .unwrap_or_default()
}

fn parse_log(out: &str) -> Vec<LogEntry> {
    out.split('\x1e')
        .filter_map(|record| {
            let mut lines = record.lines();
            let mut header = lines.next()?.splitn(3, '\x1f');
            let (sha, author, subject) = (header.next()?, header.next()?, header.next()?);
            let subject = subject.trim();
            if subject.is_empty() {
                return None;
            }
            let files = lines
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect();
            Some(LogEntry {
                sha: sha.to_string(),
                author: author.to_string(),
                subject: subject.to_string(),
                files,
            })
        })
        .collect()
}

/// Who the next commit will be by, as `Name <email>`: user.name and
/// user.email, or GIT_AUTHOR_NAME and GIT_AUTHOR_EMAIL. `None` when git
/// knows no identity.
pub fn author_ident() -> Option<String> {
    let out = run(None, &["var", "GIT_AUTHOR_IDENT"]).ok()?;
    // Name <email> 1727680000 +0800
    let end = out.rfind('>')?;
    Some(out[..=end].trim().to_string())
}

/// A commit's change as the model reads it: the patch, cut at `max_chars`
/// as the training data was, and its line counts.
pub struct CommitChange {
    pub diff: String,
    pub stats: Stats,
}

/// The changes the given commits made, keyed by hash. Commits at the edge of
/// a shallow clone are left out: git shows them as adding every file.
pub fn commit_changes(shas: &[&str], max_chars: usize) -> Result<HashMap<String, CommitChange>> {
    let shallow: Vec<String> = git_path("shallow")
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| s.lines().map(|l| l.trim().to_string()).collect())
        .unwrap_or_default();
    let shas: Vec<&str> = shas
        .iter()
        .copied()
        .filter(|s| !shallow.iter().any(|b| b == s))
        .collect();
    if shas.is_empty() {
        return Ok(HashMap::new());
    }
    let mut args = DIFF_CONFIG.to_vec();
    args.extend([
        "-c",
        "log.showSignature=false",
        "show",
        "--format=%x1e%H",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--unified=3",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--output-indicator-new=+",
        "--output-indicator-old=-",
        "--output-indicator-context= ",
    ]);
    args.extend(&shas);
    let mut child = git(None)
        .args(&args)
        // In a partial clone, fail rather than download the old contents.
        .env("GIT_NO_LAZY_FETCH", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| anyhow!("could not run git: {e}"))?;
    let stdout = child.stdout.take().context("no output from git show")?;
    let changes = read_patches(std::io::BufReader::new(stdout), max_chars);
    if !child.wait()?.success() {
        bail!("`git show` failed");
    }
    Ok(changes)
}

/// Each commit's patch, from `\x1e<hash>` lines on. Only the first
/// `max_chars` (and a margin) of a patch are kept, however large the commit,
/// and the text is cut as miner.py cuts it. The line counts are those
/// `--numstat` gives: files, and added and removed lines in the hunks.
fn read_patches(
    mut reader: impl std::io::BufRead,
    max_chars: usize,
) -> HashMap<String, CommitChange> {
    struct Reading {
        sha: String,
        text: String,
        chars: usize,
        stats: Stats,
        in_hunk: bool,
    }
    let finish = |r: Reading| {
        let diff = cut(&r.text, max_chars);
        (
            r.sha,
            CommitChange {
                diff,
                stats: r.stats,
            },
        )
    };
    let keep = max_chars + 1000;
    let mut changes = HashMap::new();
    let mut current: Option<Reading> = None;
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if let Some(sha) = line.strip_prefix(b"\x1e") {
            if let Some(done) = current.take() {
                let (sha, change) = finish(done);
                changes.insert(sha, change);
            }
            current = Some(Reading {
                sha: String::from_utf8_lossy(sha).trim().to_string(),
                text: String::new(),
                chars: 0,
                stats: Stats {
                    files_changed: 0,
                    additions: 0,
                    deletions: 0,
                },
                in_hunk: false,
            });
            continue;
        }
        let Some(r) = current.as_mut() else {
            continue;
        };
        if line.starts_with(b"diff --git ") {
            r.stats.files_changed += 1;
            r.in_hunk = false;
        } else if line.starts_with(b"@@") {
            r.in_hunk = true;
        } else if r.in_hunk {
            match line.first() {
                Some(b'+') => r.stats.additions += 1,
                Some(b'-') => r.stats.deletions += 1,
                _ => {}
            }
        }
        if r.chars <= keep {
            let text = String::from_utf8_lossy(&line);
            r.chars += text.chars().count();
            r.text.push_str(&text);
        }
    }
    if let Some(done) = current {
        let (sha, change) = finish(done);
        changes.insert(sha, change);
    }
    changes
}

fn cut(text: &str, max_chars: usize) -> String {
    text.trim().chars().take(max_chars).collect()
}

pub struct CommitRequest<'a> {
    pub header: &'a str,
    pub body: &'a [String],
    /// `git commit -a`
    pub all: bool,
    /// Commit only these paths (`git commit -- <paths>`).
    pub paths: &'a [String],
    pub edit: bool,
    pub no_verify: bool,
    pub signoff: bool,
}

/// Runs `git commit` with the terminal attached, so hooks print their output
/// and `--edit` can open the editor. Each body paragraph is another `-m`.
pub fn commit(req: &CommitRequest) -> Result<()> {
    let mut args = vec!["commit", "-m", req.header];
    for paragraph in req.body {
        args.extend(["-m", paragraph.as_str()]);
    }
    if req.all {
        args.push("-a");
    }
    if req.edit {
        args.push("--edit");
    }
    if req.no_verify {
        args.push("--no-verify");
    }
    if req.signoff {
        args.push("--signoff");
    }
    run_inherit(&with_paths(args, req.paths))
}

fn current_branch() -> Option<String> {
    run(None, &["symbolic-ref", "--quiet", "--short", "HEAD"])
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn remotes() -> Vec<String> {
    run(None, &["remote"])
        .map(|out| {
            out.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Where a branch without an upstream should go: `remote.pushDefault`, then
/// `origin`, then the only remote there is.
fn default_remote() -> Result<String> {
    if let Some(r) = get_config("remote.pushDefault") {
        return Ok(r);
    }
    let all = remotes();
    if all.iter().any(|r| r == "origin") {
        return Ok("origin".into());
    }
    match all.as_slice() {
        [only] => Ok(only.clone()),
        [] => bail!("this repository has no remote to push to"),
        _ => bail!("several remotes and no default; choose one with --remote or `gca config remote <name>`"),
    }
}

/// Pushes the current branch. A branch without an upstream gets one
/// (`git push -u <remote> HEAD`); pushing to a remote other than the upstream
/// leaves the upstream alone.
pub fn push(remote: Option<&str>) -> Result<()> {
    let Some(branch) = current_branch() else {
        bail!("HEAD is detached; push it yourself with `git push <remote> HEAD:<branch>`");
    };
    let upstream = get_config(&format!("branch.{branch}.remote"))
        .filter(|_| get_config(&format!("branch.{branch}.merge")).is_some());
    match (remote, upstream) {
        (None, Some(_)) => run_inherit(&["push"]),
        (Some(r), Some(up)) if r == up => run_inherit(&["push", r]),
        (Some(r), Some(_)) => run_inherit(&["push", r, "HEAD"]),
        (r, None) => {
            let r = match r {
                Some(r) => r.to_string(),
                None => default_remote()?,
            };
            run_inherit(&["push", "--set-upstream", &r, "HEAD"])
        }
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

pub fn set_config(key: &str, value: &str, local: bool) -> Result<()> {
    let scope = if local { "--local" } else { "--global" };
    run(None, &["config", scope, key, value]).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_index_copy_keeps_its_time() {
        let dir = tempfile::tempdir().unwrap();
        let (from, to) = (dir.path().join("index"), dir.path().join("copy"));
        std::fs::write(&from, b"DIRC").unwrap();
        // whole 100 ns, which every file system CI runs on can store
        let past = std::time::UNIX_EPOCH + std::time::Duration::new(1_700_000_000, 123_456_700);
        std::fs::File::options()
            .write(true)
            .open(&from)
            .unwrap()
            .set_modified(past)
            .unwrap();
        copy_index(&from, &to).unwrap();
        assert_eq!(std::fs::read(&to).unwrap(), b"DIRC");
        assert_eq!(std::fs::metadata(&to).unwrap().modified().unwrap(), past);
    }

    #[test]
    fn numstat_z_plain_binary_and_rename() {
        let out = "3\t1\tsrc/a.rs\0-\t-\tlogo.png\x000\t2\t\0old/b.rs\0new/b.rs\0";
        let stats = parse_numstat_z(out);
        assert_eq!(stats.files_changed, 3);
        assert_eq!(stats.additions, 3);
        assert_eq!(stats.deletions, 3);
    }

    #[test]
    fn numstat_z_empty() {
        assert_eq!(parse_numstat_z("").files_changed, 0);
    }

    #[test]
    fn name_status_z_with_rename() {
        let out = "M\0src/a.rs\0R087\0old/b.rs\0new/b.rs\0A\0c.txt\0D\0gone.md\0";
        let files = parse_name_status_z(out);
        let got: Vec<(char, &str)> = files.iter().map(|f| (f.status, f.path.as_str())).collect();
        assert_eq!(
            got,
            [
                ('M', "src/a.rs"),
                ('R', "new/b.rs"),
                ('A', "c.txt"),
                ('D', "gone.md")
            ]
        );
    }

    #[test]
    fn log_records() {
        let out = "\x1eaaa\x1fAda <a@x>\x1ffeat(cli): add -a\n\nsrc/main.rs\nREADME.md\n\x1ebbb\x1fBo <b@x>\x1fdocs: typo\n\nREADME.md\n\x1eccc\x1fCy <c@x>\x1fempty commit\n";
        let log = parse_log(out);
        assert_eq!(log.len(), 3);
        assert_eq!(log[0].sha, "aaa");
        assert_eq!(log[0].author, "Ada <a@x>");
        assert_eq!(log[0].subject, "feat(cli): add -a");
        assert_eq!(log[0].files, ["src/main.rs", "README.md"]);
        assert_eq!(log[1].files, ["README.md"]);
        assert!(log[2].files.is_empty());
    }

    #[test]
    fn patches_of_several_commits_are_cut_like_the_training_data() {
        let long: String = (0..50).map(|i| format!("+line {i}\n")).collect();
        let out = format!(
            "\x1eaaa\n\ndiff --git a/f b/f\nnew file mode 100644\n--- /dev/null\n+++ b/f\n@@ -0,0 +1,50 @@\n{long}\
             \x1ebbb\n\ndiff --git a/g b/h\nsimilarity index 90%\nrename from g\nrename to h\n--- a/g\n+++ b/h\n\
             @@ -1,3 +1,3 @@\n a\n--b\n+++b\n c\ndiff --git a/i.png b/i.png\nBinary files a/i.png and b/i.png differ\n"
        );
        let changes = read_patches(std::io::Cursor::new(out), 40);
        let a = &changes["aaa"];
        assert_eq!(a.diff.chars().count(), 40);
        assert!(a.diff.starts_with("diff --git a/f b/f\nnew file mode"));
        assert_eq!(
            (a.stats.files_changed, a.stats.additions, a.stats.deletions),
            (1, 50, 0)
        );
        // "--b" and "+++b" are lines of the file, not headers
        let b = &changes["bbb"];
        assert_eq!(
            (b.stats.files_changed, b.stats.additions, b.stats.deletions),
            (2, 1, 1)
        );
    }
}
