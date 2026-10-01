//! What the subject model reads, exactly as the training data had it
//! (draft_model/t5_format.py and draft_model/prepare.py): the type and scope
//! chosen before the subject, the files, the headers of a few earlier
//! commits, and the changed lines, cut to fit the model's 512-token input.

use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

use crate::history::{same_person, LogEntry};
use crate::message;
use crate::subject;

/// The input is cut at this many characters.
pub const MAX_CHARS: usize = 2000;
/// Files listed by name; the rest are counted.
const MAX_FILES: usize = 12;
/// Each changed line is cut at this many characters.
const MAX_LINE: usize = 160;
/// Each earlier commit's header is cut at this many characters.
const MAX_HISTORY: usize = 100;
/// Earlier commits picked by their files (then by type).
const BY_FILES: usize = 3;

static GIT_HEADER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^diff --git a/(.+?) b/(.+)$").unwrap());
/// The type of an earlier commit as the training data labeled it
/// (CONVENTIONAL_RE in miner.py): unlike message::parse_subject, it also
/// takes `fix:  two spaces`. The scope still needs the stricter form.
static LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(?:\([^)]+\))?!?:\s.",
    )
    .unwrap()
});

struct File<'a> {
    status: char,
    path: &'a str,
    old: Option<&'a str>,
    body: Vec<String>,
}

/// The model's input for a change: its diff (as `git diff` prints it), the
/// chosen type and scope, and the headers from [`history`].
pub fn format(diff: &str, kind: &str, scope: Option<&str>, history: &[String]) -> String {
    let mut files: Vec<File> = Vec::new();
    let mut in_hunk = false;
    for line in diff.split('\n') {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            let path = match GIT_HEADER.captures(line) {
                Some(caps) => caps.get(2).map_or("", |m| m.as_str()),
                None => rest,
            };
            files.push(File {
                status: 'M',
                path,
                old: None,
                body: Vec::new(),
            });
            in_hunk = false;
            continue;
        }
        let Some(cur) = files.last_mut() else {
            continue;
        };
        if line.starts_with("@@") {
            in_hunk = true;
            let parts: Vec<&str> = line.splitn(3, "@@").collect();
            let context = if parts.len() == 3 {
                py_strip(parts[2])
            } else {
                ""
            };
            cur.body.push(if context.is_empty() {
                "@@".to_string()
            } else {
                format!("@@ {context}")
            });
        } else if in_hunk {
            if line.starts_with('+') || line.starts_with('-') {
                let text = py_strip(&line[1..]);
                if !text.is_empty() {
                    cur.body
                        .push(format!("{} {}", &line[..1], take_chars(text, MAX_LINE)));
                }
            }
        } else if line.starts_with("new file mode") {
            cur.status = 'A';
        } else if line.starts_with("deleted file mode") {
            cur.status = 'D';
        } else if let Some(old) = line.strip_prefix("rename from ") {
            cur.status = 'R';
            cur.old = Some(old);
        } else if line.starts_with("Binary files") {
            cur.body.push("(binary)".to_string());
        }
    }

    let mut names: Vec<String> = files
        .iter()
        .take(MAX_FILES)
        .map(|f| match f.old.filter(|o| !o.is_empty()) {
            Some(old) => format!("{} {old} -> {}", f.status, f.path),
            None => format!("{} {}", f.status, f.path),
        })
        .collect();
    if files.len() > MAX_FILES {
        names.push(format!("and {} more", files.len() - MAX_FILES));
    }
    let scope = scope.filter(|s| !s.is_empty()).unwrap_or("none");
    let mut text = format!("type: {kind}\nscope: {scope}\nfiles: {}", names.join(" | "));
    if !history.is_empty() {
        text.push_str("\nhistory:");
        for h in history {
            text.push_str("\n- ");
            text.push_str(take_chars(h, MAX_HISTORY));
        }
    }
    let mut len = text.chars().count();
    for f in &files {
        let head = format!("--- {}", f.path);
        for piece in std::iter::once(&head).chain(&f.body) {
            let n = piece.chars().count();
            if len + 1 + n > MAX_CHARS {
                return take_chars(&text, MAX_CHARS).to_string();
            }
            text.push('\n');
            text.push_str(piece);
            len += 1 + n;
        }
    }
    take_chars(&text, MAX_CHARS).to_string()
}

/// The headers of up to four earlier commits for the model's input, picked
/// from the recent log (newest first) as the training data picked them: up
/// to three that touched a staged file (the same type first, then the
/// larger overlap of files, then the newer), the newest of the same type if
/// fewer than three did, and your own newest commit if not picked already.
/// Only commits with a Conventional Commit type count.
pub fn history(log: &[LogEntry], staged: &[String], kind: &str, me: Option<&str>) -> Vec<String> {
    struct Typed<'a> {
        kind: &'a str,
        header: String,
        files: HashSet<&'a str>,
        author: &'a str,
    }
    let typed: Vec<Typed> = log
        .iter()
        .filter_map(|e| {
            let kind = LABEL.captures(&e.subject)?.get(1)?.as_str();
            let scope = message::parse_subject(&e.subject).and_then(|(_, scope)| scope);
            let subject = subject::normalize(&e.subject);
            if subject.is_empty() {
                return None;
            }
            Some(Typed {
                kind,
                header: match scope {
                    Some(scope) => format!("{kind}({scope}): {subject}"),
                    None => format!("{kind}: {subject}"),
                },
                files: e.files.iter().map(String::as_str).collect(),
                author: &e.author,
            })
        })
        .collect();
    let mine: HashSet<&str> = staged.iter().map(String::as_str).collect();

    // (same type, overlap, index): the index is the age, 0 the newest
    let mut sharing: Vec<(bool, f64, usize)> = typed
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let shared = c.files.iter().filter(|f| mine.contains(*f)).count();
            (shared > 0).then(|| {
                let union = mine.len() + c.files.len() - shared;
                (c.kind == kind, shared as f64 / union as f64, i)
            })
        })
        .collect();
    sharing.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)).then(a.2.cmp(&b.2)));
    let mut picked: Vec<usize> = sharing.iter().take(BY_FILES).map(|s| s.2).collect();
    if picked.len() < BY_FILES {
        for (i, c) in typed.iter().enumerate() {
            if picked.len() == BY_FILES {
                break;
            }
            if c.kind == kind && !picked.contains(&i) {
                picked.push(i);
            }
        }
    }
    if let Some(me) = me {
        if let Some(i) = typed.iter().position(|c| same_person(c.author, me)) {
            if !picked.contains(&i) {
                picked.push(i);
            }
        }
    }
    picked
        .into_iter()
        .map(|i| typed[i].header.clone())
        .collect()
}

/// How many of the typed commits in the recent log have this subject,
/// compared as the evaluation compares subjects: case, spacing and a final
/// period aside.
#[cfg_attr(not(feature = "t5"), allow(dead_code))]
pub fn repeats(log: &[LogEntry], subject: &str) -> usize {
    let wanted = loose(subject);
    log.iter()
        .filter(|e| LABEL.is_match(&e.subject))
        .filter(|e| loose(&subject::normalize(&e.subject)) == wanted)
        .count()
}

#[cfg_attr(not(feature = "t5"), allow(dead_code))]
fn loose(s: &str) -> String {
    let s = s
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    match s.strip_suffix('.') {
        Some(t) => t.to_string(),
        None => s,
    }
}

/// The first `n` characters of `s`.
fn take_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `str.strip()` as Python does it: Unicode white space and also the
/// separator controls U+001C to U+001F.
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    fn entry(author: &str, subject: &str, files: &[&str]) -> LogEntry {
        LogEntry {
            sha: String::new(),
            author: author.into(),
            subject: subject.into(),
            files: files.iter().map(|f| f.to_string()).collect(),
        }
    }

    #[test]
    fn lays_out_type_scope_files_history_and_lines() {
        let diff = "diff --git a/src/a.rs b/src/a.rs\nindex 1..2 100644\n--- a/src/a.rs\n+++ b/src/a.rs\n\
                    @@ -1,3 +1,3 @@ fn main() {\n context\n-    old();\n+    new();\n\
                    diff --git a/b.txt b/c.txt\nsimilarity index 90%\nrename from b.txt\nrename to c.txt\n\
                    diff --git a/img.png b/img.png\nnew file mode 100644\nBinary files /dev/null and b/img.png differ";
        let got = format(diff, "fix", Some("cli"), &["feat(cli): add -a".to_string()]);
        assert_eq!(
            got,
            "type: fix\nscope: cli\nfiles: M src/a.rs | R b.txt -> c.txt | A img.png\n\
             history:\n- feat(cli): add -a\n--- src/a.rs\n@@ fn main() {\n- old();\n+ new();\n\
             --- c.txt\n--- img.png\n(binary)"
        );
        assert!(format(diff, "fix", None, &[]).starts_with("type: fix\nscope: none\nfiles: "));
    }

    #[test]
    fn cuts_by_characters_not_bytes() {
        let line = "é".repeat(500);
        let diff = format!("diff --git a/x b/x\n@@ -1 +1 @@\n+{line}\n+{line}\n");
        let got = format(&diff, "docs", None, &[]);
        let added: Vec<&str> = got.lines().filter(|l| l.starts_with("+ ")).collect();
        assert_eq!(added.len(), 2);
        assert_eq!(added[0].chars().count(), 2 + MAX_LINE);
        let many: String = (0..400).map(|i| format!("+línea {i}\n")).collect();
        let got = format(
            &format!("diff --git a/x b/x\n@@ -1 +1 @@\n{many}"),
            "docs",
            None,
            &[],
        );
        assert!(got.chars().count() <= MAX_CHARS);
        assert!(got.ends_with(char::is_numeric));
    }

    #[test]
    fn counts_the_commits_that_had_a_subject() {
        let log = [
            entry("Bo <bo@x>", "chore: bump version", &[]),
            entry("Bo <bo@x>", "chore(release): Bump  version.", &[]),
            entry("Bo <bo@x>", "bump version", &[]), // no type: not counted
            entry("Bo <bo@x>", "fix: handle empty input (#12)", &[]),
        ];
        assert_eq!(repeats(&log, "bump version"), 2);
        assert_eq!(repeats(&log, "Handle empty input"), 1);
        assert_eq!(repeats(&log, "something new"), 0);
    }

    #[test]
    fn strips_like_python() {
        assert_eq!(py_strip("\u{1c}\u{3000} x \u{1f}\r"), "x");
        assert_eq!(py_strip("\u{200b}x"), "\u{200b}x"); // zero width space is not white space
    }

    #[test]
    fn picks_history_by_files_type_and_author() {
        let log = [
            entry("Bo <bo@x>", "docs: newest, other files", &["README.md"]),
            entry(
                "Bo <bo@x>",
                "feat(cli): same files, other type",
                &["src/cli.rs"],
            ),
            entry("Ada <ada@x>", "WIP untyped", &["src/cli.rs"]),
            entry(
                "Bo <bo@x>",
                "fix(cli): same type, half the files",
                &["src/cli.rs", "src/x.rs"],
            ),
            entry("Bo <bo@x>", "fix: same type, no shared file", &["src/y.rs"]),
            entry(
                "Ada <ada@x>",
                "chore: my newest typed commit",
                &["Cargo.lock"],
            ),
        ];
        let staged = vec!["src/cli.rs".to_string()];
        assert_eq!(
            history(&log, &staged, "fix", Some("Ada <ada@x>")),
            [
                "fix(cli): same type, half the files",
                "feat(cli): same files, other type",
                "fix: same type, no shared file",
                "chore: my newest typed commit",
            ]
        );
        // without a sharing commit: the newest of the type; no author given
        assert_eq!(
            history(&log, &["new.rs".to_string()], "docs", None),
            ["docs: newest, other files"]
        );
        // a header with two spaces has a type but, like in training, no scope
        let spaced = [entry(
            "Bo <bo@x>",
            "fix(cli):  two spaces (#12)",
            &["src/cli.rs"],
        )];
        assert_eq!(history(&spaced, &staged, "fix", None), ["fix: two spaces"]);
    }

    #[derive(Deserialize)]
    struct FormatCase {
        diff: String,
        kind: String,
        scope: Option<String>,
        history: Vec<String>,
        expected: String,
    }

    #[derive(Deserialize)]
    struct Commit {
        author: String,
        subject: String,
        files: Vec<String>,
    }

    #[derive(Deserialize)]
    struct HistoryCase {
        kind: String,
        author: String,
        staged: Vec<String>,
        log: Vec<Commit>,
        expected: Vec<String>,
    }

    #[derive(Deserialize)]
    struct Fixtures {
        format: Vec<FormatCase>,
        history: Vec<HistoryCase>,
    }

    /// Cases written by the Python code that made the training data
    /// (eval/t5/gen_t5_fixtures.py), from held-out commits.
    #[test]
    fn matches_the_training_data_code() {
        // GCA_T5_FIXTURES names a larger file from the same script, for a one-off check.
        let path = std::env::var_os("GCA_T5_FIXTURES").map_or_else(
            || {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/t5_input_fixtures.json")
            },
            std::path::PathBuf::from,
        );
        let fx: Fixtures = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(fx.format.len() >= 50 && fx.history.len() >= 25);
        for (n, c) in fx.format.iter().enumerate() {
            let got = format(&c.diff, &c.kind, c.scope.as_deref(), &c.history);
            assert_eq!(got, c.expected, "format case {n}");
        }
        for (n, c) in fx.history.iter().enumerate() {
            let log: Vec<LogEntry> = c
                .log
                .iter()
                .map(|e| LogEntry {
                    sha: String::new(),
                    author: e.author.clone(),
                    subject: e.subject.clone(),
                    files: e.files.clone(),
                })
                .collect();
            let me = (!c.author.is_empty()).then_some(c.author.as_str());
            assert_eq!(
                history(&log, &c.staged, &c.kind, me),
                c.expected,
                "history case {n}"
            );
        }
    }
}
