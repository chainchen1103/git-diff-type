//! Scope suggestions learned from the repository's own history.
//!
//! Scopes are project vocabulary (`feat(parser): ...`), so no global model can
//! know them. Instead gca looks at recent commits: if the project uses scopes,
//! it suggests the one used most often for the same files, or failing that,
//! for other files in the same directories.

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use crate::message;

/// How many recent commits to learn from.
pub const DEPTH: usize = 500;

pub struct LogEntry {
    pub author: String,
    pub subject: String,
    pub files: Vec<String>,
}

/// Dependency bots write most of the commits in some repositories, and their
/// habits say nothing about how the people on the project write scopes.
/// Same rule as BOT_RE in miner.py.
static BOT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\[bot\]|\bbot\b|renovate|dependabot|automation|actions-user|github-actions")
        .unwrap()
});

fn is_bot(author: &str) -> bool {
    BOT.is_match(author)
}

#[derive(Debug, PartialEq)]
pub struct ScopeHint {
    /// Enough recent commits use a scope that it is worth asking for one.
    pub repo_uses_scopes: bool,
    pub suggestion: Option<String>,
}

pub fn scope_hint(log: &[LogEntry], files: &[String]) -> ScopeHint {
    let staged: HashSet<&str> = files.iter().map(String::as_str).collect();
    let dirs: HashSet<&str> = files
        .iter()
        .map(|f| parent(f))
        .filter(|d| !d.is_empty())
        .collect();

    let mut conventional = 0usize;
    let mut scoped = 0usize;
    // scope -> (commits, index of the most recent one)
    let mut same_file: HashMap<&str, (usize, usize)> = HashMap::new();
    let mut same_dir: HashMap<&str, (usize, usize)> = HashMap::new();

    for (i, entry) in log.iter().enumerate().filter(|(_, e)| !is_bot(&e.author)) {
        let Some((_, scope)) = message::parse_subject(&entry.subject) else {
            continue;
        };
        conventional += 1;
        let Some(scope) = scope else { continue };
        scoped += 1;
        if entry.files.iter().any(|f| staged.contains(f.as_str())) {
            same_file.entry(scope).or_insert((0, i)).0 += 1;
        } else if entry.files.iter().any(|f| dirs.contains(parent(f))) {
            same_dir.entry(scope).or_insert((0, i)).0 += 1;
        }
    }

    // At least five conventional commits, and one in five of them scoped.
    let repo_uses_scopes = conventional >= 5 && scoped * 5 >= conventional;
    let suggestion = if repo_uses_scopes {
        most_used(&same_file).or_else(|| most_used(&same_dir))
    } else {
        None
    };
    ScopeHint {
        repo_uses_scopes,
        suggestion,
    }
}

/// Highest count wins; a tie goes to the scope used most recently.
fn most_used(counts: &HashMap<&str, (usize, usize)>) -> Option<String> {
    counts
        .iter()
        .max_by(|a, b| a.1 .0.cmp(&b.1 .0).then(b.1 .1.cmp(&a.1 .1)))
        .map(|(scope, _)| scope.to_string())
}

fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(dir, _)| dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(subject: &str, files: &[&str]) -> LogEntry {
        LogEntry {
            author: "Ada <ada@example.com>".into(),
            subject: subject.into(),
            files: files.iter().map(|f| f.to_string()).collect(),
        }
    }

    fn staged(files: &[&str]) -> Vec<String> {
        files.iter().map(|f| f.to_string()).collect()
    }

    #[test]
    fn prefers_scope_used_for_the_same_file() {
        let log = [
            entry("fix(cli): quote paths", &["src/cli.rs"]),
            entry("feat(model): add fold", &["src/model.rs"]),
            entry("feat(model): tune", &["src/model.rs"]),
            entry("feat(cli): add -a", &["src/cli.rs", "README.md"]),
            entry("docs: typo", &["README.md"]),
        ];
        let hint = scope_hint(&log, &staged(&["src/cli.rs"]));
        assert!(hint.repo_uses_scopes);
        assert_eq!(hint.suggestion.as_deref(), Some("cli"));
    }

    #[test]
    fn falls_back_to_same_directory() {
        let log = [
            entry("feat(parser): lexer", &["src/parser/lex.rs"]),
            entry("fix(parser): eof", &["src/parser/lex.rs"]),
            entry("chore(deps): bump", &["Cargo.lock"]),
            entry("docs: readme", &["README.md"]),
            entry("docs: more", &["README.md"]),
        ];
        let hint = scope_hint(&log, &staged(&["src/parser/ast.rs"]));
        assert_eq!(hint.suggestion.as_deref(), Some("parser"));
    }

    #[test]
    fn tie_goes_to_most_recent() {
        let log = [
            entry("fix(b): x", &["a.rs"]),
            entry("fix(a): x", &["a.rs"]),
            entry("fix(a): y", &["b.rs"]),
            entry("fix(b): y", &["b.rs"]),
            entry("fix(c): z", &["c.rs"]),
        ];
        let hint = scope_hint(&log, &staged(&["a.rs"]));
        assert_eq!(hint.suggestion.as_deref(), Some("b"));
    }

    #[test]
    fn no_scope_prompt_for_repos_without_scopes() {
        let log: Vec<_> = (0..20)
            .map(|i| entry(&format!("fix: bug {i}"), &["a.rs"]))
            .collect();
        let hint = scope_hint(&log, &staged(&["a.rs"]));
        assert_eq!(
            hint,
            ScopeHint {
                repo_uses_scopes: false,
                suggestion: None
            }
        );
    }

    #[test]
    fn bots_do_not_count() {
        let mut log: Vec<_> = (0..20)
            .map(|i| LogEntry {
                author: "renovate[bot] <bot@renovateapp.com>".into(),
                subject: format!("chore: update dependency x to v{i}"),
                files: vec!["package.json".into()],
            })
            .collect();
        log.extend([
            entry("feat(cli): a", &["src/cli.ts"]),
            entry("fix(cli): b", &["src/cli.ts"]),
            entry("fix(core): c", &["src/core.ts"]),
            entry("docs: d", &["README.md"]),
            entry("feat(cli): e", &["src/cli.ts"]),
        ]);
        let hint = scope_hint(&log, &staged(&["src/cli.ts"]));
        assert!(hint.repo_uses_scopes);
        assert_eq!(hint.suggestion.as_deref(), Some("cli"));
    }

    #[test]
    fn root_files_do_not_match_by_directory() {
        let log = [
            entry("chore(deps): bump", &["Cargo.lock"]),
            entry("chore(deps): bump", &["Cargo.lock"]),
            entry("feat(cli): x", &["src/main.rs"]),
            entry("feat(cli): y", &["src/main.rs"]),
            entry("feat(cli): z", &["src/main.rs"]),
        ];
        let hint = scope_hint(&log, &staged(&["README.md"]));
        assert!(hint.repo_uses_scopes);
        assert_eq!(hint.suggestion, None);
    }
}
