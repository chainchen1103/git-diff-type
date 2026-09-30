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

/// The types people gave the recent commits: all of them, and the ones that
/// touched a file that is staged now.
#[derive(Debug, Default, PartialEq)]
pub struct Habits {
    pub project: HashMap<String, usize>,
    pub same_files: HashMap<String, usize>,
}

pub fn habits(log: &[LogEntry], files: &[String]) -> Habits {
    let staged: HashSet<&str> = files.iter().map(String::as_str).collect();
    let mut habits = Habits::default();
    for entry in log.iter().filter(|e| !is_bot(&e.author)) {
        let Some((kind, _)) = message::parse_subject(&entry.subject) else {
            continue;
        };
        *habits.project.entry(kind.to_string()).or_insert(0) += 1;
        if entry.files.iter().any(|f| staged.contains(f.as_str())) {
            *habits.same_files.entry(kind.to_string()).or_insert(0) += 1;
        }
    }
    habits
}

/// How much the project's own mix of types counts, without and with a known
/// subject, and then the mix in the commits that touched the same files.
/// Chosen on projects held out from training (eval/tune_history.py).
pub const HABIT_WEIGHT: f64 = 0.1;
pub const HABIT_WEIGHT_WITH_SUBJECT: f64 = 0.25;
pub const FILE_HABIT_WEIGHT: f64 = 0.1;
pub const FILE_HABIT_WEIGHT_WITH_SUBJECT: f64 = 0.15;
/// Each mix is smoothed with this many commits' worth of the training mix,
/// so a short history moves the ranking little.
pub const PROJECT_PSEUDO_COMMITS: f64 = 10.0;
pub const FILE_PSEUDO_COMMITS: f64 = 5.0;

/// Commits of each type the models were trained on (out/train_report.json).
const TRAINING_MIX: [(&str, f64); 11] = [
    ("fix", 128219.0),
    ("feat", 84737.0),
    ("chore", 64801.0),
    ("docs", 46886.0),
    ("refactor", 37082.0),
    ("test", 24902.0),
    ("build", 19428.0),
    ("ci", 14304.0),
    ("style", 8321.0),
    ("perf", 7923.0),
    ("revert", 1342.0),
];

/// `p ∝ p · (q / π)^weight`: q is a mix of types from the history, smoothed
/// toward the training mix π with `pseudo` commits' worth of it, so history
/// that uses the types as the training data did keeps the ranking. `None`
/// without typed history, or for a model with types the training mix does
/// not have.
pub fn weigh_by_habits(
    classes: &[String],
    probs: &[f64],
    counts: &HashMap<String, usize>,
    weight: f64,
    pseudo: f64,
) -> Option<Vec<f64>> {
    let total_train: f64 = TRAINING_MIX.iter().map(|(_, n)| n).sum();
    let prior: Vec<f64> = classes
        .iter()
        .map(|c| {
            TRAINING_MIX
                .iter()
                .find(|(kind, _)| kind == c)
                .map(|(_, n)| n / total_train)
        })
        .collect::<Option<_>>()?;
    let n: Vec<f64> = classes
        .iter()
        .map(|c| counts.get(c).copied().unwrap_or(0) as f64)
        .collect();
    let total: f64 = n.iter().sum();
    if total == 0.0 || probs.len() != classes.len() {
        return None;
    }
    let scores: Vec<f64> = probs
        .iter()
        .zip(&n)
        .zip(&prior)
        .map(|((p, n), pi)| {
            let q = (n + pseudo * pi) / (total + pseudo);
            (p + 1e-12).ln() + weight * (q / pi).ln()
        })
        .collect();
    Some(crate::subject::softmax(&scores))
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

    fn by(author: &str, subject: &str) -> LogEntry {
        LogEntry {
            author: author.into(),
            subject: subject.into(),
            files: vec![],
        }
    }

    #[test]
    fn counts_the_types_people_used() {
        let log = [
            by("Ada <a@x>", "fix: one"),
            by("Ada <a@x>", "fix(cli): two"),
            by("Ada <a@x>", "perf!: three"),
            by("renovate[bot] <r@x>", "chore(deps): bump"),
            by("Ada <a@x>", "Merge branch 'x'"),
            by("Ada <a@x>", "WIP: nothing"),
        ];
        let counts = habits(&log, &[]).project;
        assert_eq!(counts.get("fix"), Some(&2));
        assert_eq!(counts.get("perf"), Some(&1));
        assert_eq!(counts.get("chore"), None);
        assert_eq!(counts.values().sum::<usize>(), 3);
    }

    #[test]
    fn habits_of_the_same_files() {
        let log = [
            entry("docs(site): guide", &["site/guide.md", "site/nav.ts"]),
            entry("feat(site): search", &["site/search.ts"]),
            entry("fix: nav", &["site/nav.ts"]),
        ];
        let h = habits(&log, &staged(&["site/nav.ts"]));
        assert_eq!(h.project.values().sum::<usize>(), 3);
        assert_eq!(h.same_files.get("docs"), Some(&1));
        assert_eq!(h.same_files.get("fix"), Some(&1));
        assert_eq!(h.same_files.get("feat"), None);
    }

    #[test]
    fn habits_tilt_the_ranking_toward_the_project_s_types() {
        let classes: Vec<String> = TRAINING_MIX.iter().map(|(k, _)| k.to_string()).collect();
        let total: f64 = TRAINING_MIX.iter().map(|(_, n)| n).sum();
        let flat = vec![1.0 / classes.len() as f64; classes.len()];
        // no typed history: nothing to learn
        assert!(weigh_by_habits(&classes, &flat, &HashMap::new(), 0.25, 10.0).is_none());
        // a project that uses types in the training proportions keeps its ranking
        let same: HashMap<String, usize> = TRAINING_MIX
            .iter()
            .map(|(k, n)| (k.to_string(), (n / total * 10_000.0).round() as usize))
            .collect();
        let kept = weigh_by_habits(&classes, &flat, &same, 0.25, 10.0).unwrap();
        for p in &kept {
            assert!((p - flat[0]).abs() < 1e-3, "{kept:?}");
        }
        // a project that writes many perf commits lifts perf
        let perf: HashMap<String, usize> =
            [("perf".to_string(), 200), ("fix".to_string(), 50)].into();
        let tilted = weigh_by_habits(&classes, &flat, &perf, 0.25, 10.0).unwrap();
        let at = |k: &str| tilted[classes.iter().position(|c| c == k).unwrap()];
        assert!(
            at("perf") > at("fix") && at("fix") > at("feat"),
            "{tilted:?}"
        );
        assert!((tilted.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // a model with other types is left alone
        let other = vec!["feat".to_string(), "wip".to_string()];
        assert!(weigh_by_habits(&other, &[0.5, 0.5], &perf, 0.25, 10.0).is_none());
    }

    #[test]
    fn training_mix_matches_the_training_report() {
        let report: serde_json::Value =
            serde_json::from_str(include_str!("../../out/train_report.json")).unwrap();
        for (kind, n) in TRAINING_MIX {
            assert_eq!(report["labels"][kind].as_f64(), Some(n), "{kind}");
        }
        assert_eq!(
            report["labels"].as_object().unwrap().len(),
            TRAINING_MIX.len()
        );
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
