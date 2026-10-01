//! What the repository's own history says about a commit: the scope, and
//! how the project, the files and you usually choose the type.
//!
//! Scopes are project vocabulary (`feat(parser): ...`), so no global model can
//! know them. Instead gca looks at recent commits: if the project uses scopes,
//! it suggests the one, or no scope, used most often for the same files, or
//! failing that, for other files in the deepest directory they share.

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use crate::message;

/// How many recent commits to learn from.
pub const DEPTH: usize = 500;

#[derive(Clone)]
pub struct LogEntry {
    pub sha: String,
    /// `Name <email>`
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

/// A commit of yours counts this many times when suggesting a scope. Chosen
/// on projects held out from training (eval/evaluate_scope.py).
const OWN_SCOPE_VOTES: usize = 8;

/// `me` is who the commit will be by (`Name <email>`), whose own commits
/// count more.
pub fn scope_hint(log: &[LogEntry], files: &[String], me: Option<&str>) -> ScopeHint {
    let staged: HashSet<&str> = files.iter().map(String::as_str).collect();
    // Every directory above a staged file: `a` and `a/b` for `a/b/c.rs`.
    let mut staged_dirs: HashSet<&str> = HashSet::new();
    for f in files {
        staged_dirs.extend(f.match_indices('/').map(|(i, _)| &f[..i]));
    }

    let mut conventional = 0usize;
    let mut scoped = 0usize;
    // (how close, scope or none) -> (votes, index of the most recent one).
    // Closest are commits to a staged file, then those sharing the deepest
    // directory with one.
    let mut votes: HashMap<(usize, Option<&str>), (usize, usize)> = HashMap::new();
    for (i, entry) in log.iter().enumerate().filter(|(_, e)| !is_bot(&e.author)) {
        let Some((_, scope)) = message::parse_subject(&entry.subject) else {
            continue;
        };
        conventional += 1;
        scoped += usize::from(scope.is_some());
        let closeness = if entry.files.iter().any(|f| staged.contains(f.as_str())) {
            usize::MAX
        } else {
            entry
                .files
                .iter()
                .filter_map(|f| {
                    f.match_indices('/')
                        .rev()
                        .map(|(i, _)| &f[..i])
                        .find(|dir| staged_dirs.contains(dir))
                        .map(|dir| dir.matches('/').count() + 1)
                })
                .max()
                .unwrap_or(0)
        };
        if closeness == 0 {
            continue;
        }
        let weight = match me {
            Some(me) if same_person(&entry.author, me) => OWN_SCOPE_VOTES,
            _ => 1,
        };
        votes.entry((closeness, scope)).or_insert((0, i)).0 += weight;
    }

    // At least five conventional commits, and one in five of them scoped.
    let repo_uses_scopes = conventional >= 5 && scoped * 5 >= conventional;
    let closest = votes.keys().map(|(c, _)| *c).max();
    let suggestion = if repo_uses_scopes {
        // Most votes win; a tie goes to the scope used most recently.
        votes
            .iter()
            .filter(|((c, _), _)| Some(*c) == closest)
            .max_by(|a, b| a.1 .0.cmp(&b.1 .0).then(b.1 .1.cmp(&a.1 .1)))
            .and_then(|((_, scope), _)| scope.map(String::from))
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
    let prior = training_prior(classes)?;
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

/// The training mix of the given types; `None` if it lacks one of them.
fn training_prior(classes: &[String]) -> Option<Vec<f64>> {
    let total: f64 = TRAINING_MIX.iter().map(|(_, n)| n).sum();
    classes
        .iter()
        .map(|c| {
            TRAINING_MIX
                .iter()
                .find(|(kind, _)| kind == c)
                .map(|(_, n)| n / total)
        })
        .collect()
}

/// Your own recent commits against what gca would have suggested for them:
/// how many it reads again, how much the comparison counts without and with
/// a known subject, and how many commits' worth of the training mix smooths
/// it. Chosen on projects held out from training (eval/tune_history.py).
pub const OWN_COMMITS: usize = 10;
pub const OWN_WEIGHT: f64 = 0.1;
pub const OWN_WEIGHT_WITH_SUBJECT: f64 = 0.15;
pub const OWN_PSEUDO_COMMITS: f64 = 1.0;

/// The latest commits by `me` (`Name <email>`, the same email or name) with
/// a type the model knows, newest first, at most OWN_COMMITS, with the type.
pub fn own_commits<'a>(
    log: &'a [LogEntry],
    me: &str,
    classes: &[String],
) -> Vec<(&'a LogEntry, &'a str)> {
    log.iter()
        .filter(|e| same_person(&e.author, me))
        .filter_map(|e| {
            let (kind, _) = message::parse_subject(&e.subject)?;
            classes.iter().any(|c| c == kind).then_some((e, kind))
        })
        .take(OWN_COMMITS)
        .collect()
}

/// Whether two `Name <email>` idents are one person: the same email, or
/// failing that the same name.
pub(crate) fn same_person(a: &str, b: &str) -> bool {
    let split = |ident: &str| -> (String, String) {
        match ident.rsplit_once('<') {
            Some((name, email)) => (
                name.trim().to_string(),
                email.trim_end_matches('>').trim().to_lowercase(),
            ),
            None => (ident.trim().to_string(), String::new()),
        }
    };
    let ((a_name, a_email), (b_name, b_email)) = (split(a), split(b));
    (!a_email.is_empty() && a_email == b_email) || (!a_name.is_empty() && a_name == b_name)
}

/// `p ∝ p · ((chosen + s·π) / (shown + s·π))^weight`: for each type, how
/// many of your recent commits you gave it against how much probability gca
/// would have shown it for them, both smoothed toward the training mix π
/// with `pseudo` commits' worth of it. A type you choose more often than
/// gca suggests it rises. `None` without such commits, or for a model with
/// types the training mix does not have.
pub fn weigh_by_own_choices(
    classes: &[String],
    probs: &[f64],
    chosen: &[&str],
    shown: &[Vec<f64>],
    weight: f64,
    pseudo: f64,
) -> Option<Vec<f64>> {
    let prior = training_prior(classes)?;
    if chosen.is_empty() || chosen.len() != shown.len() || probs.len() != classes.len() {
        return None;
    }
    let mut observed = vec![0.0; classes.len()];
    let mut expected = vec![0.0; classes.len()];
    for (kind, probs) in chosen.iter().zip(shown) {
        let j = classes.iter().position(|c| c == kind)?;
        observed[j] += 1.0;
        if probs.len() != classes.len() {
            return None;
        }
        expected.iter_mut().zip(probs).for_each(|(e, p)| *e += p);
    }
    let scores: Vec<f64> = probs
        .iter()
        .zip(observed.iter().zip(&expected))
        .zip(&prior)
        .map(|((p, (o, e)), pi)| {
            (p + 1e-12).ln() + weight * ((o + pseudo * pi) / (e + pseudo * pi)).ln()
        })
        .collect();
    Some(crate::subject::softmax(&scores))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(subject: &str, files: &[&str]) -> LogEntry {
        LogEntry {
            sha: String::new(),
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
            sha: String::new(),
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
        let hint = scope_hint(&log, &staged(&["src/cli.rs"]), None);
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
        let hint = scope_hint(&log, &staged(&["src/parser/ast.rs"]), None);
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
        let hint = scope_hint(&log, &staged(&["a.rs"]), None);
        assert_eq!(hint.suggestion.as_deref(), Some("b"));
    }

    #[test]
    fn no_scope_prompt_for_repos_without_scopes() {
        let log: Vec<_> = (0..20)
            .map(|i| entry(&format!("fix: bug {i}"), &["a.rs"]))
            .collect();
        let hint = scope_hint(&log, &staged(&["a.rs"]), None);
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
                sha: String::new(),
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
        let hint = scope_hint(&log, &staged(&["src/cli.ts"]), None);
        assert!(hint.repo_uses_scopes);
        assert_eq!(hint.suggestion.as_deref(), Some("cli"));
    }

    #[test]
    fn no_scope_when_the_same_files_mostly_had_none() {
        let log = [
            entry("fix: a", &["src/cli.rs"]),
            entry("fix: b", &["src/cli.rs"]),
            entry("feat(cli): c", &["src/cli.rs"]),
            entry("feat(model): d", &["src/model.rs"]),
            entry("feat(model): e", &["src/model.rs"]),
        ];
        let hint = scope_hint(&log, &staged(&["src/cli.rs"]), None);
        assert!(hint.repo_uses_scopes);
        assert_eq!(hint.suggestion, None);
    }

    #[test]
    fn the_deepest_shared_directory_counts_first() {
        let log = [
            entry("feat(core): a", &["packages/core/src/x.ts"]),
            entry("feat(core): b", &["packages/core/src/x.ts"]),
            entry("feat(core): c", &["packages/core/src/y.ts"]),
            entry("fix(foo): d", &["packages/foo/test/b.ts"]),
            entry("docs: e", &["README.md"]),
        ];
        let hint = scope_hint(&log, &staged(&["packages/foo/src/a.ts"]), None);
        assert_eq!(hint.suggestion.as_deref(), Some("foo"));
        // with nothing closer, the top directory
        let hint = scope_hint(&log, &staged(&["packages/new/a.ts"]), None);
        assert_eq!(hint.suggestion.as_deref(), Some("core"));
    }

    #[test]
    fn your_own_commits_count_more_for_the_scope() {
        let mut log: Vec<LogEntry> = (0..3)
            .map(|i| by("Bo <bo@x>", &format!("fix(x): {i}")))
            .collect();
        log.push(by("Ada <ada@x>", "fix(y): mine"));
        log.push(by("Cy <cy@x>", "docs: other"));
        for e in &mut log[..4] {
            e.files = vec!["src/a.rs".into()];
        }
        let files = staged(&["src/a.rs"]);
        assert_eq!(
            scope_hint(&log, &files, None).suggestion.as_deref(),
            Some("x")
        );
        assert_eq!(
            scope_hint(&log, &files, Some("Ada <ada@x>"))
                .suggestion
                .as_deref(),
            Some("y")
        );
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
        let hint = scope_hint(&log, &staged(&["README.md"]), None);
        assert!(hint.repo_uses_scopes);
        assert_eq!(hint.suggestion, None);
    }

    #[test]
    fn own_commits_are_found_by_email_or_name() {
        let classes: Vec<String> = TRAINING_MIX.iter().map(|(k, _)| k.to_string()).collect();
        let log = [
            by("Ada <ADA@x.org>", "fix: newest"),
            by("Bo <bo@x.org>", "feat: not mine"),
            by("Ada Lovelace <ada@x.org>", "WIP: untyped"),
            by(
                "Ada Lovelace <ada@x.org>",
                "deps: a type the model does not know",
            ),
            by("Ada Lovelace <other@y.org>", "docs: same name, other email"),
            by("Ada Lovelace <ada@x.org>", "perf: oldest"),
        ];
        let mine = own_commits(&log, "Ada Lovelace <ada@x.org>", &classes);
        let kinds: Vec<&str> = mine.iter().map(|(_, k)| *k).collect();
        assert_eq!(kinds, ["fix", "docs", "perf"]);
        let many: Vec<LogEntry> = (0..30)
            .map(|i| by("Ada <ada@x.org>", &format!("fix: {i}")))
            .collect();
        assert_eq!(
            own_commits(&many, "Ada <ada@x.org>", &classes).len(),
            OWN_COMMITS
        );
        assert!(own_commits(&log, "Cy <cy@x.org>", &classes).is_empty());
    }

    #[test]
    fn own_choices_lift_the_types_chosen_more_often_than_shown() {
        let classes: Vec<String> = TRAINING_MIX.iter().map(|(k, _)| k.to_string()).collect();
        let at = |p: &[f64], k: &str| p[classes.iter().position(|c| c == k).unwrap()];
        let one_hot = |k: &str| -> Vec<f64> {
            classes
                .iter()
                .map(|c| if c == k { 1.0 } else { 0.0 })
                .collect()
        };
        let flat = vec![1.0 / classes.len() as f64; classes.len()];
        assert!(weigh_by_own_choices(&classes, &flat, &[], &[], 0.1, 1.0).is_none());
        // gca would have shown exactly what was chosen: nothing to learn
        let kept = weigh_by_own_choices(
            &classes,
            &flat,
            &["fix", "docs"],
            &[one_hot("fix"), one_hot("docs")],
            0.1,
            1.0,
        )
        .unwrap();
        for p in &kept {
            assert!((p - flat[0]).abs() < 1e-9, "{kept:?}");
        }
        // gca kept saying fix, and chore was chosen
        let shown = vec![one_hot("fix"); 5];
        let tilted =
            weigh_by_own_choices(&classes, &flat, &["chore"; 5], &shown, 0.1, 1.0).unwrap();
        assert!(
            at(&tilted, "chore") > at(&tilted, "feat") && at(&tilted, "feat") > at(&tilted, "fix")
        );
        assert!((tilted.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // a model with other types is left alone
        let other = vec!["feat".to_string(), "wip".to_string()];
        assert!(
            weigh_by_own_choices(&other, &[0.5, 0.5], &["feat"], &[vec![0.5, 0.5]], 0.1, 1.0)
                .is_none()
        );
    }
}
