use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use crate::model::{CountVecSpec, TfidfSpec};

fn for_each_token(text: &str, regex: &Regex, mut visit: impl FnMut(&str)) {
    if regex.captures_len() == 1 {
        for token in regex.find_iter(text) {
            visit(token.as_str());
        }
    } else {
        for captures in regex.captures_iter(text) {
            visit(captures.get(1).map_or("", |token| token.as_str()));
        }
    }
}

static PATH_HEADER_PLUS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\+\+\+ b/(.+)$").unwrap());
static PATH_HEADER_DIFF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^diff --git a/.+ b/(.+)$").unwrap());
static PATH_SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[/\-_.]").unwrap());

static EXT_HEADER_PLUS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\+\+\+ b/.+(\.[a-zA-Z0-9]+)$").unwrap());
static EXT_HEADER_DIFF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^diff --git a/.+ b/.+(\.[a-zA-Z0-9]+)$").unwrap());

static JACCARD_TOK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\p{L}\p{N}_]+").unwrap());

pub fn tfidf_vec(diff: &str, spec: &TfidfSpec, tok_re: &Regex) -> Vec<f64> {
    let mut vec = vec![0.0_f64; spec.vocabulary.len()];
    let lowered;
    let text: &str = if spec.lowercase {
        lowered = diff.to_lowercase();
        &lowered
    } else {
        diff
    };
    for_each_token(text, tok_re, |token| {
        if let Some(&idx) = spec.vocabulary.get(token) {
            vec[idx] += 1.0;
        }
    });
    if spec.sublinear_tf {
        for v in vec.iter_mut() {
            if *v > 0.0 {
                *v = 1.0 + v.ln();
            }
        }
    }
    for (v, idf) in vec.iter_mut().zip(&spec.idf) {
        *v *= idf;
    }
    let norm = match spec.norm.as_deref() {
        Some("l1") => vec.iter().map(|value| value.abs()).sum(),
        Some("l2") => vec.iter().map(|value| value * value).sum::<f64>().sqrt(),
        _ => 0.0,
    };
    if norm > 0.0 {
        for value in &mut vec {
            *value /= norm;
        }
    }
    vec
}

pub fn count_vec(text: &str, spec: &CountVecSpec, tok_re: &Regex) -> Vec<f64> {
    let mut vec = vec![0.0_f64; spec.vocabulary.len()];
    let lowered;
    let t: &str = if spec.lowercase {
        lowered = text.to_lowercase();
        &lowered
    } else {
        text
    };
    for_each_token(t, tok_re, |token| {
        if let Some(&idx) = spec.vocabulary.get(token) {
            if spec.binary {
                vec[idx] = 1.0;
            } else {
                vec[idx] += 1.0;
            }
        }
    });
    vec
}

// Keep this fallback aligned with train_enhanced.py.
fn header_matches<'a>(diff: &'a str, primary: &Regex, fallback: &Regex) -> Vec<&'a str> {
    let grab = |re: &Regex| -> Vec<&'a str> {
        re.captures_iter(diff)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str())
            .collect()
    };
    let found = grab(primary);
    if found.is_empty() {
        grab(fallback)
    } else {
        found
    }
}

pub fn extract_path_tokens(diff: &str) -> String {
    let mut tokens: HashSet<String> = HashSet::new();
    for path in header_matches(diff, &PATH_HEADER_PLUS, &PATH_HEADER_DIFF) {
        for p in PATH_SPLIT.split(path) {
            if p.chars().nth(2).is_some() {
                tokens.insert(p.to_lowercase());
            }
        }
    }
    tokens.into_iter().collect::<Vec<_>>().join(" ")
}

pub fn extract_extensions(diff: &str) -> String {
    let exts: HashSet<String> = header_matches(diff, &EXT_HEADER_PLUS, &EXT_HEADER_DIFF)
        .into_iter()
        .map(|e| e.trim_start_matches('.').to_lowercase())
        .collect();
    exts.into_iter().collect::<Vec<_>>().join(" ")
}

pub fn jaccard(diff: &str) -> f64 {
    let mut adds: HashSet<String> = HashSet::new();
    let mut dels: HashSet<String> = HashSet::new();
    for line in diff.lines() {
        if line.starts_with("+++") || line.starts_with("---") {
            continue;
        }
        let (set, rest) = if let Some(r) = line.strip_prefix('+') {
            (&mut adds, r)
        } else if let Some(r) = line.strip_prefix('-') {
            (&mut dels, r)
        } else {
            continue;
        };
        let lowered = rest.to_lowercase();
        for m in JACCARD_TOK.find_iter(&lowered) {
            set.insert(m.as_str().to_string());
        }
    }
    if adds.is_empty() && dels.is_empty() {
        return 0.0;
    }
    let inter = adds.intersection(&dels).count();
    let union = adds.len() + dels.len() - inter;
    inter as f64 / union as f64
}

/// Whether a change looks like it alters behavior, in the order of
/// BEHAVIOR_FEATURES in train_enhanced.py, with the same ASCII-only rules.
pub const BEHAVIOR_FEATURES: [&str; 9] = [
    "moved",
    "reshaped",
    "reformatted",
    "comments",
    "new_files",
    "deleted_files",
    "renamed_files",
    "test_files",
    "perf_words",
];

static IDENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").unwrap());
static DIGITS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[0-9]+").unwrap());
static QUOTED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""[^"]*"|'[^']*'|`[^`]*`"#).unwrap());
static COMMENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"^[ \t]*(?://|#|/\*|\*|<!--|--|;;|"""|''')"#).unwrap());
static TEST_PATH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:^|/)(?:tests?|__tests__|spec|e2e)/|[._-](?:test|spec)\.|(?:^|/)test_").unwrap()
});
static PERF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i-u)cache|memo|lazy|fast|perf|optimi|bench|alloc|capacity|reserve|pool|batch|parallel|concurren|throttl|debounc|speed|latenc|throughput",
    )
    .unwrap()
});

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')
}

/// How many items two multisets share.
fn overlap<T: std::hash::Hash + Eq>(
    a: impl IntoIterator<Item = T>,
    b: impl IntoIterator<Item = T>,
) -> usize {
    let mut counts: HashMap<T, usize> = HashMap::new();
    for item in b {
        *counts.entry(item).or_insert(0) += 1;
    }
    let mut shared = 0;
    for item in a {
        if let Some(n) = counts.get_mut(&item) {
            if *n > 0 {
                *n -= 1;
                shared += 1;
            }
        }
    }
    shared
}

fn shape(line: &str) -> String {
    let named = IDENT.replace_all(line, "x");
    let numbered = DIGITS.replace_all(&named, "0");
    QUOTED.replace_all(&numbered, "s").into_owned()
}

fn squash(line: &str) -> String {
    line.chars().filter(|c| !is_ws(*c)).collect()
}

/// BEHAVIOR_FEATURES of a diff; see behavior_features in train_enhanced.py.
pub fn behavior(diff: &str) -> [f64; 9] {
    let (mut removed, mut added) = (Vec::new(), Vec::new());
    let (mut files, mut new, mut deleted, mut renamed, mut tests) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut in_hunk = false;
    for line in diff.split('\n') {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            files += 1;
            tests += usize::from(TEST_PATH.is_match(rest));
            in_hunk = false;
        } else if line.starts_with("@@") {
            in_hunk = true;
        } else if in_hunk {
            if let Some(rest) = line.strip_prefix('+') {
                added.push(rest);
            } else if let Some(rest) = line.strip_prefix('-') {
                removed.push(rest);
            }
        } else if line.starts_with("new file mode") {
            new += 1;
        } else if line.starts_with("deleted file mode") {
            deleted += 1;
        } else if line.starts_with("rename from ") {
            renamed += 1;
        }
    }
    let kept = |lines: &[&'_ str]| -> Vec<String> {
        lines
            .iter()
            .map(|l| l.trim_matches(is_ws))
            .filter(|l| l.chars().count() >= 4)
            .map(String::from)
            .collect()
    };
    let (kept_removed, kept_added) = (kept(&removed), kept(&added));
    let moved = overlap(kept_removed.iter(), kept_added.iter());
    let reshaped = overlap(
        kept_removed.iter().map(|l| shape(l)),
        kept_added.iter().map(|l| shape(l)),
    ) - moved;
    let reformatted = overlap(
        kept_removed.iter().map(|l| squash(l)),
        kept_added.iter().map(|l| squash(l)),
    ) - moved;
    let lines = removed.len() + added.len();
    let comments = removed
        .iter()
        .chain(&added)
        .filter(|l| COMMENT.is_match(l))
        .count();
    let perf: usize = added.iter().map(|l| PERF.find_iter(l).count()).sum();
    let per_line = |k: usize| {
        if lines > 0 {
            2.0 * k as f64 / lines as f64
        } else {
            0.0
        }
    };
    let per_file = |k: usize| {
        if files > 0 {
            k as f64 / files as f64
        } else {
            0.0
        }
    };
    [
        per_line(moved),
        per_line(reshaped),
        per_line(reformatted),
        if lines > 0 {
            comments as f64 / lines as f64
        } else {
            0.0
        },
        per_file(new),
        per_file(deleted),
        per_file(renamed),
        per_file(tests),
        (perf as f64).ln_1p(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{count_vec, jaccard, tfidf_vec};
    use crate::model::{CountVecSpec, TfidfSpec};
    use regex::Regex;
    use serde_json::json;

    #[test]
    fn tfidf_supports_capture_group_and_l1_normalization() {
        let spec: TfidfSpec = serde_json::from_value(json!({
            "vocabulary": {"alpha": 0, "beta": 1}, "idf": [1.0, 2.0],
            "token_pattern": "#(\\w+)", "lowercase": true, "norm": "l1", "sublinear_tf": false
        }))
        .unwrap();
        let regex = Regex::new(&spec.token_pattern).unwrap();
        assert_eq!(tfidf_vec("#ALPHA #alpha #beta", &spec, &regex), [0.5, 0.5]);
    }

    #[test]
    fn count_vector_uses_captured_token() {
        let spec: CountVecSpec = serde_json::from_value(json!({
            "vocabulary": {"alpha": 0, "beta": 1}, "token_pattern": "#(\\w+)",
            "lowercase": true, "binary": false
        }))
        .unwrap();
        let regex = Regex::new(&spec.token_pattern).unwrap();
        assert_eq!(count_vec("#ALPHA #alpha #beta", &spec, &regex), [2.0, 1.0]);
    }

    #[test]
    fn jaccard_excludes_headers_and_deduplicates_tokens() {
        assert_eq!(
            jaccard("+++ b/only_added\n--- a/only_deleted\n+one ONE two\n-one three"),
            1.0 / 3.0
        );
        assert_eq!(jaccard("+++ b/README.md\n context"), 0.0);
    }

    #[test]
    fn jaccard_matches_python_unicode_words() {
        assert_eq!(jaccard("+cafe\u{301} _var ²\n-cafe _var ²"), 1.0);
    }
}
