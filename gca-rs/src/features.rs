use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

use crate::model::{CountVecSpec, TfidfSpec};

static PATH_HEADER_PLUS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\+\+\+ b/(.+)$").unwrap());
static PATH_HEADER_DIFF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^diff --git a/.+ b/(.+)$").unwrap());
static PATH_SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[/\-_.]").unwrap());

static EXT_HEADER_PLUS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\+\+\+ b/.+(\.[a-zA-Z0-9]+)$").unwrap());
static EXT_HEADER_DIFF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^diff --git a/.+ b/.+(\.[a-zA-Z0-9]+)$").unwrap());

static JACCARD_TOK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b\w+\b").unwrap());

pub fn tfidf_vec(diff: &str, spec: &TfidfSpec, tok_re: &Regex) -> Vec<f64> {
    let mut vec = vec![0.0_f64; spec.vocabulary.len()];
    let lowered;
    let text: &str = if spec.lowercase {
        lowered = diff.to_lowercase();
        &lowered
    } else {
        diff
    };
    for m in tok_re.find_iter(text) {
        if let Some(&idx) = spec.vocabulary.get(m.as_str()) {
            vec[idx] += 1.0;
        }
    }
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
    if spec.norm.as_deref() == Some("l2") {
        let n: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 0.0 {
            for v in vec.iter_mut() { *v /= n; }
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
    for m in tok_re.find_iter(t) {
        if let Some(&idx) = spec.vocabulary.get(m.as_str()) {
            if spec.binary {
                vec[idx] = 1.0;
            } else {
                vec[idx] += 1.0;
            }
        }
    }
    vec
}

// Group 1 of every `+++ b/...` header, or of `diff --git` headers when the
// diff has no `+++` lines. Mirrors the fallback in train_enhanced.py.
fn header_matches<'a>(diff: &'a str, primary: &Regex, fallback: &Regex) -> Vec<&'a str> {
    let grab = |re: &Regex| -> Vec<&'a str> {
        re.captures_iter(diff)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str())
            .collect()
    };
    let found = grab(primary);
    if found.is_empty() { grab(fallback) } else { found }
}

pub fn extract_path_tokens(diff: &str) -> String {
    let mut tokens: HashSet<String> = HashSet::new();
    for path in header_matches(diff, &PATH_HEADER_PLUS, &PATH_HEADER_DIFF) {
        for p in PATH_SPLIT.split(path) {
            if p.chars().count() > 2 {
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
    let union = adds.union(&dels).count();
    if union == 0 { 0.0 } else { inter as f64 / union as f64 }
}
