//! The type a commit subject suggests. Words such as "speed up" or "rename"
//! state an intent the diff often hides, so when the subject is known before
//! the type (`-m`, or a subject draft), its model's probabilities are
//! multiplied into the diff model's.
//!
//! The model is multinomial logistic regression over TF-IDF weighted words
//! and word pairs, trained by train_subject.py on the subjects of the same
//! commits as the diff model. How much weight the subject gets was chosen on
//! projects left out of training.

use anyhow::{ensure, Context, Result};
use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use crate::model::token_regex;

static EMBEDDED: &[u8] = include_bytes!("../../out/subject_model.json");

#[derive(Debug, Deserialize)]
struct Payload {
    schema_version: u32,
    classes: Vec<String>,
    /// Terms in feature order: words, and neighbouring words joined by a space.
    vocabulary: Vec<String>,
    idf: Vec<f64>,
    token_pattern: String,
    lowercase: bool,
    sublinear_tf: bool,
    ngram_range: [usize; 2],
    /// One row of weights per class.
    coef: Vec<Vec<f64>>,
    intercept: Vec<f64>,
    fusion: Fusion,
}

/// `p ∝ p_diff · p_subject^weight / prior^prior_power`
#[derive(Debug, Deserialize)]
struct Fusion {
    weight: f64,
    prior_power: f64,
    /// Each class's share of the training commits, in class order.
    prior: Vec<f64>,
}

pub struct SubjectModel {
    payload: Payload,
    index: HashMap<String, usize>,
    token_re: Regex,
}

static PREFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z]+(?:\([^()\r\n]*\))?!?:\s*").unwrap());
static PR_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s*\(#\d+(?:,\s*#\d+)*\)\s*$").unwrap());

/// The subject as the model saw it in training: the first line, without a
/// `type(scope):` prefix or a trailing pull request number such as `(#123)`.
pub fn normalize(subject: &str) -> String {
    let first = subject.lines().next().unwrap_or("").trim();
    let without_prefix = PREFIX.replace(first, "");
    PR_SUFFIX.replace(&without_prefix, "").trim().to_string()
}

impl SubjectModel {
    pub fn embedded() -> Result<Self> {
        Self::from_bytes(EMBEDDED)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let payload: Payload =
            serde_json::from_slice(bytes).context("failed to parse the subject model")?;
        validate(&payload)?;
        let token_re =
            token_regex(&payload.token_pattern).context("invalid subject model token_pattern")?;
        ensure!(
            token_re.captures_len() <= 2,
            "subject model token_pattern has more than one capture group"
        );
        let index = payload
            .vocabulary
            .iter()
            .enumerate()
            .map(|(i, term)| (term.clone(), i))
            .collect();
        Ok(Self {
            payload,
            index,
            token_re,
        })
    }

    #[allow(dead_code)] // for tests/parity.rs
    pub fn classes(&self) -> &[String] {
        &self.payload.classes
    }

    /// Nonzero TF-IDF features of a subject, as scikit-learn's
    /// TfidfVectorizer computes them (l2-normalized).
    fn features(&self, subject: &str) -> Vec<(usize, f64)> {
        let p = &self.payload;
        let text = if p.lowercase {
            subject.to_lowercase()
        } else {
            subject.to_string()
        };
        let tokens: Vec<&str> = if self.token_re.captures_len() == 1 {
            self.token_re.find_iter(&text).map(|m| m.as_str()).collect()
        } else {
            self.token_re
                .captures_iter(&text)
                .map(|c| c.get(1).map_or("", |m| m.as_str()))
                .collect()
        };
        let mut counts: HashMap<usize, f64> = HashMap::new();
        let [min_n, max_n] = p.ngram_range;
        for n in min_n..=max_n {
            for window in tokens.windows(n) {
                let term = window.join(" ");
                if let Some(&i) = self.index.get(&term) {
                    *counts.entry(i).or_default() += 1.0;
                }
            }
        }
        let mut x: Vec<(usize, f64)> = counts
            .into_iter()
            .map(|(i, tf)| {
                let tf = if p.sublinear_tf { 1.0 + tf.ln() } else { tf };
                (i, tf * p.idf[i])
            })
            .collect();
        // In index order, as scikit-learn keeps them: sums in the map's
        // random order could differ in the last bits from run to run.
        x.sort_unstable_by_key(|&(i, _)| i);
        let norm = x.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for (_, v) in &mut x {
                *v /= norm;
            }
        }
        x
    }

    /// Probability of each class given the subject alone.
    pub fn predict_proba(&self, subject: &str) -> Vec<f64> {
        let p = &self.payload;
        let x = self.features(&normalize(subject));
        let scores: Vec<f64> = p
            .coef
            .iter()
            .zip(&p.intercept)
            .map(|(row, b)| b + x.iter().map(|&(i, v)| row[i] * v).sum::<f64>())
            .collect();
        softmax(&scores)
    }

    /// The diff model's probabilities combined with what the subject says.
    /// `None` when the diff model has other classes than this one.
    pub fn combine(
        &self,
        classes: &[String],
        diff_probs: &[f64],
        subject: &str,
    ) -> Option<Vec<f64>> {
        if classes != self.payload.classes.as_slice() || diff_probs.len() != classes.len() {
            return None;
        }
        let f = &self.payload.fusion;
        let scores: Vec<f64> = diff_probs
            .iter()
            .zip(self.predict_proba(subject))
            .zip(&f.prior)
            .map(|((d, s), prior)| {
                (d + 1e-12).ln() + f.weight * (s + 1e-12).ln() - f.prior_power * prior.ln()
            })
            .collect();
        Some(softmax(&scores))
    }
}

pub fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.into_iter().map(|e| e / sum).collect()
}

fn validate(p: &Payload) -> Result<()> {
    ensure!(
        p.schema_version == 1,
        "unsupported subject model schema_version: {}",
        p.schema_version
    );
    let n = p.classes.len();
    ensure!(n >= 3, "the subject model needs at least three classes");
    let mut labels = HashSet::new();
    ensure!(
        p.classes
            .iter()
            .all(|c| !c.trim().is_empty() && labels.insert(c)),
        "subject model class labels must be nonempty and unique"
    );
    let mut terms = HashSet::new();
    ensure!(
        p.vocabulary.iter().all(|t| terms.insert(t)),
        "subject model vocabulary has duplicate terms"
    );
    ensure!(
        p.idf.len() == p.vocabulary.len() && p.idf.iter().all(|v| v.is_finite() && *v > 0.0),
        "subject model idf must be finite, positive and match the vocabulary"
    );
    let [min_n, max_n] = p.ngram_range;
    ensure!(
        (1..=max_n).contains(&min_n) && max_n <= 3,
        "unsupported subject model ngram_range"
    );
    ensure!(
        p.coef.len() == n && p.intercept.len() == n,
        "subject model weights do not match its classes"
    );
    ensure!(
        p.coef
            .iter()
            .all(|row| row.len() == p.vocabulary.len() && row.iter().all(|v| v.is_finite()))
            && p.intercept.iter().all(|v| v.is_finite()),
        "subject model weights must be finite and match the vocabulary"
    );
    let f = &p.fusion;
    ensure!(
        f.weight.is_finite()
            && f.prior_power.is_finite()
            && f.prior.len() == n
            && f.prior.iter().all(|v| v.is_finite() && *v > 0.0),
        "invalid subject model fusion settings"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Two words and one pair, three classes; fusion weight 1, no prior power.
    fn tiny(weight: f64, prior_power: f64) -> SubjectModel {
        let payload = json!({
            "schema_version": 1,
            "classes": ["feat", "fix", "perf"],
            "vocabulary": ["speed", "up", "speed up"],
            "idf": [2.0, 1.0, 3.0],
            "token_pattern": "(?u)\\b\\w\\w+\\b",
            "lowercase": true,
            "sublinear_tf": true,
            "ngram_range": [1, 2],
            "coef": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 2.0]],
            "intercept": [0.5, 0.0, -1.0],
            "fusion": {"weight": weight, "prior_power": prior_power, "prior": [0.3, 0.6, 0.1]}
        });
        SubjectModel::from_bytes(&serde_json::to_vec(&payload).unwrap()).unwrap()
    }

    #[test]
    fn normalizes_like_training() {
        assert_eq!(normalize("fix(parser)!: handle CRLF (#123)"), "handle CRLF");
        assert_eq!(
            normalize("  speed up diff reading (#1, #22)\nbody"),
            "speed up diff reading"
        );
        assert_eq!(normalize("Revert \"feat: x\""), "Revert \"feat: x\"");
        assert_eq!(
            normalize("bump zod from 3.22.0 to 3.23.8"),
            "bump zod from 3.22.0 to 3.23.8"
        );
    }

    #[test]
    fn words_and_pairs_are_weighted_like_scikit_learn() {
        let m = tiny(1.0, 0.0);
        // in index order, so that sums over them do not vary from run to run
        let x = m.features("Speed UP speed");
        // tf: speed 2, up 1, "speed up" 1; sublinear tf is 1 + ln(tf)
        let raw = [(1.0 + 2f64.ln()) * 2.0, 1.0, 3.0];
        let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
        let expected: Vec<(usize, f64)> = raw.iter().map(|v| v / norm).enumerate().collect();
        assert_eq!(x.len(), 3);
        for ((i, v), (j, e)) in x.iter().zip(&expected) {
            assert_eq!(i, j);
            assert!((v - e).abs() < 1e-12, "{v} vs {e}");
        }
        assert!(m.features("nothing known").is_empty());
    }

    #[test]
    fn probabilities_and_fusion() {
        let m = tiny(1.0, 0.0);
        let p = m.predict_proba("");
        let e: Vec<f64> = [0.5f64, 0.0, -1.0].iter().map(|s| s.exp()).collect();
        let sum: f64 = e.iter().sum();
        for (a, b) in p.iter().zip(e.iter().map(|v| v / sum)) {
            assert!((a - b).abs() < 1e-12);
        }
        let classes: Vec<String> = ["feat", "fix", "perf"].map(String::from).to_vec();
        let diff = [0.3, 0.4, 0.3];
        // weight 0 and no prior power: the diff model's ranking unchanged
        let same = tiny(0.0, 0.0).combine(&classes, &diff, "speed up").unwrap();
        for (a, b) in same.iter().zip(diff) {
            assert!((a - b).abs() < 1e-9);
        }
        // "speed up" moves perf to the top
        let fused = m.combine(&classes, &diff, "perf: speed up (#9)").unwrap();
        assert!(fused[2] > fused[1] && fused[2] > fused[0], "{fused:?}");
        assert!((fused.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // another model's classes: no fusion
        let other: Vec<String> = ["feat", "fix"].map(String::from).to_vec();
        assert!(m.combine(&other, &[0.5, 0.5], "speed up").is_none());
    }

    #[test]
    fn rejects_malformed_models() {
        let bad = json!({"schema_version": 1, "classes": ["a", "b", "c"], "vocabulary": ["x", "x"],
            "idf": [1.0, 1.0], "token_pattern": "\\w+", "lowercase": true, "sublinear_tf": false,
            "ngram_range": [1, 1], "coef": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "intercept": [0.0, 0.0, 0.0], "fusion": {"weight": 1.0, "prior_power": 0.0, "prior": [0.3, 0.3, 0.4]}});
        assert!(SubjectModel::from_bytes(&serde_json::to_vec(&bad).unwrap()).is_err());
    }

    #[test]
    fn embedded_model_loads() {
        let m = SubjectModel::embedded().unwrap();
        assert!(m.classes().iter().any(|c| c == "perf"));
    }
}
