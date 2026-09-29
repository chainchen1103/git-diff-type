use anyhow::{ensure, Context, Result};
use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::features;

#[derive(Debug, Deserialize)]
pub struct TfidfSpec {
    pub vocabulary: HashMap<String, usize>,
    pub idf: Vec<f64>,
    pub token_pattern: String,
    pub lowercase: bool,
    pub norm: Option<String>,
    pub sublinear_tf: bool,
    pub ngram_range: Option<[usize; 2]>,
}

#[derive(Debug, Deserialize)]
pub struct CountVecSpec {
    pub vocabulary: HashMap<String, usize>,
    pub token_pattern: String,
    pub lowercase: bool,
    pub binary: bool,
}

#[derive(Debug, Deserialize)]
pub struct ScalerSpec {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Fold {
    pub coef: Vec<Vec<f64>>,
    pub intercept: Vec<f64>,
    pub sigmoid_a: Vec<f64>,
    pub sigmoid_b: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Payload {
    pub schema_version: u32,
    pub classes: Vec<String>,
    pub tfidf: TfidfSpec,
    pub path_bow: CountVecSpec,
    pub ext_bow: CountVecSpec,
    pub scaler: ScalerSpec,
    pub calibrated_folds: Vec<Fold>,
    pub feature_layout: Option<HashMap<String, [usize; 2]>>,
}

pub struct Model {
    pub payload: Payload,
    pub tfidf_re: Regex,
    pub path_bow_re: Regex,
    pub ext_bow_re: Regex,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let payload: Payload =
            serde_json::from_slice(bytes).context("failed to parse model JSON")?;
        validate_payload(&payload)?;
        let tfidf_re =
            token_regex(&payload.tfidf.token_pattern).context("invalid tfidf token_pattern")?;
        let path_bow_re = token_regex(&payload.path_bow.token_pattern)
            .context("invalid path_bow token_pattern")?;
        let ext_bow_re =
            token_regex(&payload.ext_bow.token_pattern).context("invalid ext_bow token_pattern")?;
        for (name, re) in [
            ("tfidf", &tfidf_re),
            ("path_bow", &path_bow_re),
            ("ext_bow", &ext_bow_re),
        ] {
            ensure!(
                re.captures_len() <= 2,
                "{name} token_pattern has more than one capture group"
            );
        }
        Ok(Self {
            payload,
            tfidf_re,
            path_bow_re,
            ext_bow_re,
        })
    }

    pub fn n_features(&self) -> usize {
        self.payload.tfidf.vocabulary.len()
            + self.payload.path_bow.vocabulary.len()
            + self.payload.ext_bow.vocabulary.len()
            + 1
            + self.payload.scaler.mean.len()
    }

    // Block order must match feature_layout in export_model.py.
    pub fn build_features(&self, diff_text: &str, numeric: [f64; 4]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_features());

        out.extend(features::tfidf_vec(
            diff_text,
            &self.payload.tfidf,
            &self.tfidf_re,
        ));

        let path_text = features::extract_path_tokens(diff_text);
        out.extend(features::count_vec(
            &path_text,
            &self.payload.path_bow,
            &self.path_bow_re,
        ));

        let ext_text = features::extract_extensions(diff_text);
        out.extend(features::count_vec(
            &ext_text,
            &self.payload.ext_bow,
            &self.ext_bow_re,
        ));

        out.push(features::jaccard(diff_text));

        for (i, v) in numeric.iter().enumerate() {
            let m = self.payload.scaler.mean[i];
            let s = self.payload.scaler.scale[i];
            out.push((v - m) / s);
        }

        out
    }

    pub fn predict_proba(&self, features: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            features.len() == self.n_features(),
            "feature count mismatch"
        );
        ensure!(
            features.iter().all(|value| value.is_finite()),
            "features contain non-finite values"
        );
        let n_classes = self.payload.classes.len();
        let n_folds = self.payload.calibrated_folds.len();
        let mut accum = vec![0.0_f64; n_classes];

        for fold in &self.payload.calibrated_folds {
            let mut probs: Vec<f64> = (0..fold.coef.len())
                .map(|c| {
                    let d = fold.coef[c]
                        .iter()
                        .zip(features)
                        .fold(fold.intercept[c], |d, (w, x)| d + w * x);
                    ensure!(d.is_finite(), "model decision overflowed for class {c}");
                    let calibrated = fold.sigmoid_a[c] * d + fold.sigmoid_b[c];
                    ensure!(
                        calibrated.is_finite(),
                        "model calibration overflowed for class {c}"
                    );
                    Ok(sigmoid(calibrated))
                })
                .collect::<Result<_>>()?;
            if n_classes == 2 {
                probs.insert(0, 1.0 - probs[0]);
            }
            let sum: f64 = probs.iter().sum();
            for (a, p) in accum.iter_mut().zip(&probs) {
                *a += if sum > 0.0 {
                    p / sum
                } else {
                    1.0 / n_classes as f64
                };
            }
        }

        for v in accum.iter_mut() {
            *v /= n_folds as f64;
        }
        Ok(accum)
    }

    pub fn prob(&self, probs: &[f64], label: &str) -> f64 {
        self.payload
            .classes
            .iter()
            .position(|c| c == label)
            .map_or(0.0, |i| probs[i])
    }

    pub fn topk<'a>(&'a self, probs: &[f64], k: usize) -> Vec<(&'a str, f64)> {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]));
        idx.into_iter()
            .take(k)
            .map(|i| (self.payload.classes[i].as_str(), probs[i]))
            .collect()
    }
}

fn token_regex(pattern: &str) -> Result<Regex, regex::Error> {
    // sklearn's default \w excludes combining marks and includes all numeric characters.
    let pattern = match pattern {
        r"(?u)\b\w\w+\b" | r"\b\w\w+\b" => r"[\p{L}\p{N}_]{2,}",
        r"(?u)\b\w+\b" | r"\b\w+\b" => r"[\p{L}\p{N}_]+",
        other => other,
    };
    Regex::new(pattern)
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let exp = (-value).exp();
        exp / (1.0 + exp)
    } else {
        1.0 / (1.0 + value.exp())
    }
}

fn validate_vocabulary(vocabulary: &HashMap<String, usize>, name: &str) -> Result<()> {
    let mut seen = vec![false; vocabulary.len()];
    for &index in vocabulary.values() {
        ensure!(
            index < seen.len(),
            "{name} vocabulary index {index} is out of range"
        );
        ensure!(
            !seen[index],
            "{name} vocabulary contains duplicate index {index}"
        );
        seen[index] = true;
    }
    Ok(())
}

fn validate_payload(payload: &Payload) -> Result<()> {
    ensure!(
        payload.schema_version == 1,
        "unsupported model schema_version: {}",
        payload.schema_version
    );
    ensure!(
        payload.classes.len() >= 2,
        "model must contain at least two classes"
    );
    let mut labels = HashSet::new();
    for label in &payload.classes {
        ensure!(
            !label.trim().is_empty() && labels.insert(label),
            "model class labels must be nonempty and unique"
        );
    }
    validate_vocabulary(&payload.tfidf.vocabulary, "tfidf")?;
    validate_vocabulary(&payload.path_bow.vocabulary, "path_bow")?;
    validate_vocabulary(&payload.ext_bow.vocabulary, "ext_bow")?;
    ensure!(
        payload.tfidf.idf.len() == payload.tfidf.vocabulary.len(),
        "tfidf idf length does not match vocabulary"
    );
    ensure!(
        payload.tfidf.idf.iter().all(|v| v.is_finite() && *v > 0.0),
        "tfidf idf values must be finite and positive"
    );
    ensure!(
        matches!(payload.tfidf.norm.as_deref(), None | Some("l1" | "l2")),
        "unsupported tfidf norm"
    );
    ensure!(
        matches!(payload.tfidf.ngram_range, None | Some([1, 1])),
        "only word unigrams are supported"
    );
    ensure!(
        payload.scaler.mean.len() == 4 && payload.scaler.scale.len() == 4,
        "numeric scaler must have four means and scales"
    );
    ensure!(
        payload.scaler.mean.iter().all(|v| v.is_finite()),
        "scaler means must be finite"
    );
    ensure!(
        payload
            .scaler
            .scale
            .iter()
            .all(|v| v.is_finite() && *v > 0.0),
        "scaler scales must be finite and positive"
    );

    let blocks = [
        ("diff_tfidf", payload.tfidf.vocabulary.len()),
        ("path_bow", payload.path_bow.vocabulary.len()),
        ("ext_bow", payload.ext_bow.vocabulary.len()),
        ("diff_sim", 1),
        ("numeric", 4),
    ];
    let mut n_features = 0;
    for (name, width) in blocks {
        let next = n_features + width;
        if let Some(layout) = &payload.feature_layout {
            ensure!(
                layout.get(name) == Some(&[n_features, next]),
                "feature_layout for {name} does not match model dimensions"
            );
        }
        n_features = next;
    }

    ensure!(
        !payload.calibrated_folds.is_empty(),
        "model must contain at least one calibrated fold"
    );
    let n_rows = if payload.classes.len() == 2 {
        1
    } else {
        payload.classes.len()
    };
    for (index, fold) in payload.calibrated_folds.iter().enumerate() {
        ensure!(
            fold.coef.len() == n_rows
                && fold.intercept.len() == n_rows
                && fold.sigmoid_a.len() == n_rows
                && fold.sigmoid_b.len() == n_rows,
            "fold {index} class dimensions do not match model classes"
        );
        ensure!(
            fold.coef.iter().all(|row| row.len() == n_features),
            "fold {index} coefficient width does not match model features"
        );
        ensure!(
            fold.coef
                .iter()
                .flatten()
                .chain(&fold.intercept)
                .chain(&fold.sigmoid_a)
                .chain(&fold.sigmoid_b)
                .all(|v| v.is_finite()),
            "fold {index} contains non-finite values"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Model;
    use serde_json::{json, Value};

    fn binary_payload() -> Value {
        json!({
            "schema_version": 1,
            "classes": ["fix", "feat"],
            "tfidf": {"vocabulary": {"hello": 0}, "idf": [1.0], "token_pattern": "\\w+", "lowercase": true, "norm": "l2", "sublinear_tf": false},
            "path_bow": {"vocabulary": {}, "token_pattern": "\\w+", "lowercase": true, "binary": true},
            "ext_bow": {"vocabulary": {}, "token_pattern": "\\w+", "lowercase": true, "binary": true},
            "scaler": {"mean": vec![0.0; 4], "scale": vec![1.0; 4]},
            "calibrated_folds": [{"coef": vec![vec![0.0; 6]], "intercept": [0.0], "sigmoid_a": [1.0], "sigmoid_b": [0.0]}]
        })
    }

    fn load(payload: &Value) -> anyhow::Result<Model> {
        Model::from_bytes(&serde_json::to_vec(payload).unwrap())
    }

    #[test]
    fn binary_probabilities_follow_class_order() {
        let mut payload = binary_payload();
        payload["calibrated_folds"][0]["sigmoid_b"][0] = json!(3_f64.ln());
        let model = load(&payload).unwrap();
        assert_eq!(model.predict_proba(&[0.0; 6]).unwrap(), [0.75, 0.25]);
    }

    #[test]
    fn rejects_invalid_dimensions_and_vectorizer_specs() {
        let changes = [
            ("/tfidf/vocabulary/hello", json!(1)),
            ("/tfidf/idf", json!([])),
            ("/tfidf/norm", json!("max")),
            ("/tfidf/ngram_range", json!([1, 2])),
            ("/tfidf/token_pattern", json!("(a)(b)")),
            ("/scaler/scale", json!([1.0])),
            ("/scaler/scale/0", json!(0.0)),
            ("/classes", json!(["fix", "fix"])),
            ("/classes", json!([])),
            ("/calibrated_folds", json!([])),
            ("/calibrated_folds/0/coef/0", json!([0.0])),
            ("/calibrated_folds/0/intercept", json!([])),
        ];
        for (pointer, value) in changes {
            let mut payload = binary_payload();
            if pointer == "/tfidf/ngram_range" {
                payload["tfidf"]["ngram_range"] = value;
            } else {
                *payload.pointer_mut(pointer).unwrap() = value;
            }
            assert!(load(&payload).is_err(), "accepted invalid {pointer}");
        }
    }

    #[test]
    fn rejects_duplicate_vocabulary_indices_and_mismatched_layout() {
        let mut payload = binary_payload();
        payload["tfidf"]["vocabulary"]["world"] = json!(0);
        payload["tfidf"]["idf"] = json!([1.0, 1.0]);
        assert!(load(&payload).is_err());
        let mut payload = binary_payload();
        payload["feature_layout"] = json!({"diff_tfidf": [1, 2]});
        assert!(load(&payload).is_err());
    }

    #[test]
    fn multiclass_underflow_returns_uniform_probabilities() {
        let mut payload = binary_payload();
        payload["classes"] = json!(["fix", "feat", "docs"]);
        payload["calibrated_folds"] = json!([{
            "coef": vec![vec![0.0; 6]; 3], "intercept": vec![0.0; 3],
            "sigmoid_a": vec![1.0; 3], "sigmoid_b": vec![1000.0; 3]
        }]);
        let model = load(&payload).unwrap();
        assert_eq!(model.predict_proba(&[0.0; 6]).unwrap(), [1.0 / 3.0; 3]);
    }

    #[test]
    fn sigmoid_preserves_small_probabilities_without_overflow() {
        let probability = super::sigmoid(710.0);
        assert!(probability > 0.0 && probability < 1e-300);
        assert_eq!(super::sigmoid(-1000.0), 1.0);
    }

    #[test]
    fn default_token_pattern_matches_python_unicode_words() {
        let regex = super::token_regex(r"(?u)\b\w\w+\b").unwrap();
        let tokens: Vec<_> = regex
            .find_iter("cafe\u{301} _var ²² a\u{203f}b 中文")
            .map(|token| token.as_str())
            .collect();
        assert_eq!(tokens, ["cafe", "_var", "²²", "中文"]);
    }

    #[test]
    fn rejects_invalid_features_and_arithmetic_overflow() {
        let model = load(&binary_payload()).unwrap();
        assert!(model.predict_proba(&[]).is_err());
        assert!(model.predict_proba(&[f64::NAN; 6]).is_err());
        assert!(model.predict_proba(&[f64::INFINITY; 6]).is_err());

        let mut payload = binary_payload();
        payload["calibrated_folds"][0]["coef"][0][0] = json!(f64::MAX);
        let model = load(&payload).unwrap();
        assert!(model.predict_proba(&[2.0; 6]).is_err());

        let mut payload = binary_payload();
        payload["calibrated_folds"][0]["intercept"][0] = json!(2.0);
        payload["calibrated_folds"][0]["sigmoid_a"][0] = json!(f64::MAX);
        let model = load(&payload).unwrap();
        assert!(model.predict_proba(&[0.0; 6]).is_err());
    }
}
