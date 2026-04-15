use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
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
}

pub struct Model {
    pub payload: Payload,
    pub tfidf_re: Regex,
    pub path_bow_re: Regex,
    pub ext_bow_re: Regex,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let payload: Payload = serde_json::from_slice(bytes)
            .context("failed to parse model JSON")?;
        if payload.schema_version != 1 {
            anyhow::bail!("unsupported model schema_version: {}", payload.schema_version);
        }
        let tfidf_re = Regex::new(&payload.tfidf.token_pattern)
            .context("invalid tfidf token_pattern")?;
        let path_bow_re = Regex::new(&payload.path_bow.token_pattern)
            .context("invalid path_bow token_pattern")?;
        let ext_bow_re = Regex::new(&payload.ext_bow.token_pattern)
            .context("invalid ext_bow token_pattern")?;
        Ok(Self { payload, tfidf_re, path_bow_re, ext_bow_re })
    }

    pub fn n_features(&self) -> usize {
        self.payload.tfidf.vocabulary.len()
            + self.payload.path_bow.vocabulary.len()
            + self.payload.ext_bow.vocabulary.len()
            + 1
            + self.payload.scaler.mean.len()
    }

    pub fn build_features(&self, diff_text: &str, numeric: [f64; 4]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_features());

        // tfidf
        out.extend(features::tfidf_vec(diff_text, &self.payload.tfidf, &self.tfidf_re));

        // path_bow
        let path_text = features::extract_path_tokens(diff_text);
        out.extend(features::count_vec(&path_text, &self.payload.path_bow, &self.path_bow_re));

        // ext_bow
        let ext_text = features::extract_extensions(diff_text);
        out.extend(features::count_vec(&ext_text, &self.payload.ext_bow, &self.ext_bow_re));

        // diff_sim
        out.push(features::jaccard(diff_text));

        // numeric (scaled)
        for (i, v) in numeric.iter().enumerate() {
            let m = self.payload.scaler.mean[i];
            let s = self.payload.scaler.scale[i];
            out.push((v - m) / s);
        }

        out
    }

    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        let n_classes = self.payload.classes.len();
        let n_folds = self.payload.calibrated_folds.len();
        let mut accum = vec![0.0_f64; n_classes];

        for fold in &self.payload.calibrated_folds {
            let mut probs = vec![0.0_f64; n_classes];
            for c in 0..n_classes {
                // decision = coef[c] . x + intercept[c]
                let coef = &fold.coef[c];
                let mut d = fold.intercept[c];
                for (k, fv) in features.iter().enumerate() {
                    d += coef[k] * fv;
                }
                let z = fold.sigmoid_a[c] * d + fold.sigmoid_b[c];
                probs[c] = 1.0 / (1.0 + z.exp());
            }
            let sum: f64 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() { *p /= sum; }
            }
            for c in 0..n_classes {
                accum[c] += probs[c];
            }
        }

        for v in accum.iter_mut() { *v /= n_folds as f64; }
        accum
    }

    pub fn topk<'a>(&'a self, probs: &[f64], k: usize) -> Vec<(&'a str, f64)> {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.into_iter()
            .take(k)
            .map(|i| (self.payload.classes[i].as_str(), probs[i]))
            .collect()
    }
}
