use serde::Deserialize;
use std::path::{Path, PathBuf};

#[path = "../src/features.rs"]
mod features;
#[path = "../src/model.rs"]
#[allow(dead_code)]
mod model;

#[derive(Deserialize)]
struct Case {
    diff_text: String,
    numeric: [f64; 4],
    expected_probs: Vec<f64>,
}

#[derive(Deserialize)]
struct Fixtures {
    classes: Vec<String>,
    cases: Vec<Case>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn python_rust_parity() {
    let model_path = repo_root().join("out").join("model_v2.json");
    let m = model::Model::load(&model_path).expect("load model");
    let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    check_fixture(&m, &tests_dir.join("synthetic_fixtures.json"));
    let dataset_fixture = tests_dir.join("fixtures.json");
    if dataset_fixture.exists() {
        check_fixture(&m, &dataset_fixture);
    }
}

fn check_fixture(m: &model::Model, fixtures_path: &Path) {
    let raw = std::fs::read(fixtures_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", fixtures_path.display()));
    let fx: Fixtures = serde_json::from_slice(&raw).expect("parse fixtures");

    assert_eq!(fx.classes, m.payload.classes, "class order mismatch");
    assert!(
        !fx.cases.is_empty(),
        "fixtures must contain at least one case"
    );

    let tol = 1e-6_f64;
    let mut max_diff = 0.0_f64;
    let mut fails = 0usize;
    for (i, case) in fx.cases.iter().enumerate() {
        let feats = m.build_features(&case.diff_text, case.numeric);
        let probs = m.predict_proba(&feats).expect("valid fixture predictions");
        assert_eq!(probs.len(), case.expected_probs.len());
        for (c, (p, e)) in probs.iter().zip(&case.expected_probs).enumerate() {
            assert!(
                p.is_finite() && e.is_finite(),
                "case {i} class {c}: non-finite probability"
            );
            let d = (p - e).abs();
            max_diff = max_diff.max(d);
            if d > tol {
                fails += 1;
                if fails <= 5 {
                    eprintln!(
                        "case {i} class {c} ({}): rust={p:.8} py={e:.8} diff={d:.3e}",
                        fx.classes[c]
                    );
                }
            }
        }
    }
    eprintln!(
        "{}: max abs diff across {} cases: {:.3e}",
        fixtures_path.display(),
        fx.cases.len(),
        max_diff
    );
    assert_eq!(
        fails, 0,
        "{} class probabilities exceeded tol={}",
        fails, tol
    );
}
