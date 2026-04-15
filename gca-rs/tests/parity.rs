use serde::Deserialize;
use std::path::PathBuf;

#[path = "../src/features.rs"]
mod features;
#[path = "../src/model.rs"]
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

    let fixtures_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures.json");
    let raw = std::fs::read(&fixtures_path).expect("read fixtures.json");
    let fx: Fixtures = serde_json::from_slice(&raw).expect("parse fixtures");

    assert_eq!(fx.classes, m.payload.classes, "class order mismatch");

    let tol = 1e-6_f64;
    let mut max_diff = 0.0_f64;
    let mut fails = 0usize;
    for (i, case) in fx.cases.iter().enumerate() {
        let feats = m.build_features(&case.diff_text, case.numeric);
        let probs = m.predict_proba(&feats);
        assert_eq!(probs.len(), case.expected_probs.len());
        for c in 0..probs.len() {
            let d = (probs[c] - case.expected_probs[c]).abs();
            if d > max_diff { max_diff = d; }
            if d > tol {
                fails += 1;
                if fails <= 5 {
                    eprintln!(
                        "case {i} class {} ({}): rust={:.8} py={:.8} diff={:.3e}",
                        c, fx.classes[c], probs[c], case.expected_probs[c], d
                    );
                }
            }
        }
    }
    eprintln!("max abs diff across {} cases: {:.3e}", fx.cases.len(), max_diff);
    assert_eq!(fails, 0, "{} class probabilities exceeded tol={}", fails, tol);
}
