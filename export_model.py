#!/usr/bin/env python3
"""Serialize the trained sklearn pipeline to a single JSON file so a Rust
(or any other non-Python) runtime can rebuild the forward pass.

Feature order encoded in the coef matrix:
    [0                              : 10000)   diff_tfidf   (l2-normalized)
    [10000                          : 12000)   path_bow     (binary count)
    [12000                          : 12100)   ext_bow      (binary count)
    [12100                          : 12101)   diff_sim     (jaccard)
    [12101                          : 12105)   numeric      (scaled)

Usage:
    python export_model.py --model out/model_v2.joblib --out out/model_v2.json
"""
import argparse
import json
import sys
from pathlib import Path

# joblib.load needs the custom classes importable to reconstruct the pipeline.
from train_enhanced import (  # noqa: F401
    PathTokenExtractor,
    DiffSimilarityExtractor,
    FileExtensionExtractor,
)
import joblib


def extract_transformer(preprocessor, name):
    for n, t, _cols in preprocessor.transformers_:
        if n == name:
            return t
    raise KeyError(f"transformer {name!r} not found")


def to_int_vocab(vocab):
    return {str(k): int(v) for k, v in vocab.items()}


def dump_vocab(vectorizer):
    return {
        "vocabulary": to_int_vocab(vectorizer.vocabulary_),
        "token_pattern": vectorizer.token_pattern,
        "lowercase": vectorizer.lowercase,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/model_v2.joblib")
    ap.add_argument("--out", default="out/model_v2.json")
    args = ap.parse_args()

    model = joblib.load(args.model)
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]

    tfidf = extract_transformer(pre, "diff_tfidf")
    path_pipe = extract_transformer(pre, "path_bow")
    ext_pipe = extract_transformer(pre, "ext_bow")
    scaler = extract_transformer(pre, "numeric")

    path_cv = path_pipe.named_steps["vect"]
    ext_cv = ext_pipe.named_steps["vect"]

    payload = {
        "schema_version": 1,
        "classes": list(clf.classes_),
        "feature_layout": {
            "diff_tfidf": [0, len(tfidf.vocabulary_)],
            "path_bow":   [len(tfidf.vocabulary_),
                           len(tfidf.vocabulary_) + len(path_cv.vocabulary_)],
            "ext_bow":    [len(tfidf.vocabulary_) + len(path_cv.vocabulary_),
                           len(tfidf.vocabulary_) + len(path_cv.vocabulary_) + len(ext_cv.vocabulary_)],
            "diff_sim":   [len(tfidf.vocabulary_) + len(path_cv.vocabulary_) + len(ext_cv.vocabulary_),
                           len(tfidf.vocabulary_) + len(path_cv.vocabulary_) + len(ext_cv.vocabulary_) + 1],
            "numeric":    [len(tfidf.vocabulary_) + len(path_cv.vocabulary_) + len(ext_cv.vocabulary_) + 1,
                           len(tfidf.vocabulary_) + len(path_cv.vocabulary_) + len(ext_cv.vocabulary_) + 5],
        },
        "tfidf": {
            "vocabulary": to_int_vocab(tfidf.vocabulary_),
            "idf": tfidf.idf_.tolist(),
            "token_pattern": tfidf.token_pattern,
            "lowercase": tfidf.lowercase,
            "norm": tfidf.norm,
            "sublinear_tf": tfidf.sublinear_tf,
            "ngram_range": list(tfidf.ngram_range),
        },
        "path_bow": {
            **dump_vocab(path_cv),
            "binary": path_cv.binary,
        },
        "ext_bow": {
            **dump_vocab(ext_cv),
            "binary": ext_cv.binary,
        },
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "calibrated_folds": [],
    }

    for cc in clf.calibrated_classifiers_:
        est = cc.estimator
        payload["calibrated_folds"].append({
            "coef": est.coef_.tolist(),
            "intercept": est.intercept_.tolist(),
            "sigmoid_a": [float(cal.a_) for cal in cc.calibrators],
            "sigmoid_b": [float(cal.b_) for cal in cc.calibrators],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out}  ({size_mb:.2f} MB)")
    print(f"  classes      : {payload['classes']}")
    print(f"  folds        : {len(payload['calibrated_folds'])}")
    print(f"  feature dims : {sum(len(v) for k, v in [('tfidf', tfidf.vocabulary_), ('path', path_cv.vocabulary_), ('ext', ext_cv.vocabulary_)]) + 5}")


if __name__ == "__main__":
    sys.exit(main() or 0)
