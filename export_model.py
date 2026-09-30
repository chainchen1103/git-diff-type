#!/usr/bin/env python3
"""Serialize the trained sklearn pipeline to one JSON file so the Rust CLI
can rebuild the forward pass without Python.

Feature order: diff TF-IDF, path tokens, extensions, Jaccard similarity,
scaled numeric stats, and the scaled behavior features (models trained
before they existed lack them). Vocabulary sizes determine the offsets.

Usage:
    python export_model.py --model out/model_v2.joblib --out out/model_v2.json
"""
import argparse
import json
import sys
from pathlib import Path

# joblib.load needs the custom classes importable to reconstruct the pipeline.
from train_enhanced import (  # noqa: F401
    BEHAVIOR_FEATURES,
    BehaviorExtractor,
    PathTokenExtractor,
    DiffSimilarityExtractor,
    FileExtensionExtractor,
)
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def export_pipeline(model):
    if not isinstance(model, Pipeline) or list(model.named_steps) != ["preprocessor", "clf"]:
        raise ValueError("expected a preprocessor followed by the calibrated classifier")
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]
    if not isinstance(pre, ColumnTransformer):
        raise ValueError("preprocessor must be a ColumnTransformer")
    expected_order = ["diff_tfidf", "path_bow", "ext_bow", "diff_sim", "numeric"]
    active = [(name, cols) for name, transformer, cols in pre.transformers_ if not isinstance(transformer, str) or transformer != "drop"]
    names = [name for name, _ in active]
    with_behavior = names == expected_order + ["behavior"]
    if not (names == expected_order or with_behavior) or pre.transformer_weights:
        raise ValueError("unsupported feature order or transformer weights")
    if with_behavior and active[5][1] != "diff_text":
        raise ValueError("behavior features must use diff_text")
    if [cols for _, cols in active[:4]] != ["diff_text"] * 4:
        raise ValueError("text transformers must use diff_text")
    if list(active[4][1]) != ["files_changed", "additions", "deletions", "add_del_ratio"]:
        raise ValueError("unsupported numeric feature order")
    if clf.method != "sigmoid":
        raise ValueError("the Rust runtime only supports sigmoid calibration")

    tfidf = extract_transformer(pre, "diff_tfidf")
    path_pipe = extract_transformer(pre, "path_bow")
    ext_pipe = extract_transformer(pre, "ext_bow")
    scaler = extract_transformer(pre, "numeric")

    if not isinstance(tfidf, TfidfVectorizer) or not isinstance(scaler, StandardScaler):
        raise ValueError("expected TF-IDF text features and a StandardScaler")
    if type(extract_transformer(pre, "diff_sim")) is not DiffSimilarityExtractor:
        raise ValueError("unsupported diff similarity extractor")
    for pipe, extractor_type in ((path_pipe, PathTokenExtractor), (ext_pipe, FileExtensionExtractor)):
        if not isinstance(pipe, Pipeline) or list(pipe.named_steps) != ["extractor", "vect"]:
            raise ValueError("path and extension pipelines must contain extractor and vect")
        if type(pipe.named_steps["extractor"]) is not extractor_type:
            raise ValueError("unsupported path or extension extractor")
        if type(pipe.named_steps["vect"]) is not CountVectorizer:
            raise ValueError("path and extension features must use CountVectorizer")

    path_cv = path_pipe.named_steps["vect"]
    ext_cv = ext_pipe.named_steps["vect"]

    for vect in (tfidf, path_cv, ext_cv):
        if vect.analyzer != "word" or tuple(vect.ngram_range) != (1, 1):
            raise ValueError(f"{type(vect).__name__}: the Rust runtime only supports word unigrams")
        if vect.preprocessor is not None or vect.tokenizer is not None or vect.strip_accents is not None:
            raise ValueError("custom tokenization and accent stripping are not supported")
    if tfidf.binary or not tfidf.use_idf or tfidf.norm not in (None, "l1", "l2"):
        raise ValueError("unsupported TF-IDF weighting")
    if not scaler.with_mean or not scaler.with_std:
        raise ValueError("numeric features require centering and scaling")
    behavior = None
    if with_behavior:
        pipe = extract_transformer(pre, "behavior")
        if not isinstance(pipe, Pipeline) or list(pipe.named_steps) != ["extractor", "scale"]:
            raise ValueError("the behavior pipeline must contain extractor and scale")
        if type(pipe.named_steps["extractor"]) is not BehaviorExtractor:
            raise ValueError("unsupported behavior extractor")
        behavior = pipe.named_steps["scale"]
        if not isinstance(behavior, StandardScaler) or not behavior.with_mean or not behavior.with_std:
            raise ValueError("behavior features require a centering and scaling StandardScaler")
        if len(behavior.mean_) != len(BEHAVIOR_FEATURES):
            raise ValueError("behavior scaler does not match the behavior features")

    layout, offset = {}, 0
    blocks = [
        ("diff_tfidf", len(tfidf.vocabulary_)),
        ("path_bow", len(path_cv.vocabulary_)),
        ("ext_bow", len(ext_cv.vocabulary_)),
        ("diff_sim", 1),
        ("numeric", len(scaler.mean_)),
    ]
    if behavior is not None:
        blocks.append(("behavior", len(behavior.mean_)))
    for name, size in blocks:
        layout[name] = [offset, offset + size]
        offset += size

    payload = {
        "schema_version": 1,
        "classes": list(clf.classes_),
        "feature_layout": layout,
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
    if behavior is not None:
        payload["behavior"] = {
            "features": list(BEHAVIOR_FEATURES),
            "mean": behavior.mean_.tolist(),
            "scale": behavior.scale_.tolist(),
        }

    for cc in clf.calibrated_classifiers_:
        est = cc.estimator
        rows = 1 if len(clf.classes_) == 2 else len(clf.classes_)
        if est.coef_.shape != (rows, offset) or est.intercept_.shape != (rows,) or len(cc.calibrators) != rows:
            raise ValueError("classifier dimensions do not match the exported features and classes")
        if not np.array_equal(cc.classes, clf.classes_):
            raise ValueError("calibrated fold class order differs from the classifier")
        payload["calibrated_folds"].append({
            "coef": est.coef_.tolist(),
            "intercept": est.intercept_.tolist(),
            "sigmoid_a": [float(cal.a_) for cal in cc.calibrators],
            "sigmoid_b": [float(cal.b_) for cal in cc.calibrators],
        })
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/model_v2.joblib")
    ap.add_argument("--out", default="out/model_v2.json")
    args = ap.parse_args()

    payload = export_pipeline(joblib.load(args.model))
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(serialized, encoding="utf-8")

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out}  ({size_mb:.2f} MB)")
    print(f"  classes      : {payload['classes']}")
    print(f"  folds        : {len(payload['calibrated_folds'])}")
    print(f"  feature dims : {max(end for _, end in payload['feature_layout'].values())}")


if __name__ == "__main__":
    sys.exit(main() or 0)
