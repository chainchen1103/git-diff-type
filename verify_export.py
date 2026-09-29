#!/usr/bin/env python3
"""Verify the JSON-exported model within a numerical tolerance.

Re-implements the full forward pass in pure NumPy from out/model_v2.json,
then compares against the original sklearn pipeline's predict_proba on a
random sample of dataset rows. The Rust port must match this code path.

Usage:
    python verify_export.py --json out/model_v2.json --model out/model_v2.joblib \\
        --data datasets/_merged.jsonl --n 50
    python verify_export.py --check-fixtures gca-rs/tests/synthetic_fixtures.json \\
        gca-rs/tests/fixtures.json
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_enhanced import (  # noqa: F401
    PathTokenExtractor,
    DiffSimilarityExtractor,
    FileExtensionExtractor,
    FEATURE_COLUMNS,
    prepare_features,
)
from dedupe import iter_rows
from export_model import export_pipeline


def tokenize(text, token_pattern, lowercase):
    if lowercase:
        text = text.lower()
    return re.findall(token_pattern, text)


def build_tfidf_vec(diff_text, spec):
    vocab = spec["vocabulary"]
    idf = np.asarray(spec["idf"], dtype=np.float64)
    vec = np.zeros(len(vocab), dtype=np.float64)
    tokens = tokenize(diff_text, spec["token_pattern"], spec["lowercase"])
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            vec[idx] += 1.0
    if spec.get("sublinear_tf"):
        nz = vec > 0
        vec[nz] = 1.0 + np.log(vec[nz])
    vec *= idf
    norm = spec.get("norm")
    if norm in ("l1", "l2"):
        n = np.linalg.norm(vec, ord=1 if norm == "l1" else 2)
        if n > 0:
            vec /= n
    return vec


def extract_path_tokens(diff_text):
    tokens = set()
    matches = re.findall(r'^\+\+\+ b/(.+)$', diff_text, re.MULTILINE)
    if not matches:
        matches = re.findall(r'^diff --git a/.+ b/(.+)$', diff_text, re.MULTILINE)
    for path in matches:
        for p in re.split(r'[/\-_.]', path):
            if len(p) > 2:
                tokens.add(p.lower())
    return " ".join(tokens)


def extract_extensions(diff_text):
    exts = set()
    matches = re.findall(r'^\+\+\+ b/.+(\.[a-zA-Z0-9]+)$', diff_text, re.MULTILINE)
    if not matches:
        matches = re.findall(r'^diff --git a/.+ b/.+(\.[a-zA-Z0-9]+)$', diff_text, re.MULTILINE)
    for ext in matches:
        exts.add(ext.lstrip('.').lower())
    return " ".join(exts)


def build_count_vec(text, spec):
    vocab = spec["vocabulary"]
    vec = np.zeros(len(vocab), dtype=np.float64)
    tokens = tokenize(text, spec["token_pattern"], spec["lowercase"])
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            if spec.get("binary"):
                vec[idx] = 1.0
            else:
                vec[idx] += 1.0
    return vec


def compute_jaccard(diff_text):
    tp = re.compile(r'(?u)\b\w+\b')
    adds, dels = set(), set()
    for line in diff_text.splitlines():
        if line.startswith('+++') or line.startswith('---'):
            continue
        if line.startswith('+'):
            adds.update(tp.findall(line[1:].lower()))
        elif line.startswith('-'):
            dels.update(tp.findall(line[1:].lower()))
    if not adds and not dels:
        return 0.0
    inter = len(adds & dels)
    union = len(adds | dels)
    return inter / union if union > 0 else 0.0


def build_feature_vector(row, payload):
    diff = str(row["diff_text"])
    parts = [
        build_tfidf_vec(diff, payload["tfidf"]),
        build_count_vec(extract_path_tokens(diff), payload["path_bow"]),
        build_count_vec(extract_extensions(diff), payload["ext_bow"]),
        np.array([compute_jaccard(diff)], dtype=np.float64),
    ]
    numeric = np.array([
        float(row["files_changed"]),
        float(row["additions"]),
        float(row["deletions"]),
        float(row["add_del_ratio"]),
    ], dtype=np.float64)
    mean = np.asarray(payload["scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(payload["scaler"]["scale"], dtype=np.float64)
    parts.append((numeric - mean) / scale)
    return np.concatenate(parts)


def sigmoid(x):
    if x >= 0:
        exp_neg = math.exp(-x)
        return exp_neg / (1.0 + exp_neg)
    return 1.0 / (1.0 + math.exp(x))


def forward_pass(x, payload):
    n_classes = len(payload["classes"])
    accum = np.zeros(n_classes, dtype=np.float64)
    for fold in payload["calibrated_folds"]:
        coef = np.asarray(fold["coef"], dtype=np.float64)
        intercept = np.asarray(fold["intercept"], dtype=np.float64)
        decisions = coef @ x + intercept
        a = np.asarray(fold["sigmoid_a"], dtype=np.float64)
        b = np.asarray(fold["sigmoid_b"], dtype=np.float64)
        calibrated = np.array([sigmoid(a[i] * decisions[i] + b[i]) for i in range(len(a))])
        if n_classes == 2:
            probs = np.array([1.0 - calibrated[0], calibrated[0]])
        else:
            s = calibrated.sum()
            probs = calibrated / s if s > 0 else np.full(n_classes, 1.0 / n_classes)
        accum += probs
    accum /= len(payload["calibrated_folds"])
    return accum


def load_samples(data_path, n, seed=0):
    if n <= 0:
        raise ValueError("sample count must be positive")
    rows = []
    p = Path(data_path)
    files = [p] if p.is_file() else sorted(list(p.glob("*.jsonl")) + list(p.glob("*.json")))
    rng = random.Random(seed)
    for seen, row in enumerate(iter_rows(files), 1):
        if len(rows) < n:
            rows.append(row)
        else:
            index = rng.randrange(seen)
            if index < n:
                rows[index] = row
    if not rows:
        raise ValueError(f"no records found in {data_path}")
    return prepare_features(pd.DataFrame(rows))


def check_fixtures(payload, fixtures_path, tol):
    """The Rust parity test compares gca against these fixtures. Check that
    they were built from this model, so a stale model or stale fixtures fail
    loudly instead of the parity test passing against the wrong numbers."""
    fx = json.loads(Path(fixtures_path).read_text(encoding="utf-8"))
    if fx["classes"] != payload["classes"]:
        print(f"class order mismatch: fixtures={fx['classes']} json={payload['classes']}")
        return 1
    worst = 0.0
    for case in fx["cases"]:
        files, adds, dels, ratio = case["numeric"]
        row = {"diff_text": case["diff_text"], "files_changed": files,
               "additions": adds, "deletions": dels, "add_del_ratio": ratio}
        probs = forward_pass(build_feature_vector(row, payload), payload)
        worst = max(worst, float(np.abs(probs - np.asarray(case["expected_probs"])).max()))
    print(f"max abs diff across {len(fx['cases'])} fixture cases: {worst:.3e}")
    if worst > tol:
        print("FAIL: the fixtures were not produced by this model")
        return 1
    print("PASS: fixtures match this model")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/model_v2.json")
    ap.add_argument("--model", default="out/model_v2.joblib")
    ap.add_argument("--data", default="datasets/_merged.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--check-fixtures", metavar="FILE", nargs="+",
                    help="instead of comparing with sklearn, check the Rust parity fixtures")
    args = ap.parse_args()
    if args.n <= 0:
        ap.error("--n must be positive")
    if not math.isfinite(args.tol) or args.tol < 0:
        ap.error("--tol must be a finite non-negative number")

    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    if args.check_fixtures:
        sys.exit(max(check_fixtures(payload, f, args.tol) for f in args.check_fixtures))

    sk_model = joblib.load(args.model)
    expected_payload = export_pipeline(sk_model)
    if any(payload.get(key) != value for key, value in expected_payload.items()):
        ap.error("JSON parameters differ from the trained model; re-run export_model.py with this model")
    sk_classes = list(sk_model.classes_)
    json_classes = payload["classes"]
    if sk_classes != json_classes:
        print(f"class order mismatch: sk={sk_classes} json={json_classes}")
        sys.exit(1)

    df = load_samples(args.data, args.n)
    print(f"loaded {len(df)} samples; comparing forward pass...")

    sk_probs = sk_model.predict_proba(df[FEATURE_COLUMNS])

    max_diff = 0.0
    mismatched = 0
    for i, row in df.reset_index(drop=True).iterrows():
        x = build_feature_vector(row, payload)
        my_probs = forward_pass(x, payload)
        if not np.isfinite(my_probs).all() or not np.isfinite(sk_probs[i]).all():
            print(f"FAIL: non-finite probabilities for sample {i}")
            sys.exit(1)
        diff = np.abs(my_probs - sk_probs[i]).max()
        if diff > max_diff:
            max_diff = diff
        if diff > args.tol:
            mismatched += 1
            if mismatched <= 3:
                print(f"  sample {i}: max abs diff = {diff:.3e}")
                print(f"    sk : {np.round(sk_probs[i], 5)}")
                print(f"    mine: {np.round(my_probs, 5)}")

    print(f"\nmax abs diff across {len(df)} samples : {max_diff:.3e}")
    print(f"samples exceeding tol={args.tol}        : {mismatched}")
    if mismatched == 0:
        print("PASS: forward pass matches sklearn within tolerance")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
