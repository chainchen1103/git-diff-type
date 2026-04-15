#!/usr/bin/env python3
"""Bit-exact verification of the JSON-exported model.

Re-implements the full forward pass in pure NumPy from out/model_v2.json,
then compares against the original sklearn pipeline's predict_proba on a
random sample of dataset rows. The Rust port must match this code path.

Usage:
    python verify_export.py --json out/model_v2.json --model out/model_v2.joblib \\
        --data datasets/_merged.jsonl --n 50
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
)


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
    if spec.get("norm") == "l2":
        n = np.linalg.norm(vec)
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
    # CalibratedClassifierCV uses prob = 1 / (1 + exp(a*decision + b))
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
        probs = np.array([sigmoid(a[i] * decisions[i] + b[i]) for i in range(n_classes)])
        s = probs.sum()
        if s > 0:
            probs = probs / s
        accum += probs
    accum /= len(payload["calibrated_folds"])
    return accum


def load_samples(data_path, n, seed=0):
    rows = []
    p = Path(data_path)
    files = [p] if p.is_file() else sorted(list(p.glob("*.jsonl")) + list(p.glob("*.json")))
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    df = pd.DataFrame(rows)
    df["diff_text"] = df["diff_text"].fillna("").astype(str).str.slice(0, 20000)
    for col in ["files_changed", "additions", "deletions"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    df["add_del_ratio"] = df["additions"] / (df["deletions"] + 1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/model_v2.json")
    ap.add_argument("--model", default="out/model_v2.joblib")
    ap.add_argument("--data", default="datasets/_merged.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    sk_model = joblib.load(args.model)
    sk_classes = list(sk_model.classes_)
    json_classes = payload["classes"]
    if sk_classes != json_classes:
        print(f"class order mismatch: sk={sk_classes} json={json_classes}")
        sys.exit(1)

    df = load_samples(args.data, args.n)
    print(f"loaded {len(df)} samples; comparing forward pass...")

    feature_cols = ["diff_text", "files_changed", "additions", "deletions", "add_del_ratio"]
    sk_probs = sk_model.predict_proba(df[feature_cols])

    max_diff = 0.0
    mismatched = 0
    for i, row in df.reset_index(drop=True).iterrows():
        x = build_feature_vector(row, payload)
        my_probs = forward_pass(x, payload)
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
