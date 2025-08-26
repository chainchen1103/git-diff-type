#!/usr/bin/env python3
"""
train_baseline.py — Train a baseline commit classifier from a JSON/JSONL dataset
built with commit_labeler.py.

Features used (as you requested):
- diff_text (text)
- files_changed (numeric)
- additions (numeric)
- deletions (numeric)
- label (target)

Highlights
- Supports `.jsonl` (one JSON per line) **or** `.json` (array or {"data": [...]}).
- Optional preprocessing to keep only **added lines** from the diff (recommended).
- Time-based split (if `labeled_at` exists) or stratified random split.
- TF‑IDF (1–2 gram) + Logistic Regression (`class_weight=balanced`).
- Saves the whole **scikit-learn Pipeline** to a `.joblib` file.
- Prints classification report and confusion matrix; can also save them to disk.

Usage examples
--------------
# Default: JSONL in, model out
python train_baseline.py \
  --data commits_labeled.jsonl \
  --model out/model.joblib \
  --report out/report.txt \
  --cm_png out/confusion_matrix.png

# Use only added lines from diff and cap length
python train_baseline.py --data commits_labeled.jsonl \
  --only-added-lines --max-chars 15000

# If your file is a JSON array:
python train_baseline.py --data commits_label.json
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

try:
    import joblib
    
except Exception:  # pragma: no cover
    joblib = None

try:
    from skl2onnx import to_onnx
except Exception:
    to_onnx = None

# -------------------------------
# Data loading
# -------------------------------

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json_any(path: Path) -> List[dict]:
    """Load `.jsonl` or `.json` as a list of dicts.
    For `.json`, accept either an array or {"data": [...]}.
    """
    if path.suffix.lower() == '.jsonl':
        return load_jsonl(path)
    with path.open('r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and 'data' in obj and isinstance(obj['data'], list):
        return obj['data']
    raise ValueError('Unsupported JSON structure: expected array or {"data": [...]}')


# -------------------------------
# Diff preprocessing
# -------------------------------

def extract_added_lines(diff_text: str) -> str:
    """Keep only lines starting with '+' but skip diff headers like '+++ b/file'."""
    out_lines: List[str] = []
    for raw in (diff_text or '').splitlines():
        if not raw:
            continue
        if raw.startswith('+++ '):
            continue
        if raw.startswith('+'):
            out_lines.append(raw[1:])  # drop leading '+'
    return '\n'.join(out_lines)


def cap_text(s: str, max_chars: int) -> str:
    s = s or ''
    if max_chars is not None and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars]
    return s


# -------------------------------
# Main
# -------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description='Train baseline classifier from labeled commits')
    ap.add_argument('--data', required=True, help='Path to commits JSON/JSONL (from commit_labeler.py)')
    ap.add_argument('--model', default='model.joblib', help='Output path for the trained Pipeline (.joblib)')
    ap.add_argument('--report', default='', help='Optional path to save text classification report')
    ap.add_argument('--cm_png', default='', help='Optional path to save confusion matrix PNG')

    ap.add_argument('--only-added-lines', action='store_true', help='Use only added lines (+) from the diff')
    ap.add_argument('--max-chars', type=int, default=20000, help='Cap processed diff text length (chars)')

    ap.add_argument('--time-split', action='store_true', help='Sort by labeled_at (if present) and split last 20% as test')
    ap.add_argument('--test-size', type=float, default=0.2, help='Test size for random split if not using time-split')
    ap.add_argument('--random-state', type=int, default=42)

    ap.add_argument('--tfidf-max-features', type=int, default=30000)
    ap.add_argument('--tfidf-ngram-min', type=int, default=1)
    ap.add_argument('--tfidf-ngram-max', type=int, default=2)
    ap.add_argument('--tfidf-min-df', type=float, default=2)

    ap.add_argument('--onnx', default='', help='Optional path to save ONNX model (e.g., out/model.onnx)')

    args = ap.parse_args()

    data_path = Path(args.data)
    rows = load_json_any(data_path)

    # Build DataFrame and coerce fields
    # Expected keys: diff_text, files_changed, additions, deletions, label
    df = pd.DataFrame(rows)

    # Coerce numerics and fill missing
    for col in ['files_changed', 'additions', 'deletions']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    if 'label' not in df.columns:
        raise ValueError('Dataset must contain a "label" field')

    # Prepare diff text
    if 'diff_text' not in df.columns:
        raise ValueError('Dataset must contain a "diff_text" field')

    diff_proc = []
    for s in df['diff_text'].astype(str).tolist():
        t = extract_added_lines(s) if args.only_added_lines else s
        t = cap_text(t, args.max_chars)
        diff_proc.append(t)
    df['diff_proc'] = diff_proc

    if 'top_exts' in df.columns:
        df['exts_proc'] = df['top_exts'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else str(x).replace(',', ' ').strip()
        )
    else:
        df['exts_proc'] = ''
    df['exts_proc'] = df['exts_proc'].fillna('')

    # Decide split
    if args.time_split and 'labeled_at' in df.columns:
        # Sort ascending by labeled_at and split last portion as test
        # If parsing fails, fall back to random split
        try:
            df['_ts'] = pd.to_datetime(df['labeled_at'], errors='coerce')
            df = df.sort_values('_ts').drop(columns=['_ts'])
            split_idx = int(len(df) * (1 - args.test_size))
            df_train = df.iloc[:split_idx]
            df_test = df.iloc[split_idx:]
        except Exception:
            df_train, df_test = train_test_split(df, test_size=args.test_size,
                                                 stratify=df['label'], random_state=args.random_state)
    else:
        df_train, df_test = train_test_split(df, test_size=args.test_size,
                                             stratify=df['label'], random_state=args.random_state)

    X_cols = ['diff_proc', 'exts_proc', 'files_changed', 'additions', 'deletions']
    X_train = df_train[X_cols]
    X_test  = df_test [X_cols]
    y_train = df_train['label']
    y_test = df_test['label']

    # Build pipeline
    preproc = ColumnTransformer(
        transformers=[
            ('diff', TfidfVectorizer(
                max_features=args.tfidf_max_features,
                ngram_range=(args.tfidf_ngram_min, args.tfidf_ngram_max),
                min_df=args.tfidf_min_df,
            ), 'diff_proc'),
            ('exts', CountVectorizer(binary=True, min_df=1), 'exts_proc'),
            ('nums', StandardScaler(with_mean=False), ['files_changed', 'additions', 'deletions']),
        ],
        transformer_weights={
            'diff': 1.0,   
            'exts': 1.0,   
            'nums': 1.0,   
        }
    )

    clf = LogisticRegression(max_iter=3000, class_weight='balanced', n_jobs=None)
    pipe = Pipeline([
        ('prep', preproc),
        ('clf', clf),
    ])

    pipe.fit(X_train, y_train)
    # export to ONNX ===
    if args.onnx:
        if to_onnx is None:
            print('[warn] skl2onnx not installed; skip ONNX export. Install with `pip install skl2onnx onnx onnxruntime`.')
        else:
            try:
                sample = X_train.head(1)
                onx = to_onnx(pipe, sample, target_opset=15)  
                Path(args.onnx).parent.mkdir(parents=True, exist_ok=True)
                with open(args.onnx, 'wb') as f:
                    f.write(onx.SerializeToString())
                print(f'[ok] Saved ONNX model to {args.onnx}')
            except Exception as e:
                print(f'[warn] Failed to export ONNX: {e}')

    # Evaluation
    y_pred = pipe.predict(X_test)
    print('\n=== Classification Report (test) ===')
    print(classification_report(y_test, y_pred, digits=3))

    # Optionally save report
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(classification_report(y_test, y_pred, digits=3))

    # Confusion matrix
    all_labels = set(y_train) | set(y_test)
    labels_sorted = sorted(all_labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print('\n=== Confusion Matrix (rows=true, cols=pred) ===')
    print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted))

    if args.cm_png:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns  # optional, for nicer heatmap; ok to skip if not installed
            Path(args.cm_png).parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(1 + 0.5*len(labels_sorted), 1 + 0.5*len(labels_sorted)))
            if 'sns' in globals():
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels_sorted, yticklabels=labels_sorted)
            else:
                plt.imshow(cm, interpolation='nearest')
                plt.xticks(range(len(labels_sorted)), labels_sorted, rotation=45, ha='right')
                plt.yticks(range(len(labels_sorted)), labels_sorted)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(args.cm_png, dpi=160)
            plt.close()
        except Exception as e:  # pragma: no cover
            print(f"[warn] Failed to save confusion matrix PNG: {e}")

    # Save model
    if joblib is None:
        print('[warn] joblib not installed; skip saving model. Install with `pip install joblib`.')
    else:
        Path(args.model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, args.model)
        print(f'[ok] Saved model pipeline to {args.model}')
    try:
        counts_total = df['label'].value_counts().sort_index()
        counts_train = y_train.value_counts().sort_index()
        counts_test  = y_test.value_counts().sort_index()

        counts_df = pd.DataFrame({
            'total': counts_total,
            'train': counts_train,
            'test': counts_test,
        }).fillna(0).astype(int).sort_index()

        out_dir = Path('out')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'label_counts.csv'
        counts_df.to_csv(out_path, encoding='utf-8', index_label='label')
        print(f'[ok] Saved label counts to {out_path}')
    except Exception as e:
        print(f'[warn] Failed to save label counts: {e}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
