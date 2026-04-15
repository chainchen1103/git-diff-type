#!/usr/bin/env python3
"""Train the commit-type classifier.

Features: TF-IDF over diff text, path tokens, file extensions, Jaccard
similarity between added/deleted tokens, plus numeric stats. Classifier is
calibrated LinearSVC so the CLI can surface top-k probabilities.

Usage:
    python train_enhanced.py --data datasets/ --model out/model_v2.joblib \\
        --cm_out out/confusion_matrix.png
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Set

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

try:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType, FloatTensorType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class FileExtensionExtractor(BaseEstimator, TransformerMixin):
    """Extract file extensions from diff headers, e.g. 'py md json'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._extract_extensions(str(d)) for d in X]

    def _extract_extensions(self, diff_text: str) -> str:
        extensions = set()
        matches = re.findall(r'^\+\+\+ b/.+(\.[a-zA-Z0-9]+)$', diff_text, re.MULTILINE)
        if not matches:
            matches = re.findall(r'^diff --git a/.+ b/.+(\.[a-zA-Z0-9]+)$', diff_text, re.MULTILINE)
        for ext in matches:
            extensions.add(ext.lstrip('.').lower())
        return " ".join(extensions)


class DiffSimilarityExtractor(BaseEstimator, TransformerMixin):
    """Jaccard similarity between added and deleted token sets in a diff."""

    def __init__(self):
        self.token_pattern = re.compile(r'(?u)\b\w+\b')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scores = [self._compute_jaccard(str(d)) for d in X]
        return np.array(scores).reshape(-1, 1)

    def _compute_jaccard(self, diff_text: str) -> float:
        if not diff_text:
            return 0.0
        adds_tokens, dels_tokens = set(), set()
        for line in diff_text.splitlines():
            if line.startswith('+++') or line.startswith('---'):
                continue
            if line.startswith('+'):
                adds_tokens.update(self.token_pattern.findall(line[1:].lower()))
            elif line.startswith('-'):
                dels_tokens.update(self.token_pattern.findall(line[1:].lower()))
        if not adds_tokens and not dels_tokens:
            return 0.0
        intersection = len(adds_tokens & dels_tokens)
        union = len(adds_tokens | dels_tokens)
        return intersection / union if union > 0 else 0.0


class PathTokenExtractor(BaseEstimator, TransformerMixin):
    """Tokenize file paths from diff headers: src/auth/login.ts -> 'src auth login'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._extract_path_tokens(str(d)) for d in X]

    def _extract_path_tokens(self, diff_text: str) -> str:
        tokens = set()
        path_matches = re.findall(r'^\+\+\+ b/(.+)$', diff_text, re.MULTILINE)
        if not path_matches:
            path_matches = re.findall(r'^diff --git a/.+ b/(.+)$', diff_text, re.MULTILINE)
        for path in path_matches:
            for p in re.split(r'[/\-_.]', path):
                if len(p) > 2:
                    tokens.add(p.lower())
        return " ".join(tokens)


def load_data(data_path: str):
    print(f"loading data from {data_path}...")
    data = []
    path = Path(data_path)
    
    files = []
    if path.is_file():
        files = [path]
    else:
        files = sorted(list(path.glob("*.json")) + list(path.glob("*.jsonl")))
    
    print(f"   Found {len(files)} file(s).")
    
    for p in files:
        with open(p, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                if isinstance(content, list):
                    data.extend(content)
                    continue
                elif isinstance(content, dict) and 'data' in content:
                    data.extend(content['data'])
                    continue
            except json.JSONDecodeError:
                pass

            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                        
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Train enhanced commit classifier")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset(s) or directory")
    parser.add_argument("--model", default="out/model_v2.joblib", help="Output model path")
    parser.add_argument("--onnx", default="out/model_v2.onnx", help="Output ONNX path")
    parser.add_argument("--cm_out", default="out/confusion_matrix.png", help="Path to save confusion matrix image")
    parser.add_argument("--max_diff_len", type=int, default=20000, help="Truncate diff text")
    args = parser.parse_args()

    df = load_data(args.data)
    if df.empty:
        print("no data found")
        return

    df['diff_text'] = df['diff_text'].fillna('')
    df['diff_text'] = df['diff_text'].apply(lambda x: x[:args.max_diff_len])

    for col in ['files_changed', 'additions', 'deletions']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    df['add_del_ratio'] = df['additions'] / (df['deletions'] + 1)

    print(f"training on {len(df)} samples")
    print(f"   labels: {df['label'].unique()}")

    X = df[['diff_text', 'files_changed', 'additions', 'deletions', 'add_del_ratio']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('diff_tfidf', TfidfVectorizer(max_features=10000, stop_words='english'), 'diff_text'),

            ('path_bow', Pipeline([
                ('extractor', PathTokenExtractor()),
                ('vect', CountVectorizer(max_features=2000, binary=True))
            ]), 'diff_text'),

            ('ext_bow', Pipeline([
                ('extractor', FileExtensionExtractor()),
                ('vect', CountVectorizer(max_features=100, binary=True))
            ]), 'diff_text'),

            ('diff_sim', DiffSimilarityExtractor(), 'diff_text'),
            
            # 5. 數值特徵
            ('numeric', StandardScaler(), ['files_changed', 'additions', 'deletions', 'add_del_ratio']),
        ],
        remainder='drop'
    )

    # CalibratedClassifierCV wrap gives LinearSVC a true predict_proba so the
    # CLI can show top-k probabilities directly.
    base_svc = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
    clf = CalibratedClassifierCV(base_svc, method='sigmoid', cv=3)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])

    print("training model...")
    model.fit(X_train, y_train)

    print("evaluating...")
    y_pred = model.predict(X_test)
    print("\n" + classification_report(y_test, y_pred))

    labels = sorted(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    if args.cm_out:
        print(f"writing confusion matrix plot -> {args.cm_out}")
        try:
            plt.figure(figsize=(10, 8))
            if HAS_SEABORN:
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
            else:
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title("Confusion Matrix")
                plt.colorbar()
                tick_marks = np.arange(len(labels))
                plt.xticks(tick_marks, labels, rotation=45, ha='right')
                plt.yticks(tick_marks, labels)
                
                thresh = cm.max() / 2.
                for i, j in np.ndindex(cm.shape):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            Path(args.cm_out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.cm_out, dpi=150)
            plt.close()
        except Exception as e:
            print(f"failed to save plot: {e}")

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"\nmodel saved to {args.model}")

    with open(Path(args.model).parent / 'labels.txt', 'w') as f:
        f.write('\n'.join(labels))

    if HAS_ONNX and args.onnx:
        print("exporting to ONNX...")
        try:
            initial_types = [
                ('diff_text', StringTensorType([None, 1])),
                ('files_changed', FloatTensorType([None, 1])),
                ('additions', FloatTensorType([None, 1])),
                ('deletions', FloatTensorType([None, 1])),
                ('add_del_ratio', FloatTensorType([None, 1])),
            ]
            onx = to_onnx(model, X_train[:1], options={id(clf): {'zipmap': False}})
            with open(args.onnx, "wb") as f:
                f.write(onx.SerializeToString())
            print(f"   ONNX saved to {args.onnx}")
        except Exception as e:
            print(f"ONNX export failed: {e}")

if __name__ == '__main__':
    main()