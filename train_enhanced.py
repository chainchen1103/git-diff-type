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
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dedupe import iter_rows

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


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
    path = Path(data_path)
    
    files = []
    if path.is_file():
        files = [path]
    else:
        files = sorted(list(path.glob("*.json")) + list(path.glob("*.jsonl")))
    
    print(f"   Found {len(files)} files.")
    return pd.DataFrame(iter_rows(files))


FEATURE_COLUMNS = ['diff_text', 'files_changed', 'additions', 'deletions', 'add_del_ratio']


def prepare_features(df, max_diff_len=20000):
    if max_diff_len <= 0:
        raise ValueError("max_diff_len must be positive")
    df = df.copy()
    if 'diff_text' not in df:
        df['diff_text'] = ''
    df['diff_text'] = df['diff_text'].fillna('').astype(str).str.slice(0, max_diff_len)
    for col in ['files_changed', 'additions', 'deletions']:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], 0).fillna(0).clip(lower=0)
    df['add_del_ratio'] = df['additions'] / (df['deletions'] + 1)
    return df

def build_model():
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

            ('numeric', StandardScaler(), ['files_changed', 'additions', 'deletions', 'add_del_ratio']),
        ],
        remainder='drop'
    )

    base_svc = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
    clf = CalibratedClassifierCV(base_svc, method='sigmoid', cv=3)

    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])


def main():
    parser = argparse.ArgumentParser(description="Train enhanced commit classifier")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset(s) or directory")
    parser.add_argument("--model", default="out/model_v2.joblib", help="Output model path")
    parser.add_argument("--cm_out", default="out/confusion_matrix.png", help="Path to save confusion matrix image")
    parser.add_argument("--max_diff_len", type=int, default=20000, help="Truncate diff text")
    args = parser.parse_args()
    if args.max_diff_len <= 0:
        parser.error("--max_diff_len must be positive")

    df = load_data(args.data)
    if df.empty:
        parser.error("no data found")
    if 'label' not in df:
        parser.error("dataset is missing the label field")
    df = prepare_features(df, args.max_diff_len)
    df = df[df['label'].map(lambda value: isinstance(value, str) and bool(value.strip()))]
    df = df[df['diff_text'].str.strip().ne('')]
    before = len(df)
    # A directory can contain both source corpora and their merged output.
    df = df.drop_duplicates(subset=['diff_text'])
    if len(df) != before:
        print(f"removed {before - len(df)} repeated diffs before splitting")
    counts = df['label'].value_counts()
    if len(counts) < 2 or counts.min() < 4:
        parser.error("training needs at least two labels and four distinct diffs per label")

    print(f"training on {len(df)} samples")
    print(f"   labels: {df['label'].unique()}")

    X = df[FEATURE_COLUMNS]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=max(len(counts), int(np.ceil(len(df) * 0.1))), random_state=42, stratify=y
    )
    if y_train.value_counts().min() < 3:
        parser.error("each label needs at least three training rows for calibration")

    model = build_model()

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
            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
            except ImportError:
                sns = None
            plt.figure(figsize=(10, 8))
            if sns is not None:
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
            else:
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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

    with open(Path(args.model).parent / 'labels.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))

if __name__ == '__main__':
    main()
