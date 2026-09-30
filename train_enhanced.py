#!/usr/bin/env python3
"""Train the commit-type classifier.

Features: TF-IDF over diff text, path tokens, file extensions, Jaccard
similarity between added/deleted tokens, numeric stats, and signs of whether
the change alters behavior (moved or renamed lines, comments, new, deleted,
renamed and test files, words about performance). Classifier is
calibrated LinearSVC so the CLI can surface top-k probabilities.

Usage:
    python train_enhanced.py --data datasets/ --model out/model_v2.joblib \\
        --cm_out out/confusion_matrix.png
"""
import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
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


# Whether a change looks like it alters behavior. Computed on the diff text
# the model reads with ASCII-only rules, so gca-rs/src/features.rs can match
# them exactly.
BEHAVIOR_FEATURES = ("moved", "reshaped", "reformatted", "comments", "new_files", "deleted_files",
                     "renamed_files", "test_files", "perf_words")
_WS = " \t\n\r\x0b\x0c"
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DIGITS = re.compile(r"[0-9]+")
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'|`[^`]*`")
_COMMENT = re.compile(r"[ \t]*(?://|#|/\*|\*|<!--|--|;;|\"\"\"|''')")
_TEST_PATH = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|e2e)/|[._-](?:test|spec)\.|(?:^|/)test_")
_PERF = re.compile(r"cache|memo|lazy|fast|perf|optimi|bench|alloc|capacity|reserve|pool|batch|parallel|"
                   r"concurren|throttl|debounc|speed|latenc|throughput", re.ASCII | re.IGNORECASE)


def _overlap(a, b):
    """How many items two multisets share."""
    counts = Counter(b)
    shared = 0
    for item, n in Counter(a).items():
        shared += min(n, counts[item])
    return shared


def _shape(line):
    """A line with names, numbers and string contents blanked out."""
    return _QUOTED.sub("s", _DIGITS.sub("0", _IDENT.sub("x", line)))


def _squash(line):
    """A line without whitespace."""
    return "".join(ch for ch in line if ch not in _WS)


def behavior_features(diff_text):
    """BEHAVIOR_FEATURES of a diff: the shares of changed lines (a removed and
    an added line count as two) that were only moved, only renamed or changed
    in literals, or only reformatted, and that are comments; the shares of
    files that are new, deleted, renamed and tests; and log(1 + the number of
    words about performance in the added lines). Lines shorter than four
    characters, such as a closing brace, do not count as moved."""
    removed, added = [], []
    files = new = deleted = renamed = tests = 0
    in_hunk = False
    for line in diff_text.split("\n"):
        if line.startswith("diff --git "):
            files += 1
            tests += bool(_TEST_PATH.search(line[len("diff --git "):]))
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk:
            if line.startswith("+"):
                added.append(line[1:])
            elif line.startswith("-"):
                removed.append(line[1:])
        elif line.startswith("new file mode"):
            new += 1
        elif line.startswith("deleted file mode"):
            deleted += 1
        elif line.startswith("rename from "):
            renamed += 1
    kept_removed = [x for x in (line.strip(_WS) for line in removed) if len(x) >= 4]
    kept_added = [x for x in (line.strip(_WS) for line in added) if len(x) >= 4]
    moved = _overlap(kept_removed, kept_added)
    reshaped = _overlap([_shape(x) for x in kept_removed], [_shape(x) for x in kept_added]) - moved
    reformatted = _overlap([_squash(x) for x in kept_removed], [_squash(x) for x in kept_added]) - moved
    lines = len(removed) + len(added)
    comments = sum(1 for x in removed + added if _COMMENT.match(x))
    perf = sum(len(_PERF.findall(x)) for x in added)

    def per_line(k):
        return 2.0 * k / lines if lines else 0.0

    def per_file(k):
        return k / files if files else 0.0

    return [per_line(moved), per_line(reshaped), per_line(reformatted),
            comments / lines if lines else 0.0,
            per_file(new), per_file(deleted), per_file(renamed), per_file(tests), math.log1p(perf)]


class BehaviorExtractor(BaseEstimator, TransformerMixin):
    """BEHAVIOR_FEATURES of each diff."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [behavior_features(str(d)) for d in X]
        return np.array(rows, dtype=np.float64).reshape(-1, len(BEHAVIOR_FEATURES))


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

def build_model(C=1.0, class_weight="balanced", n_jobs=None):
    """The training pipeline. The defaults reproduce the original model; the
    released model was trained with C=0.1 (see eval/tune_c.py)."""
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

            ('behavior', Pipeline([
                ('extractor', BehaviorExtractor()),
                ('scale', StandardScaler())
            ]), 'diff_text'),
        ],
        remainder='drop'
    )

    weight = None if class_weight in (None, "none") else class_weight
    base_svc = LinearSVC(C=C, class_weight=weight, random_state=42, max_iter=5000)
    clf = CalibratedClassifierCV(base_svc, method='sigmoid', cv=3, n_jobs=n_jobs)

    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])


def _stream_rows(paths, include_bots):
    for row in iter_rows(paths):
        if row.get("is_bot") and not include_bots:
            continue
        if not isinstance(row.get("label"), str) or not str(row.get("diff_text") or "").strip():
            continue
        yield row


def train_streaming(args):
    """Train on corpora too large to hold as one DataFrame: fit the
    vocabularies and scaler on a uniform sample, transform everything in
    chunks, then fit the calibrated classifier on the sparse matrix. There
    is no internal test split; score the model with eval/evaluate.py on the
    held-out sets instead."""
    # Built through the module name so the pickle refers to train_enhanced.X
    # rather than __main__.X and loads from any script.
    import train_enhanced as te

    started = time.time()
    log = lambda msg: print(f"[{time.time() - started:7.1f}s] {msg}", flush=True)  # noqa: E731
    paths = []
    for item in args.data:
        p = Path(item)
        paths.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])
    rng = random.Random(args.seed)

    labels, sample, repos = [], [], Counter()
    for i, row in enumerate(_stream_rows(paths, args.include_bots)):
        labels.append(row["label"])
        repos[row.get("repo", "?")] += 1
        if len(sample) < args.fit_sample:
            sample.append(row)
        elif (j := rng.randrange(i + 1)) < args.fit_sample:
            sample[j] = row
    if len(set(labels)) < 2:
        raise SystemExit("training needs at least two labels")
    log(f"{len(labels)} commits from {len(repos)} repositories")

    model = te.build_model(C=args.C, class_weight=args.class_weight, n_jobs=args.jobs)
    pre, clf = model.named_steps["preprocessor"], model.named_steps["clf"]
    pre.fit(prepare_features(pd.DataFrame(sample), args.max_diff_len)[FEATURE_COLUMNS])
    del sample
    log("fitted vocabularies")

    blocks, buf = [], []
    for row in _stream_rows(paths, args.include_bots):
        buf.append(row)
        if len(buf) == args.chunk:
            frame = prepare_features(pd.DataFrame(buf), args.max_diff_len)[FEATURE_COLUMNS]
            blocks.append(sp.csr_matrix(pre.transform(frame)))
            buf = []
    if buf:
        frame = prepare_features(pd.DataFrame(buf), args.max_diff_len)[FEATURE_COLUMNS]
        blocks.append(sp.csr_matrix(pre.transform(frame)))
    X = sp.vstack(blocks, format="csr")
    del blocks
    log(f"features: {X.shape[0]} x {X.shape[1]}")

    clf.fit(X, np.asarray(labels))
    log("fitted classifier")

    out = Path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out, compress=3)
    (out.parent / "labels.txt").write_text("\n".join(sorted(clf.classes_)), encoding="utf-8")
    report = {
        "commits": len(labels),
        "repositories": len(repos),
        "labels": dict(Counter(labels).most_common()),
        "C": args.C,
        "class_weight": args.class_weight,
        "fit_sample": args.fit_sample,
        "include_bots": args.include_bots,
    }
    (out.parent / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"model saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Train enhanced commit classifier")
    parser.add_argument("--data", required=True, nargs="+",
                        help="JSONL dataset(s) or a directory (several only with --stream)")
    parser.add_argument("--model", default="out/model_v2.joblib", help="Output model path")
    parser.add_argument("--cm_out", default="out/confusion_matrix.png", help="Path to save confusion matrix image")
    parser.add_argument("--max_diff_len", type=int, default=20000, help="Truncate diff text")
    parser.add_argument("--C", type=float, default=1.0, help="LinearSVC regularization")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--jobs", type=int, default=None, help="calibration folds fitted in parallel")
    stream = parser.add_argument_group("large corpora")
    stream.add_argument("--stream", action="store_true",
                        help="fit on a vocabulary sample and transform in chunks; no internal split")
    stream.add_argument("--fit-sample", type=int, default=150000)
    stream.add_argument("--chunk", type=int, default=20000)
    stream.add_argument("--include-bots", action="store_true", help="keep commits marked is_bot")
    stream.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_diff_len <= 0:
        parser.error("--max_diff_len must be positive")
    if args.stream:
        return train_streaming(args)
    if len(args.data) != 1:
        parser.error("several --data paths need --stream")

    df = load_data(args.data[0])
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

    model = build_model(C=args.C, class_weight=args.class_weight, n_jobs=args.jobs)

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
