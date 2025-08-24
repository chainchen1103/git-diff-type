#!/usr/bin/env python3
"""
train_baseline.py — baseline classifier using only:
  - diff_text (TF‑IDF)
  - files_changed, additions, deletions (scaled numerics)
Target: label

Now with robust handling for rare classes & stratified split fallback.

Outputs:
  - out_dir/report.txt               (classification report)
  - out_dir/confusion_matrix.png     (normalized + counts)
  - out_dir/model.joblib             (fitted sklearn Pipeline)
  - out_dir/params.json              (run configuration)
  - out_dir/class_counts.json        (label counts after any handling)

Example:
  python train_baseline.py \
    --csv commits_labeled.csv \
    --out-dir outputs/baseline_v2 \
    --max-features 30000 --ngram-max 2 \
    --only-added-lines --limit-chars 12000 \
    --min-class-count 2 --handle-rare drop

Install deps:
  pip install scikit-learn pandas matplotlib joblib
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Text pre-processing helpers
# -----------------------------

def keep_added_lines(text: str) -> str:
    """Return only lines that start with '+' from a unified diff body.
    Also strips leading '+'. If text is NaN/None, return empty string.
    """
    if not isinstance(text, str):
        return ""
    kept: List[str] = []
    for line in text.splitlines():
        # Skip diff headers and context
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") and not line.startswith("+++ "):
            kept.append(line[1:])
    return "\n".join(kept)


def truncate_text(text: str, max_chars: int | None) -> str:
    if not isinstance(text, str):
        return ""
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def build_text_processing_fn(only_added: bool, limit_chars: int | None):
    def _proc(col: pd.Series) -> pd.Series:
        s = col.fillna("")
        if only_added:
            s_added = s.apply(keep_added_lines)
            # 若只取 '+' 後變成空字串，改回使用原始 diff 內容，避免出現空詞彙表
            s = s_added.mask(s_added.str.strip().eq(""), s)
        if limit_chars and limit_chars > 0:
            s = s.apply(lambda t: truncate_text(t, limit_chars))
        return s
    return _proc


# -----------------------------
# Rare class handling
# -----------------------------

def handle_rare_classes(df: pd.DataFrame, label_col: str, min_count: int, mode: str) -> pd.DataFrame:
    """Handle rare classes before splitting.
    mode ∈ {"drop", "merge", "ignore"}
      - drop: remove rows whose label count < min_count
      - merge: map those labels to 'other'
      - ignore: do nothing (we will later disable stratify)
    """
    counts = df[label_col].astype(str).value_counts()
    rare_labels = set(counts[counts < min_count].index)
    if not rare_labels:
        return df

    if mode == "drop":
        return df[~df[label_col].astype(str).isin(rare_labels)].copy()
    elif mode == "merge":
        df2 = df.copy()
        df2[label_col] = df2[label_col].astype(str).apply(lambda x: "other" if x in rare_labels else x)
        return df2
    elif mode == "ignore":
        return df
    else:
        raise ValueError(f"Unknown handle-rare mode: {mode}")


def write_counts(path: Path, y: pd.Series) -> Dict[str, int]:
    counts = y.astype(str).value_counts().sort_index().to_dict()
    path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


# -----------------------------
# Main training routine
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Baseline commit classifier (diff_text + numerics)")
    ap.add_argument("--csv", required=True, help="Path to commits_labeled.csv")
    ap.add_argument("--out-dir", default="outputs/baseline", help="Directory to write artifacts")

    # split
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")

    # text features
    ap.add_argument("--max-features", type=int, default=30000, help="TF‑IDF max_features")
    ap.add_argument("--ngram-max", type=int, default=2, help="TF‑IDF ngram max (1 or 2)")
    ap.add_argument("--only-added-lines", action="store_true", help="Use only '+' lines from diff")
    ap.add_argument("--limit-chars", type=int, default=12000, help="Truncate processed diff to N chars (0=disable)")

    # model
    ap.add_argument("--c", type=float, default=1.5, help="Inverse regularization strength for LogisticRegression")

    # vectorizer robustness
    ap.add_argument("--min-df", type=float, default=1, help="TfidfVectorizer min_df")
    ap.add_argument("--char-fallback", action="store_true", help="If empty vocabulary occurs, retry with char ngrams (3-5)")

    # rare class handling
    ap.add_argument("--min-class-count", type=int, default=2, help="Minimum samples per class to allow stratify")
    ap.add_argument("--handle-rare", choices=["drop", "merge", "ignore"], default="ignore",
                    help="How to treat labels with count < min-class-count before split")
    ap.add_argument("--strict-stratify", action="store_true",
                    help="If set, fail instead of auto-fallback when stratified split is not possible")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(args.csv)
    required_cols = {"diff_text", "files_changed", "additions", "deletions", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {sorted(missing)}")

    # Handle rare classes (optional)
    if args.handle_rare in {"drop", "merge"} and args.min_class_count > 1:
        df = handle_rare_classes(df, "label", args.min_class_count, args.handle_rare)

    # Process text
    text_proc_fn = build_text_processing_fn(args.only_added_lines, args.limit_chars)
    df["diff_proc"] = text_proc_fn(df["diff_text"])  # new column for vectorizer

    # Define features/target
    X = df[["diff_proc", "files_changed", "additions", "deletions"]].copy()
    y = df["label"].astype(str)

    # Save class counts after handling
    counts = write_counts(out_dir / "class_counts.json", y)
    print("Label counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

    # Determine stratify usage
    stratify = y if (y.value_counts().min() >= args.min_class_count) else None
    if stratify is None:
        print(f"[WARN] Some classes have < {args.min_class_count} samples; disabling stratified split.")
        if args.strict_stratify:
            raise SystemExit("strict-stratify set and stratified split not possible.")

    # Train/val split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=stratify, random_state=args.random_state
        )
    except ValueError as e:
        if args.strict_stratify:
            raise
        print(f"[WARN] Stratified split failed: {e}\n       Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=None, random_state=args.random_state
        )

    # Preprocessor
    vect = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, int(args.ngram_max)),
        lowercase=True,
        strip_accents=None,
        token_pattern=r"(?u)\b\w+\b",
        min_df=args.min_df,
    )

    pre = ColumnTransformer([
        ("diff", vect, "diff_proc"),
        ("nums", StandardScaler(with_mean=False), ["files_changed", "additions", "deletions"])  # with_mean False to allow sparse
    ])

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=float(args.c),
        solver="liblinear",
        multi_class="auto",
    )

    vectorizer_kind = "word"
    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    # Fit with fallback for empty vocabulary
    try:
        pipe.fit(X_train, y_train)
    except ValueError as e:
        msg = str(e).lower()
        if "empty vocabulary" in msg and args.char_fallback:
            print("[WARN] Empty vocabulary detected; retrying with character ngrams (3-5).")
            # rebuild pipeline with char analyzer
            vect_char = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=args.max_features,
                min_df=args.min_df,
                lowercase=False,
            )
            pre_char = ColumnTransformer([
                ("diff", vect_char, "diff_proc"),
                ("nums", StandardScaler(with_mean=False), ["files_changed", "additions", "deletions"])  
            ])
            pipe = Pipeline([("pre", pre_char), ("clf", clf)])
            vectorizer_kind = "char-3-5"
            pipe.fit(X_train, y_train)
        else:
            raise

    # Evaluate
    y_pred = pipe.predict(X_test)
    labels_sorted = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    report = classification_report(y_test, y_pred, digits=3, labels=labels_sorted)

    # Save report
    (out_dir / "report.txt").write_text(report, encoding="utf-8")

    # Print a short summary to stdout
    print("\n=== Classification Report (saved to report.txt) ===\n")
    print(report)

    # Plot confusion matrix (normalized + counts)
    fig = plt.figure(figsize=(10, 8))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    ax = plt.gca()
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title("Confusion Matrix (normalized)")
    tick_marks = np.arange(len(labels_sorted))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Annotate counts
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Save model pipeline & params
    joblib.dump(pipe, out_dir / "model.joblib")
    params = {
        "csv": str(args.csv),
        "out_dir": str(out_dir),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_features": args.max_features,
        "ngram_max": args.ngram_max,
        "only_added_lines": bool(args.only_added_lines),
        "limit_chars": args.limit_chars,
        "C": args.c,
        "labels_in_test": labels_sorted,
        "vectorizer": vectorizer_kind,
        "min_df": args.min_df,
        "char_fallback": bool(args.char_fallback),
        "min_class_count": args.min_class_count,
        "handle_rare": args.handle_rare,
        "strict_stratify": bool(args.strict_stratify),
    }
    (out_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nArtifacts written to: {out_dir}")
    print(" - report.txt\n - confusion_matrix.png\n - model.joblib\n - params.json\n - class_counts.json")


if __name__ == "__main__":
    main()
