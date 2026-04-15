#!/usr/bin/env python3
"""Classify the current git diff using the trained model.

Usage:
    git add <files>
    python predict.py
"""
import argparse
import sys
import re
import subprocess
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# Feature extractors must mirror the definitions in train_enhanced.py so that
# joblib can rebuild the pipeline on load.

class DiffSimilarityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.token_pattern = re.compile(r'(?u)\b\w+\b')
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        scores = []
        for diff in X:
            scores.append(self._compute_jaccard(str(diff)))
        return np.array(scores).reshape(-1, 1)
    def _compute_jaccard(self, diff_text: str) -> float:
        if not diff_text: return 0.0
        adds, dels = set(), set()
        for line in diff_text.splitlines():
            if line.startswith('+++') or line.startswith('---'): continue
            if line.startswith('+'): adds.update(self.token_pattern.findall(line[1:].lower()))
            elif line.startswith('-'): dels.update(self.token_pattern.findall(line[1:].lower()))
        if not adds and not dels: return 0.0
        intersection = len(adds & dels)
        union = len(adds | dels)
        return intersection / union if union > 0 else 0.0

class PathTokenExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [self._extract_path_tokens(str(diff)) for diff in X]
    def _extract_path_tokens(self, diff_text: str) -> str:
        tokens = set()
        path_matches = re.findall(r'^\+\+\+ b/(.+)$', diff_text, re.MULTILINE)
        if not path_matches:
            path_matches = re.findall(r'^diff --git a/.+ b/(.+)$', diff_text, re.MULTILINE)
        for path in path_matches:
            parts = re.split(r'[/\-_.]', path)
            for p in parts:
                if len(p) > 2: tokens.add(p.lower())
        return " ".join(tokens)


def get_git_diff(cached=True):
    cmd = ["git", "diff", "--cached"] if cached else ["git", "diff"]
    try:
        cmd += ["--no-color"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None

def get_git_stats(cached=True):
    cmd = ["git", "diff", "--cached", "--numstat"] if cached else ["git", "diff", "--numstat"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if not result.stdout:
            return 0, 0, 0
        files_changed = additions = deletions = 0
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                # Binary files appear as '-'
                adds = 0 if parts[0] == '-' else int(parts[0])
                dels = 0 if parts[1] == '-' else int(parts[1])
                files_changed += 1
                additions += adds
                deletions += dels
        return files_changed, additions, deletions
    except Exception:
        return 0, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Predict commit type from git diff")
    parser.add_argument("--model", default="out/model_v2.joblib", help="Path to trained model")
    parser.add_argument("--unstaged", action="store_true", help="Look at unstaged changes (default is staged)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model not found at {model_path}")
        print("   run: python train_enhanced.py --data ...")
        sys.exit(1)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"failed to load model: {e}")
        sys.exit(1)

    print("analyzing git changes...")
    diff_text = get_git_diff(cached=not args.unstaged)

    if not diff_text:
        target = "unstaged" if args.unstaged else "staged"
        print(f"no {target} changes found")
        if not args.unstaged:
            print("   (try 'git add <file>' first, or pass --unstaged)")
        sys.exit(0)

    files_changed, additions, deletions = get_git_stats(cached=not args.unstaged)
    add_del_ratio = additions / (deletions + 1)

    input_df = pd.DataFrame([{
        'diff_text': diff_text[:20000],
        'files_changed': files_changed,
        'additions': additions,
        'deletions': deletions,
        'add_del_ratio': add_del_ratio
    }])

    try:
        pred_label = model.predict(input_df)[0]
        print("\n" + "=" * 40)
        print(f"Suggested type: \033[1;32m{pred_label}\033[0m")
        print("=" * 40)
        print(f"\nStats: +{additions} / -{deletions} lines in {files_changed} files")
        print(f"\nReady to commit? Copy this:\n")
        print(f"git commit -m \"{pred_label}: <description>\"")
    except Exception as e:
        print(f"prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()