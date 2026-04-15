#!/usr/bin/env python3
import sys
import os
import subprocess
import joblib
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# PyInstaller needs these explicit imports so frozen builds don't miss submodules.
import sklearn.pipeline
import sklearn.compose
import sklearn.preprocessing
import sklearn.feature_extraction.text
import sklearn.svm
import sklearn.tree
import sklearn.neighbors
import sklearn.calibration
import sklearn.linear_model

from heuristics import heuristic_classify

try:
    from prompt_toolkit import prompt
except ImportError:
    print("Missing dependency. Please run: pip install prompt_toolkit")
    sys.exit(1)

# Feature extractors — must match the definitions in train_enhanced.py so that
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

class FileExtensionExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        exts_list = []
        for diff in X:
            exts_list.append(self._extract_extensions(str(diff)))
        return exts_list
    def _extract_extensions(self, diff_text: str) -> str:
        extensions = set()
        matches = re.findall(r'^\+\+\+ b/.+(\.[a-zA-Z0-9]+)$', diff_text, re.MULTILINE)
        if not matches:
             matches = re.findall(r'^diff --git a/.+ b/.+(\.[a-zA-Z0-9]+)$', diff_text, re.MULTILINE)
        for ext in matches:
            extensions.add(ext.lstrip('.').lower())
        return " ".join(extensions)


def is_git_repo():
    try:
        return subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                            capture_output=True).returncode == 0
    except FileNotFoundError:
        print("git executable not found")
        sys.exit(1)
    except Exception:
        return False

def get_staged_diff():
    try:
        # --no-ext-diff avoids interference from user-configured external diff tools.
        result = subprocess.run(["git", "diff", "--cached", "--no-color", "--no-ext-diff"],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        return result.stdout.strip()
    except Exception:
        return ""

def get_staged_files():
    try:
        result = subprocess.run(["git", "diff", "--cached", "--name-only"],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        return [f for f in result.stdout.splitlines() if f.strip()]
    except Exception:
        return []

def get_git_stats():
    try:
        result = subprocess.run(["git", "diff", "--cached", "--numstat"], 
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        if not result.stdout: return 0, 0, 0
        files, adds, dels = 0, 0, 0
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                a = 0 if parts[0] == '-' else int(parts[0])
                d = 0 if parts[1] == '-' else int(parts[1])
                files += 1
                adds += a
                dels += d
        return files, adds, dels
    except:
        return 0, 0, 0

def get_model_path():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, 'model_v2.joblib')

def predict_topk(model, input_df, k=3):
    """Return top-k (label, score) pairs. Falls back to softmax over decision
    margins when the classifier does not expose predict_proba."""
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
    else:
        margins = model.decision_function(input_df)[0]
        margins = np.asarray(margins, dtype=np.float64)
        if margins.ndim == 0:
            margins = np.array([-margins, margins])
        shifted = margins - margins.max()
        exp = np.exp(shifted)
        probs = exp / exp.sum()
    order = np.argsort(-probs)[:k]
    return [(classes[i], float(probs[i])) for i in order]


def main():
    try:
        if not is_git_repo():
            print("not a git repository")
            input("Press Enter to exit...")
            sys.exit(1)

        diff_text = get_staged_diff()
        if not diff_text:
            print("no staged changes. run: git add <file> ...")
            input("Press Enter to exit...")
            sys.exit(0)

        dry_run = "--dry-run" in sys.argv

        files_changed, additions, deletions = get_git_stats()
        staged_files = get_staged_files()

        print("\n" + "=" * 50)
        print("  GCA - Git Commit Assistant")
        print(f"  +{additions} / -{deletions} across {files_changed} file(s)")
        print("=" * 50 + "\n")

        candidates = []
        hit = heuristic_classify(staged_files)
        if hit:
            label, reason = hit
            print(f"heuristic hit: {label}  ({reason})")
            candidates.append((label, 1.0, "heuristic"))

        if not candidates:
            model_path = get_model_path()
            if not os.path.exists(model_path):
                dev_path = os.path.join("out", "model_v2.joblib")
                if os.path.exists(dev_path):
                    model_path = dev_path
                else:
                    print(f"model file not found: {model_path}")
                    input("Press Enter to exit...")
                    sys.exit(1)

            print("analyzing staged changes...")
            model = joblib.load(model_path)
            add_del_ratio = additions / (deletions + 1)
            input_df = pd.DataFrame([{
                'diff_text': diff_text[:20000],
                'files_changed': files_changed,
                'additions': additions,
                'deletions': deletions,
                'add_del_ratio': add_del_ratio
            }])
            topk = predict_topk(model, input_df, k=3)
            candidates = [(lbl, sc, "model") for lbl, sc in topk]

        print("\nSuggested type:")
        for i, (lbl, sc, src) in enumerate(candidates, start=1):
            tag = "rule" if src == "heuristic" else "model"
            print(f"  [{i}] {lbl:<10} {sc*100:5.1f}%  ({tag})")
        print("  [Enter] accept [1], or type 1/2/3 to switch\n")

        choice = prompt("Type > ", default="1").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            pred_label = candidates[int(choice) - 1][0]
        else:
            pred_label = candidates[0][0]

        default_msg = f"{pred_label}: "
        commit_msg = prompt("Commit Message > ", default=default_msg).strip()

        if not commit_msg or commit_msg == default_msg.strip():
            print("empty message, aborting")
            sys.exit(0)

        if dry_run:
            print(f"\n[dry-run] skipping commit. message: {commit_msg}")
            sys.exit(0)

        print(f"\n> git commit -m \"{commit_msg}\"")
        subprocess.run(["git", "commit", "-m", commit_msg])

    except Exception as e:
        print(f"\nunexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()