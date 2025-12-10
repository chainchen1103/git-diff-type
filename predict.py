#!/usr/bin/env python3
"""
predict.py — Use the trained model to classify current git changes.

Usage:
  1. git add <files>
  2. python predict.py
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

# -----------------------------------------------------------------------------
# 必須包含與 train_enhanced.py 完全相同的類別定義
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 核心功能
# -----------------------------------------------------------------------------

def get_git_diff(cached=True):
    """獲取 git diff 內容"""
    cmd = ["git", "diff", "--cached"] if cached else ["git", "diff"]
    try:
        # 加上 --no-color 避免 ANSI code 干擾
        # 加上 --unified=3 (預設) 確保格式標準
        cmd += ["--no-color"] 
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None

def get_git_stats(cached=True):
    """獲取數值統計 (files_changed, additions, deletions)"""
    cmd = ["git", "diff", "--cached", "--numstat"] if cached else ["git", "diff", "--numstat"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if not result.stdout:
            return 0, 0, 0
            
        files_changed = 0
        additions = 0
        deletions = 0
        
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                # Binary files show as '-'
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
        print(f"❌ Model not found at {model_path}")
        print("   Please run: python train_enhanced.py --data ...")
        sys.exit(1)

    # 1. 載入模型
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   (Ensure you have defined the same custom extractor classes in this script)")
        sys.exit(1)

    # 2. 獲取 Git 資訊
    print("🔍 Analyzing git changes...")
    diff_text = get_git_diff(cached=not args.unstaged)
    
    if not diff_text:
        target = "Unstaged" if args.unstaged else "Staged"
        print(f"⚠️  No {target.lower()} changes found.")
        if not args.unstaged:
            print("   (Try 'git add <file>' first, or use --unstaged)")
        sys.exit(0)

    files_changed, additions, deletions = get_git_stats(cached=not args.unstaged)
    add_del_ratio = additions / (deletions + 1)

    # 3. 準備輸入資料 (必須是 DataFrame 且欄位名稱與訓練時一致)
    input_df = pd.DataFrame([{
        'diff_text': diff_text[:20000], # 記得做同樣的截斷
        'files_changed': files_changed,
        'additions': additions,
        'deletions': deletions,
        'add_del_ratio': add_del_ratio
    }])

    # 4. 預測
    try:
        pred_label = model.predict(input_df)[0]
        
        # 如果模型支援機率估算 (例如有設 probability=True 的 SVC 或 LR)，可以顯示信心度
        # LinearSVC 預設沒有 predict_proba，這裡用 decision_function 簡單判斷信心 (可選)
        confidence_msg = ""
        if hasattr(model, "decision_function"):
             # 簡單看一下 decision function 的最大值
             pass 

        print("\n" + "="*40)
        print(f"🤖 Suggested Type: \033[1;32m{pred_label}\033[0m")
        print("="*40)
        
        print(f"\nStats: +{additions} / -{deletions} lines in {files_changed} files")
        
        # 簡單的 Commit 訊息生成模板
        print(f"\nReady to commit? Copy this:\n")
        print(f"git commit -m \"{pred_label}: <description>\"")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()