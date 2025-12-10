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

# =========================================================
# [新增] PyInstaller 強制引用區 (解決 hidden import 問題)
# 這些模組雖然沒直接用到，但 joblib 還原模型時需要它們
# =========================================================
import sklearn.pipeline
import sklearn.compose
import sklearn.preprocessing
import sklearn.feature_extraction.text
import sklearn.svm
import sklearn.tree
import sklearn.neighbors
# =========================================================

# 引入 UI 庫...
try:
    from prompt_toolkit import prompt
# ... (後面程式碼保持不變)
    from prompt_toolkit.shortcuts import confirm
except ImportError:
    print("❌ Missing dependency. Please run: pip install prompt_toolkit")
    sys.exit(1)

# ==========================================
# 1. 必須包含與訓練時完全相同的特徵提取類別
# ==========================================

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

# ==========================================
# 2. Git 操作工具函數
# ==========================================

def is_git_repo():
    return subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True).returncode == 0

def get_staged_diff():
    # 獲取 diff，注意編碼錯誤處理
    try:
        result = subprocess.run(["git", "diff", "--cached", "--no-color"], 
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        return result.stdout.strip()
    except Exception:
        return ""

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
    # 處理 PyInstaller 打包後的路徑問題
    if getattr(sys, 'frozen', False):
        # 如果是打包後的執行檔，資源會在 _MEIPASS 暫存目錄
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, 'model_v2.joblib')

# ==========================================
# 3. 主程式邏輯
# ==========================================

def main():
    # A. 檢查 Git 環境
    if not is_git_repo():
        print("❌ 錯誤: 當前目錄不是 Git 儲存庫 (Not a git repository)")
        sys.exit(1)

    # B. 檢查 Staged Changes
    diff_text = get_staged_diff()
    if not diff_text:
        print("⚠️  沒有偵測到暫存的變更 (No staged changes).")
        print("   請先執行: git add <file> ...")
        sys.exit(0)

    # C. 載入模型
    model_path = get_model_path()
    if not os.path.exists(model_path):
        # 嘗試找 out 資料夾 (開發模式)
        dev_path = os.path.join("out", "model_v2.joblib")
        if os.path.exists(dev_path):
            model_path = dev_path
        else:
            print(f"❌ 找不到模型檔案: {model_path}")
            sys.exit(1)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        sys.exit(1)

    # D. 執行預測
    files_changed, additions, deletions = get_git_stats()
    add_del_ratio = additions / (deletions + 1)
    
    input_df = pd.DataFrame([{
        'diff_text': diff_text[:20000],
        'files_changed': files_changed,
        'additions': additions,
        'deletions': deletions,
        'add_del_ratio': add_del_ratio
    }])

    print("🤖 正在分析程式碼變更...")
    try:
        pred_label = model.predict(input_df)[0]
    except Exception as e:
        print(f"❌ 預測失敗: {e}")
        pred_label = "feat" # Fallback

    # E. 互動式介面 (The Magic Part)
    # 我們將預測結果設為 default，使用者可以移動游標修改
    default_msg = f"{pred_label}: "
    
    print("\n" + "="*50)
    print(f"  GCA - Git Commit Assistant")
    print(f"  變更統計: +{additions}/-{deletions} 行, 共 {files_changed} 個檔案")
    print("="*50 + "\n")

    # 使用 prompt_toolkit 讓使用者編輯
    try:
        commit_msg = prompt(
            "Commit Message > ", 
            default=default_msg,
        ).strip()
    except KeyboardInterrupt:
        print("\n🚫 已取消")
        sys.exit(0)

    if not commit_msg or commit_msg == default_msg.strip():
        print("⚠️  訊息為空或未填寫描述，取消 Commit。")
        sys.exit(0)

    # F. 確認並執行
    # 再次確認 (因為直接 commit 有點危險)
    # 這裡直接執行 git commit -m "..."
    
    print(f"\n準備執行: git commit -m \"{commit_msg}\"")
    
    # 簡單的 Enter 確認
    check = input("按 [Enter] 執行，或按 [n] 取消: ")
    if check.lower() in ['n', 'no']:
        print("🚫 已取消")
        sys.exit(0)

    subprocess.run(["git", "commit", "-m", commit_msg])
    print("✅ Commit 完成！")

if __name__ == "__main__":
    main()