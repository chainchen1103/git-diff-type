#!/usr/bin/env python3
"""
train_enhanced.py — Improved commit classifier training script.

Improvements over baseline:
1. Feature Engineering:
   - Diff Similarity: Jaccard index between added and deleted tokens (targets 'refactor').
   - Path Tokens: Tokenizes file paths from diff headers (targets 'test', 'ci', 'docs').
2. Model:
   - Replaced LogisticRegression with LinearSVC (better for high-dimensional sparse text data).
3. Data Handling:
   - Supports loading all .json/.jsonl files from a directory.
   - robustly handles both JSON arrays and JSONL formats.
4. Visualization:
   - Generates a Confusion Matrix heatmap image.

Usage:
  python train_enhanced.py --data datasets/ --model out/model_v2.joblib --cm_out out/confusion_matrix.png
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

# 嘗試匯入 ONNX 相關套件 (非必要)
try:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType, FloatTensorType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# -----------------------------------------------------------------------------
# Custom Feature Extractors
# -----------------------------------------------------------------------------

class DiffSimilarityExtractor(BaseEstimator, TransformerMixin):
    """
    計算 Diff 中新增部分與刪除部分的文字 Jaccard Similarity。
    Refactor 通常具有較高的相似度（搬移代碼、改名），而 Feat 通常很低。
    """
    def __init__(self):
        self.token_pattern = re.compile(r'(?u)\b\w+\b')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is a list/series of diff_text strings
        scores = []
        for diff in X:
            scores.append(self._compute_jaccard(str(diff)))
        return np.array(scores).reshape(-1, 1)

    def _compute_jaccard(self, diff_text: str) -> float:
        if not diff_text:
            return 0.0
        
        adds_tokens = set()
        dels_tokens = set()
        
        for line in diff_text.splitlines():
            # 跳過 header
            if line.startswith('+++') or line.startswith('---'):
                continue
            
            # 簡單分詞
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
    """
    從 Diff Text 中提取檔案路徑，並進行分詞。
    例如: "diff --git a/src/auth/login.spec.ts" -> "src auth login spec ts"
    這能捕捉 'spec', 'test', 'github', 'workflows' 等關鍵字。
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        paths_list = []
        for diff in X:
            paths_list.append(self._extract_path_tokens(str(diff)))
        return paths_list

    def _extract_path_tokens(self, diff_text: str) -> str:
        # 抓取 diff --git a/path/to/file b/...
        # 或者 +++ b/path/to/file
        tokens = set()
        
        # 簡單策略：抓取 +++ b/ 之後的路徑
        # 或是 diff --git a/ 之後的路徑
        # 這裡用一個簡單的 regex 來抓取可能的路徑字串
        path_matches = re.findall(r'^\+\+\+ b/(.+)$', diff_text, re.MULTILINE)
        if not path_matches:
            # 嘗試抓 diff --git
            path_matches = re.findall(r'^diff --git a/.+ b/(.+)$', diff_text, re.MULTILINE)
            
        for path in path_matches:
            # 將路徑拆解為 token: src/utils/foo.py -> src, utils, foo, py
            parts = re.split(r'[/\-_.]', path)
            for p in parts:
                if len(p) > 2: # 過濾太短的
                    tokens.add(p.lower())
                    
        return " ".join(tokens)


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------

def load_data(data_path: str):
    print(f"📂 Loading data from {data_path}...")
    data = []
    path = Path(data_path)
    
    # 支援讀取單一檔案 或 資料夾內所有 .json / .jsonl
    files = []
    if path.is_file():
        files = [path]
    else:
        # 遞迴或非遞迴抓取皆可，這裡抓取當層
        files = sorted(list(path.glob("*.json")) + list(path.glob("*.jsonl")))
    
    print(f"   Found {len(files)} file(s).")
    
    for p in files:
        with open(p, 'r', encoding='utf-8') as f:
            # 策略：先嘗試當作整個 JSON Array 讀取，失敗則當作 JSONL 讀取
            try:
                content = json.load(f)
                if isinstance(content, list):
                    data.extend(content)
                    continue
                elif isinstance(content, dict) and 'data' in content:
                     # 相容某些 { "data": [...] } 格式
                    data.extend(content['data'])
                    continue
            except json.JSONDecodeError:
                # 可能是 JSONL 或空檔，重置 cursor 逐行讀取
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

    # 1. 載入資料
    df = load_data(args.data)
    if df.empty:
        print("❌ No data found.")
        return

    # 簡單清理
    df['diff_text'] = df['diff_text'].fillna('')
    df['diff_text'] = df['diff_text'].apply(lambda x: x[:args.max_diff_len])
    
    # 計算基礎數值特徵
    for col in ['files_changed', 'additions', 'deletions']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
    
    # 增加特徵: Add/Del Ratio (對 feat/fix 有用)
    df['add_del_ratio'] = df['additions'] / (df['deletions'] + 1)

    print(f"📊 Training on {len(df)} samples...")
    print(f"   Labels: {df['label'].unique()}")

    X = df[['diff_text', 'files_changed', 'additions', 'deletions', 'add_del_ratio']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # 2. 定義 Pipeline
    
    # 特徵工程組合
    preprocessor = ColumnTransformer(
        transformers=[
            # A. Diff 內容本身的文字特徵 (TF-IDF)
            ('diff_tfidf', TfidfVectorizer(max_features=10000, stop_words='english'), 'diff_text'),
            
            # B. 路徑關鍵字 (Path Tokens) - 從 diff_text 提取
            ('path_bow', Pipeline([
                ('extractor', PathTokenExtractor()),
                ('vect', CountVectorizer(max_features=2000, binary=True))
            ]), 'diff_text'),
            
            # C. Diff 相似度 (Jaccard) - 解決 Refactor
            ('diff_sim', DiffSimilarityExtractor(), 'diff_text'),
            
            # D. 數值特徵標準化
            ('numeric', StandardScaler(), ['files_changed', 'additions', 'deletions', 'add_del_ratio']),
        ],
        remainder='drop'
    )

    # 模型: LinearSVC (比 LR 更適合稀疏高維特徵，且輕量)
    clf = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])

    # 3. 訓練
    print("🚀 Training model...")
    model.fit(X_train, y_train)

    # 4. 評估
    print("⚖️  Evaluating...")
    y_pred = model.predict(X_test)
    print("\n" + classification_report(y_test, y_pred))

    # 產生混淆矩陣
    labels = sorted(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix (Text):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # 5. 繪製混淆矩陣圖表
    if args.cm_out:
        print(f"🎨 Generating confusion matrix plot -> {args.cm_out}")
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
                
                # 手動標註數字
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
            print(f"⚠️  Failed to save plot: {e}")

    # 6. 儲存模型
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"\n💾 Model saved to {args.model}")
    
    # 儲存 labels 對照表
    with open(Path(args.model).parent / 'labels.txt', 'w') as f:
        f.write('\n'.join(labels))

    # 7. 匯出 ONNX (Optional)
    if HAS_ONNX and args.onnx:
        print("📦 Exporting to ONNX...")
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
            print(f"⚠️  ONNX export skipped/failed: {e}")

if __name__ == '__main__':
    main()