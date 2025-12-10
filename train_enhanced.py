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
   - Parses paths directly from diff_text (no need to change miner.py schema).

Usage:
  python train_enhanced.py --data datasets/combined_train.jsonl --model out/model_v2.joblib
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
    
    # 支援讀取單一 jsonl 或資料夾內所有 jsonl
    files = [path] if path.is_file() else list(path.glob("*.jsonl"))
    
    for p in files:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Train enhanced commit classifier")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset(s)")
    parser.add_argument("--model", default="out/model_v2.joblib", help="Output model path")
    parser.add_argument("--onnx", default="out/model_v2.onnx", help="Output ONNX path")
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

    # 顯示 Confusion Matrix (Text)
    labels = sorted(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # 5. 儲存
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"\n💾 Model saved to {args.model}")
    
    # 儲存 labels 對照表
    with open(Path(args.model).parent / 'labels.txt', 'w') as f:
        f.write('\n'.join(labels))

    # 6. 匯出 ONNX (Optional)
    if HAS_ONNX and args.onnx:
        print("📦 Exporting to ONNX...")
        try:
            # 定義輸入型別
            # 注意: 這裡必須與 ColumnTransformer 的輸入對齊
            # 雖然我們傳入 DataFrame，但在 ONNX 中通常定義為幾個 Tensor
            initial_types = [
                ('diff_text', StringTensorType([None, 1])),
                ('files_changed', FloatTensorType([None, 1])),
                ('additions', FloatTensorType([None, 1])),
                ('deletions', FloatTensorType([None, 1])),
                ('add_del_ratio', FloatTensorType([None, 1])),
            ]
            
            # ONNX export 對於自定義 Transformer (DiffSimilarityExtractor) 可能會遇到困難
            # 因為它包含 Python code。
            # 為了讓它能被 export，通常需要註冊 custom converter，這比較複雜。
            # 如果只是要在 Python 環境用，joblib 就夠了。
            # 如果一定要 ONNX，這裡可能需要簡化特徵或寫 converter。
            # 為了保持腳本簡單，這裡先做一個 try-catch 提醒。
            
            onx = to_onnx(model, X_train[:1], options={id(clf): {'zipmap': False}})
            with open(args.onnx, "wb") as f:
                f.write(onx.SerializeToString())
            print(f"   ONNX saved to {args.onnx}")
        except Exception as e:
            print(f"⚠️  ONNX export skipped/failed (likely due to custom transformers): {e}")
            print("   (To fix this, you'd need to register custom ONNX converters or rely on Python runtime)")

if __name__ == '__main__':
    main()