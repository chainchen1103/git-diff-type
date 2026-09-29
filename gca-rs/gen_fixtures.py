#!/usr/bin/env python3
"""Generate fixtures for the Rust parity test.

Usage:
    python gca-rs/gen_fixtures.py --n 20
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from train_enhanced import (  # noqa: F401
    PathTokenExtractor,
    DiffSimilarityExtractor,
    FileExtensionExtractor,
    FEATURE_COLUMNS,
    prepare_features,
)
from verify_export import load_samples


def synthetic_samples():
    cases = [
        ("", 0, 0, 0),
        ("diff --git a/src/parser.py b/src/parser.py\n--- a/src/parser.py\n+++ b/src/parser.py\n@@ -1 +1 @@\n-return None\n+return value\n", 1, 1, 1),
        ("diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -0,0 +1 @@\n+Document installation and usage.\n", 1, 1, 0),
        ("diff --git a/old.rs b/old.rs\ndeleted file mode 100644\n--- a/old.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-fn obsolete() {}\n", 1, 0, 1),
        ("diff --git a/icon.png b/icon.png\nBinary files a/icon.png and b/icon.png differ\n", 1, 0, 0),
        ("diff --git a/src/測試.ts b/src/測試.ts\n--- a/src/測試.ts\n+++ b/src/測試.ts\n@@ -1 +1 @@\n-const café = '舊值';\n+const café = '新值';\n", 1, 1, 1),
        ("diff --git a/tests/test_math.py b/tests/test_math.py\n--- a/tests/test_math.py\n+++ b/tests/test_math.py\n@@ -1 +1 @@\n-assert add(1, 1) == 3\n+assert add(1, 1) == 2\n", 1, 1, 1),
        ("diff --git a/package.json b/package.json\n--- a/package.json\n+++ b/package.json\n@@ -1 +1 @@\n-1.0.0\n+2.0.0\n", 1, 1, 1),
        ("diff --git a/unicode.py b/unicode.py\n--- a/unicode.py\n+++ b/unicode.py\n@@ -1 +1 @@\n-cafe _var ²\n+cafe\u0301 _var ² +\n", 1, 1, 1),
    ]
    return prepare_features(pd.DataFrame(cases, columns=FEATURE_COLUMNS[:4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "out/model_v2.joblib"))
    ap.add_argument("--data", default=str(ROOT / "datasets/_merged.jsonl"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--synthetic", action="store_true", help="Use built-in cases without a dataset")
    args = ap.parse_args()
    if args.n <= 0:
        ap.error("--n must be positive")
    df = synthetic_samples().iloc[:args.n] if args.synthetic else load_samples(args.data, args.n, args.seed)

    model = joblib.load(args.model)
    probs = model.predict_proba(
        df[FEATURE_COLUMNS]
    )
    classes = list(model.classes_)

    cases = []
    for i, row in df.reset_index(drop=True).iterrows():
        cases.append({
            "diff_text": row["diff_text"],
            "numeric": [
                float(row["files_changed"]),
                float(row["additions"]),
                float(row["deletions"]),
                float(row["add_del_ratio"]),
            ],
            "expected_probs": probs[i].tolist(),
        })

    out = Path(args.out) if args.out else Path(__file__).parent / "tests" / ("synthetic_fixtures.json" if args.synthetic else "fixtures.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"classes": classes, "cases": cases}, f, ensure_ascii=False, allow_nan=False)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB, {len(cases)} cases)")


if __name__ == "__main__":
    main()
