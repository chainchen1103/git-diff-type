#!/usr/bin/env python3
"""Generate fixtures for the Rust parity test.

Usage:
    python gca-rs/gen_fixtures.py --n 20
    python gca-rs/gen_fixtures.py --synthetic --out gca-rs/tests/synthetic_fixtures.json
    python gca-rs/gen_fixtures.py --behavior
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
    BEHAVIOR_FEATURES,
    BehaviorExtractor,
    behavior_features,
    PathTokenExtractor,
    DiffSimilarityExtractor,
    FileExtensionExtractor,
    FEATURE_COLUMNS,
    prepare_features,
)
from verify_export import load_samples


# Diffs that exercise the edges of the behavior features: whitespace and
# line endings, non-ASCII text, quotes, comment syntaxes, file headers, test
# paths, short and cut lines.
BEHAVIOR_CASES = [
    "",
    "diff --git a/x b/x\n",
    "diff --git a/src/a.rs b/src/b.rs\nsimilarity index 90%\nrename from src/a.rs\nrename to src/b.rs\n--- a/src/a.rs\n+++ b/src/b.rs\n@@ -1,4 +1,4 @@\n-fn old_name(x: u32) -> u32 { x + 1 }\n+fn new_name(y: u32) -> u32 { y + 2 }\n-    let cache = compute();\n+  let  cache  =  compute();\n context\n-// a comment here\n+// a comment here\ndiff --git a/tests/t.rs b/tests/t.rs\nnew file mode 100644\n--- /dev/null\n+++ b/tests/t.rs\n@@ -0,0 +1 @@\n+assert!(fast_path());\n",
    "diff --git a/w.txt b/w.txt\n--- a/w.txt\n+++ b/w.txt\n@@ -1,3 +1,3 @@\n-hello world\r\n+hello  world\r\n-\tindented line\n+    indented line\n-\x0bvertical\x0ctab form\n+vertical tab form\n",
    "diff --git a/src/café.ts b/src/café.ts\n--- a/src/café.ts\n+++ b/src/café.ts\n@@ -1,3 +1,3 @@\n-const café = '舊值';\n+const café = '新值';\n-let 變數 = 1;\n+let 變數 = 2;\n-ſpeed Keep\n+ſpeed CACHE Fast \n",
    "diff --git a/q.py b/q.py\n--- a/q.py\n+++ b/q.py\n@@ -1,4 +1,4 @@\n-x = \"a b\" + 'c' + `d`\n+y = \"e f\" + 'g' + `h`\n-s = \"unterminated\n+t = \"unterminated\n-v1 = 10.5\n+v22 = 3.25\n",
    "diff --git a/c.sql b/c.sql\n--- a/c.sql\n+++ b/c.sql\n@@ -1,8 +1,8 @@\n--- a sql comment\n+-- a sql comment!\n-  # python comment\n+/* c comment */\n- * doc line\n+<!-- html -->\n-;; lisp\n+\"\"\"docstring\"\"\"\n-'''single'''\n+not a comment\n",
    "diff --git a/old.rs b/old.rs\ndeleted file mode 100644\n--- a/old.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-fn obsolete() {}\ndiff --git a/icon.png b/icon.png\nBinary files a/icon.png and b/icon.png differ\ndiff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n",
    "diff --git a/src/foo_test.go b/src/foo_test.go\n--- a/src/foo_test.go\n+++ b/src/foo_test.go\n@@ -1 +1 @@\n-a\n+b\ndiff --git a/pkg/spec/x.rb b/pkg/spec/x.rb\ndiff --git a/web/__tests__/y.js b/web/__tests__/y.js\ndiff --git a/test_z.py b/test_z.py\ndiff --git a/app/attest/x.ts b/app/attest/x.ts\ndiff --git a/src/contest.ts b/src/contest.ts\ndiff --git a/src/a.test.ts b/src/a.test.ts\ndiff --git a/e2e/login.cy.js b/e2e/login.cy.js\n",
    "diff --git a/m.c b/m.c\n--- a/m.c\n+++ b/m.c\n@@ -1,6 +1,6 @@\n-}\n+}\n-end\n+end\n-x\n+y\n-    return value;\n+    return value;\n\\ No newline at end of file\n@@ -20,2 +20,2 @@\n-int batch_size = 64;\n+int batch_size = 128; // parallel pool\n",
    "diff --git a/cut.py b/cut.py\n--- a/cut.py\n+++ b/cut.py\n@@ -1,2 +1,2 @@\n-def compute(self, memo):\n+def compute(self, memo, lazy=Tru",
    "diff --git a/u.txt b/u.txt\n--- a/u.txt\n+++ b/u.txt\n@@ -1 +1 @@\n-line one still one\n+line one still one\n",
    "diff --git a/big.md b/big.md\n--- a/big.md\n+++ b/big.md\n@@ -1,3 +1,3 @@\n+Throughput and LATENCY improved; OPTIMIZED allocation, reserve capacity.\n+benchmark: perf_counter, debounce, throttle, concurrency\n",
]


def behavior_fixtures(out):
    cases = [{"diff_text": d, "expected": behavior_features(d)} for d in BEHAVIOR_CASES]
    out.write_text(json.dumps({"features": list(BEHAVIOR_FEATURES), "cases": cases}, ensure_ascii=False,
                              allow_nan=False), encoding="utf-8")
    print(f"wrote {out} ({len(cases)} cases)")


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
        # moved and renamed code, a renamed file and a comment: the behavior features
        ("diff --git a/src/util.ts b/src/helpers.ts\nsimilarity index 88%\nrename from src/util.ts\nrename to src/helpers.ts\n--- a/src/util.ts\n+++ b/src/helpers.ts\n@@ -1,3 +1,3 @@\n-export function parseDate(input: string) {\n+export function parseDay(value: string) {\n-  return new Date(input);\n+  return new Date(value);\n // keep the old behavior\n", 1, 2, 2),
        ("diff --git a/src/cache.rs b/src/cache.rs\n--- a/src/cache.rs\n+++ b/src/cache.rs\n@@ -3,2 +3,3 @@\n-    let items = load();\n+    let items = CACHE.get_or_init(load);\n+    // reuse the parsed items: faster startup\n", 1, 2, 1),
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
    ap.add_argument("--behavior", action="store_true",
                    help="write the behavior feature fixtures instead (no model needed)")
    args = ap.parse_args()
    if args.behavior:
        return behavior_fixtures(Path(args.out) if args.out else Path(__file__).parent / "tests" / "behavior_fixtures.json")
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
