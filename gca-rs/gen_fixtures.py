#!/usr/bin/env python3
"""Generate fixtures for the Rust parity test.

Usage:
    python gca-rs/gen_fixtures.py --n 20
"""
import argparse
import json
import random
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
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "out/model_v2.joblib"))
    ap.add_argument("--data", default=str(ROOT / "datasets/_merged.jsonl"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).parent / "tests/fixtures.json"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]

    df = pd.DataFrame(rows)
    df["diff_text"] = df["diff_text"].fillna("").astype(str).str.slice(0, 20000)
    for col in ["files_changed", "additions", "deletions"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    df["add_del_ratio"] = df["additions"] / (df["deletions"] + 1)

    model = joblib.load(args.model)
    probs = model.predict_proba(
        df[["diff_text", "files_changed", "additions", "deletions", "add_del_ratio"]]
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"classes": classes, "cases": cases}, f, ensure_ascii=False)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB, {len(cases)} cases)")


if __name__ == "__main__":
    main()
