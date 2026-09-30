#!/usr/bin/env python3
"""Score the type suggestion when the subject is known too.

For each held-out set, compares the diff model alone, the subject model alone
and the two combined as gca combines them (train_subject.py), on commits
written by people. The subject is the one the author wrote, without its type
prefix. The subject model runs from its exported JSON, the way the CLI runs it.

Usage:
    python eval/evaluate_subject.py \\
        --sets datasets/test_unseen_recent.jsonl datasets/test_unseen_older.jsonl \\
               datasets/test_seen_recent.jsonl \\
        --out eval/results_subject.json
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dedupe import iter_rows  # noqa: E402
from train_enhanced import FEATURE_COLUMNS, prepare_features  # noqa: E402
from train_subject import Exported, subject_of  # noqa: E402


def score(p, classes, y):
    order = np.argsort(-p, axis=1)
    first = classes[order[:, 0]]
    top3 = (classes[order[:, :3]] == y[:, None]).any(axis=1)
    recall = {c: float(np.mean(first[y == c] == c)) for c in classes if (y == c).sum() >= 20}
    return {"top1": float((first == y).mean()), "top3": float(top3.mean()),
            "macro_recall": float(np.mean(list(recall.values()))), "recall": recall}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--diff-model", default=str(ROOT / "out/model_v2.joblib"))
    ap.add_argument("--subject-model", default=str(ROOT / "out/subject_model.json"))
    ap.add_argument("--out", default=str(ROOT / "eval/results_subject.json"))
    args = ap.parse_args()

    diff_model = joblib.load(args.diff_model)
    subject_model = Exported(json.loads(Path(args.subject_model).read_text(encoding="utf-8")))
    classes = np.array(subject_model.classes)
    assert [str(c) for c in diff_model.named_steps["clf"].classes_] == list(classes)

    results = {}
    for path in args.sets:
        rows = [r for r in iter_rows([Path(path)])
                if not r.get("is_bot") and isinstance(r.get("label"), str)
                and str(r.get("diff_text") or "").strip()]
        y = np.array([r["label"] for r in rows])
        p_diff = diff_model.predict_proba(prepare_features(pd.DataFrame(rows), 20000)[FEATURE_COLUMNS])
        subjects = [subject_of(r.get("message")) for r in rows]
        p_subject = np.array([subject_model.predict_proba(s) for s in subjects])
        p_both = np.array([subject_model.combine(d, s) for d, s in zip(p_diff, subjects)])
        name = Path(path).stem
        results[name] = {"commits": len(rows), "diff": score(p_diff, classes, y),
                         "subject": score(p_subject, classes, y), "both": score(p_both, classes, y)}
        line = "  ".join(f"{k} {v['top1']:.1%} / {v['top3']:.1%} / {v['macro_recall']:.3f}"
                         for k, v in results[name].items() if k != "commits")
        print(f"{name:20} {len(rows):6}  {line}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
