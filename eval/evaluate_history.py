#!/usr/bin/env python3
"""Score the ranking with the project's own mix of types (gca-rs/src/history.rs).

For each commit by people in the held-out sets, the types of the 500 commits
before it in the same repository tilt the ranking, as gca does with the
repository's log. Reports the diff model alone and with the subject, each
without and with that history.

Usage:
    python eval/evaluate_history.py \\
        --sets datasets/test_unseen_recent.jsonl datasets/test_unseen_older.jsonl \\
               datasets/test_seen_recent.jsonl \\
        --history datasets/train.jsonl --out eval/results_history.json

--history adds commits that come before the scored ones, such as the
training projects' older history for test_seen_recent.
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
sys.path.insert(0, str(ROOT / "eval"))
from dedupe import iter_rows  # noqa: E402
from evaluate_subject import score  # noqa: E402
from history_sim import counts_before, timelines, weigh  # noqa: E402
from train_enhanced import FEATURE_COLUMNS, prepare_features  # noqa: E402
from train_subject import Exported, subject_of  # noqa: E402

# As in gca-rs/src/history.rs.
WEIGHT, WEIGHT_WITH_SUBJECT = 0.1, 0.25


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--history", nargs="*", default=[], help="more commits to take history from")
    ap.add_argument("--diff-model", default=str(ROOT / "out/model_v2.joblib"))
    ap.add_argument("--subject-model", default=str(ROOT / "out/subject_model.json"))
    ap.add_argument("--out", default=str(ROOT / "eval/results_history.json"))
    args = ap.parse_args()

    diff_model = joblib.load(args.diff_model)
    subject_model = Exported(json.loads(Path(args.subject_model).read_text(encoding="utf-8")))
    classes = np.array(subject_model.classes)
    labels = json.loads((Path(args.diff_model).parent / "train_report.json").read_text())["labels"]
    prior = np.array([labels[c] for c in classes], dtype=float)
    prior /= prior.sum()
    by_repo = timelines(args.sets + args.history)

    results = {}
    for path in args.sets:
        rows = [r for r in iter_rows([Path(path)])
                if not r.get("is_bot") and isinstance(r.get("label"), str)
                and str(r.get("diff_text") or "").strip()]
        y = np.array([r["label"] for r in rows])
        p_diff = diff_model.predict_proba(prepare_features(pd.DataFrame(rows), 20000)[FEATURE_COLUMNS])
        p_both = np.array([subject_model.combine(d, subject_of(r.get("message"))) for d, r in zip(p_diff, rows)])
        n = counts_before(by_repo, rows, classes)
        log_diff, log_both = np.log(p_diff + 1e-12), np.log(p_both + 1e-12)
        name = Path(path).stem
        results[name] = {
            "commits": len(rows),
            "median_history": float(np.median(n.sum(axis=1))),
            "diff": score(log_diff, classes, y),
            "diff_history": score(weigh(log_diff, n, prior, WEIGHT), classes, y),
            "both": score(log_both, classes, y),
            "both_history": score(weigh(log_both, n, prior, WEIGHT_WITH_SUBJECT), classes, y),
        }
        line = "  ".join(f"{k} {v['top1']:.1%} / {v['top3']:.1%} / {v['macro_recall']:.3f}"
                         for k, v in results[name].items() if isinstance(v, dict))
        print(f"{name:20} {len(rows):6}  {line}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
