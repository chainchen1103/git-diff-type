#!/usr/bin/env python3
"""Score the ranking with the project's own history (gca-rs/src/history.rs).

For each commit by people in the held-out sets, the types of the 500 commits
before it in the same repository tilt the ranking, then the types of those
that touched a file it touches, and then how the types its author gave their
latest ones among them differ from what the models say about those commits,
as gca does with the repository's log. Reports the diff model alone and with
the subject, each without history and with each of those steps in turn.

Usage:
    python eval/evaluate_history.py \\
        --sets datasets/test_unseen_recent.jsonl datasets/test_unseen_older.jsonl \\
               datasets/test_seen_recent.jsonl \\
        --history datasets/train.jsonl --out eval/results_history.json

--history adds commits that come before the scored ones, such as the
training projects' older history for test_seen_recent. The models read again
the latest commits of each author found there, as gca reads again the
user's own.
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
from history_sim import (FILE_PSEUDO_COMMITS, counts_before, latest_by_author, own_choices,  # noqa: E402
                         scored, timelines, weigh, weigh_own, window_start)
from train_enhanced import FEATURE_COLUMNS, prepare_features  # noqa: E402
from train_subject import Exported, subject_of  # noqa: E402

# As in gca-rs/src/history.rs.
WEIGHT, WEIGHT_WITH_SUBJECT = 0.1, 0.25
FILE_WEIGHT, FILE_WEIGHT_WITH_SUBJECT = 0.1, 0.15
OWN_WEIGHT, OWN_WEIGHT_WITH_SUBJECT = 0.1, 0.15


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

    def predict(rows):
        p_diff = diff_model.predict_proba(prepare_features(pd.DataFrame(rows), 20000)[FEATURE_COLUMNS])
        p_both = np.array([subject_model.combine(d, subject_of(r.get("message"))) for d, r in zip(p_diff, rows)])
        return p_diff, p_both

    # The commits the models could read again as someone's own: every scored
    # commit in the sets, and the latest ones of their authors in --history.
    pool = {"keys": [], "times": [], "labels": [], "diff": [], "both": []}

    def add(rows, p_diff, p_both):
        pool["keys"] += [(r.get("repo"), r.get("author") or "") for r in rows]
        pool["times"] += [r.get("committed_at") or "" for r in rows]
        pool["labels"] += [r["label"] for r in rows]
        pool["diff"].append(p_diff)
        pool["both"].append(p_both)

    sets, shas = {}, set()
    for path in args.sets:
        rows = [r for r in iter_rows([Path(path)]) if scored(r)]
        p_diff, p_both = predict(rows)
        n, same = counts_before(by_repo, rows, classes)
        first = len(pool["keys"])
        add(rows, p_diff, p_both)
        shas.update(r.get("sha") for r in rows)
        sets[Path(path).stem] = {"targets": range(first, first + len(rows)), "y": np.array([r["label"] for r in rows]),
                                 "p_diff": p_diff, "p_both": p_both, "n": n, "same": same,
                                 "starts": window_start(by_repo, rows)}
    if args.history:
        wanted = latest_by_author(by_repo, set(pool["keys"]), exclude=shas)
        rows = [r for r in iter_rows([Path(p) for p in args.history]) if r.get("sha") in wanted and scored(r)]
        if rows:
            add(rows, *predict(rows))
    shown_by = {k: np.vstack(pool[k]) for k in ("diff", "both")}

    results = {}
    for name, s in sets.items():
        y, n, same = s["y"], s["n"], s["same"]
        own = {k: own_choices(pool["keys"], pool["times"], pool["labels"], shown_by[k], classes,
                              s["targets"], s["starts"]) for k in ("diff", "both")}
        log_diff, log_both = np.log(s["p_diff"] + 1e-12), np.log(s["p_both"] + 1e-12)
        diff_history = weigh(log_diff, n, prior, WEIGHT)
        both_history = weigh(log_both, n, prior, WEIGHT_WITH_SUBJECT)
        diff_files = weigh(diff_history, same, prior, FILE_WEIGHT, FILE_PSEUDO_COMMITS)
        both_files = weigh(both_history, same, prior, FILE_WEIGHT_WITH_SUBJECT, FILE_PSEUDO_COMMITS)
        results[name] = {
            "commits": len(y),
            "median_history": float(np.median(n.sum(axis=1))),
            "with_same_file_history": float(np.mean(same.sum(axis=1) > 0)),
            "with_own_commits": float(np.mean(own["diff"][0].sum(axis=1) > 0)),
            "diff": score(log_diff, classes, y),
            "diff_history": score(diff_history, classes, y),
            "diff_history_files": score(diff_files, classes, y),
            "diff_history_files_own": score(weigh_own(diff_files, *own["diff"], prior, OWN_WEIGHT), classes, y),
            "both": score(log_both, classes, y),
            "both_history": score(both_history, classes, y),
            "both_history_files": score(both_files, classes, y),
            "both_history_files_own": score(weigh_own(both_files, *own["both"], prior, OWN_WEIGHT_WITH_SUBJECT),
                                            classes, y),
        }
        line = "  ".join(f"{k} {v['top1']:.1%} / {v['top3']:.1%} / {v['macro_recall']:.3f}"
                         for k, v in results[name].items() if isinstance(v, dict))
        print(f"{name:20} {len(y):6}  {line}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
