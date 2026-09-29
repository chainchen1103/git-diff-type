#!/usr/bin/env python3
"""Choose the classifier settings on a split of the training set.

With --split project (the default) about 10% of the training projects are
held out and scored, which mirrors how gca is used: on a repository the model
has never seen. With --split time the newest 10% of commits are held out
instead. Only training data is used, so the held-out test sets stay untouched
by tuning. Both parts are sampled to
keep it quick. With --calibrated the candidates are fitted exactly as the
shipped model is (sigmoid-calibrated, 3 folds) and ranked by probability.

The default criterion is macro recall: the first suggestion's recall averaged
over the types. Plain top-1 accuracy rewards a model that nearly always
suggests the most common type (`fix`), which is useless for pre-selection.

Usage:
    python eval/tune_c.py --data datasets/train.jsonl --C 0.1 0.3 1 3
    python eval/tune_c.py --data datasets/train.jsonl --C 0.3 \\
        --class-weight balanced none --calibrated
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_enhanced import build_preprocessor, iter_rows, to_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--C", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--class-weight", nargs="+", default=["balanced"], choices=["balanced", "none"])
    ap.add_argument("--calibrated", action="store_true", help="fit exactly like the shipped model")
    ap.add_argument("--train-sample", type=int, default=100000)
    ap.add_argument("--val-sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select", choices=["macro_recall", "top1"], default="macro_recall")
    ap.add_argument("--split", choices=["project", "time"], default="project")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    paths = [Path(args.data)]

    if args.split == "time":
        landed = sorted(r["committed_at"] for r in iter_rows(paths, False))
        split = landed[int(len(landed) * 0.9)]
        held_out = lambda r: r["committed_at"] >= split  # noqa: E731
        print(f"{len(landed)} commits; validating on those landed from {split}", flush=True)
    else:
        sizes = {}
        for r in iter_rows(paths, False):
            sizes[r["repo"]] = sizes.get(r["repo"], 0) + 1
        repos = sorted(sizes)
        random.Random(args.seed).shuffle(repos)
        chosen, total = set(), sum(sizes.values())
        for name in repos:
            if sum(sizes[c] for c in chosen) >= 0.1 * total:
                break
            chosen.add(name)
        held_out = lambda r: r["repo"] in chosen  # noqa: E731
        print(f"validating on {len(chosen)} held-out projects: {sorted(chosen)}", flush=True)

    older, newer, n_old, n_new = [], [], 0, 0
    for r in iter_rows(paths, False):
        if not held_out(r):
            n_old += 1
            bucket, cap, n = older, args.train_sample, n_old
        else:
            n_new += 1
            bucket, cap, n = newer, args.val_sample, n_new
        if len(bucket) < cap:
            bucket.append(r)
        elif (j := rng.randrange(n)) < cap:
            bucket[j] = r

    pre = build_preprocessor()
    Xtr = sp.csr_matrix(pre.fit_transform(to_frame(older, 20000)))
    ytr = np.array([r["label"] for r in older])
    Xva = sp.csr_matrix(pre.transform(to_frame(newer, 20000)))
    yva = np.array([r["label"] for r in newer])

    results = []
    for weight in args.class_weight:
        for c in args.C:
            t = time.time()
            clf = LinearSVC(C=c, class_weight=None if weight == "none" else weight,
                            random_state=args.seed, max_iter=5000)
            if args.calibrated:
                clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
                scores = clf.fit(Xtr, ytr).predict_proba(Xva)
            else:
                scores = clf.fit(Xtr, ytr).decision_function(Xva)
            order = np.argsort(-scores, axis=1)
            classes = clf.classes_
            top1 = float((classes[order[:, 0]] == yva).mean())
            top3 = float(np.mean([y in classes[o[:3]] for y, o in zip(yva, order)]))
            pred = classes[order[:, 0]]
            recalls = [float((pred[yva == k] == k).mean()) for k in classes if (yva == k).sum() >= 20]
            macro = float(np.mean(recalls))
            results.append({"C": c, "class_weight": weight, "top1": top1, "top3": top3,
                            "macro_recall": macro})
            print(f"C={c:<6} class_weight={weight:<9} top-1 {top1:.2%}  top-3 {top3:.2%}"
                  f"  macro recall {macro:.3f}   ({time.time() - t:.0f}s)", flush=True)
    best = max(results, key=lambda r: (r[args.select], r["top3"]))
    print(json.dumps({"best": best, "all": results}))


if __name__ == "__main__":
    main()
