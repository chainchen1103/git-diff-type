#!/usr/bin/env python3
"""Choose how much the subject counts next to the diff (train_subject.py).

Trains the subject model on the training projects minus a few held out, and
scores p ∝ p_diff · p_subject^weight / prior^prior_power on the held-out
projects' commits (by people) for a grid of weights and prior powers. The
diff probabilities come from a diff model trained without those projects
(eval/holdout_split.py), with the same prior correction as the shipped one.

Usage (with the split and diff model from eval/holdout_split.py):
    python eval/tune_fusion.py --diff-model datasets/holdout/model_v2.joblib --alpha 0.9 \\
        --train datasets/holdout/train.jsonl datasets/holdout/external.jsonl \\
        --val datasets/holdout/validation.jsonl
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
from train_enhanced import FEATURE_COLUMNS, prepare_features  # noqa: E402
from train_subject import fit, subject_of, training_rows  # noqa: E402
import tune_prior  # noqa: E402


def metrics(scores, classes, y):
    order = np.argsort(-scores, axis=1)
    first = classes[order[:, 0]]
    top3 = (classes[order[:, :3]] == y[:, None]).any(axis=1)
    recall = [np.mean(first[y == c] == c) for c in classes if (y == c).sum() >= 20]
    return float((first == y).mean()), float(top3.mean()), float(np.mean(recall))


def held_out(diff_model, alpha, train, val, C=0.5):
    """Commits by people in the held-out projects, with the probabilities a
    diff model and a subject model trained without those projects give, and
    the subject model's training prior."""
    model = joblib.load(diff_model)
    if alpha:
        report = json.loads((Path(diff_model).parent / "train_report.json").read_text())
        tune_prior.apply(model, report["labels"], alpha)
    classes = np.array([str(c) for c in model.named_steps["clf"].classes_])

    rows = [r for r in iter_rows([Path(val)])
            if not r.get("is_bot") and isinstance(r.get("label"), str) and str(r.get("diff_text") or "").strip()]
    y = np.array([r["label"] for r in rows])
    p_diff = model.predict_proba(prepare_features(pd.DataFrame(rows), 20000)[FEATURE_COLUMNS])

    pairs = list(training_rows(train))
    subjects, labels = [s for s, _ in pairs], np.array([t for _, t in pairs])
    vec, clf = fit(subjects, labels, C=C)
    assert list(clf.classes_) == list(classes)
    p_subject = clf.predict_proba(vec.transform([subject_of(r.get("message")) for r in rows]))
    prior = np.array([(labels == c).mean() for c in classes])
    return rows, classes, y, p_diff, p_subject, prior


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--diff-model", required=True, help="diff model trained without the --val projects")
    ap.add_argument("--alpha", type=float, default=0.0, help="its prior correction, as shipped")
    ap.add_argument("--train", nargs="+", required=True, help="training data without the --val projects")
    ap.add_argument("--val", required=True, help="the held-out projects' commits")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    ap.add_argument("--prior-powers", type=float, nargs="+", default=[0.0, 0.1, 0.15, 0.2, 0.3])
    args = ap.parse_args()
    rows, classes, y, p_diff, p_subject, prior = held_out(args.diff_model, args.alpha, args.train,
                                                          args.val, args.C)

    print(f"{len(rows)} held-out commits by people; first suggestion right / right type in top 3 / "
          f"average recall per type")
    print("diff only        %.3f / %.3f / %.3f" % metrics(p_diff, classes, y))
    print("subject only     %.3f / %.3f / %.3f" % metrics(p_subject, classes, y))
    best = None
    for w in args.weights:
        for g in args.prior_powers:
            s = np.log(p_diff + 1e-12) + w * np.log(p_subject + 1e-12) - g * np.log(prior)
            m = metrics(s, classes, y)
            print(f"weight {w:<4} prior power {g:<4}  %.3f / %.3f / %.3f" % m)
            if best is None or m[0] > best[0][0]:
                best = (m, w, g)
    print(f"best first suggestion: weight {best[1]}, prior power {best[2]}")


if __name__ == "__main__":
    main()
