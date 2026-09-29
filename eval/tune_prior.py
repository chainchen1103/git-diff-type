#!/usr/bin/env python3
"""Choose how strongly to correct the model for how common each type is.

The calibrated classifier learns how often each type occurs in its training
data, so when a diff is ambiguous it falls back on `fix`, the most common
type, and it almost never suggests `refactor` or `perf` first. Dividing each
type's probability by its training frequency raised to a power alpha (logit
adjustment) removes that pull: alpha = 0 keeps the model as trained, alpha = 1
removes the frequencies entirely. The correction shifts the intercepts of the
calibration sigmoids, so the corrected model is still an ordinary
scikit-learn model: export_model.py, verify_export.py and the CLI need no
changes.

The frequencies come from train_report.json next to the model. Choose alpha
on data that is not a test set you report.

Usage:
    python eval/tune_prior.py --model out/model_v2.joblib --data datasets/test_seen_recent.jsonl
    python eval/tune_prior.py --model out/model_v2.joblib --write 0.5
"""
import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dedupe import iter_rows  # noqa: E402
from train_enhanced import FEATURE_COLUMNS, prepare_features  # noqa: E402


def shifts(model, counts, alpha):
    """Per calibrated fold, the change to each calibration intercept, in the
    order of the fold's calibrators."""
    total = sum(counts.values())
    log_prior = {c: math.log(n / total) for c, n in counts.items()}
    out = []
    for cc in model.named_steps["clf"].calibrated_classifiers_:
        classes = list(cc.estimator.classes_)
        if len(cc.calibrators) == 1 and len(classes) == 2:
            # One sigmoid for the second class; the first gets the rest.
            out.append(np.array([alpha * (log_prior[classes[1]] - log_prior[classes[0]])]))
        else:
            out.append(np.array([alpha * log_prior[c] for c in classes]))
    return out


def apply(model, counts, alpha):
    for cc, delta in zip(model.named_steps["clf"].calibrated_classifiers_, shifts(model, counts, alpha)):
        for calibrator, d in zip(cc.calibrators, delta):
            calibrator.b_ = calibrator.b_ + d


def probabilities(scores, calibrators, delta):
    """Mirror of scikit-learn's calibrated predict_proba for one fold, with
    the intercepts shifted by delta. scores: (n, classes) decision values."""
    a = np.array([float(c.a_) for c in calibrators])
    b = np.array([float(c.b_) for c in calibrators]) + delta
    proba = expit(-(scores * a + b))
    total = proba.sum(axis=1, keepdims=True)
    return np.divide(proba, total, out=np.full_like(proba, 1 / proba.shape[1]), where=total != 0)


def metrics(proba, classes, y):
    order = np.argsort(-proba, axis=1)
    first = classes[order[:, 0]]
    top3 = np.array([label in classes[o[:3]] for label, o in zip(y, order)])
    recall = {c: float((first[y == c] == c).mean()) for c in classes if (y == c).sum() >= 20}
    return {"top1": float((first == y).mean()), "top3": float(top3.mean()),
            "macro_recall": float(np.mean(list(recall.values()))), "recall": recall}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(ROOT / "out/model_v2.joblib"))
    ap.add_argument("--data", help="JSONL commits to choose alpha on (bot commits are skipped)")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--write", type=float, metavar="ALPHA",
                    help="apply this alpha to the model file and record it in train_report.json")
    args = ap.parse_args()

    model_path = Path(args.model)
    report_path = model_path.parent / "train_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    counts = report["labels"]
    model = joblib.load(model_path)
    if report.get("prior_correction"):
        raise SystemExit(f"{model_path} is already corrected (alpha {report['prior_correction']})")

    if args.write is not None:
        apply(model, counts, args.write)
        joblib.dump(model, model_path, compress=3)
        report["prior_correction"] = args.write
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"applied alpha {args.write} to {model_path}")
        return
    if not args.data:
        ap.error("--data is needed to compare values of alpha")

    if len(model.named_steps["clf"].classes_) < 3:
        ap.error("comparing values of alpha needs a model with three or more types")
    rows = [r for r in iter_rows([Path(args.data)]) if not r.get("is_bot")]
    frame = prepare_features(pd.DataFrame(rows))[FEATURE_COLUMNS]
    X = model.named_steps["preprocessor"].transform(frame)
    y = np.array([r["label"] for r in rows])
    clf = model.named_steps["clf"]
    classes = clf.classes_
    folds = [(cc.estimator.decision_function(X), cc.calibrators) for cc in clf.calibrated_classifiers_]
    print(f"{len(rows)} commits by people in {args.data}")
    print(f"{'alpha':>6} {'top-1':>7} {'top-3':>7} {'macro':>7}   first-suggestion recall per type")
    results = []
    for alpha in args.alpha:
        proba = sum(probabilities(s, cal, d) for (s, cal), d in zip(folds, shifts(model, counts, alpha)))
        m = metrics(proba / len(folds), classes, y)
        results.append({"alpha": alpha, **m})
        per_type = "  ".join(f"{c} {v:.0%}" for c, v in sorted(m["recall"].items(), key=lambda kv: -counts[kv[0]]))
        print(f"{alpha:6.2f} {m['top1']:7.1%} {m['top3']:7.1%} {m['macro_recall']:7.3f}   {per_type}", flush=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
