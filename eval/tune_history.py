#!/usr/bin/env python3
"""Choose how much a project's own mix of types counts (gca-rs/src/history.rs).

gca reads the last 500 commits and counts the types people gave them, then
tilts the ranking toward that mix: p ∝ p · (q / prior)^weight, where q is the
project's mix smoothed with 10 commits' worth of the training mix. This
scores weights on the held-out training projects, each commit seeing only
the commits before it, for the ranking without and with the subject.

Usage (with the split and diff model from eval/holdout_split.py):
    python eval/tune_history.py --diff-model datasets/holdout/model_v2.joblib --alpha 0.9 \\
        --train datasets/holdout/train.jsonl datasets/holdout/external.jsonl \\
        --val datasets/holdout/validation.jsonl
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from history_sim import counts_before, timelines, weigh  # noqa: E402
from tune_fusion import held_out, metrics  # noqa: E402


def log_softmax(s):
    s = s - s.max(axis=1, keepdims=True)
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--diff-model", required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--fusion-weight", type=float, default=0.25, help="as in out/subject_model.json")
    ap.add_argument("--fusion-prior-power", type=float, default=0.15)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    args = ap.parse_args()

    rows, classes, y, p_diff, p_subject, prior = held_out(args.diff_model, args.alpha, args.train, args.val)
    n = counts_before(timelines([args.val]), rows, classes)
    print(f"{len(rows)} held-out commits by people, a median of {np.median(n.sum(axis=1)):.0f} typed "
          f"commits before each; first suggestion right / right type in top 3 / average recall per type")
    log_diff = np.log(p_diff + 1e-12)
    both = np.exp(log_softmax(log_diff + args.fusion_weight * np.log(p_subject + 1e-12)
                              - args.fusion_prior_power * np.log(prior)))
    for name, log_p in [("diff", log_diff), ("diff and subject", np.log(both + 1e-12))]:
        print(f"{name}: without history  %.3f / %.3f / %.3f" % metrics(log_p, classes, y))
        best = None
        for w in args.weights:
            m = metrics(weigh(log_p, n, prior, w), classes, y)
            print(f"  weight {w:<5} %.3f / %.3f / %.3f" % m)
            if best is None or m[0] > best[0][0]:
                best = (m, w)
        print(f"  best first suggestion: weight {best[1]}")


if __name__ == "__main__":
    main()
