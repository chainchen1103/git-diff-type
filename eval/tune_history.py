#!/usr/bin/env python3
"""Choose how much a project's own history counts (gca-rs/src/history.rs).

gca reads the last 500 commits and counts the types people gave them, then
tilts the ranking toward that mix: p ∝ p · (q / prior)^weight, where q is the
project's mix smoothed with 10 commits' worth of the training mix. It then
tilts it the same way toward the mix of the commits among them that touched
a file the change touches. This scores weights on the held-out training
projects, each commit seeing only the commits before it, for the ranking
without and with the subject: first the project's mix, then, with the best
weight for it, the same files' mix, smoothed with each of a few numbers of
commits' worth of the training mix (gca uses 5).

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


def best_of(results):
    """The entry with the best first suggestion, the first one on a tie."""
    best = None
    for m, *settings in results:
        if best is None or m[0] > best[0][0]:
            best = (m, *settings)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--diff-model", required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--fusion-weight", type=float, default=0.25, help="as in out/subject_model.json")
    ap.add_argument("--fusion-prior-power", type=float, default=0.15)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--file-weights", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3])
    ap.add_argument("--file-pseudo-commits", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    args = ap.parse_args()

    rows, classes, y, p_diff, p_subject, prior = held_out(args.diff_model, args.alpha, args.train, args.val)
    n, same = counts_before(timelines([args.val]), rows, classes)
    print(f"{len(rows)} held-out commits by people, a median of {np.median(n.sum(axis=1)):.0f} typed "
          f"commits before each; {np.mean(same.sum(axis=1) > 0):.1%} have some that touched the same "
          f"files; first suggestion right / right type in top 3 / average recall per type")
    log_diff = np.log(p_diff + 1e-12)
    both = np.exp(log_softmax(log_diff + args.fusion_weight * np.log(p_subject + 1e-12)
                              - args.fusion_prior_power * np.log(prior)))
    for name, log_p in [("diff", log_diff), ("diff and subject", np.log(both + 1e-12))]:
        print(f"{name}: without history  %.3f / %.3f / %.3f" % metrics(log_p, classes, y))
        results = [(metrics(weigh(log_p, n, prior, w), classes, y), w) for w in args.weights]
        for m, w in results:
            print(f"  weight {w:<5} %.3f / %.3f / %.3f" % m)
        _, weight = best_of(results)
        print(f"  best first suggestion: weight {weight}")

        tilted = weigh(log_p, n, prior, weight)
        print(f"  then the commits to the same files, with weight {weight} for the project's mix:")
        for pseudo in args.file_pseudo_commits:
            results = [(metrics(weigh(tilted, same, prior, w, pseudo), classes, y), w)
                       for w in args.file_weights]
            for m, w in results:
                print(f"    {pseudo:g} pseudo-commits, weight {w:<5} %.4f / %.4f / %.4f" % m)
            print(f"    best first suggestion with {pseudo:g} pseudo-commits: weight {best_of(results)[1]}")


if __name__ == "__main__":
    main()
