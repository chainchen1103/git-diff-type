#!/usr/bin/env python3
"""Choose how much a project's own history counts (gca-rs/src/history.rs).

gca reads the last 500 commits and counts the types people gave them, then
tilts the ranking toward that mix: p ∝ p · (q / prior)^weight, where q is the
project's mix smoothed with 10 commits' worth of the training mix. It then
tilts it the same way toward the mix of the commits among them that touched
a file the change touches, and then by the user's own latest commits among
them: p ∝ p · ((chosen + s · prior) / (shown + s · prior))^weight, where
chosen counts the types the user gave them and shown sums the probabilities
the models give them. This scores the settings on the held-out training
projects, each commit seeing only the commits before it and its author
standing for the user, for the ranking without and with the subject: first
the project's mix; with the best weight for it, the same files' mix,
smoothed with each of a few numbers of commits' worth of the training mix
(gca uses 5); and with the best weight for that, the author's own commits,
for a few numbers of them and of pseudo-commits (gca reads 10, with 1).

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
from history_sim import (FILE_PSEUDO_COMMITS, OWN_COMMITS, OWN_PSEUDO_COMMITS, counts_before,  # noqa: E402
                         own_choices, timelines, weigh, weigh_own, window_start)
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
    ap.add_argument("--own-commits", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--own-weights", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--own-pseudo-commits", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0])
    args = ap.parse_args()

    rows, classes, y, p_diff, p_subject, prior = held_out(args.diff_model, args.alpha, args.train, args.val)
    by_repo = timelines([args.val])
    n, same = counts_before(by_repo, rows, classes)
    starts = window_start(by_repo, rows)
    keys = [(r.get("repo"), r.get("author") or "") for r in rows]
    times = [r.get("committed_at") or "" for r in rows]
    labels = [r["label"] for r in rows]
    print(f"{len(rows)} held-out commits by people, a median of {np.median(n.sum(axis=1)):.0f} typed "
          f"commits before each; {np.mean(same.sum(axis=1) > 0):.1%} have some that touched the same "
          f"files; first suggestion right / right type in top 3 / average recall per type")
    log_diff = np.log(p_diff + 1e-12)
    both = np.exp(log_softmax(log_diff + args.fusion_weight * np.log(p_subject + 1e-12)
                              - args.fusion_prior_power * np.log(prior)))
    for name, log_p, shown in [("diff", log_diff, p_diff), ("diff and subject", np.log(both + 1e-12), both)]:
        print(f"{name}: without history  %.3f / %.3f / %.3f" % metrics(log_p, classes, y))
        results = [(metrics(weigh(log_p, n, prior, w), classes, y), w) for w in args.weights]
        for m, w in results:
            print(f"  weight {w:<5} %.3f / %.3f / %.3f" % m)
        _, weight = best_of(results)
        print(f"  best first suggestion: weight {weight}")

        tilted = weigh(log_p, n, prior, weight)
        print(f"  then the commits to the same files, with weight {weight} for the project's mix:")
        file_weight = None
        for pseudo in args.file_pseudo_commits:
            results = [(metrics(weigh(tilted, same, prior, w, pseudo), classes, y), w)
                       for w in args.file_weights]
            for m, w in results:
                print(f"    {pseudo:g} pseudo-commits, weight {w:<5} %.4f / %.4f / %.4f" % m)
            best = best_of(results)[1]
            print(f"    best first suggestion with {pseudo:g} pseudo-commits: weight {best}")
            if pseudo == FILE_PSEUDO_COMMITS:
                file_weight = best
        if file_weight is None:
            continue

        files = weigh(tilted, same, prior, file_weight, FILE_PSEUDO_COMMITS)
        print(f"  then the author's own commits, with weight {file_weight} and {FILE_PSEUDO_COMMITS:g} "
              f"pseudo-commits for the same files:")
        for depth in args.own_commits:
            chosen, shown_sum = own_choices(keys, times, labels, shown, classes, range(len(rows)), starts, depth)
            print(f"    the latest {depth}: {np.mean(chosen.sum(axis=1) > 0):.1%} of the commits have some")
            for pseudo in args.own_pseudo_commits:
                results = [(metrics(weigh_own(files, chosen, shown_sum, prior, w, pseudo), classes, y), w)
                           for w in args.own_weights]
                if depth == OWN_COMMITS and pseudo == OWN_PSEUDO_COMMITS:
                    for m, w in results:
                        print(f"      {pseudo:g} pseudo-commits, weight {w:<5} %.4f / %.4f / %.4f" % m)
                m, w = best_of(results)
                print(f"      best first suggestion with {pseudo:g} pseudo-commits: weight {w}  "
                      "%.4f / %.4f / %.4f" % m)

if __name__ == "__main__":
    main()
