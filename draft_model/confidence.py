#!/usr/bin/env python3
"""How well the model's confidence (the mean log-probability of its greedy
subject's tokens) picks the drafts worth offering: for each threshold, how
many commits get a draft and how good those drafts are.

    python draft_model/confidence.py predictions.jsonl

gca offers a draft at THRESHOLD in gca-rs/src/t5draft/mod.rs and up.
"""
import argparse

from score import TEST, measures, read_predictions, read_tests

THRESHOLDS = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.8, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions")
    ap.add_argument("--test", default=str(TEST))
    args = ap.parse_args()
    tests = read_tests(args.test)
    preds = read_predictions(args.predictions)
    rows = [(preds[k]["confidence"], measures(preds[k]["greedy"], t["t"])) for k, t in tests.items() if k in preds]
    print(f"{'confidence':>12} {'offered':>8} {'exact':>6} {'<=1 word':>9} {'<=2 words':>10} {'half+':>6} {'first':>6}")
    for th in THRESHOLDS:
        sel = [m for c, m in rows if th is None or c >= th]
        if not sel:
            continue
        pct = lambda k: 100 * sum(m[k] for m in sel) / len(sel)  # noqa: E731
        name = "any" if th is None else f">= {th:.2f}"
        print(f"{name:>12} {100 * len(sel) / len(rows):7.1f}% {pct('exact'):6.1f} {pct('w1'):9.1f} "
              f"{pct('w2'):10.1f} {pct('half'):6.1f} {pct('first'):6.1f}")


if __name__ == "__main__":
    main()
