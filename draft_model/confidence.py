#!/usr/bin/env python3
"""How well the model's confidence (the mean log-probability of its greedy
subject's tokens) picks the drafts worth offering: for each threshold, how
many commits get a draft and how good those drafts are.

    python draft_model/confidence.py predictions.jsonl [--datasets datasets]

The second table also holds back a draft that repeats the subject of a
single earlier commit among the 500 before it, as gca does
(ModelDraft::why_not_offered in gca-rs/src/t5draft/mod.rs): the model
sometimes copies the commit before, which is rarely right, while a subject
several earlier commits share (`bump version`) usually is. It needs the
datasets the test commits come from.
"""
import argparse
import sys
from pathlib import Path

from score import TEST, measures, norm, read_predictions, read_tests

HERE = Path(__file__).resolve().parent
THRESHOLDS = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.8, None]


def table(rows, title):
    """rows: (confidence, measures, offered apart from the threshold)."""
    print(f"\n{title}")
    print(f"{'confidence':>12} {'offered':>8} {'exact':>6} {'<=1 word':>9} {'<=2 words':>10} {'half+':>6} {'first':>6}")
    for th in THRESHOLDS:
        sel = [m for c, m, ok in rows if ok and (th is None or c >= th)]
        if not sel:
            continue
        pct = lambda k: 100 * sum(m[k] for m in sel) / len(sel)  # noqa: E731
        name = "any" if th is None else f">= {th:.2f}"
        print(f"{name:>12} {100 * len(sel) / len(rows):7.1f}% {pct('exact'):6.1f} {pct('w1'):9.1f} "
              f"{pct('w2'):10.1f} {pct('half'):6.1f} {pct('first'):6.1f}")


def repeats(tests, preds, datasets):
    """For each test commit, how many of the 500 typed commits before it in
    its project have the subject the model wrote."""
    sys.path.insert(0, str(HERE))
    from prepare import DEPTH, Timelines

    timelines = Timelines([datasets / "test_unseen_older.jsonl", datasets / "test_unseen_recent.jsonl"])
    out = {}
    for k, t in tests.items():
        commits, position = timelines.repos[t["repo"]][:2]
        i = position[k]
        wanted = norm(preds[k]["greedy"])
        out[k] = sum(1 for c in commits[max(0, i - DEPTH):i] if norm(c[3].split(": ", 1)[-1]) == wanted)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions")
    ap.add_argument("--test", default=str(TEST))
    ap.add_argument("--datasets", default=str(HERE.parent / "datasets"))
    args = ap.parse_args()
    tests = read_tests(args.test)
    preds = read_predictions(args.predictions)
    tests = {k: t for k, t in tests.items() if k in preds}
    scored = {k: (preds[k]["confidence"], measures(preds[k]["greedy"], t["t"])) for k, t in tests.items()}
    table([(c, m, True) for c, m in scored.values()], "By confidence alone:")
    datasets = Path(args.datasets)
    if (datasets / "test_unseen_older.jsonl").exists():
        seen = repeats(tests, preds, datasets)
        table([(c, m, seen[k] != 1) for k, (c, m) in scored.items()],
              "Holding back drafts that repeat a single earlier commit's subject, as gca does:")


if __name__ == "__main__":
    main()
