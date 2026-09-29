#!/usr/bin/env python3
"""Score an exported model (the JSON the CLI embeds) on held-out commits.

Uses verify_export.py's NumPy forward pass. gca-rs/tests/parity.rs checks the
Rust CLI against the same code path, so these numbers are what `gca` shows.

The headline numbers are for commits written by people; bot-authored commits
(dependency updates and the like) are reported separately. Besides top-1 and
top-3 accuracy it reports macro recall, the first suggestion's recall averaged
over the types.

Usage:
    python eval/evaluate.py --json out/model_v2.json \\
        --sets datasets/test_unseen_recent.jsonl datasets/test_unseen_older.jsonl \\
               datasets/test_seen_recent.jsonl \\
        --out eval/results.json
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from verify_export import build_feature_vector, forward_pass  # noqa: E402


def score(rows, preds, classes):
    """Top-1 / top-3 accuracy, per type and per repository."""
    n = len(rows)
    if n == 0:
        return {"commits": 0}
    y = [r["label"] for r in rows]
    top1 = [p[0] for p in preds]
    maj_label, maj_n = Counter(y).most_common(1)[0]
    out = {
        "commits": n,
        "top1": sum(a == b for a, b in zip(y, top1)) / n,
        "top3": sum(a in p[:3] for a, p in zip(y, preds)) / n,
        "baseline_most_common": {"type": maj_label, "top1": maj_n / n},
        "baseline_random_top3": 3 / len(classes),
        "types": dict(Counter(y).most_common()),
        "per_type": {},
        "per_repo": {},
    }
    for c in classes:
        support = sum(a == c for a in y)
        predicted = sum(b == c for b in top1)
        hit = sum(a == c and b == c for a, b in zip(y, top1))
        in3 = sum(a == c and c in p[:3] for a, p in zip(y, preds))
        out["per_type"][c] = {
            "support": support,
            "precision": hit / predicted if predicted else None,
            "recall": hit / support if support else None,
            "top3_recall": in3 / support if support else None,
        }
    # Recall averaged over the types (with at least 20 commits): a model that
    # always suggests the most common type scores well on top-1 but not here.
    recalls = [v["recall"] for v in out["per_type"].values() if v["support"] >= 20]
    out["macro_recall"] = sum(recalls) / len(recalls) if recalls else None
    # confusion[true type][first suggestion] = commits
    confusion = defaultdict(Counter)
    for a, b in zip(y, top1):
        confusion[a][b] += 1
    out["confusion"] = {a: dict(confusion[a]) for a in classes if a in confusion}
    by_repo = defaultdict(lambda: [0, 0, 0])
    for r, p in zip(rows, preds):
        s = by_repo[r.get("repo", "?")]
        s[0] += 1
        s[1] += p[0] == r["label"]
        s[2] += r["label"] in p[:3]
    out["per_repo"] = {k: {"commits": v[0], "top1": v[1] / v[0], "top3": v[2] / v[0]}
                       for k, v in sorted(by_repo.items())}
    return out


_PAYLOAD = None


def _init(json_path):
    global _PAYLOAD
    _PAYLOAD = json.load(open(json_path))


def _top3(lines):
    classes = _PAYLOAD["classes"]
    out = []
    for line in lines:
        r = json.loads(line)
        x = dict(r)
        x["diff_text"] = (r.get("diff_text") or "")[:20000]
        x["add_del_ratio"] = r["additions"] / (r["deletions"] + 1)
        p = forward_pass(build_feature_vector(x, _PAYLOAD), _PAYLOAD)
        out.append(({"label": r["label"], "repo": r.get("repo"), "sha": r.get("sha"),
                     "is_bot": bool(r.get("is_bot"))},
                    [classes[i] for i in np.argsort(-np.asarray(p))[:3]]))
    return out


def evaluate_set(path, json_path, workers, predictions=None):
    classes = json.load(open(json_path))["classes"]
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    chunks = [lines[i:i + 500] for i in range(0, len(lines), 500)]
    with Pool(workers, initializer=_init, initargs=(json_path,)) as pool:
        scored = [item for part in pool.imap(_top3, chunks) for item in part]
    rows = [r for r, _ in scored]
    preds = [p for _, p in scored]
    if predictions is not None:
        for r, p in scored:
            predictions.write(json.dumps({"set": Path(path).stem, **r, "top3": p}) + "\n")
    humans = [i for i, r in enumerate(rows) if not r["is_bot"]]
    bots = [i for i, r in enumerate(rows) if r["is_bot"]]
    pick = lambda idx: ([rows[i] for i in idx], [preds[i] for i in idx])  # noqa: E731
    return {
        "humans": score(*pick(humans), classes),
        "bots": score(*pick(bots), classes),
        "all": score(rows, preds, classes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "out/model_v2.json"), help="exported model")
    ap.add_argument("--sets", nargs="+", required=True, help="held-out JSONL files")
    ap.add_argument("--out", default=str(ROOT / "eval/results.json"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--predictions", metavar="FILE",
                    help="also write each commit's label and top three suggestions as JSONL")
    args = ap.parse_args()

    results = {}
    predictions = open(args.predictions, "w", encoding="utf-8") if args.predictions else None
    for path in args.sets:
        name = Path(path).stem
        results[name] = evaluate_set(path, args.json, args.workers, predictions)
        h = results[name]["humans"]
        if h["commits"]:
            print(f"{name:22} {h['commits']:>7} commits by people   "
                  f"top-1 {h['top1']:.1%}   top-3 {h['top3']:.1%}   macro recall {h['macro_recall']:.3f}   "
                  f"(always '{h['baseline_most_common']['type']}': {h['baseline_most_common']['top1']:.1%})")
    if predictions is not None:
        predictions.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
