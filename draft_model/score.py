#!/usr/bin/env python3
"""Score subject suggestions against the subjects the authors wrote.

    python draft_model/score.py predictions.jsonl [baseline.json ...] [--drafts FILE]

A predictions file has one JSON line per commit from generate_t5.py ({"id",
"greedy", "beam", "c1", "chalf", "confidence"}); a baseline file maps id to
a subject. --drafts names JSON lines {"sha", "draft"} of other drafts to
compare with on the commits they cover.

Measures, after lower-casing, collapsing spaces and dropping a final period:
exact; within one or two words (word-level edit distance); "saves half or
more" (the character edits needed to reach the author's subject are at most
half its length); the first word; and the typing saved on average.
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = HERE / "data/test_unseen_recent.jsonl.gz"


def norm(s):
    s = " ".join(str(s or "").split()).lower()
    return s[:-1] if s.endswith(".") else s


def lev(a, b):
    """Edit distance between two strings or two lists of words."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def measures(pred, ref):
    """How close one suggestion is; None (no suggestion) misses everything."""
    if pred is None:
        return {"exact": 0, "w1": 0, "w2": 0, "half": 0, "first": 0, "saved": 0.0}
    p, r = norm(pred), norm(ref)
    d = lev(p, r)
    words = lev(p.split(), r.split())
    saved = max(0, len(r) - d) / max(1, len(r))
    return {"exact": int(p == r), "w1": int(words <= 1), "w2": int(words <= 2), "half": int(saved >= 0.5),
            "first": int(bool(r) and p.split()[:1] == r.split()[:1]), "saved": saved}


def summarize(rows):
    n = len(rows)
    if not n:
        return "(n=0)"
    pct = lambda k: 100 * sum(r[k] for r in rows) / n  # noqa: E731
    return (f"exact {pct('exact'):5.1f}  <=1 word {pct('w1'):5.1f}  <=2 words {pct('w2'):5.1f}  "
            f"half+ {pct('half'):5.1f}  first word {pct('first'):5.1f}  saved {pct('saved'):5.1f}  (n={n})")


def best(preds, ref):
    ms = [measures(p, ref) for p in preds or []] or [measures(None, ref)]
    return max(ms, key=lambda m: (m["exact"], m["w1"], m["saved"]))


def read_tests(path=TEST):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def read_predictions(path):
    """id -> record; a baseline's subjects become {"greedy": subject}."""
    if str(path).endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return {r["id"]: r for r in map(json.loads, f)}
    with open(path, encoding="utf-8") as f:
        return {k: {"greedy": v} for k, v in json.load(f).items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions", nargs="+")
    ap.add_argument("--drafts", help="JSON lines {sha, draft} to compare with")
    ap.add_argument("--test", default=str(TEST))
    args = ap.parse_args()
    tests = read_tests(args.test)
    drafts = {}
    if args.drafts:
        with open(args.drafts, encoding="utf-8") as f:
            drafts = {r["sha"]: r["draft"] for r in map(json.loads, f)}
    for path in args.predictions:
        preds = read_predictions(path)
        print(f"== {path}")
        scored = {k: measures(preds[k]["greedy"] if k in preds else None, t["t"]) for k, t in tests.items()}
        print("all             ", summarize(list(scored.values())))
        by_type = defaultdict(list)
        for k, m in scored.items():
            by_type[tests[k]["type"]].append(m)
        for kind, rows in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            if len(rows) >= 100:
                print(f"  {kind:14}", summarize(rows))
        if drafts:
            covered = [k for k in scored if k in drafts]
            print("drafts' commits ", summarize([scored[k] for k in covered]))
            print("  the drafts    ", summarize([measures(drafts[k], tests[k]["t"]) for k in covered]))
        if any("beam" in p for p in preds.values()):
            print("best of 3 beams ", summarize([best((preds.get(k) or {}).get("beam"), t["t"])
                                                for k, t in tests.items()]))
        for name, label in [("c1", "after the first word"), ("chalf", "after half the words")]:
            hits = whole = n = 0
            for k, t in tests.items():
                c = (preds.get(k) or {}).get(name)
                if not c or not t["t"].startswith(c["prefix"]):
                    continue
                n += 1
                got, want = norm(c["rest"]).split(), norm(t["t"][len(c["prefix"]):]).split()
                hits += bool(got[:1] and got[:1] == want[:1])
                whole += got == want
            if n:
                print(f"completion {label}: next word {100 * hits / n:5.1f}  whole rest {100 * whole / n:5.1f}  (n={n})")


if __name__ == "__main__":
    main()
