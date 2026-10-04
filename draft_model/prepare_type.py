#!/usr/bin/env python3
"""Build the type model's data from the mined datasets: one JSON line per
commit by a person, {"i": input, "y": type}, gzip-compressed in parts.

    python draft_model/prepare_type.py [--datasets datasets] [--out draft_model/type_data]

The input is the subject model's (t5_format.py) without its first two lines,
the type and the scope, which are what the model is to tell. The history
lines are picked without looking at the commit's type, as gca can pick them
before you choose one (gca-rs/src/t5draft/input.rs does the same):

- up to three of the 500 earlier commits that touched a staged file, the
  larger file overlap (Jaccard) first, then the newer;
- if fewer than three, the newest commits of any type;
- the author's own newest commit, if not picked already.

Their headers carry their types: the project's habits, as gca reads them.

Training data: the same commits, shuffle and validation split as prepare.py
(436,000 commits, 2,000 held back). Test data: every commit gca's evaluation
scores (by a person, typed, with a diff) in the three held-out sets, with
"id" (the commit), "repo" and "y".
"""
import argparse
import bisect
import gzip
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(HERE)]
from dedupe import iter_rows  # noqa: E402
from history_sim import scored  # noqa: E402
from prepare import DEPTH, VALID, Timelines, write_parts  # noqa: E402
from t5_format import t5_input  # noqa: E402
from train_subject import subject_of  # noqa: E402

BY_FILES = 3
# Held-out sets, each with the files its commits' history comes from.
TESTS = {
    "test_unseen_recent": ["test_unseen_older.jsonl", "test_unseen_recent.jsonl"],
    "test_unseen_older": ["test_unseen_older.jsonl", "test_unseen_recent.jsonl"],
    "test_seen_recent": ["train.jsonl", "test_seen_recent.jsonl"],
}


class AnyTypeTimelines(Timelines):
    def picks(self, repo, sha, depth=DEPTH):
        """Indexes of the earlier commits picked for this one, in order,
        whatever its type."""
        if repo not in self.repos:
            return []
        commits, position, by_file, by_type, by_author = self.repos[repo]
        i = position.get(sha)
        if i is None:
            return []
        start = max(0, i - depth)
        me = commits[i]
        shared = defaultdict(int)
        for f in me[4]:
            p = by_file[f]
            for j in p[bisect.bisect_left(p, start):bisect.bisect_left(p, i)]:
                shared[j] += 1
        ranked = sorted(shared, key=lambda j: (shared[j] / (len(me[4]) + len(commits[j][4]) - shared[j]), j),
                        reverse=True)
        picked = ranked[:BY_FILES]
        for j in range(i - 1, start - 1, -1):
            if len(picked) >= BY_FILES:
                break
            if j not in picked:
                picked.append(j)
        if me[5]:
            p = by_author[me[5]]
            for j in reversed(p[bisect.bisect_left(p, start):bisect.bisect_left(p, i)]):
                if j not in picked:
                    picked.append(j)
                break
        return picked


def type_input(diff, history):
    """t5_input without its first two lines (type and scope)."""
    return t5_input(diff, "x", None, history=history).split("\n", 2)[2]


def train_examples(path, timelines):
    for r in iter_rows([path]):
        if r.get("is_bot") or not isinstance(r.get("label"), str):
            continue
        diff = str(r.get("diff_text") or "")
        subject = subject_of(r.get("message") or "")
        if not diff.strip() or not subject or len(subject) > 200:
            continue  # the same commits as the subject model's data
        yield {"i": type_input(diff, timelines.history(r.get("repo"), r.get("sha"))), "y": r["label"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--out", default=str(HERE / "type_data"))
    args = ap.parse_args()
    data, out = Path(args.datasets), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train = []
    for name in ["train.jsonl", "external.jsonl"]:
        train.extend(train_examples(data / name, AnyTypeTimelines([data / name])))
        print(f"{name}: {len(train)}  {time.time() - t0:.0f}s", flush=True)
    random.Random(7).shuffle(train)
    valid, train = train[:VALID], train[VALID:]
    with gzip.open(out / "valid.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in valid)
    parts = write_parts(train, "train", out)
    print(f"train {len(train)} in {len(parts)} parts  {time.time() - t0:.0f}s", flush=True)
    for name, history in TESTS.items():
        timelines = AnyTypeTimelines([data / h for h in history])
        rows = []
        for r in iter_rows([data / f"{name}.jsonl"]):
            if scored(r):
                hist = timelines.history(r.get("repo"), r.get("sha"))
                rows.append({"id": r.get("sha"), "repo": r.get("repo"), "y": r["label"],
                             "i": type_input(str(r.get("diff_text") or ""), hist)})
        write_parts(rows, name, out)
        print(f"{name}: {len(rows)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
