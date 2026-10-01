#!/usr/bin/env python3
"""Build the subject model's data from the mined datasets: one JSON line per
commit by a person, {"i": input, "t": subject}, with the input laid out by
t5_format.py, gzip-compressed in parts of about 18 MB.

    python draft_model/prepare.py [--datasets datasets] [--out draft_model/data] [--no-history]

Training data: datasets/train.jsonl and datasets/external.jsonl, shuffled,
with 2,000 commits held back for validation. Test data:
datasets/test_unseen_recent.jsonl, with "id", "repo" and "type" added.

Each input also lists the headers of up to four earlier commits in the same
repository (--no-history leaves them out), picked from the 500 before it the
way gca can pick them from what it already reads (subjects, files and
authors, no diffs); gca-rs/src/t5draft/input.rs does the same:

- up to three that touched a staged file, same type first, then the larger
  file overlap (Jaccard), then the newer;
- if fewer than three, the newest of the same type;
- the author's own newest commit, if not picked already.
"""
import argparse
import bisect
import gzip
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(HERE)]
from dedupe import iter_rows  # noqa: E402
from history_sim import files_of, scope_of  # noqa: E402
from train_subject import subject_of  # noqa: E402
from t5_format import t5_input  # noqa: E402

DEPTH = 500
BY_FILES = 3
PART_BYTES = 18_000_000  # compressed; small enough to copy around one by one
VALID = 2000


def when(r):
    """A sortable commit time, or None (CommitBench rows only have the import time)."""
    if r.get("committed_at"):
        return r["committed_at"]
    if r.get("source") == "cc" and r.get("labeled_at"):
        try:
            return datetime.strptime(r["labeled_at"], "%d.%m.%Y %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
    return None


def header(r):
    subject = subject_of(r.get("message") or "")
    scope = scope_of(r.get("message"))
    return f"{r['label']}({scope}): {subject}" if scope else f"{r['label']}: {subject}"


class Timelines:
    """Each repository's commits in time order, with indexes by file, type
    and author, to pick an earlier commit's history quickly."""

    def __init__(self, paths):
        repos = defaultdict(list)
        for path in paths:
            for r in iter_rows([path]):
                t = when(r)
                subject = subject_of(r.get("message") or "")
                if t is None or not isinstance(r.get("label"), str) or not subject:
                    continue
                repos[r.get("repo")].append((t, r.get("sha"), r["label"], header(r), files_of(r), r.get("author") or ""))
        self.repos = {}
        for name, commits in repos.items():
            commits.sort(key=lambda c: c[:2])
            by_file, by_type, by_author = defaultdict(list), defaultdict(list), defaultdict(list)
            for i, c in enumerate(commits):
                for f in c[4]:
                    by_file[f].append(i)
                by_type[c[2]].append(i)
                if c[5]:
                    by_author[c[5]].append(i)
            self.repos[name] = (commits, {c[1]: i for i, c in enumerate(commits)}, by_file, by_type, by_author)

    def picks(self, repo, sha, depth=DEPTH):
        """Indexes of the earlier commits picked for this one, in order."""
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
        ranked = sorted(shared, key=lambda j: (commits[j][2] == me[2],
                                               shared[j] / (len(me[4]) + len(commits[j][4]) - shared[j]), j),
                        reverse=True)
        picked = ranked[:BY_FILES]
        if len(picked) < BY_FILES:
            p = by_type[me[2]]
            for j in reversed(p[bisect.bisect_left(p, start):bisect.bisect_left(p, i)]):
                if j not in picked:
                    picked.append(j)
                if len(picked) == BY_FILES:
                    break
        if me[5]:
            p = by_author[me[5]]
            for j in reversed(p[bisect.bisect_left(p, start):bisect.bisect_left(p, i)]):
                if j not in picked:
                    picked.append(j)
                break
        return picked

    def history(self, repo, sha, depth=DEPTH):
        commits = self.repos[repo][0] if repo in self.repos else []
        return [commits[j][3] for j in self.picks(repo, sha, depth)]


def examples(path, timelines):
    """(row, example, history) for each commit by a person with a type, a
    diff and a subject of at most 200 characters."""
    for r in iter_rows([path]):
        if r.get("is_bot") or not isinstance(r.get("label"), str):
            continue
        diff = str(r.get("diff_text") or "")
        subject = subject_of(r.get("message") or "")
        if not diff.strip() or not subject or len(subject) > 200:
            continue
        hist = timelines.history(r.get("repo"), r.get("sha")) if timelines else []
        yield r, {"i": t5_input(diff, r["label"], scope_of(r.get("message")), history=hist), "t": subject}, hist


def write_parts(rows, stem, out):
    """gzip JSONL parts of about PART_BYTES compressed each."""
    names, part, buf = [], 0, []

    def flush():
        nonlocal part, buf
        name = f"{stem}-{part:03d}.jsonl.gz"
        with gzip.open(out / name, "wt", encoding="utf-8", compresslevel=9) as f:
            f.writelines(buf)
        names.append(name)
        part, buf = part + 1, []

    # compress a sample to estimate how many rows fit in a part
    sample = [json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:5000]]
    ratio = len(gzip.compress("".join(sample).encode(), 9)) / max(1, len("".join(sample).encode()))
    budget = PART_BYTES / ratio * 0.97
    size = 0
    for r in rows:
        line = json.dumps(r, ensure_ascii=False) + "\n"
        if buf and size + len(line.encode()) > budget:
            flush()
            size = 0
        buf.append(line)
        size += len(line.encode())
    if buf:
        flush()
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--no-history", action="store_true", help="leave the earlier commits out of the inputs")
    args = ap.parse_args()
    data, out = Path(args.datasets), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train, with_history = [], 0
    for name in ["train.jsonl", "external.jsonl"]:
        timelines = None if args.no_history else Timelines([data / name])
        for _, ex, hist in examples(data / name, timelines):
            train.append(ex)
            with_history += bool(hist)
        print(f"{name}: {len(train)} examples  {time.time() - t0:.0f}s", flush=True)
    random.Random(7).shuffle(train)
    valid, train = train[:VALID], train[VALID:]
    with gzip.open(out / "valid.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in valid)
    parts = write_parts(train, "train", out)
    timelines = None if args.no_history else Timelines([data / "test_unseen_older.jsonl",
                                                          data / "test_unseen_recent.jsonl"])
    test, test_with = [], 0
    for r, ex, hist in examples(data / "test_unseen_recent.jsonl", timelines):
        test.append({"id": r.get("sha"), "repo": r.get("repo"), "type": r["label"], **ex})
        test_with += bool(hist)
    with gzip.open(out / "test_unseen_recent.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in test)
    total = len(train) + len(valid)
    print(f"train {total} ({with_history / total:.1%} with history) in {len(parts)} parts; "
          f"test {len(test)} ({test_with / len(test):.1%} with history)  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
