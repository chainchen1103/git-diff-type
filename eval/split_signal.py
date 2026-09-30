#!/usr/bin/env python3
"""Could gca tell that the staged files hold two unrelated changes?

Joins the files of a change that share their top directory (or directory, or
name, as `parser.rs` and `parser_test.rs`), or that one of the 500 commits
before it changed together; changelogs and lockfiles join anything. More
than one group means "these look like separate changes". Scored on real
commits by people, which it should mostly leave alone, and on two
consecutive commits by the same author, a day apart at most and without
files in common, put together as if they had been committed at once, which
it should flag and split along the two commits.

The answer so far: such pairs are flagged only three to five times as often
as real commits (39.8% against 12.4% on the unseen projects, 33.6% against
6.9% on the six held-out training projects). Unless more than one commit in
six mixes unrelated work, most hints on the unseen projects would be wrong,
so gca does not suggest splitting.

Usage:
    python eval/split_signal.py datasets/test_unseen_older.jsonl datasets/test_unseen_recent.jsonl
"""
import bisect
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from history_sim import DEPTH, timelines  # noqa: E402

MAX_GAP_HOURS = 24
TEST_AFFIX = re.compile(r"(^test_|_test$|_spec$|\.test$|\.spec$|^test$|Test$|Tests$|_tests$)")
SATELLITE = re.compile(r"(^|/)(CHANGELOG[^/]*|CHANGES[^/]*|HISTORY[^/]*|package-lock\.json|yarn\.lock|"
                       r"pnpm-lock\.yaml|Cargo\.lock|go\.sum|poetry\.lock|uv\.lock|composer\.lock|Gemfile\.lock|"
                       r"\.changeset/[^/]+)$", re.I)


def name_stem(path):
    name = path.rsplit("/", 1)[-1]
    base = name if name.startswith(".") else name.split(".", 1)[0]
    return TEST_AFFIX.sub("", base).lower()


def groups(files, window):
    """The staged files in groups that nothing joins."""
    core = sorted(f for f in files if not SATELLITE.search(f))
    if len(core) <= 1:
        return [set(files)]
    parent = {f: f for f in core}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    def join(members):
        for f in members[1:]:
            parent[find(f)] = find(members[0])

    keys = defaultdict(list)
    for f in core:
        dirs = f.split("/")[:-1]
        if dirs:
            keys[("dir", "/".join(dirs))].append(f)
            keys[("top", dirs[0])].append(f)
        keys[("name", name_stem(f))].append(f)
    for members in keys.values():
        join(members)
    staged = set(core)
    for h in window:
        if not h[3]:  # people's commits only
            join(sorted(staged & h[4]))
    found = defaultdict(set)
    for f in core:
        found[find(f)].add(f)
    out = list(found.values())
    out[0] |= set(files) - staged  # changelogs and lockfiles go with any group
    return out


def when(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def main():
    by_repo = timelines(sys.argv[1:])
    real = flagged = 0
    pairs = pairs_flagged = clean = 0
    other_type = other_type_flagged = 0
    for commits in by_repo.values():
        times = [c[0] for c in commits]
        last = {}
        for k, c in enumerate(commits):
            if not (c[6] and c[4]):
                continue
            end = bisect.bisect_left(times, c[0])
            real += 1
            flagged += len(groups(c[4], commits[max(0, end - DEPTH):end])) > 1
            j = last.get(c[5])
            last[c[5]] = k
            if j is None:
                continue
            p = commits[j]
            if (p[4] & c[4]) or not 0 <= (when(c[0]) - when(p[0])).total_seconds() <= MAX_GAP_HOURS * 3600:
                continue
            start = bisect.bisect_left(times, p[0])
            found = groups(p[4] | c[4], commits[max(0, start - DEPTH):start])
            pairs += 1
            hit = len(found) > 1
            pairs_flagged += hit
            clean += hit and all(g <= p[4] or g <= c[4] for g in found)
            if p[2] != c[2]:
                other_type += 1
                other_type_flagged += hit
    print(f"{real} real commits by people: {flagged / real:.1%} would be flagged")
    print(f"{pairs} pairs of consecutive commits put together: {pairs_flagged / pairs:.1%} flagged, "
          f"{clean / pairs:.1%} split exactly along the two commits; "
          f"{other_type_flagged / max(1, other_type):.1%} of the {other_type} with two types flagged")


if __name__ == "__main__":
    main()
