"""The history gca reads (gca-rs/src/history.rs), simulated from dataset
files: for a commit, the DEPTH commits before it in the same repository, by
commit time, the types people gave them, and which of them touched a file
the commit touches."""
import bisect
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dedupe import iter_rows

DEPTH = 500
# As in gca-rs/src/history.rs.
PROJECT_PSEUDO_COMMITS = 10.0
FILE_PSEUDO_COMMITS = 5.0
# The files a stored diff touches, as they are after the commit. The miner
# cuts diffs at 20,000 characters, so a large commit's last files are missing.
FILE_HEADER = re.compile(r"^diff --git a/.+? b/(.+)$", re.M)


def files_of(row):
    return frozenset(FILE_HEADER.findall(str(row.get("diff_text") or "")))


def timelines(paths):
    """Each repository's commits, oldest first: (time, sha, type, by a bot, files)."""
    by_repo = defaultdict(list)
    for r in iter_rows([Path(p) for p in paths]):
        if isinstance(r.get("label"), str):
            by_repo[r.get("repo")].append(
                (r.get("committed_at") or "", r.get("sha"), r["label"], bool(r.get("is_bot")), files_of(r)))
    for commits in by_repo.values():
        commits.sort(key=lambda c: c[:4])
    return by_repo


def counts_before(by_repo, rows, classes, depth=DEPTH):
    """n[i, c]: commits by people of type c among the `depth` before rows[i];
    same[i, c]: the ones of them that touched a file rows[i] touches."""
    index = {c: j for j, c in enumerate(classes)}
    times = {repo: [c[0] for c in commits] for repo, commits in by_repo.items()}
    n = np.zeros((len(rows), len(classes)))
    same = np.zeros_like(n)
    for i, r in enumerate(rows):
        repo = r.get("repo")
        if repo not in by_repo:
            continue
        files = files_of(r)
        end = bisect.bisect_left(times[repo], r.get("committed_at") or "")
        for _, _, label, bot, touched in by_repo[repo][max(0, end - depth):end]:
            if not bot and label in index:
                n[i, index[label]] += 1
                if files & touched:
                    same[i, index[label]] += 1
    return n, same


def weigh(log_p, n, prior, weight, pseudo=PROJECT_PSEUDO_COMMITS):
    """log p + weight · log(q / prior), q being a mix of types from the
    history smoothed toward the prior. Without history a row stays as it is."""
    q = (n + pseudo * prior) / (n.sum(axis=1, keepdims=True) + pseudo)
    return log_p + weight * np.log(q / prior)
