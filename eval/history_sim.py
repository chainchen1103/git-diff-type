"""The history gca reads (gca-rs/src/history.rs), simulated from dataset
files: for a commit, the DEPTH commits before it in the same repository, by
commit time, and the types people gave them."""
import bisect
from collections import defaultdict
from pathlib import Path

import numpy as np

from dedupe import iter_rows

DEPTH = 500
# As in gca-rs/src/history.rs.
PSEUDO_COMMITS = 10.0


def timelines(paths):
    """Each repository's commits, oldest first: (time, sha, type, by a bot)."""
    by_repo = defaultdict(list)
    for r in iter_rows([Path(p) for p in paths]):
        if isinstance(r.get("label"), str):
            by_repo[r.get("repo")].append(
                (r.get("committed_at") or "", r.get("sha"), r["label"], bool(r.get("is_bot"))))
    for commits in by_repo.values():
        commits.sort()
    return by_repo


def counts_before(by_repo, rows, classes, depth=DEPTH):
    """n[i, c]: commits by people of type c among the `depth` before rows[i]."""
    index = {c: j for j, c in enumerate(classes)}
    times = {repo: [c[0] for c in commits] for repo, commits in by_repo.items()}
    n = np.zeros((len(rows), len(classes)))
    for i, r in enumerate(rows):
        repo = r.get("repo")
        if repo not in by_repo:
            continue
        end = bisect.bisect_left(times[repo], r.get("committed_at") or "")
        for _, _, label, bot in by_repo[repo][max(0, end - depth):end]:
            if not bot and label in index:
                n[i, index[label]] += 1
    return n


def weigh(log_p, n, prior, weight, pseudo=PSEUDO_COMMITS):
    """log p + weight · log(q / prior), q being the project's mix of types
    smoothed toward the prior. Without history a row stays as it is."""
    q = (n + pseudo * prior) / (n.sum(axis=1, keepdims=True) + pseudo)
    return log_p + weight * np.log(q / prior)
