"""The history gca reads (gca-rs/src/history.rs), simulated from dataset
files: for a commit, the DEPTH commits before it in the same repository, by
commit time, the types people gave them, which of them touched a file the
commit touches, and the latest ones by the same author, which gca reads again
with its models."""
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
OWN_COMMITS = 10
OWN_PSEUDO_COMMITS = 1.0
# The files a stored diff touches, as they are after the commit. The miner
# cuts diffs at 20,000 characters, so a large commit's last files are missing.
FILE_HEADER = re.compile(r"^diff --git a/.+? b/(.+)$", re.M)


def files_of(row):
    return frozenset(FILE_HEADER.findall(str(row.get("diff_text") or "")))


def scored(row):
    """A commit the models score: by a person, typed, with a diff."""
    return (not row.get("is_bot") and isinstance(row.get("label"), str)
            and bool(str(row.get("diff_text") or "").strip()))


def timelines(paths):
    """Each repository's commits, oldest first: (time, sha, type, by a bot,
    files, author, scored)."""
    by_repo = defaultdict(list)
    for r in iter_rows([Path(p) for p in paths]):
        if isinstance(r.get("label"), str):
            by_repo[r.get("repo")].append(
                (r.get("committed_at") or "", r.get("sha"), r["label"], bool(r.get("is_bot")), files_of(r),
                 r.get("author") or "", scored(r)))
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
        for _, _, label, bot, touched, *_ in by_repo[repo][max(0, end - depth):end]:
            if not bot and label in index:
                n[i, index[label]] += 1
                if files & touched:
                    same[i, index[label]] += 1
    return n, same


def window_start(by_repo, rows, depth=DEPTH):
    """The time of the oldest of the `depth` commits before each row, as far
    back as gca's log reaches; None when there is none."""
    times = {repo: [c[0] for c in commits] for repo, commits in by_repo.items()}
    starts = []
    for r in rows:
        t = times.get(r.get("repo"), [])
        end = bisect.bisect_left(t, r.get("committed_at") or "")
        starts.append(t[max(0, end - depth)] if end else None)
    return starts


def latest_by_author(by_repo, authors, exclude=frozenset(), depth=OWN_COMMITS):
    """Hashes of the latest `depth` scored commits, other than those in
    `exclude`, of each (repository, author) in `authors`: the ones gca may
    read again for their next commit."""
    out = set()
    for repo, commits in by_repo.items():
        per = defaultdict(list)
        for c in commits:
            if c[6] and (repo, c[5]) in authors and c[1] not in exclude:
                per[c[5]].append(c[1])
        for shas in per.values():
            out.update(shas[-depth:])
    return out


def own_choices(keys, times, labels, probs, classes, targets, starts, depth=OWN_COMMITS):
    """keys, times, labels and probs describe every commit gca could read
    again. For each index i in `targets`, with starts[n] for its n-th entry:
    of the latest `depth` commits with its key (repository, author) before
    it, and not before starts[n], how many had each type (chosen), and the
    sum of the probabilities `probs` gives them (shown)."""
    onehot = (np.asarray(labels)[:, None] == np.asarray(classes)[None, :]).astype(float)
    chosen = np.zeros((len(targets), len(classes)))
    shown = np.zeros_like(chosen)
    by_key = defaultdict(list)
    for j, k in enumerate(keys):
        by_key[k].append(j)
    groups = {}
    for k, members in by_key.items():
        members.sort(key=lambda j: times[j])
        m = np.array(members)
        groups[k] = ([times[j] for j in members],
                     np.vstack([np.zeros(len(classes)), np.cumsum(onehot[m], axis=0)]),
                     np.vstack([np.zeros(len(classes)), np.cumsum(probs[m], axis=0)]))
    for n, i in enumerate(targets):
        if starts[n] is None:
            continue
        ts, co, cs = groups[keys[i]]
        hi = bisect.bisect_left(ts, times[i])
        lo = min(hi, max(hi - depth, bisect.bisect_left(ts, starts[n])))
        chosen[n] = co[hi] - co[lo]
        shown[n] = cs[hi] - cs[lo]
    return chosen, shown


def weigh(log_p, n, prior, weight, pseudo=PROJECT_PSEUDO_COMMITS):
    """log p + weight · log(q / prior), q being a mix of types from the
    history smoothed toward the prior. Without history a row stays as it is."""
    q = (n + pseudo * prior) / (n.sum(axis=1, keepdims=True) + pseudo)
    return log_p + weight * np.log(q / prior)


def weigh_own(log_p, chosen, shown, prior, weight, pseudo=OWN_PSEUDO_COMMITS):
    """log p + weight · log((chosen + pseudo · prior) / (shown + pseudo · prior)):
    types the author chose more often than the models suggested them rise.
    Without such commits a row stays as it is."""
    return log_p + weight * np.log((chosen + pseudo * prior) / (shown + pseudo * prior))
