#!/usr/bin/env python3
"""Score gca's scope suggestion (gca-rs/src/history.rs).

For each commit by people in the sets whose project uses scopes, going by the
500 commits before it (at least five Conventional Commits, one in five of
them with a scope), the scope gca would pre-fill, or none, against the scope
the commit has. Scores the suggestion of gca 0.3 and earlier (the scope used
most often for the same files, else for other files in the same directory)
and the current one: no scope counts as a choice, commits sharing the deepest
directory with the change come after those to the same files, and the
user's own commits count several times. Each commit's author stands for the
user.

Usage:
    python eval/evaluate_scope.py \\
        --sets datasets/test_unseen_recent.jsonl datasets/test_unseen_older.jsonl \\
               datasets/test_seen_recent.jsonl \\
        --history datasets/train.jsonl --out eval/results_scope.json

To choose how much the user's own commits count, on the held-out training
projects (eval/holdout_split.py):
    python eval/evaluate_scope.py --sets datasets/holdout/validation.jsonl \\
        --own-votes 1 2 3 4 8 16 --out /tmp/scope_holdout.json
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from dedupe import iter_rows  # noqa: E402
from history_sim import DEPTH, timelines  # noqa: E402

# As in gca-rs/src/history.rs.
OWN_SCOPE_VOTES = 8


def parent(path):
    return path.rsplit("/", 1)[0] if "/" in path else ""


def dirs_above(path):
    """`a` and `a/b` for `a/b/c.rs`."""
    return [path[:i] for i, ch in enumerate(path) if ch == "/"]


def related(by_repo, wanted, depth=DEPTH):
    """For each scored commit in `wanted`: its scope, how many of the people's
    commits before it are Conventional (all in these datasets) and scoped, and
    those related to it as (index from the most recent, scope, to a same
    file, in a same directory, depth of the deepest shared directory, by
    the same author)."""
    for repo, commits in by_repo.items():
        times = [c[0] for c in commits]
        for c in commits:
            if (repo, c[1]) not in wanted:
                continue
            files, author = c[4], c[5]
            parents = {parent(f) for f in files} - {""}
            staged_dirs = {d for f in files for d in dirs_above(f)}
            end = bisect.bisect_left(times, c[0])
            conventional = scoped = 0
            entries = []
            for idx, h in enumerate(reversed(commits[max(0, end - depth):end])):
                if h[3]:
                    continue
                conventional += 1
                scoped += h[7] is not None
                same_file = bool(files & h[4])
                same_dir = any(parent(f) in parents for f in h[4])
                shared = [next((d for d in reversed(dirs_above(f)) if d in staged_dirs), None) for f in h[4]]
                deepest = max((d.count("/") + 1 for d in shared if d), default=0)
                if same_file or deepest:
                    entries.append((idx, h[7], same_file, same_dir, deepest, h[5] == author))
            yield {"scope": c[7], "asked": conventional >= 5 and scoped * 5 >= conventional, "entries": entries}


def most_votes(votes):
    """Most votes win; a tie goes to the one used most recently."""
    return max(votes.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]


def before(r):
    """gca 0.3: the scope used most often for the same files, else for other
    files in the same directories; commits without a scope do not count."""
    for group in ([e for e in r["entries"] if e[2]], [e for e in r["entries"] if not e[2] and e[3]]):
        votes = {}
        for idx, scope, *_ in group:
            if scope is not None:
                n, first = votes.get(scope, (0, idx))
                votes[scope] = (n + 1, min(first, idx))
        if votes:
            return most_votes(votes)
    return None


def now(r, own_votes=OWN_SCOPE_VOTES):
    """The scope, or none, used most often by the closest commits: those to
    the same files, else those sharing the deepest directory. The user's own
    commits count `own_votes` times."""
    if not r["entries"]:
        return None
    closeness = [float("inf") if e[2] else e[4] for e in r["entries"]]
    closest = max(closeness)
    votes = {}
    for e, c in zip(r["entries"], closeness):
        if c == closest:
            n, first = votes.get(e[1], (0, e[0]))
            votes[e[1]] = (n + (own_votes if e[5] else 1), min(first, e[0]))
    return most_votes(votes)


def score(asked, suggest):
    guesses = [suggest(r) for r in asked]
    right = [g == r["scope"] for g, r in zip(guesses, asked)]
    made = [g is not None for g in guesses]
    return {"right": sum(right) / len(asked), "suggested": sum(made) / len(asked),
            "right_when_suggested": sum(r and m for r, m in zip(right, made)) / max(1, sum(made))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--history", nargs="*", default=[], help="more commits to take history from")
    ap.add_argument("--own-votes", type=int, nargs="*", default=[], help="also score these")
    ap.add_argument("--out", default=str(ROOT / "eval/results_scope.json"))
    args = ap.parse_args()

    by_repo = timelines(args.sets + args.history)
    scored = {(repo, c[1]) for repo, commits in by_repo.items() for c in commits if c[6]}
    results = {}
    for path in args.sets:
        in_set = {(r.get("repo"), r.get("sha")) for r in iter_rows([Path(path)])}
        rel = list(related(by_repo, scored & in_set))
        asked = [r for r in rel if r["asked"]]
        name = Path(path).stem
        res = {"commits": len(rel), "asked": len(asked) / max(1, len(rel)),
               "scoped": sum(r["scope"] is not None for r in asked) / max(1, len(asked)),
               "before": score(asked, before), "now": score(asked, now)}
        for k in args.own_votes:
            res[f"now_own_{k}"] = score(asked, lambda r, k=k: now(r, k))
        results[name] = res
        line = "  ".join(f"{k} {v['right']:.1%} right, {v['suggested']:.1%} suggested, "
                         f"{v['right_when_suggested']:.1%} right when suggested"
                         for k, v in res.items() if isinstance(v, dict))
        print(f"{name:20} {len(asked):6} asked ({res['scoped']:.1%} scoped)  {line}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
