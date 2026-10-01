"""A baseline for the subject model that gca could compute from what it
already reads (subjects and file names of the last 500 commits, no diffs):
the subject of the earlier commit of the same type whose files overlap the
staged ones most (Jaccard), the most recent on a tie; nothing when no
earlier commit of that type shares a file.

    python draft_model/baseline_files.py   # writes draft_model/baseline_files.json
    python draft_model/score.py draft_model/baseline_files.json
"""
import bisect
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from dedupe import iter_rows  # noqa: E402
from history_sim import files_of  # noqa: E402
from train_subject import subject_of  # noqa: E402

DEPTH = 500


def main():
    test = [json.loads(line) for line in gzip.open(HERE / "data/test_unseen_recent.jsonl.gz", "rt", encoding="utf-8")]
    history = defaultdict(list)
    for name in ["test_unseen_older.jsonl", "test_unseen_recent.jsonl"]:
        for r in iter_rows([ROOT / "datasets" / name]):
            subject = subject_of(r.get("message") or "")
            if isinstance(r.get("label"), str) and subject:
                history[r.get("repo")].append((r.get("committed_at") or "", r.get("sha"), r["label"], subject, files_of(r)))
    out = {}
    for repo, commits in history.items():
        commits.sort(key=lambda c: c[:2])
        times = [c[0] for c in commits]
        index = {c[1]: c for c in commits}
        for r in (t for t in test if t["repo"] == repo):
            me = index[r["id"]]
            end = bisect.bisect_left(times, me[0])
            best, best_score = None, 0.0
            for j in range(end - 1, max(0, end - DEPTH) - 1, -1):  # newest first
                c = commits[j]
                if c[2] != r["type"] or not c[4]:
                    continue
                score = len(me[4] & c[4]) / len(me[4] | c[4])
                if score > best_score:
                    best, best_score = c, score
            if best:
                out[r["id"]] = best[3]
    json.dump(out, open(HERE / "baseline_files.json", "w"), ensure_ascii=False)
    print(f"{len(out)} of {len(test)} commits get a suggestion")


if __name__ == "__main__":
    main()
