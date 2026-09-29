#!/usr/bin/env python3
"""Split mined commits into a training set and three out-of-sample test sets.

    train.jsonl               training repos, landed before the cutoff
    test_unseen_recent.jsonl  test repos (never trained on), landed on/after the cutoff
    test_unseen_older.jsonl   test repos, landed before the cutoff
    test_seen_recent.jsonl    training repos, landed on/after the cutoff

"Landed" is the committer date, i.e. when the commit reached the branch.
Training and test repositories belong to different organizations. Any test
commit whose normalized diff also occurs in the training set is dropped, and
each set is deduplicated on its own. Bot-authored commits are left out of the
training set and kept, flagged, in the test sets.

Usage:
    python split_dataset.py --mined data/mined --train-repos repos_train.txt \\
        --test-repos repos_test.txt --cutoff 2026-04-20 --out datasets/
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dedupe import md5_of, normalize_diff


def repo_names(path):
    """`org/repo` lines -> the names miner.py records (`org_repo`)."""
    return [ln.strip().replace("/", "_") for ln in Path(path).read_text().splitlines() if ln.strip()]


def rows_of(mined_dir, name):
    path = Path(mined_dir) / f"{name}.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def landed(row):
    return datetime.fromisoformat(row["committed_at"].replace("Z", "+00:00"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mined", required=True, help="directory of miner.py output, one file per repo")
    ap.add_argument("--train-repos", required=True)
    ap.add_argument("--test-repos", required=True)
    ap.add_argument("--cutoff", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-bots", action="store_true", help="keep bot commits in the training set")
    args = ap.parse_args()

    cutoff = datetime.fromisoformat(args.cutoff).replace(tzinfo=timezone.utc)
    train_repos, test_repos = repo_names(args.train_repos), repo_names(args.test_repos)
    orgs = lambda names: {n.split("_", 1)[0].lower() for n in names}  # noqa: E731
    shared = orgs(train_repos) & orgs(test_repos)
    if shared:
        raise SystemExit(f"training and test repos share organizations: {sorted(shared)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {k: (out / f"{k}.jsonl").open("w", encoding="utf-8") for k in
             ("train", "test_unseen_recent", "test_unseen_older", "test_seen_recent")}
    counts = {k: {"commits": 0, "humans": 0} for k in files}
    seen = {k: set() for k in files}
    dropped = {"bots_from_train": 0, "duplicates": 0, "overlap_with_train": 0}
    missing = []

    def emit(kind, row, h):
        if h in seen[kind]:
            dropped["duplicates"] += 1
            return
        seen[kind].add(h)
        files[kind].write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[kind]["commits"] += 1
        counts[kind]["humans"] += not row.get("is_bot", False)

    # Training repos first, so the test sets can be checked against them.
    later = []
    for name in train_repos:
        n = 0
        for row in rows_of(args.mined, name):
            n += 1
            h = md5_of(normalize_diff(row["diff_text"]))
            if landed(row) >= cutoff:
                later.append((row, h))
            elif row.get("is_bot") and not args.include_bots:
                dropped["bots_from_train"] += 1
            else:
                emit("train", row, h)
        if n == 0:
            missing.append(name)

    for row, h in later:
        if h in seen["train"]:
            dropped["overlap_with_train"] += 1
        else:
            emit("test_seen_recent", row, h)

    for name in test_repos:
        n = 0
        for row in rows_of(args.mined, name):
            n += 1
            h = md5_of(normalize_diff(row["diff_text"]))
            if h in seen["train"]:
                dropped["overlap_with_train"] += 1
                continue
            emit("test_unseen_recent" if landed(row) >= cutoff else "test_unseen_older", row, h)
        if n == 0:
            missing.append(name)

    for f in files.values():
        f.close()
    manifest = {
        "cutoff": args.cutoff,
        "train_repos": [n for n in train_repos if n not in missing],
        "test_repos": [n for n in test_repos if n not in missing],
        "missing_repos": missing,
        "sets": counts,
        "dropped": dropped,
    }
    (out / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sets": counts, "dropped": dropped, "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
