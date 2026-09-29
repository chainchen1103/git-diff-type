#!/usr/bin/env python3
"""Add the imported public datasets to the training data without leaking into
the held-out sets.

import_external.py writes CommitChronicle and CommitBench commits from
thousands of repositories. Before they are used for training:

- commits from any organization that owns a test repository are dropped, so
  the "projects never seen" sets stay unseen;
- a commit whose normalized diff already occurs in the mined training set, a
  test set, or earlier in the imported data is dropped;
- `repo` becomes the source and the repository from the URL (for example
  "cc:owner/name"), so that it cannot clash with a mined repository, and
  `is_bot` is set to false (neither dataset keeps the author).

Usage (after eval/collect.sh and import_external.py):
    python eval/prepare_external.py \\
        --inputs datasets/commitchronicle.jsonl datasets/commitbench.jsonl \\
        --test-repos eval/repos_test.txt \\
        --against datasets/train.jsonl datasets/test_unseen_recent.jsonl \\
                  datasets/test_unseen_older.jsonl datasets/test_seen_recent.jsonl \\
        --out datasets/external.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dedupe import iter_rows, md5_of, normalize_diff  # noqa: E402


def org_of(owner):
    """CommitBench stores `org_repo` as the owner. GitHub names cannot contain
    underscores, so the organization is the part before the first one."""
    return str(owner).split("_", 1)[0].lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", nargs="+", required=True, help="import_external.py output files")
    ap.add_argument("--test-repos", required=True, help="org/repo lines of the held-out projects")
    ap.add_argument("--against", nargs="+", required=True,
                    help="the split_dataset.py sets whose diffs must not be repeated")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    test_orgs = {line.split("/", 1)[0].strip().lower()
                 for line in Path(args.test_repos).read_text().splitlines() if line.strip()}
    seen = {md5_of(normalize_diff(row.get("diff_text") or ""))
            for row in iter_rows(map(Path, args.against))}

    counts = Counter()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for path in args.inputs:
            for row in iter_rows([Path(path)]):
                source = str(row.get("url", "")).split("://", 1)[0] or Path(path).stem
                if org_of(row.get("owner", "")) in test_orgs:
                    counts[source, "test organization"] += 1
                    continue
                digest = md5_of(normalize_diff(row.get("diff_text") or ""))
                if digest in seen:
                    counts[source, "repeated diff"] += 1
                    continue
                seen.add(digest)
                # The URL is <source>://<repository>/<sha>.
                row["repo"] = f"{source}:{row['url'].split('://', 1)[1].rsplit('/', 1)[0]}"
                row["source"] = source
                row["is_bot"] = False
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[source, "kept"] += 1

    for (source, what), n in sorted(counts.items()):
        print(f"{source}: {n} {what}")


if __name__ == "__main__":
    main()
