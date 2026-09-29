#!/usr/bin/env bash
# Build the training set and the held-out test sets from public repositories.
#
#   bash eval/collect.sh        (from the repo root; needs git and python3)
#
# Clones each repository in eval/repos_*.txt (history since $SINCE), mines its
# Conventional Commits with `miner.py --stream`, deletes the clone (set
# KEEP_CLONES=1 to keep it), and finally splits everything with
# split_dataset.py into datasets/. Repositories keep growing, so a later run
# finds more commits.
set -euo pipefail
WORK=${WORK:-eval/work}
SINCE=${SINCE:-2018-01-01}
CUTOFF=${CUTOFF:-2026-04-20}
mkdir -p "$WORK/repos" "$WORK/mined"

for r in $(cat eval/repos_test.txt eval/repos_train.txt); do
  d=$(echo "$r" | tr / _)
  [ -f "$WORK/mined/$d.jsonl" ] && continue
  [ -d "$WORK/repos/$d" ] || git clone -q --shallow-since="$SINCE" --single-branch --no-tags \
    "https://github.com/$r" "$WORK/repos/$d"
  rm -f "$WORK/mined/$d.jsonl.part"   # miner.py appends; start clean after an interrupted run
  python3 miner.py --stream --repo "$WORK/repos/$d" --out "$WORK/mined/$d.jsonl.part"
  mv "$WORK/mined/$d.jsonl.part" "$WORK/mined/$d.jsonl"
  [ -n "${KEEP_CLONES:-}" ] || rm -rf "$WORK/repos/$d"
done

python3 split_dataset.py --mined "$WORK/mined" \
  --train-repos eval/repos_train.txt --test-repos eval/repos_test.txt \
  --cutoff "$CUTOFF" --out datasets/
