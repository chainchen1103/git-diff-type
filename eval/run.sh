#!/usr/bin/env bash
# Score the shipped model on the held-out sets built by eval/collect.sh and
# redraw the README chart.
#   bash eval/run.sh   (from the repo root)
set -euo pipefail
MODEL=${MODEL:-out/model_v2.json}
python3 eval/evaluate.py --json "$MODEL" --out "${OUT:-eval/results.json}" --sets \
  datasets/test_unseen_recent.jsonl \
  datasets/test_unseen_older.jsonl \
  datasets/test_seen_recent.jsonl
if [ -z "${OUT:-}" ]; then python3 eval/plot.py; fi
