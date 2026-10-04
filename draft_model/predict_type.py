#!/usr/bin/env python3
"""Write the trained type model's probabilities for the held-out commits:
type_predictions/<set>.jsonl with {"id", "repo", "p": {type: probability}}
for each test set in type_data/ (test_*-NNN.jsonl.gz). eval_type.py scores
them.

Usage (PowerShell, in this folder, after train_type.py):
    .venv\\Scripts\\python predict_type.py
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from t5_common import HERE, encode_inputs, keep_awake, latest_model, pick_device, read_rows, run_logged
from train_type import TYPES, load


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="checkpoint folder (default: the latest under runs/)")
    ap.add_argument("--runs", default=str(HERE / "type_runs"))
    ap.add_argument("--data", default=str(HERE / "type_data"))
    ap.add_argument("--out", default=str(HERE / "type_predictions"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-input", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    path = Path(args.model) if args.model else latest_model(args.runs)
    device, dtype = pick_device()
    tok, model = load(str(path), device)
    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for p in sorted(Path(args.data).glob("test_*-*.jsonl.gz")):
        groups[p.name.rsplit("-", 1)[0]].append(p)
    t0 = time.time()
    for name, parts in groups.items():
        target = out / f"{name}.jsonl"
        if target.exists():
            print(f"{name}: already written", flush=True)
            continue
        print(f"{name}: reading", flush=True)
        rows = read_rows(parts)
        if args.limit:
            rows = rows[:args.limit]
        partial = target.with_suffix(".partial")
        with open(partial, "w", encoding="utf-8") as f:
            for k in range(0, len(rows), args.batch):
                batch = rows[k:k + args.batch]
                ids, mask = encode_inputs(tok, [r["i"] for r in batch], args.max_input, device)
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    probs = torch.softmax(model(ids, mask).float(), dim=1).tolist()
                for r, p in zip(batch, probs):
                    f.write(json.dumps({"id": r["id"], "repo": r["repo"], "p": {t: round(v, 6) for t, v in zip(TYPES, p)}}) + "\n")
                if (k // args.batch) % 100 == 0:
                    print(f"{name}: {k + len(batch):,}/{len(rows):,}  {time.time() - t0:.0f}s", flush=True)
        partial.replace(target)
        print(f"{name}: wrote {target.name}", flush=True)


if __name__ == "__main__":
    keep_awake()
    run_logged(main, HERE / "type_predict_error.log")
