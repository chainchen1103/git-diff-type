#!/usr/bin/env python3
"""Fine-tune CodeT5-small's encoder to tell a commit's type from its diff,
its files and the headers of related earlier commits, as prepare_type.py
lays them out: the encoder's states, averaged over the tokens, go through
one linear layer to the eleven types.

It starts from the encoder of the subject model (--base runs, the newest
checkpoint there), which has read these inputs before, or from any CodeT5
checkpoint or name. Reads train-*.jsonl.gz and valid.jsonl.gz from
type_data/ (prepare_type.py writes them) and writes checkpoints under
type_runs/: Ctrl+C saves one and stops, --resume continues.

Usage (PowerShell, in this folder; run_type.bat does it all):
    .venv\\Scripts\\python train_type.py --base runs
    .venv\\Scripts\\python train_type.py --resume
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, T5EncoderModel

from t5_common import HERE, encode_inputs, keep_awake, pick_device, read_rows, run_logged

TYPES = ["build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test"]


class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.d_model, len(TYPES))

    def forward(self, ids, mask):
        states = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.head(pooled.float())


def checkpoint(source):
    """A checkpoint folder, or the newest one in a runs folder (latest.txt)."""
    pointer = Path(source) / "latest.txt"
    return str(pointer.parent / pointer.read_text(encoding="utf-8").strip()) if pointer.exists() else source


def load(source, device):
    tok = AutoTokenizer.from_pretrained(source)
    model = Classifier(T5EncoderModel.from_pretrained(source)).to(device)
    head = Path(source) / "head.pt"
    if head.exists():
        model.head.load_state_dict(torch.load(head, map_location=device))
    return tok, model


def log(out, text):
    print(text, flush=True)
    with open(out / "train.log", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def save(model, tok, opt, sched, step, out):
    """Write a checkpoint to the slot not holding the latest one, then point
    latest.txt at it."""
    pointer = out / "latest.txt"
    previous = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else ""
    name = "ckpt-b" if previous == "ckpt-a" else "ckpt-a"
    path = out / name
    if path.exists():
        shutil.rmtree(path)
    model.encoder.save_pretrained(path)
    tok.save_pretrained(path)
    torch.save(model.head.state_dict(), path / "head.pt")
    (path / "types.json").write_text(json.dumps(TYPES), encoding="utf-8")
    torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, path / "state.pt")
    pointer.write_text(name, encoding="utf-8")
    return path


def labels(rows, device):
    return torch.tensor([TYPES.index(r["y"]) for r in rows], device=device)


@torch.no_grad()
def validate(model, tok, valid, args, device, dtype):
    model.eval()
    loss, right, n = 0.0, 0, 0
    for k in range(0, len(valid), args.batch):
        rows = valid[k:k + args.batch]
        ids, mask = encode_inputs(tok, [r["i"] for r in rows], args.max_input, device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(ids, mask)
        y = labels(rows, device)
        loss += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
        right += int((logits.argmax(dim=1) == y).sum())
        n += len(rows)
    model.train()
    return loss / n, right / n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="Salesforce/codet5-small",
                    help="a checkpoint folder, a runs folder with latest.txt, or a model name")
    ap.add_argument("--data", default=str(HERE / "type_data"))
    ap.add_argument("--out", default=str(HERE / "type_runs"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-input", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    print("Reading the training data (about a minute)...", flush=True)
    train = read_rows(sorted(data.glob("train-*.jsonl.gz")))
    if not train:
        raise SystemExit(f"no train-*.jsonl.gz in {data}")
    if args.limit:
        train = train[:args.limit]
    valid = read_rows([data / "valid.jsonl.gz"])
    if args.limit:
        valid = valid[:max(64, args.limit // 10)]
    device, dtype = pick_device()
    torch.manual_seed(args.seed)
    if args.resume and not (out / "latest.txt").exists():
        raise SystemExit(f"nothing to resume under {out}")
    start = out / (out / "latest.txt").read_text(encoding="utf-8").strip() if args.resume else None
    base = str(start) if start else checkpoint(args.base)
    tok, model = load(base, device)
    model.train()

    total_steps = math.ceil(len(train) * args.epochs / args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=device.type == "cuda")

    def schedule(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - args.warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    step = 0
    if start:
        state = torch.load(start / "state.pt", map_location=device, weights_only=False)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step = state["step"]
        log(out, f"resumed from {start.name} at step {step}")
    log(out, f"{len(train):,} training examples, {len(valid):,} for validation; {total_steps:,} steps of "
             f"{args.batch}; device {device}{', bf16' if dtype else ''}; model {base}")

    per_epoch = math.ceil(len(train) / args.batch)

    def batches_from(step):
        while True:
            epoch, k = divmod(step, per_epoch)
            order = list(range(len(train)))
            random.Random(args.seed + epoch).shuffle(order)
            for j in range(k, per_epoch):
                yield [train[i] for i in order[j * args.batch:(j + 1) * args.batch]]
                step += 1

    batches = batches_from(step)
    t0, done, losses = time.time(), 0, []
    try:
        while step < total_steps:
            rows = next(batches)
            ids, mask = encode_inputs(tok, [r["i"] for r in rows], args.max_input, device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(ids, mask)
            loss = nn.functional.cross_entropy(logits, labels(rows, device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            done += len(rows)
            step += 1
            if step % 50 == 0 or step == total_steps:
                rate = done / (time.time() - t0)
                left = (total_steps - step) * args.batch / max(rate, 1e-9)
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                log(out, f"step {step:,}/{total_steps:,}  loss {sum(losses[-100:]) / len(losses[-100:]):.3f}  "
                         f"lr {sched.get_last_lr()[0]:.2e}  {rate:.1f} examples/s  about {left / 3600:.1f} h left  "
                         f"GPU memory {mem:.1f} GB")
            if step % args.save_every == 0 or step == total_steps:
                vloss, vacc = validate(model, tok, valid, args, device, dtype)
                path = save(model, tok, opt, sched, step, out)
                log(out, f"step {step:,}: validation loss {vloss:.3f}, accuracy {vacc:.1%}, saved {path.name}")
    except KeyboardInterrupt:
        path = save(model, tok, opt, sched, step, out)
        log(out, f"stopped at step {step:,}; saved {path.name}. Continue with --resume.")
        sys.exit(130)
    log(out, "done")


if __name__ == "__main__":
    keep_awake()
    run_logged(main, HERE / "type_runs/error.log")
