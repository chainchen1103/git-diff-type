#!/usr/bin/env python3
"""Fine-tune a T5-style model (CodeT5-small by default) to write a commit's
subject from its type, scope and diff, as t5_format.py lays them out.

Reads train-*.jsonl.gz and valid.jsonl.gz from data/ (prepare.py writes
them) and writes checkpoints under runs/: Ctrl+C saves one and stops,
--resume continues. run.bat sets up the environment and runs everything.

Usage (PowerShell, in this folder):
    .venv\\Scripts\\python train_t5.py
    .venv\\Scripts\\python train_t5.py --resume
"""
import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from t5_common import (HERE, encode_inputs, encode_targets, fix_special_ids, keep_awake, pick_device,
                       read_rows, run_logged)


def log(out, text):
    print(text, flush=True)
    with open(out / "train.log", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def save(model, tok, opt, sched, step, out):
    """Write a checkpoint to the slot not holding the latest one, then point
    latest.txt at it, so an interrupted save never loses the previous one."""
    pointer = out / "latest.txt"
    previous = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else ""
    name = "ckpt-b" if previous == "ckpt-a" else "ckpt-a"
    path = out / name
    if path.exists():
        shutil.rmtree(path)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, path / "state.pt")
    pointer.write_text(name, encoding="utf-8")
    return path


@torch.no_grad()
def validate(model, tok, valid, args, device, dtype, out, step):
    model.eval()
    total, count = 0.0, 0
    for k in range(0, len(valid), args.batch):
        rows = valid[k:k + args.batch]
        ids, mask = encode_inputs(tok, [r["i"] for r in rows], args.max_input, device)
        labels = encode_targets(tok, [r["t"] for r in rows], args.max_target, device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        n = int((labels != -100).sum())
        total += float(loss) * n
        count += n
    rows = valid[:8]
    ids, mask = encode_inputs(tok, [r["i"] for r in rows], args.max_input, device)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=args.max_target)
    with open(out / "samples.txt", "a", encoding="utf-8") as f:
        f.write(f"--- step {step}\n")
        for r, g in zip(rows, tok.batch_decode(gen, skip_special_tokens=True)):
            f.write(f"real:  {r['t']}\nmodel: {g.strip()}\n")
    model.train()
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="Salesforce/codet5-small", help="model to start from")
    ap.add_argument("--data", default=str(HERE / "data"), help="folder with train-*.jsonl.gz and valid.jsonl.gz")
    ap.add_argument("--out", default=str(HERE / "runs"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2, help="batches per optimizer step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500, help="optimizer steps")
    ap.add_argument("--max-input", type=int, default=512)
    ap.add_argument("--max-target", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="train on this many examples only")
    ap.add_argument("--save-every", type=int, default=1000, help="optimizer steps between checkpoints")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    parts = sorted(data.glob("train-*.jsonl.gz"))
    if not parts:
        raise SystemExit(f"no train-*.jsonl.gz in {data}")
    train = read_rows(parts)
    if args.limit:
        train = train[:args.limit]
    valid = read_rows([data / "valid.jsonl.gz"])
    device, dtype = pick_device()
    if device.type == "cpu":
        log(out, "warning: no CUDA GPU found, training on the CPU will be very slow")

    torch.manual_seed(args.seed)
    if args.resume and not (out / "latest.txt").exists():
        raise SystemExit(f"nothing to resume under {out}")
    start = out / (out / "latest.txt").read_text(encoding="utf-8").strip() if args.resume else None
    source = str(start) if start else args.base
    tok = AutoTokenizer.from_pretrained(source)
    model = T5ForConditionalGeneration.from_pretrained(source).to(device)
    fix_special_ids(model, tok)
    model.train()

    per_step = args.batch * args.accum
    total_steps = math.ceil(len(train) * args.epochs / per_step)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            fused=device.type == "cuda")

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
    log(out, f"{len(train):,} training examples, {len(valid):,} for validation; {total_steps:,} optimizer "
             f"steps of {per_step}; device {device}{', bf16' if dtype else ''}; model {source}")

    def batches_from(step):
        """The training batches from optimizer step `step` on, in a fixed
        shuffled order per epoch, so a resumed run sees the same data."""
        micro = step * args.accum
        per_epoch = math.ceil(len(train) / args.batch)
        while True:
            epoch, k = divmod(micro, per_epoch)
            order = list(range(len(train)))
            random.Random(args.seed + epoch).shuffle(order)
            for j in range(k, per_epoch):
                yield [train[i] for i in order[j * args.batch:(j + 1) * args.batch]]
                micro += 1

    batches = batches_from(step)
    t0, done_examples, losses = time.time(), 0, []
    try:
        while step < total_steps:
            for _ in range(args.accum):
                rows = next(batches)
                ids, mask = encode_inputs(tok, [r["i"] for r in rows], args.max_input, device)
                labels = encode_targets(tok, [r["t"] for r in rows], args.max_target, device)
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                (loss / args.accum).backward()
                losses.append(loss.item())
                done_examples += len(rows)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == total_steps:
                rate = done_examples / (time.time() - t0)
                left = (total_steps - step) * per_step / max(rate, 1e-9)
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                log(out, f"step {step:,}/{total_steps:,}  loss {sum(losses[-100:]) / len(losses[-100:]):.3f}  "
                         f"lr {sched.get_last_lr()[0]:.2e}  {rate:.1f} examples/s  "
                         f"about {left / 3600:.1f} h left  GPU memory {mem:.1f} GB")
            if step % args.save_every == 0 or step == total_steps:
                vloss = validate(model, tok, valid, args, device, dtype, out, step)
                path = save(model, tok, opt, sched, step, out)
                log(out, f"step {step:,}: validation loss {vloss:.3f}, saved {path.name}")
    except KeyboardInterrupt:
        path = save(model, tok, opt, sched, step, out)
        log(out, f"stopped at step {step:,}; saved {path.name}. Continue with --resume.")
        sys.exit(130)
    log(out, "done. Next: .venv\\Scripts\\python generate_t5.py")


if __name__ == "__main__":
    keep_awake()
    run_logged(main, HERE / "runs/error.log")
