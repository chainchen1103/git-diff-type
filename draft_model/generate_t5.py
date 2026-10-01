#!/usr/bin/env python3
"""Write subjects for the test commits with the trained model: the greedy
one, three by beam search, and completions of the author's own subject
after its first word and after half of its words.

Usage (PowerShell, in this folder, after train_t5.py):
    .venv\\Scripts\\python generate_t5.py
writes predictions.jsonl next to this file.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from t5_common import (HERE, encode_inputs, fix_special_ids, keep_awake, latest_model, pick_device,
                       read_rows, run_logged)


def word_prefixes(subject):
    """Character lengths of the subject's first word and of its first half
    of words, when something is left to complete after them."""
    starts = [0] + [k + 1 for k, ch in enumerate(subject) if ch == " " and k + 1 < len(subject)]
    words = len(starts)
    ends = [(starts[w + 1] - 1 if w + 1 < words else len(subject)) for w in range(words)]
    out = {}
    if words >= 2:
        out["c1"] = ends[0]
    if words >= 4:
        out["chalf"] = ends[words // 2 - 1]
    return out


def mean_log_prob(model, out, eos_id):
    """Per sequence, the mean log-probability of its tokens up to and
    including the end token: how sure the model was of what it wrote."""
    scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float()
    tokens = out.sequences[:, 1:]
    is_eos = tokens == eos_id
    end = torch.where(is_eos.any(dim=1), is_eos.float().argmax(dim=1), torch.full_like(is_eos[:, 0], tokens.shape[1] - 1,
                                                                                    dtype=torch.long))
    keep = torch.arange(tokens.shape[1], device=tokens.device)[None, :] <= end[:, None]
    return ((scores * keep).sum(dim=1) / keep.sum(dim=1)).tolist()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="checkpoint folder (default: the latest under runs/)")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--data", default=str(HERE / "data/test_unseen_recent.jsonl.gz"))
    ap.add_argument("--out", default=str(HERE / "predictions.jsonl"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-input", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.model) if args.model else latest_model(args.runs)
    device, dtype = pick_device()
    tok = AutoTokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path).to(device)
    if dtype is not None:
        model = model.to(dtype)
    fix_special_ids(model, tok)
    model.eval()
    rows = read_rows([Path(args.data)])
    if args.limit:
        rows = rows[:args.limit]
    start_id = model.generation_config.decoder_start_token_id
    print(f"{len(rows):,} commits, model {path}, device {device}", flush=True)
    t0 = time.time()

    results = [{"id": r.get("id"), "greedy": None, "beam": None} for r in rows]
    for k in range(0, len(rows), args.batch):
        part = rows[k:k + args.batch]
        ids, mask = encode_inputs(tok, [r["i"] for r in part], args.max_input, device)
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=args.max_new,
                             output_scores=True, return_dict_in_generate=True)
        confidence = mean_log_prob(model, out, tok.eos_token_id)
        beams = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=args.max_new,
                               num_beams=3, num_return_sequences=3, early_stopping=True)
        greedy = [g.strip() for g in tok.batch_decode(out.sequences, skip_special_tokens=True)]
        beams = [b.strip() for b in tok.batch_decode(beams, skip_special_tokens=True)]
        for j, g in enumerate(greedy):
            results[k + j]["greedy"] = g
            results[k + j]["confidence"] = round(confidence[j], 4)
            results[k + j]["beam"] = beams[3 * j:3 * j + 3]
        if (k // args.batch) % 20 == 0:
            print(f"subjects {k + len(part):,}/{len(rows):,}  {time.time() - t0:.0f}s", flush=True)

    # Completions: the decoder is given the subject's own first tokens, cut
    # where the prefix ends, and writes the rest.
    groups = defaultdict(list)
    for n, r in enumerate(rows):
        enc = tok(r["t"], add_special_tokens=False, return_offsets_mapping=True)
        for name, end in word_prefixes(r["t"]).items():
            prefix = [t for t, (_, stop) in zip(enc["input_ids"], enc["offset_mapping"]) if stop <= end]
            if prefix:
                groups[len(prefix)].append((n, name, prefix, end))
    jobs = 0
    for length in sorted(groups):  # a batch's prefixes must have the same length
        group = groups[length]
        for k in range(0, len(group), args.batch):
            part = group[k:k + args.batch]
            ids, mask = encode_inputs(tok, [rows[n]["i"] for n, _, _, _ in part], args.max_input, device)
            dec = torch.tensor([[start_id] + p for _, _, p, _ in part], device=device)
            gen = model.generate(input_ids=ids, attention_mask=mask, decoder_input_ids=dec,
                                 max_new_tokens=args.max_new)
            rest = tok.batch_decode(gen[:, dec.shape[1]:], skip_special_tokens=True)
            for (n, name, _, end), text in zip(part, rest):
                results[n][name] = {"prefix": rows[n]["t"][:end], "rest": text}
            jobs += len(part)
    print(f"completions {jobs:,}  {time.time() - t0:.0f}s", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    keep_awake()
    run_logged(main, HERE / "generate_error.log")
