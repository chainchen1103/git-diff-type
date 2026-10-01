"""Shared by train_t5.py and generate_t5.py."""
import gzip
import json
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def read_rows(paths):
    """The JSON lines of gzip files, in order."""
    rows = []
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def pick_device():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
        return torch.device("cuda"), dtype
    return torch.device("cpu"), None


def fix_special_ids(model, tok):
    """T5 decodes from the pad token; make sure the configs say so."""
    for cfg in (model.config, model.generation_config):
        if cfg.decoder_start_token_id is None:
            cfg.decoder_start_token_id = tok.pad_token_id
        if cfg.pad_token_id is None:
            cfg.pad_token_id = tok.pad_token_id
        if cfg.eos_token_id is None:
            cfg.eos_token_id = tok.eos_token_id


def encode_inputs(tok, texts, max_input, device):
    enc = tok(texts, max_length=max_input, truncation=True, padding=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def encode_targets(tok, texts, max_target, device):
    """Labels: the subject's tokens and the end token, -100 for padding."""
    ids = tok(texts, max_length=max_target - 1, truncation=True, add_special_tokens=False)["input_ids"]
    width = max(len(t) for t in ids) + 1
    labels = torch.full((len(texts), width), -100, dtype=torch.long)
    for k, t in enumerate(ids):
        row = t + [tok.eos_token_id]
        labels[k, :len(row)] = torch.tensor(row, dtype=torch.long)
    return labels.to(device)


def latest_model(out):
    """The newest complete checkpoint under out (see train_t5.py)."""
    pointer = Path(out) / "latest.txt"
    if not pointer.exists():
        raise SystemExit(f"no trained model under {out}: run train_t5.py first")
    return Path(out) / pointer.read_text(encoding="utf-8").strip()


def keep_awake():
    """Stop Windows from sleeping while this process runs (the screen may
    still turn off)."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def run_logged(main, log_path):
    """Run main(); write any crash to log_path as well, to read it remotely."""
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(traceback.format_exc() + "\n")
        raise
