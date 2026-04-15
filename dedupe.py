#!/usr/bin/env python3
"""Merge and deduplicate the commit corpus.

Pass A: exact (repo, sha).
Pass B: MD5 of normalized diff (strips volatile index/hunk/whitespace).
Pass C: SimHash near-dup with 4x16-bit LSH banding (opt-in, --near-dup).

First-seen wins, so list --input in priority order (curated first, external
last) to keep the trusted copy on tie.

Usage:
    python dedupe.py --input datasets/ --output datasets/_merged.jsonl
    python dedupe.py --input datasets/*.jsonl --output datasets/_merged.jsonl \\
                    --near-dup --hamming 3
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

# Force UTF-8 so emoji in print() works under Windows cp950.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def expand_inputs(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*.jsonl")) + sorted(p.glob("*.json")))
        elif p.exists():
            out.append(p)
        else:
            print(f"[warn] skipping missing: {item}", file=sys.stderr)
    return out


def iter_rows(paths: Iterable[Path]):
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            # Try JSON-array first; fall back to JSONL.
            head = f.read(1)
            f.seek(0)
            if head == "[":
                try:
                    for obj in json.load(f):
                        if isinstance(obj, dict):
                            yield p.name, obj
                    continue
                except Exception:
                    f.seek(0)
            for i, ln in enumerate(f, 1):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield p.name, json.loads(ln)
                except Exception as e:
                    print(f"[warn] {p.name}:{i} invalid JSON: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Normalization / hashing
# ---------------------------------------------------------------------------

_VOLATILE = re.compile(r"^(index [0-9a-f]+\.\.[0-9a-f]+.*|@@ .*|commit [0-9a-f]+|"
                        r"Author:.*|Date:.*|\s*)$")

def normalize_diff(diff_text: str) -> str:
    keep = []
    for line in diff_text.splitlines():
        if _VOLATILE.match(line):
            continue
        keep.append(line.rstrip())
    return "\n".join(keep)


def md5_of(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# SimHash (pure python, 64-bit)
# ---------------------------------------------------------------------------

def _shingles(text: str, n: int = 5) -> List[str]:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < n:
        return toks or [text[:32]]
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def simhash64(text: str) -> int:
    v = [0] * 64
    for sh in _shingles(text):
        h = int.from_bytes(hashlib.md5(sh.encode("utf-8")).digest()[:8], "big")
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def banded_near_dup(sigs: List[int], threshold: int, bands: int = 4):
    """Return set of indices to drop (near-dup of an earlier index).
    Uses 4x16-bit LSH banding — any shared band triggers Hamming check."""
    assert 64 % bands == 0
    band_bits = 64 // bands
    mask = (1 << band_bits) - 1
    buckets = [defaultdict(list) for _ in range(bands)]
    drop = set()
    for idx, sig in enumerate(sigs):
        candidates = set()
        for b in range(bands):
            key = (sig >> (b * band_bits)) & mask
            candidates.update(buckets[b].get(key, ()))
        for c in candidates:
            if c in drop:
                continue
            if hamming(sig, sigs[c]) <= threshold:
                drop.add(idx)
                break
        if idx not in drop:
            for b in range(bands):
                key = (sig >> (b * band_bits)) & mask
                buckets[b][key].append(idx)
    return drop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True,
                    help="Files or dirs (priority order — first-seen wins)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--near-dup", action="store_true",
                    help="Enable SimHash pass (slower, needs all diffs in memory)")
    ap.add_argument("--hamming", type=int, default=3,
                    help="Max Hamming distance for near-dup (default 3 / 64 bits)")
    args = ap.parse_args()

    paths = expand_inputs(args.input)
    if not paths:
        print("no inputs found", file=sys.stderr)
        sys.exit(1)
    print(f"reading {len(paths)} file(s):")
    for p in paths:
        print(f"   - {p}")

    seen_sha = set()
    seen_diff = set()
    kept = []        # records
    diffs_norm = []  # parallel, for SimHash
    before = Counter()
    dropped_sha = dropped_diff = 0
    total = 0

    for src, row in iter_rows(paths):
        total += 1
        label = row.get("label")
        if label:
            before[label] += 1

        # Pass A: (repo, sha)
        sha = str(row.get("sha") or "").strip()
        repo = str(row.get("repo") or "").strip()
        key_a = f"{repo}|{sha}" if sha else None
        if key_a and key_a in seen_sha:
            dropped_sha += 1
            continue
        if key_a:
            seen_sha.add(key_a)

        # Pass B: normalized diff MD5
        norm = normalize_diff(row.get("diff_text") or "")
        if not norm:
            dropped_diff += 1
            continue
        key_b = md5_of(norm)
        if key_b in seen_diff:
            dropped_diff += 1
            continue
        seen_diff.add(key_b)

        kept.append(row)
        if args.near_dup:
            diffs_norm.append(norm)

    dropped_near = 0
    if args.near_dup and kept:
        print(f"computing SimHash for {len(kept)} records...")
        sigs = [simhash64(d) for d in diffs_norm]
        drop_idx = banded_near_dup(sigs, args.hamming)
        dropped_near = len(drop_idx)
        kept = [r for i, r in enumerate(kept) if i not in drop_idx]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    after = Counter(r.get("label") for r in kept if r.get("label"))
    print(f"\nwrote {len(kept)} records to {out_path}")
    print(f"   input            : {total}")
    print(f"   dropped (repo,sha): {dropped_sha}")
    print(f"   dropped norm-diff: {dropped_diff}")
    if args.near_dup:
        print(f"   dropped near-dup : {dropped_near}  (hamming <= {args.hamming})")
    print(f"\n   label before: {dict(before)}")
    print(f"   label after : {dict(after)}")


if __name__ == "__main__":
    main()
