#!/usr/bin/env python3
"""Merge and deduplicate the commit corpus.

Deduplicate by commit identity, normalized diff, and optional SimHash distance.

First-seen wins, so list --input in priority order (curated first, external
last) to keep the trusted copy on tie.

Usage:
    python dedupe.py --input datasets/ --output datasets/_merged.jsonl
    python dedupe.py --input datasets/*.jsonl --output datasets/_merged.jsonl \\
                    --near-dup --hamming 3
"""
import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

def expand_inputs(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*.jsonl")) + sorted(p.glob("*.json")))
        elif p.is_file():
            out.append(p)
        else:
            matches = sorted(Path(match) for match in glob.glob(item) if Path(match).is_file())
            if matches:
                out.extend(matches)
            else:
                print(f"[warn] skipping missing: {item}", file=sys.stderr)
    return list({p.resolve(): p for p in out}.values())


def iter_rows(paths: Iterable[Path]):
    for p in paths:
        with p.open("r", encoding="utf-8-sig") as f:
            head = next((char for char in iter(lambda: f.read(1), "") if not char.isspace()), "")
            f.seek(0)
            if head == "[" or p.suffix.lower() == ".json":
                try:
                    content = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                else:
                    if isinstance(content, dict):
                        content = content.get("data", [content])
                    if isinstance(content, list):
                        for obj in content:
                            if isinstance(obj, dict):
                                yield obj
                    continue
            for i, ln in enumerate(f, 1):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        yield obj
                    else:
                        print(f"[warn] {p.name}:{i} expected a JSON object", file=sys.stderr)
                except json.JSONDecodeError as e:
                    print(f"[warn] {p.name}:{i} invalid JSON: {e}", file=sys.stderr)


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
    return (a ^ b).bit_count()


def banded_near_dup(sigs: List[int], threshold: int, bands=None):
    """Keep the first signature within the requested Hamming distance."""
    if not 0 <= threshold <= 64:
        raise ValueError("Hamming distance must be between 0 and 64")
    if threshold == 64:
        return set(range(1, len(sigs)))
    bands = threshold + 1 if bands is None else bands
    if not threshold < bands <= 64:
        raise ValueError("Use more bands than the Hamming threshold, up to 64")
    # More bands than differing bits guarantees at least one shared band.
    slices = []
    offset = 0
    for b in range(bands):
        width = 64 // bands + (b < 64 % bands)
        slices.append((offset, (1 << width) - 1))
        offset += width
    buckets = [defaultdict(list) for _ in range(bands)]
    drop = set()
    for idx, sig in enumerate(sigs):
        candidates = set()
        for b, (offset, mask) in enumerate(slices):
            key = (sig >> offset) & mask
            candidates.update(buckets[b].get(key, ()))
        for c in candidates:
            if hamming(sig, sigs[c]) <= threshold:
                drop.add(idx)
                break
        if idx not in drop:
            for b, (offset, mask) in enumerate(slices):
                key = (sig >> offset) & mask
                buckets[b][key].append(idx)
    return drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True,
                    help="Files or dirs in priority order; first-seen wins")
    ap.add_argument("--output", required=True)
    ap.add_argument("--near-dup", action="store_true",
                    help="Enable SimHash pass (slower)")
    ap.add_argument("--hamming", type=int, default=3,
                    help="Max Hamming distance for near-dup (default 3 / 64 bits)")
    args = ap.parse_args()
    if not 0 <= args.hamming <= 64:
        ap.error("--hamming must be between 0 and 64")

    out_path = Path(args.output)
    # A glob like datasets/*.jsonl also matches the previous output.
    paths = [p for p in expand_inputs(args.input) if p.resolve() != out_path.resolve()]
    if not paths:
        print("no inputs found", file=sys.stderr)
        sys.exit(1)
    print(f"reading {len(paths)} file(s):")
    for p in paths:
        print(f"   - {p}")

    seen_sha = set()
    seen_diff = set()
    kept = []
    sigs = []
    before = Counter()
    dropped_sha = dropped_diff = 0
    total = 0

    for row in iter_rows(paths):
        total += 1
        label = row.get("label")
        if label:
            before[label] += 1

        sha = str(row.get("sha") or "").strip()
        owner = str(row.get("owner") or "").strip()
        repo = str(row.get("repo") or "").strip()
        key_a = (owner, repo, sha) if sha else None
        if key_a and key_a in seen_sha:
            dropped_sha += 1
            continue
        diff_text = row.get("diff_text")
        norm = normalize_diff(diff_text) if isinstance(diff_text, str) else ""
        if not norm:
            dropped_diff += 1
            continue
        key_b = md5_of(norm)
        if key_b in seen_diff:
            dropped_diff += 1
            continue
        seen_diff.add(key_b)
        if key_a:
            seen_sha.add(key_a)

        kept.append(row)
        if args.near_dup:
            sigs.append(simhash64(norm))

    dropped_near = 0
    if args.near_dup and kept:
        print(f"checking near-dups among {len(kept)} records...")
        drop_idx = banded_near_dup(sigs, args.hamming)
        dropped_near = len(drop_idx)
        kept = [r for i, r in enumerate(kept) if i not in drop_idx]

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
