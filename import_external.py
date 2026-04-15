#!/usr/bin/env python3
"""Stream external commit datasets and convert to the project JSONL schema.

Supported sources: JetBrains-Research/commit-chronicle, Maxscha/commitbench.
Uses HF datasets streaming mode to avoid full download; filters to Conventional
Commit format; caps per-label for balance.

Usage:
    pip install datasets
    python import_external.py --source commitchronicle \\
        --out datasets/commitchronicle.jsonl --limit 80000 --per-label 8000
"""
import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CONVENTIONAL_RE = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\((?P<scope>[^)]+)\))?!?:\s(?P<desc>.+)"
)

SOURCES = {
    "commitchronicle": {"hf_name": "JetBrains-Research/commit-chronicle", "split": "train"},
    "commitbench":     {"hf_name": "Maxscha/commitbench",                 "split": "train"},
}

ALL_LABELS = ("feat", "fix", "docs", "style", "refactor", "perf",
              "test", "build", "ci", "chore", "revert")

MAX_DIFF_CHARS = 20000


def count_diff_lines(diff_text: str):
    add = dele = 0
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            add += 1
        elif line.startswith("-") and not line.startswith("---"):
            dele += 1
    return add, dele


def top_exts_from_paths(paths):
    exts = [Path(p).suffix.lower() for p in paths if p]
    exts = [e for e in exts if e]
    if not exts:
        return ""
    return ",".join(e for e, _ in Counter(exts).most_common(3))


def rebuild_diff_from_mods(mods):
    """CommitChronicle: mods = list of {old_path, new_path, change_type, diff}."""
    chunks, paths = [], []
    for m in mods:
        old = m.get("old_path") or "/dev/null"
        new = m.get("new_path") or "/dev/null"
        paths.append(new if new != "/dev/null" else old)
        body = m.get("diff") or ""
        header = f"diff --git a/{old} b/{new}\n--- a/{old}\n+++ b/{new}\n"
        chunks.append(header + body)
    return "\n".join(chunks), paths


def normalize_commitchronicle(row):
    msg = (row.get("message") or "").strip()
    if not msg:
        return None
    first = msg.splitlines()[0]
    m = CONVENTIONAL_RE.match(first)
    if not m:
        return None
    mods = row.get("mods") or []
    if not mods:
        return None
    diff_text, paths = rebuild_diff_from_mods(mods)
    if not diff_text.strip():
        return None
    additions, deletions = count_diff_lines(diff_text)
    repo = row.get("repo") or "unknown/unknown"
    owner, _, name = str(repo).partition("/")
    return {
        "url": f"cc://{repo}/{row.get('hash', '')}",
        "owner": owner or "unknown",
        "repo": name or repo,
        "sha": row.get("hash", ""),
        "message": first,
        "diff_text": diff_text[:MAX_DIFF_CHARS],
        "files_changed": len(mods),
        "additions": additions,
        "deletions": deletions,
        "top_exts": top_exts_from_paths(paths),
        "label": m.group("type"),
        "labeled_at": str(row.get("date") or datetime.now(timezone.utc).isoformat()),
    }


def normalize_commitbench(row):
    # CommitBench field names vary by release; try common keys.
    msg = (row.get("message") or row.get("commit_message") or row.get("subject") or "").strip()
    if not msg:
        return None
    first = msg.splitlines()[0]
    m = CONVENTIONAL_RE.match(first)
    if not m:
        return None
    diff_text = row.get("diff") or row.get("patch") or row.get("diff_text") or ""
    if not diff_text.strip():
        return None
    additions, deletions = count_diff_lines(diff_text)
    paths = re.findall(r"^\+\+\+ b/(.+)$", diff_text, re.MULTILINE)
    if not paths:
        paths = re.findall(r"^diff --git a/.+ b/(.+)$", diff_text, re.MULTILINE)
    files_changed = len(set(paths)) or 1
    project = row.get("project") or row.get("repo") or "unknown/unknown"
    owner, _, name = str(project).partition("/")
    return {
        "url": f"cb://{project}/{row.get('hash', '')}",
        "owner": owner or "unknown",
        "repo": name or str(project) or "unknown",
        "sha": row.get("hash", ""),
        "message": first,
        "diff_text": diff_text[:MAX_DIFF_CHARS],
        "files_changed": files_changed,
        "additions": additions,
        "deletions": deletions,
        "top_exts": top_exts_from_paths(paths),
        "label": m.group("type"),
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }


NORMALIZERS = {
    "commitchronicle": normalize_commitchronicle,
    "commitbench":     normalize_commitbench,
}


def main():
    parser = argparse.ArgumentParser(description="Import external commit datasets -> GCA JSONL")
    parser.add_argument("--source", choices=SOURCES.keys(), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=100000,
                        help="Max total samples to write")
    parser.add_argument("--per-label", type=int, default=0,
                        help="Cap per label (0 = no cap). Recommended: limit/11")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (if dataset is gated)")
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency. Install: pip install datasets")
        sys.exit(1)

    info = SOURCES[args.source]
    normalize = NORMALIZERS[args.source]

    print(f"Streaming {info['hf_name']} (split={info['split']})...")
    kwargs = {"streaming": True, "split": info["split"]}
    if args.hf_token:
        kwargs["token"] = args.hf_token
    try:
        ds = load_dataset(info["hf_name"], **kwargs)
    except Exception as e:
        print(f"load_dataset failed: {e}")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_label = Counter()
    total = scanned = skipped_cap = 0

    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            scanned += 1
            if scanned % args.progress_every == 0:
                print(f"  scanned={scanned:>8}  kept={total:>6}  "
                      f"capped={skipped_cap:>6}  dist={dict(per_label)}")

            rec = normalize(row)
            if rec is None:
                continue

            lbl = rec["label"]
            if args.per_label and per_label[lbl] >= args.per_label:
                skipped_cap += 1
                # Early exit once every label has hit its cap.
                if all(per_label[k] >= args.per_label for k in ALL_LABELS):
                    print("  all labels at cap, stopping early")
                    break
                continue

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            per_label[lbl] += 1
            total += 1
            if total >= args.limit:
                break

    print(f"\nWrote {total} records to {out_path}")
    print(f"   Scanned: {scanned}  |  Skipped (per-label cap): {skipped_cap}")
    print(f"   Label distribution: {dict(per_label)}")


if __name__ == "__main__":
    main()
