from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import sys

def load_json_any(path: Path) -> Tuple[List[dict], List[int]]:

    rows: List[dict] = []
    line_numbers: List[int] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                    line_numbers.append(i)
                except Exception as e:
                    print(f"[warn] skip invalid JSON at line {i}: {e}", file=sys.stderr)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, obj in enumerate(data, 1):
                if isinstance(obj, dict):
                    rows.append(obj)
                    line_numbers.append(i)
                else:
                    print(f"[warn] skip non-dict element at index {i}", file=sys.stderr)
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for i, obj in enumerate(data["data"], 1):
                if isinstance(obj, dict):
                    rows.append(obj)
                    line_numbers.append(i)
                else:
                    print(f"[warn] skip non-dict element at index {i} in data[]", file=sys.stderr)
        else:
            raise ValueError("Unsupported JSON structure: expected array or {'data': [...]}")

    return rows, line_numbers

def collect_dups(rows: List[dict], lines: List[int], keys: List[str]) -> Dict[str, Dict[str, List[int]]]:

    result: Dict[str, Dict[str, List[int]]] = {}
    for k in keys:
        buckets: Dict[str, List[int]] = {}
        for obj, ln in zip(rows, lines):
            if k not in obj:
                continue
            v = obj.get(k)
            if v is None:
                continue
            v = str(v)
            buckets.setdefault(v, []).append(ln)

        result[k] = {v: lns for v, lns in buckets.items() if len(lns) >= 2}
    return result

def write_report(out_path: Path, dups: Dict[str, Dict[str, List[int]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value", "count", "line_numbers"])
        for k, mapping in dups.items():
            for v, lns in sorted(mapping.items(), key=lambda x: (-len(x[1]), x[0])):
                w.writerow([k, v, len(lns), ";".join(map(str, sorted(lns)))])

def main() -> int:
    ap = argparse.ArgumentParser(description="Find duplicated url/sha and output their line numbers.")
    ap.add_argument("--input", required=True, help="Path to JSONL or JSON file")
    ap.add_argument("--out", default="out/duplicates_report.csv", help="Output CSV path")
    ap.add_argument("--keys", nargs="+", default=["url", "sha"], help="Which keys to deduplicate (default: url sha)")
    args = ap.parse_args()

    rows, lines = load_json_any(Path(args.input))
    dups = collect_dups(rows, lines, args.keys)

    total_dup_pairs = sum(len(m) for m in dups.values())
    print(f"[i] scanned rows: {len(rows)}")
    for k in args.keys:
        cnt = len(dups.get(k, {}))
        print(f"[i] duplicates for '{k}': {cnt}")

        shown = 0
        for v, lns in sorted(dups.get(k, {}).items(), key=lambda x: (-len(x[1]), x[0]))[:10]:
            print(f"    - {k}={v}  count={len(lns)}  lines={sorted(lns)}")
            shown += 1
        if cnt > shown:
            print(f"    ... and {cnt - shown} more")

    write_report(Path(args.out), dups)
    print(f"[ok] report written to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
