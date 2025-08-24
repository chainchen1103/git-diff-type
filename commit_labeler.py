#!/usr/bin/env python3
"""
commit_labeler.py — Fetch a GitHub commit by URL, show its message & diff,
then interactively record a human-provided label (feat/fix/docs/...)

Outputs a growing dataset CSV/JSONL that you can later use for training.

Key update: ensure CSV writing is fully robust even when diff contains commas, quotes, or newlines.
We always use csv.QUOTE_ALL and proper escaping.

Usage examples:
  python commit_labeler.py --outfile commits_labeled.csv
"""
from __future__ import annotations
import argparse
import csv
import dataclasses as dc
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import requests

DEFAULT_LABELS = [
    "feat", "fix", "docs", "style", "refactor", "test", "chore", "ci", "build", "perf"
]

GITHUB_COMMIT_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/commit/(?P<sha>[0-9a-f]{7,40})(?:\\b|/)?"
)

@dc.dataclass
class LabeledCommit:
    url: str
    owner: str
    repo: str
    sha: str
    message: str
    diff_text: str
    files_changed: int
    additions: int
    deletions: int
    top_exts: str
    label: str
    labeled_at: str  # ISO8601

    def to_row(self):
        return [
            self.url,
            self.owner,
            self.repo,
            self.sha,
            self.message.replace("\r", " "),
            self.diff_text.replace("\r", " "),
            self.files_changed,
            self.additions,
            self.deletions,
            self.top_exts,
            self.label,
            self.labeled_at,
        ]

    @staticmethod
    def csv_header():
        return [
            "url","owner","repo","sha","message","diff_text",
            "files_changed","additions","deletions","top_exts","label","labeled_at"
        ]


def parse_github_commit_url(url: str) -> Tuple[str, str, str]:
    m = GITHUB_COMMIT_URL_RE.match(url.strip())
    if not m:
        raise ValueError(f"Not a valid GitHub commit URL: {url}")
    return m.group("owner"), m.group("repo"), m.group("sha")


def gh_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    hdrs = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "commit-labeler/1.0",
    }
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    return hdrs


def fetch_commit(owner: str, repo: str, sha: str) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    r = requests.get(url, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_diff_text(commit_url: str) -> str:
    diff_url = commit_url.rstrip("/") + ".diff"
    r = requests.get(diff_url, headers={"User-Agent": "commit-labeler/1.0"}, timeout=60)
    if r.status_code == 404:
        return ""
    r.raise_for_status()
    text = r.text
    MAX_CHARS = 200_000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n\n[TRUNCATED]"
    return text


def compute_top_exts(files: List[dict], k: int = 3) -> str:
    from collections import Counter
    c = Counter()
    for f in files:
        filename = (f.get("filename") or "").lower()
        ext = Path(filename).suffix or "(noext)"
        c[ext] += 1
    return ",".join([e for e,_ in c.most_common(k)])


def print_preview(message: str, files_changed: int, additions: int, deletions: int, diff_text: str):
    print("\n================ Commit Message ================")
    print("\n".join(message.strip().splitlines()[:20]))
    print("\n================ Stats ================")
    print(f"files_changed={files_changed}, additions={additions}, deletions={deletions}")
    print("\n================ Diff Preview (first 200 lines) ================")
    for line in diff_text.splitlines()[:200]:
        print(line)


def ask_label(labels: List[str]) -> str:
    labels_lower = [l.lower() for l in labels]
    prompt = (
        "\nEnter label (one of: " + ", ".join(labels_lower) + ")\n"
        "[You may also type a new label; empty to skip]: "
    )
    while True:
        ans = input(prompt).strip()
        if ans == "":
            return ""
        if ans.lower() in labels_lower:
            return ans.lower()
        confirm = input(f"Use custom label '{ans}'? [y/N]: ").strip().lower()
        if confirm == "y":
            return ans


def ensure_parent_dirs(path: Path):
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def append_csv(csv_path: Path, items: List[LabeledCommit]):
    ensure_parent_dirs(csv_path)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(
            f,
            delimiter="," if csv_path.suffix.lower() != ".tsv" else "\t",
            quotechar='"',
            quoting=csv.QUOTE_ALL,  # always quote fields
            escapechar="\\",
            doublequote=True,
            lineterminator="\n",
        )
        if not file_exists:
            w.writerow(LabeledCommit.csv_header())
        for it in items:
            w.writerow(it.to_row())


def append_jsonl(jsonl_path: Path, items: List[LabeledCommit]):
    ensure_parent_dirs(jsonl_path)
    with jsonl_path.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(dc.asdict(it), ensure_ascii=False) + "\n")


def interactive_loop(labels: List[str], outbase: Path):
    csv_path = outbase.with_suffix(".csv") if outbase.suffix != ".csv" else outbase
    jsonl_path = outbase.with_suffix(".jsonl") if outbase.suffix != ".jsonl" else outbase
    print("\n[Interactive mode] Paste a GitHub commit URL each round. Type ':q' or Ctrl+C to quit.")
    i = 0
    try:
        while True:
            url = input("\nCommit URL (:q to quit): ").strip()
            if not url:
                print("[i] Empty input, skipping.")
                continue
            if url == ":q":
                print("[i] Quit requested.")
                break
            i += 1
            print(f"\n===== [{i}] {url} =====")
            item = label_one(url, labels)
            if item:
                append_csv(csv_path, [item])
                append_jsonl(jsonl_path, [item])
                print(f"[✓] Saved → {csv_path} and {jsonl_path}")
    except KeyboardInterrupt:
        print("\n[i] Interrupted by user. Bye!")


def label_one(url: str, labels: List[str]) -> Optional[LabeledCommit]:
    try:
        owner, repo, sha = parse_github_commit_url(url)
    except Exception as e:
        print(f"[!] {e}")
        return None
    try:
        data = fetch_commit(owner, repo, sha)
    except Exception as e:
        print(f"[!] Failed to fetch commit meta: {e}")
        return None
    message = data.get("commit", {}).get("message", "").strip()
    files = data.get("files", []) or []
    files_changed = len(files)
    additions = sum(int(f.get("additions", 0)) for f in files)
    deletions = sum(int(f.get("deletions", 0)) for f in files)
    top_exts = compute_top_exts(files)
    diff_text = fetch_diff_text(url)
    print_preview(message, files_changed, additions, deletions, diff_text)
    label = ask_label(labels)
    if not label:
        print("[i] Skipped.")
        return None
    labeled_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return LabeledCommit(url, owner, repo, sha, message, diff_text, files_changed, additions, deletions, top_exts, label, labeled_at)


def read_lines(p: Path) -> List[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Interactive GitHub commit labeler")
    ap.add_argument("--url", help="A single GitHub commit URL", default=None)
    ap.add_argument("--url-file", help="A text file containing one commit URL per line", default=None)
    ap.add_argument("--labels", help="Comma-separated list of allowed labels", default=",".join(DEFAULT_LABELS))
    ap.add_argument("--outfile", help="Output base path (default=commits_labeled)", default="commits_labeled")
    args = ap.parse_args(argv)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    urls: List[str] = []
    if args.url:
        urls.append(args.url)
    if args.url_file:
        urls.extend(read_lines(Path(args.url_file)))
    if not urls:
        interactive_loop(labels, Path(args.outfile))
        return 0

    outbase = Path(args.outfile)
    csv_path = outbase.with_suffix(".csv")
    jsonl_path = outbase.with_suffix(".jsonl")
    for i, url in enumerate(urls, 1):
        print(f"\n===== [{i}/{len(urls)}] {url} =====")
        item = label_one(url, labels)
        if item:
            append_csv(csv_path, [item])
            append_jsonl(jsonl_path, [item])
            print(f"[✓] Saved → {csv_path} and {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
