#!/usr/bin/env python3
"""Mine Conventional Commits from a local git repository in parallel."""
import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

CONVENTIONAL_RE = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(?:\((?P<scope>[^)]+)\))?!?:\s(?P<desc>.+)"
)

def run_git(cmd_list, cwd):
    try:
        result = subprocess.run(
            cmd_list, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except OSError:
        return None

def parse_stats(stat_text):
    """Parse 'git show --numstat' output. Binary files count as changed with
    zero lines, matching what the CLI sees at inference time."""
    files_changed = 0
    additions = 0
    deletions = 0
    top_exts = []

    entries = iter(stat_text.split('\0') if '\0' in stat_text else stat_text.splitlines())
    for line in entries:
        if not line.strip():
            continue
        parts = line.split('\t', 2)
        if len(parts) == 3:
            adds, dels, filename = parts
            if not filename and '\0' in stat_text:
                next(entries, '')
                filename = next(entries, '')
            files_changed += 1
            additions += 0 if adds == '-' else int(adds)
            deletions += 0 if dels == '-' else int(dels)

            ext = Path(filename).suffix.lower()
            if ext:
                top_exts.append(ext)

    top_ext_str = ""
    if top_exts:
        c = Counter(top_exts)
        top_ext_str = ",".join([e for e, _ in c.most_common(3)])

    return files_changed, additions, deletions, top_ext_str

def process_commit(sha, timestamp, subject, label, repo_path, max_diff_chars):
    stat_cmd = ["git", "show", sha, "--numstat", "-z", "--format=", "--no-relative", "--no-ext-diff", "--no-textconv"]
    stat_out = run_git(stat_cmd, repo_path)
    if stat_out is None:
        return None
    files_changed, additions, deletions, top_exts = parse_stats(stat_out)

    # --format= drops the commit header. Without it the message, and so the
    # label, would leak into diff_text.
    diff_cmd = [
        "git", "show", sha, "--format=", "--no-color", "--no-ext-diff", "--no-textconv",
        "--no-relative", "--src-prefix=a/", "--dst-prefix=b/", "--unified=3",
        "--output-indicator-new=+", "--output-indicator-old=-", "--output-indicator-context= ",
    ]
    diff_full = run_git(diff_cmd, repo_path)

    if not diff_full:
        return None

    return {
        "url": f"local://{Path(repo_path).name}/{sha}",
        "owner": "local",
        "repo": Path(repo_path).name,
        "sha": sha,
        "message": subject,
        "diff_text": diff_full.strip()[:max_diff_chars],
        "files_changed": files_changed,
        "additions": additions,
        "deletions": deletions,
        "top_exts": top_exts,
        "label": label,
        "labeled_at": datetime.fromtimestamp(int(timestamp), timezone.utc).isoformat().replace("+00:00", "Z"),
    }

def mine_repo(repo_path, output_file, limit=None, max_workers=10):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if max_workers <= 0:
        raise ValueError("workers must be positive")
    repo_path = Path(repo_path).resolve()
    if run_git(["git", "rev-parse", "--git-dir"], repo_path) is None:
        raise ValueError(f"{repo_path} is not a valid git repository")

    print(f"mining {repo_path} using {max_workers} threads...")

    log_cmd = ["git", "log", "--pretty=format:%H|~|%at|~|%s", "--no-merges"]
    log_out = run_git(log_cmd, repo_path)
    if not log_out:
        print("No commits found.")
        return

    tasks = []
    lines = log_out.splitlines()
    print(f"scanning {len(lines)} commits for Conventional Commits format...")

    for line in lines:
        parts = line.split("|~|", 2)
        if len(parts) != 3:
            continue
        sha, timestamp, subject = parts

        match = CONVENTIONAL_RE.match(subject)
        if match:
            label = match.group("type")
            tasks.append((sha, timestamp, subject, label))
            if limit and len(tasks) >= limit:
                break

    print(f"found {len(tasks)} candidates; fetching diffs in parallel...")

    extracted_count = 0
    processed_count = 0
    start_time = time.time()

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor, out_path.open("a", encoding="utf-8") as f:
        pending = deque()
        task_iter = iter(tasks)
        for task in task_iter:
            pending.append(executor.submit(process_commit, *task, repo_path, 20000))
            if len(pending) >= max_workers:
                break
        total_tasks = len(tasks)
        while pending:
            res = pending.popleft().result()
            task = next(task_iter, None)
            if task is not None:
                pending.append(executor.submit(process_commit, *task, repo_path, 20000))
            processed_count += 1

            if processed_count % 100 == 0 or processed_count == total_tasks:
                elapsed = time.time() - start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\rProgress: {processed_count}/{total_tasks} | Speed: {speed:.1f} commits/s")
                sys.stdout.flush()

            if res:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                extracted_count += 1

    print(f"\nfinished mining; extracted {extracted_count} records")

    print(f"saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Mine Conventional Commits from a local git repository")
    parser.add_argument("--repo", required=True, help="Path to local git repository")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Max commits to extract")
    parser.add_argument("--workers", type=int, default=16, help="Number of threads (default: 16)")

    args = parser.parse_args()
    try:
        mine_repo(args.repo, args.out, args.limit, args.workers)
    except ValueError as error:
        parser.error(str(error))

if __name__ == "__main__":
    main()
