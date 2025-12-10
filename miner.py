#!/usr/bin/env python3
"""
miner.py — Mining existing git repositories for Conventional Commits (Multi-threaded).

Optimized for speed using ThreadPoolExecutor to run git commands in parallel.
"""
import argparse
import json
import re
import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Regex to match Conventional Commits headers
CONVENTIONAL_RE = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(?:\((?P<scope>[^)]+)\))?!?:\s(?P<desc>.+)"
)

def run_git(cmd_list, cwd):
    """Run a git command in the specified directory."""
    try:
        # close_fds=True helps in high concurrency on some unix systems
        result = subprocess.run(
            cmd_list, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace', close_fds=True
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None

def parse_stats(stat_text):
    """Parse 'git show --numstat' output."""
    files_changed = 0
    additions = 0
    deletions = 0
    top_exts = []
    
    for line in stat_text.splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=2)
        if len(parts) == 3:
            adds, dels, filename = parts
            if adds == '-' or dels == '-':
                continue
            
            files_changed += 1
            additions += int(adds)
            deletions += int(dels)
            
            ext = Path(filename).suffix.lower()
            if ext:
                top_exts.append(ext)
                
    from collections import Counter
    top_ext_str = ""
    if top_exts:
        c = Counter(top_exts)
        top_ext_str = ",".join([e for e, _ in c.most_common(3)])
        
    return files_changed, additions, deletions, top_ext_str

def process_commit(sha, timestamp, subject, label, repo_path, max_diff_chars):
    """Worker function to fetch details for a single commit."""
    
    # Get stats (files changed, +/-)
    stat_cmd = ["git", "show", sha, "--numstat", "--format="]
    stat_out = run_git(stat_cmd, repo_path)
    if stat_out is None:
        return None
    files_changed, additions, deletions, top_exts = parse_stats(stat_out)
    
    # Get full diff
    diff_cmd = ["git", "show", sha, "--no-color"]
    diff_full = run_git(diff_cmd, repo_path)
    
    if not diff_full:
        return None

    if len(diff_full) > max_diff_chars:
        diff_text = diff_full[:max_diff_chars] + "\n[TRUNCATED]"
    else:
        diff_text = diff_full

    return {
        "url": f"local://{Path(repo_path).name}/{sha}",
        "owner": "local",
        "repo": Path(repo_path).name,
        "sha": sha,
        "message": subject,
        "diff_text": diff_text,
        "files_changed": files_changed,
        "additions": additions,
        "deletions": deletions,
        "top_exts": top_exts,
        "label": label,
        "labeled_at": datetime.fromtimestamp(int(timestamp)).isoformat() + "Z"
    }

def mine_repo(repo_path, output_file, limit=None, max_workers=10):
    repo_path = Path(repo_path).resolve()
    if not (repo_path / ".git").exists():
        print(f"Error: {repo_path} is not a valid git repository.")
        return

    print(f"⛏️  Mining {repo_path} using {max_workers} threads...")
    
    # Capture Log
    log_cmd = ["git", "log", "--pretty=format:%H|~|%at|~|%s", "--no-merges"]
    if limit:
        log_cmd.extend(["-n", str(limit * 2) if limit else str(100000)])
        
    log_out = run_git(log_cmd, repo_path)
    if not log_out:
        print("No commits found.")
        return

    # Pre-filtering
    tasks = []
    lines = log_out.splitlines()
    print(f"🔍 Scanning {len(lines)} commits for Conventional Commits format...")

    for line in lines:
        parts = line.split("|~|", 2)
        if len(parts) != 3:
            continue
        sha, timestamp, subject = parts
        
        match = CONVENTIONAL_RE.match(subject)
        if match:
            label = match.group("type")
            tasks.append((sha, timestamp, subject, label))
            
    if limit:
        tasks = tasks[:limit]

    print(f"🚀 Found {len(tasks)} candidates. Fetching diffs in parallel...")
    
    # Parallel Execution
    results = []
    processed_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sha = {
            executor.submit(process_commit, sha, ts, sub, lbl, repo_path, 20000): sha 
            for sha, ts, sub, lbl in tasks
        }
        
        # Use as_completed to display the progress
        total_tasks = len(tasks)
        for future in as_completed(future_to_sha):
            res = future.result()
            processed_count += 1
            
            # Simple progress bar
            if processed_count % 100 == 0 or processed_count == total_tasks:
                elapsed = time.time() - start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\rProgress: {processed_count}/{total_tasks} | Speed: {speed:.1f} commits/s")
                sys.stdout.flush()

            if res:
                results.append(res)

    print(f"\n✅ Finished mining. Extracted {len(results)} valid datasets.")
    
    # Write to file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"💾 Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Mine Conventional Commits (Multi-threaded)")
    parser.add_argument("--repo", required=True, help="Path to local git repository")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Max commits to extract")
    parser.add_argument("--workers", type=int, default=16, help="Number of threads (default: 16)")
    
    args = parser.parse_args()
    mine_repo(args.repo, args.out, args.limit, args.workers)

if __name__ == "__main__":
    main()