#!/usr/bin/env python3
"""Mine Conventional Commits from a local git repository.

By default each candidate commit is read with its own `git show` calls, spread
over a thread pool. With --stream the whole history is read in two `git log`
passes instead (stats, then patches), which is much faster on large
repositories and also records when each commit landed, who wrote it and whether
the author is a bot; eval/collect.sh uses it. Both modes produce the same diff
text, pinned to git's default format whatever the local git config says.

Usage:
    python miner.py --repo <path> --out datasets/<name>.jsonl
    python miner.py --repo <path> --out out.jsonl --stream --since 2023-01-01 --until 2026-04-20
"""
import argparse
import codecs
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

def mine_repo(repo_path, output_file, limit=None, max_workers=10, max_diff_chars=20000):
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
            pending.append(executor.submit(process_commit, *task, repo_path, max_diff_chars))
            if len(pending) >= max_workers:
                break
        total_tasks = len(tasks)
        while pending:
            res = pending.popleft().result()
            task = next(task_iter, None)
            if task is not None:
                pending.append(executor.submit(process_commit, *task, repo_path, max_diff_chars))
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

# --- Streaming mode (--stream) ---------------------------------------------

TYPES = "feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert"
# CONVENTIONAL_RE as an extended regex for `git log --grep`, so git skips
# other commits without Python ever seeing them. It is only a prefilter: the
# subject is still checked with CONVENTIONAL_RE.
GREP_PATTERN = rf"^({TYPES})(\([^)]+\))?!?:[[:space:]]"

BOT_RE = re.compile(
    r"\[bot\]|\bbot\b|renovate|dependabot|automation|actions-user|github-actions",
    re.IGNORECASE,
)

# Keep in sync with DIFF_CONFIG and staged_diff() in gca-rs/src/git.rs.
DIFF_CONFIG = [
    "-c", "diff.noprefix=false",
    "-c", "diff.mnemonicPrefix=false",
    "-c", "diff.relative=false",
    "-c", "core.quotePath=true",
]
DIFF_FLAGS = [
    "--no-color", "--no-ext-diff", "--no-textconv", "--unified=3",
    "--src-prefix=a/", "--dst-prefix=b/",
    "--output-indicator-new=+", "--output-indicator-old=-", "--output-indicator-context= ",
]

RS, US = "\x1e", "\x1f"


def git_output(repo, args):
    """Run a git command with the pinned diff settings and yield its stdout as
    it arrives. Bytes are decoded as the CLI decodes them: UTF-8, invalid
    sequences replaced, line endings left alone."""
    proc = subprocess.Popen(
        ["git", *DIFF_CONFIG, *args], cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    with proc.stdout:
        yield proc.stdout
    if proc.wait() != 0:
        raise RuntimeError(f"`git {args[0]}` failed in {repo}")


def git_lines(repo, args):
    """Yield stdout lines, without their newline."""
    for stdout in git_output(repo, args):
        for raw in stdout:
            yield raw.decode("utf-8", errors="replace").removesuffix("\n")


def git_records(repo, args):
    """Yield the pieces of stdout that start with RS, without the RS."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    for stdout in git_output(repo, args):
        for block in iter(lambda: stdout.read(1 << 16), b""):
            *done, buf = (buf + decoder.decode(block)).split(RS)
            yield from (record for record in done if record)
    buf += decoder.decode(b"", final=True)
    if buf:
        yield buf


def shallow_boundary(repo):
    """Commits at the edge of a shallow clone. git shows them as if the whole
    tree was added, so their diffs are meaningless."""
    out = run_git(["git", "rev-parse", "--git-path", "shallow"], repo)
    path = Path(repo) / out.strip() if out and out.strip() else None
    if path and path.exists():
        return set(path.read_text().split())
    return set()


def read_meta(repo, rev_args):
    """Pass 1: header fields and --numstat counts of every candidate commit,
    counted by parse_stats() exactly as the per-commit mode counts them."""
    fmt = f"--format={RS}%H{US}%at{US}%ct{US}%an{US}%ae{US}%s"
    meta = {}
    args = ["log", "-z", *rev_args, fmt, "--numstat", "--no-ext-diff", "--no-textconv"]
    for record in git_records(repo, args):
        header, _, stats = record.partition("\0")
        sha, at, ct, name, email, subject = header.split(US, 5)
        match = CONVENTIONAL_RE.match(subject)
        if not match:
            continue
        files_changed, additions, deletions, exts = parse_stats(stats.lstrip("\n"))
        meta[sha] = {
            "at": int(at), "ct": int(ct), "author": name, "email": email,
            "subject": subject, "label": match.group("type"),
            "files_changed": files_changed, "additions": additions,
            "deletions": deletions, "top_exts": exts,
        }
    return meta


def read_patches(repo, rev_args, wanted, max_chars):
    """Pass 2: the patch of each wanted commit, cut at max_chars exactly as
    process_commit() cuts the output of `git show --format=`. Only the first
    max_chars (plus a margin, so that stripping never shortens the cut) of a
    patch is kept in memory, however large the commit."""
    keep = max_chars + 1000
    sha, buf, size, cut = None, [], 0, False

    def flush():
        if sha in wanted:
            text = "\n".join(buf)
            yield sha, (text.lstrip() if cut else text.strip())[:max_chars]

    for line in git_lines(repo, ["log", *rev_args, f"--format={RS}%H", "-p", *DIFF_FLAGS]):
        if line.startswith(RS):
            yield from flush()
            sha, buf, size, cut = line[1:].strip(), [], 0, False
        elif size <= keep:
            buf.append(line)
            size += len(line) + 1
        else:
            cut = True
    yield from flush()


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def mine_repo_stream(repo_path, output_file, limit=None, since=None, until=None,
                     max_diff_chars=20000):
    """Mine with two streamed `git log` passes. since and until are Unix times
    compared with the committer date; limit keeps the newest commits."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    repo_path = Path(repo_path).resolve()
    if run_git(["git", "rev-parse", "--git-dir"], repo_path) is None:
        raise ValueError(f"{repo_path} is not a valid git repository")
    if run_git(["git", "rev-parse", "--verify", "--quiet", "HEAD"], repo_path) is None:
        print("No commits found.")
        return
    name = repo_path.name
    start = time.time()

    rev_args = ["--no-merges", "--extended-regexp", f"--grep={GREP_PATTERN}"]
    meta = read_meta(repo_path, rev_args)
    boundary = shallow_boundary(repo_path)
    for sha in boundary & meta.keys():
        del meta[sha]
    wanted = {sha for sha, m in meta.items()
              if (since is None or m["ct"] >= since) and (until is None or m["ct"] < until)}
    if limit:
        wanted = set(sorted(wanted, key=lambda s: meta[s]["ct"], reverse=True)[:limit])

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for sha, diff_text in read_patches(repo_path, rev_args, wanted, max_diff_chars):
            if not diff_text:
                continue
            m = meta[sha]
            f.write(json.dumps({
                "url": f"local://{name}/{sha}",
                "owner": "local",
                "repo": name,
                "sha": sha,
                "message": m["subject"],
                "diff_text": diff_text,
                "files_changed": m["files_changed"],
                "additions": m["additions"],
                "deletions": m["deletions"],
                "top_exts": m["top_exts"],
                "label": m["label"],
                "labeled_at": iso(m["at"]),
                "committed_at": iso(m["ct"]),
                "author": m["author"],
                "is_bot": bool(BOT_RE.search(f"{m['author']} {m['email']}")),
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"{name}: {written} commits -> {out_path} "
          f"({time.time() - start:.1f}s, {len(boundary)} shallow boundary skipped)")


def parse_date(text):
    """YYYY-MM-DD (UTC) as a Unix time, or None."""
    if not text:
        return None
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def main():
    parser = argparse.ArgumentParser(description="Mine Conventional Commits from a local git repository")
    parser.add_argument("--repo", required=True, help="Path to local git repository")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Max commits to extract")
    parser.add_argument("--workers", type=int, default=16, help="Number of threads (default: 16)")
    parser.add_argument("--max-diff-chars", type=int, default=20000,
                        help="cut each diff at this many characters, as the CLI does (default: 20000)")
    stream = parser.add_argument_group("streaming mode")
    stream.add_argument("--stream", action="store_true",
                        help="read the history in two git log passes; adds committed_at, author and is_bot")
    stream.add_argument("--since", help="only commits that landed on or after this date (UTC, YYYY-MM-DD)")
    stream.add_argument("--until", help="only commits that landed before this date (UTC, YYYY-MM-DD)")

    args = parser.parse_args()
    if (args.since or args.until) and not args.stream:
        parser.error("--since and --until need --stream")
    try:
        if args.stream:
            mine_repo_stream(args.repo, args.out, args.limit, parse_date(args.since),
                             parse_date(args.until), args.max_diff_chars)
        else:
            mine_repo(args.repo, args.out, args.limit, args.workers, args.max_diff_chars)
    except ValueError as error:
        parser.error(str(error))

if __name__ == "__main__":
    main()
