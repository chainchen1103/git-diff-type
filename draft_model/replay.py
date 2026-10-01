"""Replay a repository's recent commits through gca built with the subject
model, as if each author were about to commit: HEAD at the parent, the
commit's tree staged, the author's identity, and the author's type and
scope chosen. Records what gca would offer as the subject.

    python draft_model/replay.py GCA REPO MODEL N OUT.jsonl [--rev REV] [--since DATE]

GCA is a gca built with the t5 feature, MODEL its model file. The replay
works in a temporary worktree of REPO, removed at the end.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import time

LABEL = re.compile(r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(?:\(([^)]+)\))?(!?):\s(.+)$")
PREFIX = re.compile(r"^[A-Za-z]+(?:\([^()\r\n]*\))?!?:\s*")
PR_SUFFIX = re.compile(r"\s*\(#\d+(?:,\s*#\d+)*\)\s*$")
BOT = re.compile(r"(?i)\[bot\]|\bbot\b|renovate|dependabot|automation|actions-user|github-actions")


def git(repo, *args, env=None):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True, env=env).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gca")
    ap.add_argument("repo")
    ap.add_argument("model")
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--since")
    args = ap.parse_args()
    log_args = ["log", "--no-merges", "--format=%H%x1f%P%x1f%an%x1f%ae%x1f%s", "-n", "2000", args.rev]
    if args.since:
        log_args.insert(1, f"--since={args.since}")
    commits = [line.split("\x1f") for line in git(args.repo, *log_args).splitlines()]
    home = tempfile.mkdtemp(prefix="replay-home-")
    open(os.path.join(home, ".gitconfig"), "w").close()
    wt = tempfile.mkdtemp(prefix="replay-wt-")
    os.rmdir(wt)
    git(args.repo, "worktree", "add", "-q", "--detach", wt, args.rev)
    done = 0
    try:
        with open(args.out, "w", encoding="utf-8") as out:
            for sha, parents, name, email, subject in commits:
                if done >= args.n:
                    break
                m = LABEL.match(subject)
                if not m or BOT.search(f"{name} <{email}>") or len(parents.split()) != 1:
                    continue
                kind, scope = m.group(1), (m.group(2) or "").strip()
                text = PR_SUFFIX.sub("", PREFIX.sub("", subject, count=1)).strip()
                # HEAD at the parent, the commit's tree staged
                git(wt, "update-ref", "--no-deref", "HEAD", parents)
                git(wt, "read-tree", sha)
                env = dict(os.environ, HOME=home, GIT_CONFIG_GLOBAL=os.path.join(home, ".gitconfig"),
                           GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME=name, GIT_AUTHOR_EMAIL=email,
                           GIT_COMMITTER_NAME=name, GIT_COMMITTER_EMAIL=email, NO_COLOR="1")
                t0 = time.perf_counter()
                res = subprocess.run([args.gca, "--dry-run", "--json", "-t", kind, "--scope", scope,
                                      "--draft-model", args.model], cwd=wt, capture_output=True, text=True, env=env)
                ms = 1000 * (time.perf_counter() - t0)
                if res.returncode != 0:
                    out.write(json.dumps({"sha": sha, "error": res.stderr.strip()[-300:]}) + "\n")
                    continue
                v = json.loads(res.stdout)
                out.write(json.dumps({
                    "sha": sha, "type": kind, "scope": scope or None, "ref": text, "header": subject,
                    "files": len(v["files"]), "offered": v["subject_draft"], "source": v["subject_draft_source"],
                    "model": v["model_draft"], "ms": round(ms), "stderr": res.stderr.strip()[-200:],
                }, ensure_ascii=False) + "\n")
                out.flush()
                done += 1
    finally:
        git(args.repo, "worktree", "remove", "--force", wt)
    print(f"{done} commits replayed into {args.out}")


if __name__ == "__main__":
    main()
