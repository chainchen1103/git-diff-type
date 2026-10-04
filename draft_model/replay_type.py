#!/usr/bin/env python3
"""Replay a repository's newest typed commits by people through gca, with
and without the type model, as if each author were about to commit: HEAD at
the parent, the commit's tree staged, the author's identity. Counts how
often the pre-selected type and the top three hold the type the author
chose, and times the dry runs.

    python draft_model/replay_type.py GCA TYPE_MODEL REPO N

GCA must be built with the t5 feature. It adds and removes a temporary git
worktree of REPO.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

LABEL = re.compile(r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(?:\([^)]+\))?!?:\s")
BOT = re.compile(r"(?i)\[bot\]|\bbot\b|renovate|dependabot|automation|actions-user|github-actions")


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    gca, model, repo, n = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    log = git(repo, "log", "--no-merges", "--format=%H%x1f%P%x1f%an%x1f%ae%x1f%s", "-n", "5000")
    commits = [line.split("\x1f") for line in log.splitlines()]
    home = tempfile.mkdtemp(prefix="replay-home-")
    open(os.path.join(home, ".gitconfig"), "w").close()
    wt = tempfile.mkdtemp(prefix="replay-wt-")
    os.rmdir(wt)
    git(repo, "worktree", "add", "-q", "--detach", wt, "HEAD")
    right = {"without": [0, 0], "with": [0, 0]}
    ms = {"without": [], "with": []}
    done = 0
    try:
        for sha, parents, name, email, subject in commits:
            if done >= n:
                break
            m = LABEL.match(subject)
            if not m or BOT.search(f"{name} <{email}>") or len(parents.split()) != 1:
                continue
            git(wt, "update-ref", "--no-deref", "HEAD", parents)
            git(wt, "read-tree", sha)
            env = dict(os.environ, HOME=home, GIT_CONFIG_GLOBAL=os.path.join(home, ".gitconfig"),
                       GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME=name, GIT_AUTHOR_EMAIL=email,
                       GIT_COMMITTER_NAME=name, GIT_COMMITTER_EMAIL=email, NO_COLOR="1", GCA_TYPE_MODEL="")
            for mode, extra in (("without", []), ("with", ["--type-model", model])):
                t0 = time.perf_counter()
                res = subprocess.run([gca, "--dry-run", "--json", *extra], cwd=wt, capture_output=True, text=True,
                                     env=env, check=True)
                ms[mode].append(1000 * (time.perf_counter() - t0))
                v = json.loads(res.stdout)
                right[mode][0] += v["preselected"] == m.group(1)
                right[mode][1] += m.group(1) in [s["type"] for s in v["suggestions"]]
            done += 1
    finally:
        git(repo, "worktree", "remove", "--force", wt)
    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
    print(f"{done} commits: pre-selected type right {right['without'][0]} -> {right['with'][0]}, "
          f"in the top three {right['without'][1]} -> {right['with'][1]}; dry run median "
          f"{median(ms['without']):.0f} -> {median(ms['with']):.0f} ms")


if __name__ == "__main__":
    main()
