#!/usr/bin/env python3
"""Write gca-rs/tests/t5_input_fixtures.json: what t5_format.py and
prepare.py make of held-out commits and of edge cases, for the Rust port
(gca-rs/src/t5draft/input.rs) to reproduce exactly.

    python draft_model/gen_fixtures.py [--n 40] [--n-history 30] [--out FILE]

--n and --n-history are the numbers of real commits for the two kinds of
case; a larger --out file checks the port more thoroughly once
(GCA_T5_FIXTURES=FILE cargo test t5draft).
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(HERE)]
from dedupe import iter_rows  # noqa: E402
from history_sim import scope_of  # noqa: E402
from prepare import Timelines  # noqa: E402
from t5_format import t5_input  # noqa: E402

# Real diffs are cut here to keep the file small; the cases are about the rules.
MAX_DIFF = 2400
# Earlier commits given to each history case, and the most files any of
# them (or the case itself) may have.
LOG_DEPTH = 15
MAX_CASE_FILES = 25

EDGE_DIFFS = [
    "",
    "preamble before any diff\n+not a change\n",
    "diff --git a/w.txt b/w.txt\r\n--- a/w.txt\r\n+++ b/w.txt\r\n@@ -1,2 +1,2 @@ ctx\r\n-hello world\r\n+hello  world\r\n",
    'diff --git "a/\\303\\244.txt" "b/\\303\\244.txt"\nnew file mode 100644\n--- /dev/null\n+++ "b/\\303\\244.txt"\n'
    "@@ -0,0 +1 @@\n+äää\n",
    "diff --git a/x b/y b/x b/y\n--- a/x b/y\n+++ b/x b/y\n@@ -1 +1 @@\n-a\n+b\n",
    "".join(f"diff --git a/f{i}.py b/f{i}.py\n--- a/f{i}.py\n+++ b/f{i}.py\n@@ -1 +1 @@ def f{i}():\n-    return {i}\n"
            f"+    return {i + 1}\n" for i in range(15)),
    "diff --git a/s.txt b/s.txt\n--- a/s.txt\n+++ b/s.txt\n@@ -1,4 +1,4 @@\n-\x1c sep \x1f\n+　wide space　\n"
    "+​zero width\n+++counter\n---dashes\n\\ No newline at end of file\n+\t\n-   \n",
    "diff --git a/m.c b/m.c\n@@ -1 +1 @@\n+a\n@@ -5 +5 @@   \n+b\n@@@ -1,2 -1,2 +1,3 @@@ combined\n+c\n@@\n+d\n",
    "diff --git a/img.png b/img.png\nnew file mode 100644\nBinary files /dev/null and b/img.png differ\n"
    "diff --git a/old.bin b/old.bin\ndeleted file mode 100644\nBinary files a/old.bin and /dev/null differ\n",
    "diff --git a/a.txt b/b.txt\nsimilarity index 100%\nrename from a.txt\nrename to b.txt\n"
    "diff --git a/c.txt b/d.txt\nsimilarity index 80%\nrename from \nrename to d.txt\n@@ -1 +1 @@\n-x\n+y\n",
    "diff --git a/n.txt b/n.txt\n--- a/n.txt\n+++ b/n.txt\n@@ -1 +1 @@\n+first\nnew file mode 100644\ndeleted file mode 100644\n"
    "Binary files x and y differ\n",
    "diff --git a/e.md b/e.md\n@@ -1 +1 @@\n" + "".join(f"+\U0001f600 emoji line {i} éè " + "x" * 150 + "\n"
                                                for i in range(30)),
    "".join(f"diff --git a/{'very/long/directory/name/' * 6}file{i}.ts b/{'very/long/directory/name/' * 6}file{i}.ts\n"
            for i in range(14)),
]
EDGE_HISTORIES = [
    [],
    ["fix(parser): handle empty input", "feat: add ünicode support — " + "y" * 120, "docs: x"],
]
UNTYPED = ["Merge branch 'main' into feature", "WIP: not ready", "Update README.md", "deps: bump x", "Fix: capital"]


def ascii_safe(s):
    return not any(0xD800 <= ord(c) <= 0xDFFF for c in s)


def first_line(message):
    return (message or "").split("\n", 1)[0].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-history", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "gca-rs/tests/t5_input_fixtures.json"))
    args = ap.parse_args()
    data = Path(args.datasets)
    paths = [data / "test_unseen_older.jsonl", data / "test_unseen_recent.jsonl"]
    timelines = Timelines(paths)
    rows, lines = [], {}
    for r in iter_rows(paths):
        lines[r.get("sha")] = first_line(r.get("message"))
    for r in iter_rows([data / "test_unseen_recent.jsonl"]):
        if not r.get("is_bot") and isinstance(r.get("label"), str) and str(r.get("diff_text") or "").strip():
            rows.append(r)
    rng = random.Random(1)

    fmt = []
    for diff in EDGE_DIFFS:
        for kind, scope, hist in [("fix", None, EDGE_HISTORIES[0]), ("feat", "", EDGE_HISTORIES[1]),
                                  ("docs", "api", EDGE_HISTORIES[1])]:
            fmt.append({"diff": diff, "kind": kind, "scope": scope, "history": hist,
                        "expected": t5_input(diff, kind, scope, history=hist)})
    for r in rng.sample(rows, min(len(rows), args.n * 3)):
        if sum(1 for c in fmt if c.get("real")) >= args.n:
            break
        diff = str(r["diff_text"])[:MAX_DIFF]
        hist = timelines.history(r.get("repo"), r.get("sha"))
        case = {"diff": diff, "kind": r["label"], "scope": scope_of(r.get("message")), "history": hist,
                "expected": t5_input(diff, r["label"], scope_of(r.get("message")), history=hist), "real": True}
        if ascii_safe(json.dumps(case, ensure_ascii=False)):
            fmt.append(case)

    hist_cases = []
    for r in rng.sample(rows, len(rows)):
        if len(hist_cases) >= args.n_history:
            break
        repo, sha = r.get("repo"), r.get("sha")
        if repo not in timelines.repos or sha not in timelines.repos[repo][1]:
            continue
        commits, position = timelines.repos[repo][:2]
        i = position[sha]
        window = commits[max(0, i - LOG_DEPTH):i][::-1]  # newest first
        if not window:
            continue
        me = commits[i]
        if max(len(c[4]) for c in window + [me]) > MAX_CASE_FILES:
            continue
        log = [{"author": c[5], "subject": lines[c[1]], "files": sorted(c[4])} for c in window]
        # commits gca reads but that have no type: the port must skip them
        for k in range(0, len(log) + 1, 6):
            log.insert(k, {"author": rng.choice([me[5], "Someone Else"]), "subject": rng.choice(UNTYPED),
                           "files": sorted(me[4])[:2]})
        case = {"kind": me[2], "author": me[5], "staged": sorted(me[4]), "log": log,
                "expected": [commits[j][3] for j in timelines.picks(repo, sha, depth=LOG_DEPTH)]}
        if ascii_safe(json.dumps(case, ensure_ascii=False)):
            hist_cases.append(case)

    for c in fmt:
        c.pop("real", None)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"format": fmt, "history": hist_cases}, f, ensure_ascii=True, indent=0)
        f.write("\n")
    picked = sum(len(c["expected"]) for c in hist_cases)
    print(f"{len(fmt)} format cases, {len(hist_cases)} history cases ({picked} picks) -> {args.out}")


if __name__ == "__main__":
    main()
