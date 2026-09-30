#!/usr/bin/env python3
"""Would a removed public definition tell gca that a change is breaking?

Counts, on commits by people, how many are marked breaking (`type!:` or a
`BREAKING CHANGE:` footer) and how often a diff that removes a public
definition (an export in JS/TS, `pub` in Rust, an exported Go name, a
top-level Python def or class), outside tests and examples, is one of them.
The answer so far is: rarely, so gca does not suggest `!`.

Usage:
    python eval/breaking_signal.py datasets/test_unseen_recent.jsonl datasets/test_unseen_older.jsonl
"""
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dedupe import iter_rows  # noqa: E402

BANG = re.compile(r"^[A-Za-z]+(\([^()\r\n]*\))?!:")
FOOTER = re.compile(r"^BREAKING[ -]CHANGE:", re.M)
DEFS = [
    (re.compile(r"\.(ts|tsx|js|jsx|mjs|cjs|mts|cts)$"),
     re.compile(r"^export\s+(?:default\s+)?(?:declare\s+)?(?:async\s+)?"
                r"(?:function\*?|class|const|let|var|interface|type|enum|abstract\s+class)\s+([A-Za-z_$][\w$]*)")),
    (re.compile(r"\.rs$"),
     re.compile(r"^\s*pub\s+(?:async\s+)?(?:unsafe\s+)?(?:fn|struct|enum|trait|type|const|static|mod|union)\s+([A-Za-z_]\w*)")),
    (re.compile(r"\.go$"), re.compile(r"^(?:func\s+(?:\([^)]*\)\s*)?|type\s+)([A-Z]\w*)")),
    (re.compile(r"\.py$"), re.compile(r"^(?:async\s+)?(?:def|class)\s+([A-Za-z]\w*)")),
]
NOT_API = re.compile(r"(^|/)(tests?|__tests__|spec|e2e|examples?|playground|fixtures?|scripts|docs?|bench(es|marks)?)/"
                     r"|\.(test|spec)\.[jt]sx?$|_test\.(go|py)$|(^|/)test_[^/]*\.py$")


def removed_definitions(diff):
    """Public definitions the diff removes and does not add back."""
    removed, added = set(), set()
    pattern = None
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            path = line.rsplit(" b/", 1)[-1]
            pattern = None if NOT_API.search(path) else next((p for f, p in DEFS if f.search(path)), None)
        elif pattern and line[:1] in "+-" and not line.startswith(("+++", "---")):
            m = pattern.match(line[1:])
            if m:
                (added if line[0] == "+" else removed).add(m.group(1))
    return removed - added


def main():
    n = marked = fired = hits = 0
    by_repo = collections.Counter()
    for row in iter_rows([Path(p) for p in sys.argv[1:]]):
        if row.get("is_bot") or not isinstance(row.get("label"), str):
            continue
        message = row.get("message") or ""
        breaking = bool(BANG.match(message.split("\n", 1)[0]) or FOOTER.search(message))
        signal = bool(removed_definitions(row.get("diff_text") or ""))
        n += 1
        marked += breaking
        fired += signal
        hits += signal and breaking
        by_repo[row.get("repo")] += breaking
    print(f"{n} commits by people; {marked} marked breaking ({marked / n:.2%}), "
          f"in {sum(1 for v in by_repo.values() if v)} of {len(by_repo)} repositories")
    print(f"a public definition removed: {fired} commits ({fired / n:.1%}); "
          f"marked breaking {hits / max(fired, 1):.1%} of those; finds {hits / max(marked, 1):.1%} of the breaking ones")


if __name__ == "__main__":
    main()
