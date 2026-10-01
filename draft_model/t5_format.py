"""What the subject model reads: the type and scope chosen before the
subject, the files, the headers of a few earlier commits, and the changed
lines of the diff, compacted to fit a 512-token encoder.

gca-rs/src/t5draft/input.rs is the Rust port; gen_fixtures.py checks that
the two agree."""
import re

MAX_CHARS = 2000
MAX_FILES = 12
MAX_LINE = 160
MAX_HISTORY = 100
GIT_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def t5_input(diff, kind, scope, max_chars=MAX_CHARS, history=None):
    """history: headers of earlier commits ("fix(parser): handle empty
    input"), most relevant first; they go before the changed lines."""
    files = []
    cur = None
    in_hunk = False
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            m = GIT_HEADER.match(line)
            cur = {"status": "M", "path": m.group(2) if m else line[11:], "old": None, "body": []}
            files.append(cur)
            in_hunk = False
        elif cur is None:
            continue
        elif line.startswith("@@"):
            in_hunk = True
            parts = line.split("@@", 2)
            context = parts[2].strip() if len(parts) == 3 else ""
            cur["body"].append("@@ " + context if context else "@@")
        elif in_hunk:
            if line[:1] in ("+", "-"):
                text = line[1:].strip()
                if text:
                    cur["body"].append(line[0] + " " + text[:MAX_LINE])
        elif line.startswith("new file mode"):
            cur["status"] = "A"
        elif line.startswith("deleted file mode"):
            cur["status"] = "D"
        elif line.startswith("rename from "):
            cur["status"] = "R"
            cur["old"] = line[len("rename from "):]
        elif line.startswith("Binary files"):
            cur["body"].append("(binary)")
    names = [f"{f['status']} {f['old'] + ' -> ' if f['old'] else ''}{f['path']}" for f in files[:MAX_FILES]]
    if len(files) > MAX_FILES:
        names.append(f"and {len(files) - MAX_FILES} more")
    text = f"type: {kind}\nscope: {scope or 'none'}\nfiles: " + " | ".join(names)
    if history:
        text += "\nhistory:" + "".join("\n- " + h[:MAX_HISTORY] for h in history)
    for f in files:
        for piece in ["--- " + f["path"]] + f["body"]:
            if len(text) + 1 + len(piece) > max_chars:
                return text[:max_chars]
            text += "\n" + piece
    return text[:max_chars]
