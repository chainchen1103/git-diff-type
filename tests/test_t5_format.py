"""The subject model's input layout and history picks still match the cases
the Rust port is tested against (gca-rs/tests/t5_input_fixtures.json)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "draft_model")]

from miner import CONVENTIONAL_RE  # noqa: E402
from prepare import Timelines  # noqa: E402
from t5_format import t5_input  # noqa: E402

FIXTURES = json.loads((ROOT / "gca-rs/tests/t5_input_fixtures.json").read_text(encoding="utf-8"))


class T5Input(unittest.TestCase):
    def test_layout(self):
        for n, c in enumerate(FIXTURES["format"]):
            self.assertEqual(t5_input(c["diff"], c["kind"], c["scope"], history=c["history"]), c["expected"],
                             f"format case {n}")

    def test_history_picks(self):
        """prepare.py's picks over each case's log, rebuilt as dataset rows:
        oldest first, the case's own commit last."""
        for n, c in enumerate(FIXTURES["history"]):
            rows = []
            log = list(reversed(c["log"])) + [{"author": c["author"], "subject": f"{c['kind']}: this commit",
                                               "files": c["staged"]}]
            for i, e in enumerate(log):
                m = CONVENTIONAL_RE.match(e["subject"])
                rows.append({"repo": "r", "sha": f"{i:06d}", "committed_at": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
                             "label": m.group("type") if m else None, "message": e["subject"],
                             "author": e["author"],
                             "diff_text": "".join(f"diff --git a/{f} b/{f}\n" for f in e["files"])})
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "rows.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                timelines = Timelines([path])
            self.assertEqual(timelines.history("r", rows[-1]["sha"]), c["expected"], f"history case {n}")


if __name__ == "__main__":
    unittest.main()
