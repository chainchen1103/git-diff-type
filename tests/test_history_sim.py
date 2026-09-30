import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from history_sim import (FILE_PSEUDO_COMMITS, counts_before, files_of, latest_by_author,  # noqa: E402
                         own_choices, timelines, weigh, weigh_own, window_start)


def commit(sha, when, label, files, bot=False, author="Ada"):
    diff = "".join(f"diff --git a/{f} b/{f}\n--- a/{f}\n+++ b/{f}\n@@ -1 +1 @@\n-a\n+b\n" for f in files)
    return {"repo": "demo", "sha": sha, "committed_at": when, "label": label, "is_bot": bot, "diff_text": diff,
            "author": author}


class HistorySimTests(unittest.TestCase):
    def test_counts_the_project_and_the_same_files_before_each_commit(self):
        rows = [
            commit("a", "2026-01-01", "docs", ["site/guide.md", "site/nav.ts"]),
            commit("b", "2026-01-02", "feat", ["site/search.ts"]),
            commit("c", "2026-01-03", "fix", ["site/nav.ts"]),
            commit("d", "2026-01-04", "chore", ["site/nav.ts"], bot=True),
            commit("e", "2026-01-05", "refactor", ["site/nav.ts", "site/new.ts"]),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "commits.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            by_repo = timelines([path])
        classes = ["chore", "docs", "feat", "fix", "refactor"]
        n, same = counts_before(by_repo, rows, classes)
        # the last commit sees the four before it, the bot's left out
        np.testing.assert_array_equal(n[4], [0, 1, 1, 1, 0])
        # of those, the ones that touched site/nav.ts
        np.testing.assert_array_equal(same[4], [0, 1, 0, 1, 0])
        # the first sees nothing
        self.assertEqual(n[0].sum(), 0)
        self.assertEqual(files_of(rows[4]), frozenset({"site/nav.ts", "site/new.ts"}))
        # gca's log reaches back to the first commit; the latest ones of an
        # author are the ones gca may read again
        self.assertEqual(window_start(by_repo, rows[4:], depth=500), ["2026-01-01"])
        self.assertEqual(latest_by_author(by_repo, {("demo", "Ada")}, exclude={"e"}, depth=2), {"b", "c"})

    def test_own_choices_are_the_author_s_latest_commits_in_the_log(self):
        keys = [("r", "ada")] * 4 + [("r", "bo")]
        times = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-02"]
        labels = ["fix", "feat", "fix", "docs", "fix"]
        classes = ["docs", "feat", "fix"]
        probs = np.array([[0.2, 0.3, 0.5]] * 5)
        # ada's last commit reads her two latest before it again; bo has none
        chosen, shown = own_choices(keys, times, labels, probs, classes, [3, 4], ["2026-01-01"] * 2, depth=2)
        np.testing.assert_array_equal(chosen[0], [0, 1, 1])
        np.testing.assert_allclose(shown[0], [0.4, 0.6, 1.0])
        self.assertEqual(chosen[1].sum(), 0)
        # where gca's log starts later, only what it reaches
        chosen, _ = own_choices(keys, times, labels, probs, classes, [3], ["2026-01-03"], depth=2)
        np.testing.assert_array_equal(chosen[0], [0, 0, 1])
        # a type chosen more often than the models suggested it rises
        prior = np.array([0.3, 0.3, 0.4])
        flat = np.log(np.full((1, 3), 1 / 3))
        tilted = weigh_own(flat, np.array([[2.0, 0, 0]]), np.array([[0.2, 0.8, 1.0]]), prior, 0.1)
        self.assertEqual(int(np.argmax(tilted)), 0)
        np.testing.assert_allclose(weigh_own(flat, np.zeros((1, 3)), np.zeros((1, 3)), prior, 0.1), flat)

    def test_weighing_leaves_commits_without_history_alone(self):
        prior = np.array([0.5, 0.3, 0.2])
        log_p = np.log(np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]))
        n = np.array([[0, 0, 0], [0, 0, 20]])
        tilted = weigh(log_p, n, prior, 0.1, FILE_PSEUDO_COMMITS)
        np.testing.assert_allclose(tilted[0], log_p[0])
        self.assertGreater(tilted[1, 2] - tilted[1, 0], log_p[1, 2] - log_p[1, 0])


if __name__ == "__main__":
    unittest.main()
