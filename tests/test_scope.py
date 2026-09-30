import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from evaluate_scope import before, dirs_above, now  # noqa: E402


def commit(entries):
    return {"scope": None, "asked": True, "entries": entries}


class ScopeTests(unittest.TestCase):
    # entries: (index from the most recent, scope, to a same file, in a same
    # directory, depth of the deepest shared directory, by the same author)

    def test_directories_above_a_file(self):
        self.assertEqual(dirs_above("a/b/c.rs"), ["a", "a/b"])
        self.assertEqual(dirs_above("README.md"), [])

    def test_no_scope_is_a_choice_now(self):
        e = [(0, None, True, True, 1, False), (1, None, True, True, 1, False), (2, "cli", True, True, 1, False)]
        self.assertEqual(before(commit(e)), "cli")
        self.assertIsNone(now(commit(e)))

    def test_the_deepest_shared_directory_counts_first(self):
        e = [(0, "core", False, False, 1, False), (1, "core", False, False, 1, False),
             (2, "foo", False, False, 2, False)]
        self.assertEqual(now(commit(e)), "foo")
        # gca 0.3 only looked at the same directory
        self.assertIsNone(before(commit(e)))

    def test_own_commits_count_more_and_ties_go_to_the_latest(self):
        e = [(0, "x", True, True, 1, False), (1, "x", True, True, 1, False), (2, "y", True, True, 1, True)]
        self.assertEqual(now(commit(e), own_votes=1), "x")
        self.assertEqual(now(commit(e), own_votes=8), "y")
        tie = [(0, "b", True, True, 1, False), (1, "a", True, True, 1, False)]
        self.assertEqual(now(commit(tie)), "b")


if __name__ == "__main__":
    unittest.main()
