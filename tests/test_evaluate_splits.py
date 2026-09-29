import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from evaluate_splits import (
    LABELS, chronological_split, clean_diff, deduplicate, evaluate_fold,
    load_corpus, metrics, remote_identity, repository_identity, validate_split,
)
from train_enhanced import FEATURE_COLUMNS, prepare_features


def diff(body="return result"):
    return (
        "diff --git a/src/parser.py b/src/parser.py\n"
        "index 1234567..7654321 100644\n"
        "--- a/src/parser.py\n+++ b/src/parser.py\n"
        f"@@ -1 +1 @@\n-return old\n+{body}\n"
    )


def split_frames():
    rows = [
        {
            "row_id": index,
            "sha": f"{index:040x}",
            "diff_hash": f"diff-{index}",
            "repository": "acme/train" if index < 6 else "acme/test",
            "label": ("fix" if index < 3 else "feat") if index < 6 else "docs",
            "commit_time": 100 + index if index < 6 else 200 + index,
            "diff_text": diff(str(index)),
        }
        for index in range(8)
    ]
    frame = prepare_features(pd.DataFrame(rows))
    return frame.iloc[:6].copy(), frame.iloc[6:].copy()


class CorpusCleaningTests(unittest.TestCase):
    def test_commit_preamble_is_removed_before_truncation(self):
        preamble = "commit " + "a" * 40 + "\nAuthor: Person\nDate: yesterday\n\nfix: leaked label\n"
        text = preamble + diff()
        self.assertEqual(clean_diff(text), diff().strip())
        self.assertEqual(clean_diff(text, max_diff_len=40), diff()[:40])
        self.assertEqual(clean_diff(preamble * 1000 + diff(), max_diff_len=40), diff()[:40])
        self.assertIn("+return result", clean_diff(text))

    def test_missing_or_invalid_patch_is_rejected(self):
        for text in (None, 42, "", "fix: change behavior", "prefix diff --git a/a b/a"):
            with self.subTest(text=text):
                self.assertIsNone(clean_diff(text))

    def test_repository_aliases_match_local_commitbench_and_commitchronicle(self):
        aliases = {"widget-clone": "acme/widget"}
        rows = [
            {"url": "local://widget-clone/commit", "owner": "local", "repo": "Widget-Clone.git"},
            {"url": "cb://acme_widget/commit", "owner": "Acme_Widget", "repo": "Acme_Widget"},
            {"url": "cc://acme/widget/commit", "owner": "Acme", "repo": "Widget.git"},
            {"url": "cc://acme/widget/commit", "owner": "unknown", "repo": "Acme/Widget.git"},
        ]
        for row in rows:
            with self.subTest(row=row):
                self.assertEqual(repository_identity(row, aliases), "acme/widget")
        self.assertEqual(repository_identity({"url": "cb://acme_widget_core", "owner": "acme_widget_core", "repo": "acme_widget_core"}, {}), "acme/widget_core")
        for row in ({"owner": "local", "repo": "missing"}, {"owner": "unknown", "repo": "widget"}, {"owner": "acme", "repo": "too/many/parts"}):
            with self.subTest(row=row):
                self.assertIsNone(repository_identity(row, aliases))

    def test_remote_identity_normalizes_git_transport_urls(self):
        for remote in (
            "https://github.com/Acme/Widget.git\n",
            "git@github.com:Acme/Widget.git",
            "ssh://git@github.com/Acme/Widget.git",
        ):
            self.assertEqual(remote_identity(remote), "acme/widget")
        self.assertIsNone(remote_identity("../widget"))

    def test_corpus_uses_git_times_and_deduplicates_after_cleaning(self):
        first_sha, second_sha, unknown_sha = "a" * 40, "b" * 40, "c" * 40
        rows = [
            {"label": "fix", "owner": "local", "repo": "widget-clone", "url": "local://widget-clone", "sha": first_sha, "diff_text": "commit metadata\nfix: leaked label\n" + diff(), "timestamp": 1},
            {"label": "fix", "owner": "acme_widget", "repo": "acme_widget", "url": "cb://acme_widget", "sha": second_sha, "diff_text": diff().replace("@@ -1 +1 @@", "@@ -9 +9 @@"), "timestamp": 999999},
            {"label": "fix", "owner": "acme", "repo": "widget", "url": "cc://acme/widget", "sha": unknown_sha, "diff_text": diff("return another_result"), "timestamp": 1, "commit_time": 1, "labeled_at": "2000-01-01"},
            {"label": "fix", "owner": "acme", "repo": "widget", "diff_text": "no patch"},
            {"label": "fix", "owner": "unknown", "repo": "widget", "diff_text": diff("return unknown")},
            {"label": "not-a-type", "diff_text": diff()},
        ]
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.jsonl"
            path.write_bytes(raw)
            with patch("evaluate_splits.git_metadata", return_value=({"widget-clone": "acme/widget"}, {first_sha: 200, second_sha: 100})):
                frame, audit = load_corpus(path, "unused")
        self.assertEqual(frame.row_id.tolist(), [2, 3])
        self.assertEqual(frame.repository.tolist(), ["acme/widget", "acme/widget"])
        self.assertEqual(frame.iloc[0].commit_time, 100)
        self.assertTrue(pd.isna(frame.iloc[1].commit_time))
        self.assertEqual(audit["trusted_time_rows"], 1)
        self.assertEqual(audit["rows_without_trusted_time"], 1)
        self.assertEqual(audit["preamble_removed_rows"], 1)
        self.assertEqual(audit["duplicate_rows_removed"], 1)
        self.assertEqual(audit["missing_patch_rows"], 1)
        self.assertEqual(audit["unknown_repository_rows"], 1)
        self.assertEqual(audit["invalid_label_rows"], 1)
        self.assertEqual(audit["data_sha256"], hashlib.sha256(raw).hexdigest())


class DeduplicationTests(unittest.TestCase):
    def test_transitive_sha_and_diff_links_keep_earliest_trusted_row(self):
        frame = pd.DataFrame([
            {"row_id": 1, "sha": "a", "diff_hash": "x", "label": "fix", "commit_time": 200},
            {"row_id": 2, "sha": "a", "diff_hash": "y", "label": "fix", "commit_time": None},
            {"row_id": 3, "sha": "b", "diff_hash": "y", "label": "fix", "commit_time": 100},
            {"row_id": 4, "sha": "", "diff_hash": "z", "label": "fix", "commit_time": None},
            {"row_id": 5, "sha": "", "diff_hash": "w", "label": "fix", "commit_time": None},
            {"row_id": 6, "sha": "c", "diff_hash": "v", "label": "fix", "commit_time": 150},
            {"row_id": 7, "sha": "c", "diff_hash": "v", "label": "fix", "commit_time": 150},
        ], index=range(20, 27))
        retained, audit = deduplicate(frame)
        self.assertEqual(retained.row_id.tolist(), [3, 4, 5, 6])
        self.assertEqual(audit["duplicate_rows_removed"], 3)
        self.assertEqual(audit["conflicting_label_rows_removed"], 0)

    def test_label_conflict_removes_entire_transitive_component(self):
        frame = pd.DataFrame([
            {"row_id": 1, "sha": "a", "diff_hash": "x", "label": "fix", "commit_time": 100},
            {"row_id": 2, "sha": "a", "diff_hash": "y", "label": "fix", "commit_time": 200},
            {"row_id": 3, "sha": "b", "diff_hash": "y", "label": "feat", "commit_time": 300},
            {"row_id": 4, "sha": "c", "diff_hash": "z", "label": "fix", "commit_time": 400},
        ])
        retained, audit = deduplicate(frame)
        self.assertEqual(retained.row_id.tolist(), [4])
        self.assertEqual(audit, {"duplicate_rows_removed": 0, "conflicting_label_groups_removed": 1, "conflicting_label_rows_removed": 3})


class SplitTests(unittest.TestCase):
    def test_chronological_boundary_keeps_equal_times_together(self):
        frame = pd.DataFrame({"row_id": [1, 2, 3, 4, 5], "commit_time": [30, 10, 20, 20, 40]}, index=[10, 20, 30, 40, 50])
        train, test, cutoff = chronological_split(frame, 0.6)
        self.assertEqual(cutoff, 20)
        self.assertEqual(train.tolist(), [20])
        self.assertEqual(test.tolist(), [30, 40, 10, 50])
        self.assertLess(frame.loc[train].commit_time.max(), frame.loc[test].commit_time.min())

    def test_chronological_split_rejects_unknown_or_identical_times(self):
        for times in ([100, 100, 100], [100, None, 300]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                chronological_split(pd.DataFrame({"row_id": [1, 2, 3], "commit_time": times}), 0.5)

    def test_overlap_is_rejected_for_each_identity(self):
        train, test = split_frames()
        validate_split(train, test, repository_split=True, temporal_split=True)
        for column in ("row_id", "sha", "diff_hash"):
            bad = test.copy()
            bad.loc[bad.index[0], column] = train.iloc[0][column]
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, column):
                validate_split(train, bad)
        train["sha"] = ""
        test["sha"] = ""
        validate_split(train, test)

    def test_repository_and_temporal_overlap_are_rejected(self):
        train, test = split_frames()
        bad = test.copy()
        bad["repository"] = train.iloc[0].repository
        with self.assertRaisesRegex(ValueError, "repositories"):
            validate_split(train, bad, repository_split=True)
        for time in (train.commit_time.max(), train.commit_time.min() - 1, None):
            bad = test.copy()
            bad["commit_time"] = time
            with self.subTest(time=time), self.assertRaisesRegex(ValueError, "temporal"):
                validate_split(train, bad, temporal_split=True)

    def test_empty_splits_and_insufficient_calibration_rows_are_rejected(self):
        train, test = split_frames()
        with self.assertRaisesRegex(ValueError, "both contain"):
            validate_split(train, test.iloc[:0])
        with self.assertRaisesRegex(ValueError, "three training rows"):
            validate_split(train.iloc[:5], test)


class EvaluationMetricsTests(unittest.TestCase):
    def test_unseen_label_errors_remain_in_all_metrics(self):
        result = metrics(np.array(["fix", "docs", "docs"]), np.array(["fix", "fix", "fix"]), [True, False, False])
        self.assertEqual(result["rows"], 3)
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertAlmostEqual(result["top3_accuracy"], 1 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 0.5 / len(LABELS))
        self.assertEqual(result["per_class"]["docs"]["support"], 2)
        self.assertEqual(result["per_class"]["docs"]["recall"], 0)
        matrix = np.asarray(result["confusion_matrix"])
        self.assertEqual(matrix.sum(), 3)
        self.assertEqual(matrix[LABELS.index("docs"), LABELS.index("fix")], 2)

    def test_evaluate_fold_counts_unseen_labels_as_incorrect_top3(self):
        train, test = split_frames()
        test.loc[test.index[-1], "label"] = "fix"
        frame = pd.concat([train, test])
        model = Mock()
        model.classes_ = np.array(["feat", "fix"])
        model.predict_proba.return_value = np.array([[0.25, 0.75], [0.5, 0.5]])
        with tempfile.TemporaryDirectory() as folder:
            with patch("evaluate_splits.build_model", return_value=model), contextlib.redirect_stdout(io.StringIO()):
                result, predictions = evaluate_fold(frame, train.index, test.index, "heldout", Path(folder), 42, grouped=True)
            self.assertTrue((Path(folder) / "heldout.json").exists())
            self.assertTrue((Path(folder) / "heldout_predictions.csv.gz").exists())
        pd.testing.assert_frame_equal(model.fit.call_args.args[0], train[FEATURE_COLUMNS])
        self.assertEqual(result["unseen_test_labels"], ["docs"])
        self.assertEqual(result["test_rows"], 2)
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["top3_accuracy"], 0.5)
        self.assertEqual(predictions.top3_correct.tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
