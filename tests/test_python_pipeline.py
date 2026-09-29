import contextlib
import copy
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from dedupe import banded_near_dup, expand_inputs, iter_rows, main as dedupe_main
from import_external import count_diff_lines, normalize_commitbench, rebuild_diff_from_mods
from miner import mine_repo, mine_repo_stream, parse_date, parse_stats, process_commit
from export_model import export_pipeline
from train_enhanced import (
    DiffSimilarityExtractor, FileExtensionExtractor, PathTokenExtractor,
    FEATURE_COLUMNS, load_data, prepare_features,
)
from utils.find_dups import load_json_any
from verify_export import build_feature_vector, build_tfidf_vec, forward_pass, load_samples, sigmoid, main as verify_main


class DatasetTests(unittest.TestCase):
    def test_read_supported_formats_and_skip_non_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            array = root / "array.json"
            array.write_text('\ufeff \n [{"id": 1}, null]', encoding="utf-8")
            wrapped = root / "wrapped.json"
            wrapped.write_text('{"data": [{"id": 2}]}', encoding="utf-8")
            lines = root / "lines.jsonl"
            lines.write_text('{"id": 3}\n[]\nnull\nbroken\n', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                rows = list(iter_rows([array, wrapped, lines]))
            self.assertEqual(rows, [{"id": 1}, {"id": 2}, {"id": 3}])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(len(load_data(root)), 3)

    def test_expand_globs_without_repeating_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.jsonl"
            path.touch()
            self.assertEqual(expand_inputs([str(Path(folder) / "*.jsonl"), str(path)]), [path])

    def test_reservoir_sampling_is_bounded_and_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text("".join(json.dumps({"diff_text": str(i)}) + "\n" for i in range(100)), encoding="utf-8")
            sampled = load_samples(path, 7, seed=4)
            self.assertEqual(len(sampled), 7)
            self.assertEqual(len(set(sampled.diff_text)), 7)
            pd.testing.assert_frame_equal(sampled, load_samples(path, 7, seed=4))
            self.assertTrue((sampled.files_changed == 0).all())

    def test_empty_sampling_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "no records"):
                load_samples(folder, 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            load_samples("unused", 0)

    def test_missing_and_invalid_numeric_fields(self):
        result = prepare_features(pd.DataFrame({"diff_text": [None, 42], "deletions": [-1, np.inf], "additions": ["bad", 3]}))
        self.assertEqual(result.diff_text.tolist(), ["", "42.0"])
        self.assertEqual(result.files_changed.tolist(), [0, 0])
        self.assertEqual(result.add_del_ratio.tolist(), [0, 3])
        self.assertTrue(np.isfinite(result.select_dtypes("number").to_numpy()).all())

    def test_duplicate_report_skips_non_object_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text('null\n{"sha":"a"}\n[]\n{"sha":"a"}\n', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                rows, lines = load_json_any(path)
            self.assertEqual(lines, [2, 4])
            self.assertEqual(len(rows), 2)


class DedupeTests(unittest.TestCase):
    def test_hamming_threshold_four_does_not_miss_distributed_bits(self):
        signature = sum(1 << bit for bit in (0, 16, 32, 48))
        self.assertEqual(banded_near_dup([0, signature], 4), {1})
        self.assertEqual(banded_near_dup([0, signature], 3), set())

    def test_hamming_bounds_and_keep_first(self):
        self.assertEqual(banded_near_dup([0, 1, 3], 1), {1})
        self.assertEqual(banded_near_dup([0, (1 << 64) - 1], 64), {1})
        for value in (-1, 65):
            with self.assertRaises(ValueError):
                banded_near_dup([0], value)

    def test_empty_record_does_not_hide_valid_commit_and_owner_is_part_of_key(self):
        rows = [
            {"owner": "a", "repo": "core", "sha": "1", "diff_text": ""},
            {"owner": "a", "repo": "core", "sha": "1", "diff_text": "+one"},
            {"owner": "b", "repo": "core", "sha": "1", "diff_text": "+two"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            src, out = Path(folder) / "in.json", Path(folder) / "out.jsonl"
            src.write_text(json.dumps(rows), encoding="utf-8")
            with patch("sys.argv", ["dedupe", "--input", str(src), "--output", str(out)]), contextlib.redirect_stdout(io.StringIO()):
                dedupe_main()
            self.assertEqual(list(iter_rows([out])), rows[1:])


class DiffTests(unittest.TestCase):
    def test_miner_ignores_git_diff_display_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=str(root / "empty-config"))
            env.pop("GIT_INDEX_FILE", None)
            def git(*args):
                subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)
            git("init", "--quiet")
            (root / "docs").mkdir()
            (root / "docs/guide.md").write_text("text\n", encoding="utf-8")
            git("add", "docs/guide.md")
            git("-c", "user.name=Tests", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "docs: guide")
            for key, value in (("diff.noprefix", "true"), ("diff.relative", "true"), ("diff.context", "0"), ("diff.outputIndicatorNew", ">")):
                git("config", key, value)
            with patch.dict(os.environ, env, clear=True):
                row = process_commit("HEAD", "0", "docs: guide", "docs", root / "docs", 20000)
            self.assertIn("diff --git a/docs/guide.md b/docs/guide.md", row["diff_text"])
            self.assertIn("+++ b/docs/guide.md", row["diff_text"])
            self.assertIn("\n+text", row["diff_text"])
            self.assertNotIn("docs: guide", row["diff_text"])
            self.assertEqual((row["files_changed"], row["additions"]), (1, 1))

    def test_numstat_handles_rename_binary_and_newline_in_path(self):
        stats = "2\t1\tsrc/a.py\0-\t-\timage.png\0" "0\t0\t\0old.md\0new\nname.txt\0"
        self.assertEqual(parse_stats(stats), (3, 2, 1, ".py,.png,.txt"))

    def test_diff_counts_code_that_begins_with_header_characters(self):
        diff = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n---counter\n+++counter\n"
        self.assertEqual(count_diff_lines(diff), (1, 1))

    def test_commitbench_counts_deleted_and_binary_files(self):
        diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\ndiff --git a/b.rs b/b.rs\n--- a/b.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-z\ndiff --git a/c.png b/c.png\nBinary files differ\n"
        row = normalize_commitbench({"message": "fix: all files", "diff": diff})
        self.assertEqual(row["files_changed"], 3)
        self.assertEqual(row["top_exts"], ".py,.rs,.png")

    def test_reconstructed_add_delete_headers_use_dev_null(self):
        diff, paths = rebuild_diff_from_mods([
            {"old_path": None, "new_path": "new.py", "diff": "@@ -0,0 +1 @@\n+pass\n"},
            {"old_path": "old.rs", "new_path": None, "diff": "@@ -1 +0,0 @@\n-old\n"},
        ])
        self.assertIn("diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py", diff)
        self.assertIn("diff --git a/old.rs b/old.rs\n--- a/old.rs\n+++ /dev/null", diff)
        self.assertEqual(paths, ["new.py", "old.rs"])

    def test_miner_preserves_log_order_and_searches_past_nonconventional_commits(self):
        log = "\n".join([f"ordinary{i}|~|0|~|message" for i in range(5)] + ["older|~|0|~|fix: issue", "oldest|~|0|~|feat: feature"])
        def fake_git(cmd, cwd):
            return ".git" if cmd[1] == "rev-parse" else log
        def fake_commit(sha, *args):
            return {"sha": sha}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "out.jsonl"
            with patch("miner.run_git", side_effect=fake_git), patch("miner.process_commit", side_effect=fake_commit), contextlib.redirect_stdout(io.StringIO()):
                mine_repo(folder, out, limit=1, max_workers=2)
            self.assertEqual(list(iter_rows([out])), [{"sha": "older"}])


    def test_streaming_miner_matches_per_commit_miner(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=str(root / "empty-config"))
            env.pop("GIT_INDEX_FILE", None)
            def git(*args, date="2026-01-01T00:00:00Z", author="Tests <test@example.invalid>"):
                name, email = author[:-1].split(" <")
                run_env = dict(env, GIT_AUTHOR_NAME=name, GIT_AUTHOR_EMAIL=email, GIT_COMMITTER_NAME=name,
                               GIT_COMMITTER_EMAIL=email, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
                subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=root, env=run_env, check=True, capture_output=True)
            git("init", "--quiet")
            (root / "src").mkdir()
            (root / "src/a.py").write_text("x = 1\n", encoding="utf-8")
            (root / "we ird.md").write_text("text\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "--quiet", "-m", "feat(core): add parser", date="2026-01-01T00:00:00Z")
            (root / "notes.txt").write_text("note\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "--quiet", "-m", "update notes", date="2026-02-01T00:00:00Z")
            git("mv", "src/a.py", "src/b.py")
            (root / "src/b.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
            (root / "img.png").write_bytes(b"\x89PNG\x00\x01")
            git("add", "-A")
            git("commit", "--quiet", "-m", "fix: rename and add an image", date="2026-03-01T00:00:00Z",
                author="dependabot[bot] <bot@example.invalid>")
            (root / "big.txt").write_text("".join(f"line {i}\n" for i in range(100)), encoding="utf-8")
            git("add", "-A")
            git("commit", "--quiet", "-m", "chore!: add a large file", date="2026-04-01T00:00:00Z")
            for key, value in (("diff.noprefix", "true"), ("diff.mnemonicPrefix", "true"), ("diff.context", "0")):
                git("config", key, value)
            with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()):
                mine_repo(root, root / "threads.jsonl", max_workers=2, max_diff_chars=300)
                mine_repo_stream(root, root / "stream.jsonl", max_diff_chars=300)
                mine_repo_stream(root, root / "range.jsonl", since=parse_date("2026-02-01"), until=parse_date("2026-04-01"))
            threads = {row["sha"]: row for row in iter_rows([root / "threads.jsonl"])}
            stream = {row["sha"]: row for row in iter_rows([root / "stream.jsonl"])}
            self.assertEqual(stream.keys(), threads.keys())
            self.assertEqual(sorted(row["label"] for row in stream.values()), ["chore", "feat", "fix"])
            for sha, row in threads.items():
                self.assertEqual({key: stream[sha][key] for key in row}, row)
            by_label = {row["label"]: row for row in stream.values()}
            self.assertEqual(len(by_label["chore"]["diff_text"]), 300)
            self.assertEqual(by_label["fix"]["top_exts"], ".png,.py")
            self.assertEqual(by_label["fix"]["committed_at"], "2026-03-01T00:00:00Z")
            self.assertEqual([row["is_bot"] for row in (by_label["feat"], by_label["fix"])], [False, True])
            self.assertEqual([row["label"] for row in iter_rows([root / "range.jsonl"])], ["fix"])

    def test_streaming_miner_handles_a_repository_without_commits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                mine_repo_stream(root, root / "out.jsonl")
            self.assertIn("No commits found", stdout.getvalue())
            self.assertFalse((root / "out.jsonl").exists())

class ExportVerificationTests(unittest.TestCase):
    def test_tfidf_norms_and_capture_groups_match_sklearn(self):
        for norm in (None, "l1", "l2"):
            vect = TfidfVectorizer(norm=norm, token_pattern=r"(?u)\b([a-z]+)\d*\b", sublinear_tf=True)
            vect.fit(["hello1 hello2 world", "other world"])
            spec = {"vocabulary": vect.vocabulary_, "idf": vect.idf_, "token_pattern": vect.token_pattern, "lowercase": True, "norm": norm, "sublinear_tf": True}
            text = "HELLO1 hello2 world world missing"
            np.testing.assert_allclose(build_tfidf_vec(text, spec), vect.transform([text]).toarray()[0])

    def test_sigmoid_extremes(self):
        self.assertEqual(sigmoid(1000), 0)
        self.assertEqual(sigmoid(-1000), 1)
        self.assertEqual(sigmoid(0), 0.5)

    def test_binary_classifier_probabilities(self):
        payload = {"classes": ["feat", "fix"], "calibrated_folds": [{"coef": [[2]], "intercept": [1], "sigmoid_a": [-1], "sigmoid_b": [0]}]}
        positive = 1 / (1 + math.exp(-3))
        np.testing.assert_allclose(forward_pass(np.array([1]), payload), [1 - positive, positive])

    def test_underflowing_multiclass_calibration_is_uniform(self):
        payload = {"classes": ["a", "b", "c"], "calibrated_folds": [{"coef": [[0], [0], [0]], "intercept": [0, 0, 0], "sigmoid_a": [1, 1, 1], "sigmoid_b": [1000, 1000, 1000]}]}
        np.testing.assert_allclose(forward_pass(np.array([1]), payload), [1 / 3] * 3)


class ExportPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = []
        labels = []
        for i in range(12):
            label, path, body = ("docs", "docs/guide.md", "describe usage") if i % 2 else ("fix", "src/parser.py", "return result")
            rows.append({"diff_text": f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -0,0 +1 @@\n+{body} {i}\n", "files_changed": 1, "additions": 1})
            labels.append(label)
        cls.samples = prepare_features(pd.DataFrame(rows))
        pre = ColumnTransformer([
            ("diff_tfidf", TfidfVectorizer(), "diff_text"),
            ("path_bow", Pipeline([("extractor", PathTokenExtractor()), ("vect", CountVectorizer(binary=True))]), "diff_text"),
            ("ext_bow", Pipeline([("extractor", FileExtensionExtractor()), ("vect", CountVectorizer(binary=True))]), "diff_text"),
            ("diff_sim", DiffSimilarityExtractor(), "diff_text"),
            ("numeric", StandardScaler(), FEATURE_COLUMNS[1:]),
        ])
        cls.model = Pipeline([("preprocessor", pre), ("clf", CalibratedClassifierCV(LinearSVC(random_state=42), cv=3))])
        cls.model.fit(cls.samples[FEATURE_COLUMNS], labels)

    def test_binary_pipeline_export_matches_sklearn(self):
        payload = export_pipeline(self.model)
        actual = [forward_pass(build_feature_vector(row, payload), payload) for _, row in self.samples.iterrows()]
        np.testing.assert_allclose(actual, self.model.predict_proba(self.samples[FEATURE_COLUMNS]), atol=1e-12)

    def test_extra_pipeline_steps_are_rejected(self):
        model = copy.deepcopy(self.model)
        model.steps.insert(0, ("extra", FunctionTransformer()))
        with self.assertRaisesRegex(ValueError, "preprocessor"):
            export_pipeline(model)
        model = copy.deepcopy(self.model)
        model.named_steps["preprocessor"].named_transformers_["path_bow"].steps.append(("extra", FunctionTransformer()))
        with self.assertRaisesRegex(ValueError, "extractor and vect"):
            export_pipeline(model)

    def test_custom_extractor_and_wrong_fold_width_are_rejected(self):
        model = copy.deepcopy(self.model)
        model.named_steps["preprocessor"].named_transformers_["path_bow"].steps[0] = ("extractor", FunctionTransformer())
        with self.assertRaisesRegex(ValueError, "unsupported path"):
            export_pipeline(model)
        model = copy.deepcopy(self.model)
        model.named_steps["clf"].calibrated_classifiers_[0].estimator.coef_ = np.zeros((1, 1))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            export_pipeline(model)

    def test_prior_correction_shifts_calibration_and_still_exports(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
        import tune_prior
        model = copy.deepcopy(self.model)
        before = model.predict_proba(self.samples[FEATURE_COLUMNS])
        tune_prior.apply(model, {"docs": 6, "fix": 6}, 1.0)
        np.testing.assert_allclose(model.predict_proba(self.samples[FEATURE_COLUMNS]), before)
        tune_prior.apply(model, {"docs": 2, "fix": 8}, 1.0)
        after = model.predict_proba(self.samples[FEATURE_COLUMNS])
        self.assertTrue((after[:, 0] > before[:, 0]).all())
        payload = export_pipeline(model)
        actual = [forward_pass(build_feature_vector(row, payload), payload) for _, row in self.samples.iterrows()]
        np.testing.assert_allclose(actual, after, atol=1e-12)

    def test_verification_rejects_stale_model_before_sampling(self):
        payload = export_pipeline(self.model)
        payload["scaler"]["mean"][0] += 1
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stale.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            stderr = io.StringIO()
            with patch("sys.argv", ["verify_export", "--json", str(path)]), patch("verify_export.joblib.load", return_value=self.model), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as error:
                    verify_main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("JSON parameters differ", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
