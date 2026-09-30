import unittest
from collections import Counter

import numpy as np

from train_subject import Exported, export, fit, subject_of


class SubjectTests(unittest.TestCase):
    def test_subject_is_read_without_prefix_or_pull_request(self):
        self.assertEqual(subject_of("fix(parser)!: handle CRLF (#123)\n\nbody"), "handle CRLF")
        self.assertEqual(subject_of("perf: speed up reading (#1, #22)"), "speed up reading")
        self.assertEqual(subject_of('Revert "feat: add x"'), 'Revert "feat: add x"')
        self.assertEqual(subject_of(""), "")
        self.assertEqual(subject_of(None), "")

    def test_export_matches_scikit_learn(self):
        subjects = ["speed up rendering", "fix crash on start", "add dark mode", "rename foo to bar",
                    "make startup faster", "fix typo in parser", "add export button", "move helpers out"] * 10
        labels = np.array(["perf", "fix", "feat", "refactor"] * 20)
        vec, clf = fit(subjects, labels, C=1.0, min_df=1)
        payload = export(vec, clf, Counter(labels), weight=0.5, prior_power=0.1)
        model = Exported(payload)
        self.assertEqual(payload["classes"], sorted(set(labels)))
        self.assertEqual(payload["ngram_range"], [1, 2])
        self.assertIn("speed up", payload["vocabulary"])
        ours = np.array([model.predict_proba(s) for s in subjects[:8]])
        theirs = clf.predict_proba(vec.transform(subjects[:8]))
        np.testing.assert_allclose(ours, theirs, atol=1e-5)

        diff = np.full(4, 0.25)
        combined = model.combine(diff, "speed up rendering")
        self.assertAlmostEqual(combined.sum(), 1.0)
        self.assertEqual(payload["classes"][int(np.argmax(combined))], "perf")


if __name__ == "__main__":
    unittest.main()
