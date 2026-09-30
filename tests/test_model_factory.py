import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from export_model import export_pipeline
from train_enhanced import BEHAVIOR_FEATURES, FEATURE_COLUMNS, build_model, prepare_features
from verify_export import build_feature_vector, forward_pass


class ModelFactoryTests(unittest.TestCase):
    def test_factory_trains_and_exports_a_multiclass_model(self):
        rows, labels = [], []
        for label, path, body in [
            ("docs", "docs/guide.md", "describe installation usage"),
            ("fix", "src/parser.py", "return corrected result"),
            ("test", "tests/parser_test.py", "assert expected result"),
        ]:
            for index in range(6):
                rows.append({
                    "diff_text": f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -0,0 +1 @@\n+{body} {index}\n",
                    "files_changed": 1,
                    "additions": 1,
                })
                labels.append(label)
        samples = prepare_features(pd.DataFrame(rows))
        model = build_model()
        self.assertIsInstance(model, Pipeline)
        model.fit(samples[FEATURE_COLUMNS], labels)

        probabilities = model.predict_proba(samples[FEATURE_COLUMNS])
        self.assertEqual(probabilities.shape, (18, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        payload = export_pipeline(model)
        self.assertEqual(payload["classes"], list(model.classes_))
        width = model.named_steps["preprocessor"].transform(samples[FEATURE_COLUMNS]).shape[1]
        behavior = len(BEHAVIOR_FEATURES)
        self.assertEqual(payload["feature_layout"]["numeric"][1], width - behavior)
        self.assertEqual(payload["feature_layout"]["behavior"], [width - behavior, width])
        self.assertEqual(payload["behavior"]["features"], list(BEHAVIOR_FEATURES))
        self.assertEqual(len(payload["calibrated_folds"]), 3)
        for fold in payload["calibrated_folds"]:
            self.assertEqual(np.asarray(fold["coef"]).shape, (3, width))
        exported = [
            forward_pass(build_feature_vector(row, payload), payload)
            for _, row in samples.iterrows()
        ]
        np.testing.assert_allclose(exported, probabilities, atol=1e-12)
        self.assertFalse(hasattr(build_model().named_steps["clf"], "classes_"))


if __name__ == "__main__":
    unittest.main()
