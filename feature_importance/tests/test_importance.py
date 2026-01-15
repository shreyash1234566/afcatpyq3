import unittest
import numpy as np
import pandas as pd
from feature_importance.importance_extractor import FeatureImportanceExtractor

class DummyModel:
    def __init__(self, importances):
        self.feature_importances_ = np.array(importances)

class TestFeatureImportanceExtractor(unittest.TestCase):
    def setUp(self):
        self.features = [f"f{i}" for i in range(5)]
        self.model = DummyModel([0.1, 0.2, 0.3, 0.25, 0.15])
        self.X_val = pd.DataFrame(np.random.rand(10, 5), columns=self.features)
        self.y_val = pd.Series(np.random.randint(0, 2, 10))
        self.extractor = FeatureImportanceExtractor(self.model, "randomforest", self.features, self.X_val, self.y_val)

    def test_native_importance(self):
        imp = self.extractor.extract_native_importance()
        self.assertEqual(len(imp), 5)
        self.assertAlmostEqual(sum(imp.values()), 1.0, places=1)

if __name__ == "__main__":
    unittest.main()
