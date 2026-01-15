import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.inspection import permutation_importance

try:
    import shap
except ImportError:
    shap = None

class FeatureImportanceExtractor:
    def __init__(self, model, model_type: str, feature_names: List[str], X_val: pd.DataFrame = None, y_val: pd.Series = None):
        self.model = model
        self.model_type = model_type.lower()
        self.feature_names = feature_names
        self.X_val = X_val
        self.y_val = y_val

    def extract_native_importance(self) -> Dict[str, float]:
        if self.model_type in ["randomforest", "random_forest", "rf"]:
            importances = self.model.feature_importances_
        elif self.model_type in ["xgboost", "xgb"]:
            importances = self.model.feature_importances_
        elif self.model_type in ["lightgbm", "lgbm", "lgb"]:
            importances = self.model.feature_importances_
        else:
            raise ValueError(f"Model type {self.model_type} not supported for native importance.")
        return dict(zip(self.feature_names, importances))

    def extract_permutation_importance(self, n_repeats=5, random_state=42) -> Dict[str, float]:
        if self.X_val is None or self.y_val is None:
            raise ValueError("Validation data required for permutation importance.")
        result = permutation_importance(self.model, self.X_val, self.y_val, n_repeats=n_repeats, random_state=random_state)
        return dict(zip(self.feature_names, result.importances_mean))

    def extract_shap_importance(self, max_samples=1000) -> Dict[str, float]:
        if shap is None:
            raise ImportError("shap library is not installed.")
        if self.X_val is None:
            raise ValueError("Validation data required for SHAP importance.")
        explainer = shap.TreeExplainer(self.model)
        X_sample = self.X_val.iloc[:max_samples]
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            # For multiclass, sum absolute SHAP values across classes
            shap_vals = np.sum([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            shap_vals = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_names, shap_vals))

    def save_importance(self, importance: Dict[str, float], out_json: str, out_csv: str):
        # Save as JSON
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(importance, f, indent=2)
        # Save as CSV
        pd.DataFrame(list(importance.items()), columns=["feature", "importance"]).to_csv(out_csv, index=False)

    @staticmethod
    def load_model(model_path: str):
        with open(model_path, "rb") as f:
            return pickle.load(f)

# Example usage:
# extractor = FeatureImportanceExtractor(model, "randomforest", feature_names, X_val, y_val)
# native = extractor.extract_native_importance()
# perm = extractor.extract_permutation_importance()
# if shap: shap_imp = extractor.extract_shap_importance()
# extractor.save_importance(native, "outputs/importance_scores.json", "outputs/importance_scores.csv")
