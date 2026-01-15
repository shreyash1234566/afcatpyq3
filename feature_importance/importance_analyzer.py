# importance_analyzer.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_importance(importance_path):
    if importance_path.endswith('.json'):
        with open(importance_path, 'r', encoding='utf-8') as f:
            importance = json.load(f)
        df = pd.DataFrame(list(importance.items()), columns=["feature", "importance"])
    elif importance_path.endswith('.csv'):
        df = pd.read_csv(importance_path)
    else:
        raise ValueError("Unsupported file format for importance_path.")
    return df

def compare_importances(importance_files, model_names=None, top_n=20, output_path=None):
    """
    Compare feature importances from multiple models.
    Args:
        importance_files: List of paths to importance files (json/csv)
        model_names: List of model names (optional)
        top_n: Number of top features to plot
        output_path: Path to save comparison plot
    """
    dfs = []
    for i, imp_file in enumerate(importance_files):
        df = load_importance(imp_file)
        name = model_names[i] if model_names else f"model_{i+1}"
        df = df.rename(columns={"importance": name})
        dfs.append(df.set_index("feature"))
    merged = pd.concat(dfs, axis=1).fillna(0)
    # Get top_n features by mean importance
    merged["mean"] = merged.mean(axis=1)
    merged = merged.sort_values("mean", ascending=False).head(top_n)

    plt.figure(figsize=(12, min(0.5*top_n+2, 14)))
    merged.drop(columns=["mean"]).plot(kind="barh", ax=plt.gca())
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances Across Models")
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    plt.close()

# Example usage:
# compare_importances([
#   'outputs/rf_importance.json',
#   'outputs/xgb_importance.json',
#   'outputs/lgb_importance.json'],
#   model_names=['RandomForest', 'XGBoost', 'LightGBM'],
#   top_n=20,
#   output_path='outputs/plots/compare_top20.png')
import pandas as pd
from typing import Dict, List

def compare_importances(importances: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare feature importances from multiple models.
    Args:
        importances: Dict of model_name -> {feature: importance}
    Returns:
        DataFrame with features as rows and models as columns.
    """
    df = pd.DataFrame(importances)
    df["mean_importance"] = df.mean(axis=1)
    return df.sort_values("mean_importance", ascending=False)

# Example usage:
# importances = {"rf": rf_imp, "xgb": xgb_imp, "lgb": lgb_imp}
# df = compare_importances(importances)
# print(df.head(15))
