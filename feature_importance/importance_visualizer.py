# importance_visualizer.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_features(importance_path, top_n=20, output_path=None, title=None):
    """
    Plot top N features from a JSON or CSV importance file.
    """
    if importance_path.endswith('.json'):
        with open(importance_path, 'r', encoding='utf-8') as f:
            importance = json.load(f)
        df = pd.DataFrame(list(importance.items()), columns=["feature", "importance"])
    elif importance_path.endswith('.csv'):
        df = pd.read_csv(importance_path)
    else:
        raise ValueError("Unsupported file format for importance_path.")

    df = df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, min(0.5*top_n+2, 12)))
    sns.barplot(x="importance", y="feature", data=df, palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title or f"Top {top_n} Features")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    plt.close()

# Example usage:
# plot_top_features('outputs/importance_scores.json', top_n=20, output_path='outputs/plots/top20_features.png')
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def plot_top_features(importance: Dict[str, float], top_n: int = 15, title: str = "Feature Importance", out_path: str = None):
    df = pd.DataFrame(list(importance.items()), columns=["feature", "importance"])
    df = df.sort_values("importance", ascending=False).head(top_n)
    plt.figure(figsize=(8, max(5, top_n // 2)))
    sns.barplot(x="importance", y="feature", data=df, palette="viridis")
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()

# Example usage:
# with open("outputs/importance_scores.json") as f:
#     imp = json.load(f)
# plot_top_features(imp, top_n=15, title="Random Forest Feature Importance", out_path="outputs/plots/rf_importance.png")
