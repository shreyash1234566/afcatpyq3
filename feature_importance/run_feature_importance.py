# run_feature_importance.py
# Example integration script for feature importance extraction and visualization
import os
import json
import pandas as pd
import yaml
from importance_extractor import FeatureImportanceExtractor
from importance_visualizer import plot_top_features
from importance_analyzer import compare_importances

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    config = load_config(os.path.join(os.path.dirname(__file__), 'config', 'importance_config.yaml'))
    # Load feature names
    with open(config['features_path'], 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    # Load validation data
    X_val = pd.read_csv(config['X_val_path'])
    y_val = pd.read_csv(config['y_val_path']).squeeze()

    importance_files = []
    model_names = []
    for model_cfg in config['models']:
        name = model_cfg['name']
        model_path = model_cfg['path']
        model_type = model_cfg['type']
        # Load model
        model = FeatureImportanceExtractor.load_model(model_path)
        extractor = FeatureImportanceExtractor(model, model_type, feature_names, X_val, y_val)
        # Native importance
        native = extractor.extract_native_importance()
        out_json = os.path.join(config['output_dir'], f'{name}_importance.json')
        out_csv = os.path.join(config['output_dir'], f'{name}_importance.csv')
        extractor.save_importance(native, out_json, out_csv)
        importance_files.append(out_json)
        model_names.append(name)
        # Plot top features
        plot_path = os.path.join(config['plot_dir'], f'{name}_top{config["top_n"]}.png')
        plot_top_features(out_json, top_n=config['top_n'], output_path=plot_path, title=f"{name} Top {config['top_n']} Features")

    # Compare importances across models
    compare_plot = os.path.join(config['plot_dir'], f'compare_top{config["top_n"]}.png')
    compare_importances(importance_files, model_names=model_names, top_n=config['top_n'], output_path=compare_plot)

if __name__ == "__main__":
    main()