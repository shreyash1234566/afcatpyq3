import sys, collections, numpy as np, math
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

Q_PATH = Path("e:/afcatpyq3/data/processed/Q.json")
try:
    sys.path.insert(0, "e:/afcatpyq3/scripts")
    from topic_normalization_map import TOPIC_NORMALIZATION
except ImportError:
    TOPIC_NORMALIZATION = {}

import json, re
def _year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

def load_counts():
    data = json.load(open(Q_PATH, encoding="utf-8"))
    cnt = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    syt = collections.defaultdict(collections.Counter)
    SEC_TARGET = {"Verbal Ability": 30, "General Awareness": 25, "Reasoning": 25, "Numerical Ability": 20}
    for q in data:
        y, s, t = _year(q.get("file_name")), q.get("section"), q.get("topic")
        if y and s in SEC_TARGET and t:
            t = TOPIC_NORMALIZATION.get(t, t)
            cnt[s][t][y] += 1
            syt[s][y] += 1
    years = sorted({y for s in syt for y in syt[s]})
    return cnt, syt, years

cnt, syt, years = load_counts()
SEC_TARGET = {"Verbal Ability": 30, "General Awareness": 25, "Reasoning": 25, "Numerical Ability": 20}
MIN_SEC_QS = 5
MIN_HISTORY = 3

# We will use the exact marginalised_mean we have right now as the "Base Prediction"
import sys
sys.path.insert(0, "e:/afcatpyq3/models")
from dirichlet_forecaster import marginalised_mean, tune_params, share_matrix, actual_vec

# Now let's implement the Hybrid boosting loop
def build_features_and_targets(cnt, syt, sec, topics, hist_years, target_year):
    """
    For a given target_year, we can use years < target_year to generate predictions
    for target_year. The features will be based on data < target_year.
    """
    # Tune params using years strictly before target_year
    g, lam = tune_params(cnt, syt, sec, topics, hist_years)
    
    # Base prediction for target_year
    S = share_matrix(cnt, syt, sec, hist_years, topics)
    base_preds = marginalised_mean(S, SEC_TARGET[sec], g, lam=lam)
    
    # Actuals for target_year
    actuals = actual_vec(cnt, syt, sec, target_year, topics)
    
    X = []
    y = []
    for j, topic in enumerate(topics):
        base_p = base_preds[j]
        # Lags
        S_raw = S[:, j] * SEC_TARGET[sec] # pseudo count scale
        lag1 = S_raw[-1] if len(S_raw) >= 1 else 0
        lag2 = S_raw[-2] if len(S_raw) >= 2 else 0
        lag3 = S_raw[-3] if len(S_raw) >= 3 else 0
        
        # Topic encoding (we can just use the base_pred as a strong feature)
        # We predict the RESIDUAL
        residual = actuals[j] - base_p
        
        # Slope
        slope = lag1 - lag2
        
        feats = [
            base_p,
            lag1,
            lag2,
            lag3,
            slope,
            float(np.mean(S_raw))
        ]
        X.append(feats)
        y.append(residual)
        
    return X, y, base_preds, actuals


print("--- Testing Hybrid SOTA Model ---")
for sec in ["Reasoning", "Verbal Ability"]:
    topics = sorted(cnt[sec])
    yrs = [y for y in sorted(syt[sec]) if syt[sec][y] >= MIN_SEC_QS]
    
    # We need to simulate walk forward.
    # For a given test year, we train the RF on all prior available "target_years".
    # e.g., to predict 2024, we generate features/targets for 2017..2023.
    
    test_results_base = []
    test_results_hybrid = []
    test_hits_base = []
    test_hits_hybrid = []
    
    # Start testing from MIN_HISTORY + 3 so we have at least 3 years of train data for the ML model
    START_TEST = MIN_HISTORY + 3
    for test_idx in range(START_TEST, len(yrs)):
        test_year = yrs[test_idx]
        
        # Build training set for ML model (using years prior to test_idx)
        X_train, y_train = [], []
        for train_target_idx in range(MIN_HISTORY, test_idx):
            train_target_year = yrs[train_target_idx]
            train_hist_years = yrs[:train_target_idx]
            
            X_t, y_t, _, _ = build_features_and_targets(cnt, syt, sec, topics, train_hist_years, train_target_year)
            X_train.extend(X_t)
            y_train.extend(y_t)
            
        # Train ML Model
        ml_model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        ml_model.fit(X_train, y_train)
        
        # Predict for Test Year
        test_hist_years = yrs[:test_idx]
        X_test, y_test, base_preds, actuals = build_features_and_targets(cnt, syt, sec, topics, test_hist_years, test_year)
        
        predicted_residuals = ml_model.predict(X_test)
        hybrid_preds = base_preds + (predicted_residuals * 0.5) # damped hybrid to prevent overfitting
        
        # Normalize hybrid
        hybrid_preds = np.clip(hybrid_preds, 0, SEC_TARGET[sec])
        if hybrid_preds.sum() > 0:
            hybrid_preds = hybrid_preds / hybrid_preds.sum() * SEC_TARGET[sec]
            
        # Evaluate
        mae_base = float(np.abs(base_preds - actuals).mean())
        mae_hybrid = float(np.abs(hybrid_preds - actuals).mean())
        
        test_results_base.append(mae_base)
        test_results_hybrid.append(mae_hybrid)
        
        k = len(topics)
        act_top = set(np.argsort(actuals)[-max(k // 2, 1):])
        
        base_top = set(np.argsort(base_preds)[-max(k // 2, 1):])
        hit_base = len(base_top & act_top) / max(len(act_top), 1)
        test_hits_base.append(hit_base)
        
        hybrid_top = set(np.argsort(hybrid_preds)[-max(k // 2, 1):])
        hit_hybrid = len(hybrid_top & act_top) / max(len(act_top), 1)
        test_hits_hybrid.append(hit_hybrid)

    print(f"{sec:20} BASE MAE={np.mean(test_results_base):.3f} (Hit {np.mean(test_hits_base)*100:.1f}%) | HYBRID MAE={np.mean(test_results_hybrid):.3f} (Hit {np.mean(test_hits_hybrid)*100:.1f}%)")
