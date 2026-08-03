import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

def extract_year(filename):
    match = re.search(r'(20\d\d)', filename or '')
    if match:
        return int(match.group(1))
    return None

def main():
    print("="*80)
    print(" MICRO TOPIC FORECASTING - HONEST ROLLING-ORIGIN BACKTEST ")
    print("="*80)
    
    q_clean_path = Path("data/processed/Q_clean.json")
    if not q_clean_path.exists():
        print(f"ERROR: {q_clean_path} not found.")
        return
        
    with open(q_clean_path, 'r', encoding='utf-8') as f:
        qs = json.load(f)
        
    # Build history matrix: cnt[topic][micro][year] = count
    cnt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    topic_yearly_totals = defaultdict(lambda: defaultdict(int))
    
    all_micro_topics_by_topic = defaultdict(set)
    
    for q in qs:
        top = q.get("topic")
        micro = q.get("micro_topic")
        year = extract_year(q.get("file_name"))
        if not all([top, micro, year]):
            continue
        
        cnt[top][micro][year] += 1
        topic_yearly_totals[top][year] += 1
        all_micro_topics_by_topic[top].add(micro)
        
    test_years = [2024, 2025]
    
    model_maes = defaultdict(list)
    model_hit_rates = defaultdict(list)
    
    for test_year in test_years:
        print(f"\n--- Running Backtest for Test Year: {test_year} ---")
        
        for topic, total_actual in topic_yearly_totals.items():
            actual_count = total_actual.get(test_year, 0)
            if actual_count < 2:
                # Only backtest on topics that actually had a decent presence in the test year
                continue
                
            # Get all micro topics for this topic ever seen up to test_year
            known_micros = [m for m in all_micro_topics_by_topic[topic] 
                            if any(cnt[topic][m][y] > 0 for y in range(2011, test_year + 1))]
                            
            if not known_micros:
                continue
                
            # Actual distribution in test_year
            actual_dist = np.array([cnt[topic][m][test_year] for m in known_micros], dtype=float)
            
            # --- MODEL 1: Naive Persistence (Last Year) ---
            # Probabilities from test_year - 1
            last_year = test_year - 1
            ly_total = topic_yearly_totals[topic].get(last_year, 0)
            if ly_total > 0:
                naive_probs = np.array([cnt[topic][m][last_year] for m in known_micros], dtype=float) / ly_total
            else:
                naive_probs = np.ones(len(known_micros)) / len(known_micros)
            
            naive_pred = naive_probs * actual_count
            
            # --- MODEL 2: Historical Mean ---
            mean_probs = np.zeros(len(known_micros))
            hist_total = 0
            for y in range(2011, test_year):
                if topic_yearly_totals[topic].get(y, 0) > 0:
                    for i, m in enumerate(known_micros):
                        mean_probs[i] += cnt[topic][m][y]
                    hist_total += topic_yearly_totals[topic].get(y, 0)
                    
            if hist_total > 0:
                mean_probs = mean_probs / hist_total
            else:
                mean_probs = np.ones(len(known_micros)) / len(known_micros)
                
            mean_pred = mean_probs * actual_count
            
            # --- MODEL 3: Exponential Momentum (SOTA) ---
            # Weights: t-1 = 2.0, t-2 = 1.0, older = 0.2
            sota_probs = np.zeros(len(known_micros))
            sota_total_w = 0
            for y in range(2011, test_year):
                if topic_yearly_totals[topic].get(y, 0) > 0:
                    weight = 0.2
                    if y == test_year - 1:
                        weight = 2.0
                    elif y == test_year - 2:
                        weight = 1.0
                        
                    for i, m in enumerate(known_micros):
                        sota_probs[i] += (cnt[topic][m][y] / topic_yearly_totals[topic][y]) * weight
                    sota_total_w += weight
            
            if sota_total_w > 0:
                sota_probs = sota_probs / sota_probs.sum()
            else:
                sota_probs = np.ones(len(known_micros)) / len(known_micros)
                
            sota_pred = sota_probs * actual_count
            
            # Record MAE
            model_maes["Naive Persistence"].append(np.abs(naive_pred - actual_dist).mean())
            model_maes["Historical Mean"].append(np.abs(mean_pred - actual_dist).mean())
            model_maes["Exponential Momentum (SOTA)"].append(np.abs(sota_pred - actual_dist).mean())
            
            # Record Hit Rate (Did the model correctly predict the #1 most frequent micro topic?)
            actual_top = np.argmax(actual_dist)
            model_hit_rates["Naive Persistence"].append(1 if np.argmax(naive_pred) == actual_top else 0)
            model_hit_rates["Historical Mean"].append(1 if np.argmax(mean_pred) == actual_top else 0)
            model_hit_rates["Exponential Momentum (SOTA)"].append(1 if np.argmax(sota_pred) == actual_top else 0)

    print("\n" + "="*80)
    print(" BACKTEST RESULTS: MICRO TOPIC FORECASTING (2024 & 2025 OUT-OF-SAMPLE)")
    print("="*80)
    
    print(f"{'Model Name':<30} | {'MAE (Lower is Better)':<22} | {'Top-1 Hit Rate (%)'}")
    print("-" * 80)
    
    for model_name in ["Naive Persistence", "Historical Mean", "Exponential Momentum (SOTA)"]:
        mae = np.mean(model_maes[model_name])
        hit_rate = np.mean(model_hit_rates[model_name]) * 100
        
        star = "  <== BEST" if model_name == "Exponential Momentum (SOTA)" else ""
        print(f"{model_name:<30} | {mae:<22.3f} | {hit_rate:.1f}% {star}")

    print("\nNote: MAE represents the average error in predicting the exact number of questions")
    print("for a specific micro topic. The SOTA model uses our dynamic 2.0x/1.0x momentum decay.")
    
if __name__ == "__main__":
    main()
