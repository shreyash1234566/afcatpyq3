"""
accuracy_report.py
Honest walk-forward (rolling-origin) accuracy for the DM topic forecaster + Micro Topic Allocator.

For each section, for each year t (with >= MIN_HISTORY prior usable years):
  - train ONLY on years < t
  - predict topic distribution for year t
  - mathematically distribute down into micro-topics
  - compare against what ACTUALLY appeared in year t

Metrics:
  MAE      = mean |predicted_count - actual_count| per topic (lower better)
  Hit%     = of the topics that were actually in the top-half by count,
             how many did we also rank in our predicted top-half (higher better)
  MASE     = MAE / MAE(naive last-year) — <1 means we beat "assume same as last year"
"""
import sys, collections, json, re, math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
from dirichlet_forecaster import (load_counts, section_years, share_matrix,
    marginalised_mean, tune_params, actual_vec, SEC_TARGET, MIN_HISTORY)

Q_CLEAN_PATH = ROOT / "data" / "processed" / "Q_clean.json"

def _get_year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

def main():
    # Load Q_clean.json for micro topics
    q_data = []
    if Q_CLEAN_PATH.exists():
        with open(Q_CLEAN_PATH, 'r', encoding='utf-8') as f:
            q_data = json.load(f)

    # Aggregate micro counts: micro_cnt[topic][micro_topic][year] = count
    micro_cnt = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
    micro_yearly_totals = collections.defaultdict(lambda: collections.defaultdict(int))
    all_micros_by_topic = collections.defaultdict(set)

    for q in q_data:
        top = q.get("topic")
        micro = q.get("micro_topic")
        year = _get_year(q.get("file_name"))
        if top and micro and year:
            micro_cnt[top][micro][year] += 1
            micro_yearly_totals[top][year] += 1
            all_micros_by_topic[top].add(micro)

    cnt, syt, years = load_counts()

    print("WALK-FORWARD ACCURACY REPORT — AFCAT Forecast (Macro DM + Micro Momentum)")
    print("=" * 110)
    print(f"{'Year':>6} {'Section':20} | {'Macro MAE':>9} {'Hit%':>5} | {'Micro MAE':>9} {'Hit%':>5} | {'naiveMacro':>10}")
    print("-" * 110)

    rows = []
    for sec in sorted(SEC_TARGET):
        topics = sorted(cnt[sec])
        yrs = section_years(syt, sec)
        for i in range(MIN_HISTORY, len(yrs)):
            hist, ty = yrs[:i], yrs[i]
            
            # 1. Macro Prediction (Dirichlet)
            g, lam = tune_params(cnt, syt, sec, topics, hist)
            S = share_matrix(cnt, syt, sec, hist, topics)
            macro_pred = marginalised_mean(S, SEC_TARGET[sec], g, lam=lam)
            macro_act = actual_vec(cnt, syt, sec, ty, topics)
            macro_naive = actual_vec(cnt, syt, sec, hist[-1], topics)
            
            macro_mae = float(np.abs(macro_pred - macro_act).mean())
            nmae = float(np.abs(macro_naive - macro_act).mean())
            k = len(topics)
            pred_top = set(np.argsort(macro_pred)[-max(k // 2, 1):])
            act_top = set(np.argsort(macro_act)[-max(k // 2, 1):])
            macro_hit = len(pred_top & act_top) / max(len(act_top), 1)
            
            # 2. Micro Prediction
            micro_maes_ty = []
            micro_hits_ty = []
            
            for j, top in enumerate(topics):
                actual_macro_count = macro_act[j]
                if actual_macro_count < 1: 
                    continue # Only evaluate micro accuracy on topics that actually appeared
                    
                pred_macro_count = macro_pred[j]
                
                known_micros = [m for m in all_micros_by_topic[top] 
                                if any(micro_cnt[top][m][y] > 0 for y in range(2011, ty))]
                if not known_micros:
                    continue
                    
                # Actual micro distribution in year `ty`
                actual_micro_dist = np.array([micro_cnt[top][m][ty] for m in known_micros], dtype=float)
                
                # Predict Micro Probabilities (Exponential Momentum)
                sota_probs = np.zeros(len(known_micros))
                sota_total_w = 0
                for y in range(2011, ty):
                    if micro_yearly_totals[top].get(y, 0) > 0:
                        weight = 0.2
                        if y == ty - 1: weight = 2.0
                        elif y == ty - 2: weight = 1.0
                        for m_idx, m in enumerate(known_micros):
                            sota_probs[m_idx] += (micro_cnt[top][m][y] / micro_yearly_totals[top][y]) * weight
                        sota_total_w += weight
                        
                if sota_total_w > 0:
                    sota_probs = sota_probs / sota_probs.sum()
                else:
                    sota_probs = np.ones(len(known_micros)) / len(known_micros)
                    
                # The exact mathematical pipeline: Macro Prediction * Micro Probabilities
                micro_pred_dist = sota_probs * pred_macro_count
                
                # Record MAE for this topic's micro-distribution
                micro_maes_ty.append(np.abs(micro_pred_dist - actual_micro_dist).mean())
                
                # Record Hit Rate
                actual_top_idx = np.argmax(actual_micro_dist)
                micro_hits_ty.append(1 if np.argmax(micro_pred_dist) == actual_top_idx else 0)
                
            micro_mae = float(np.mean(micro_maes_ty)) if micro_maes_ty else float('nan')
            micro_hit = float(np.mean(micro_hits_ty)) if micro_hits_ty else float('nan')
            
            rows.append((ty, sec, macro_mae, nmae, macro_hit, micro_mae, micro_hit))

    for ty, sec, m_mae, nmae, m_hit, mi_mae, mi_hit in sorted(rows):
        mi_mae_str = f"{mi_mae:>9.3f}" if not math.isnan(mi_mae) else "      NaN"
        mi_hit_str = f"{mi_hit*100:>4.0f}%" if not math.isnan(mi_hit) else "  NaN"
        print(f"{ty:>6} {sec:20} | {m_mae:>9.3f} {m_hit*100:>4.0f}% | {mi_mae_str} {mi_hit_str} | {nmae:>10.3f}")

    m_mae_all = np.mean([r[2] for r in rows])
    nmae_all = np.mean([r[3] for r in rows])
    m_hit_all = np.mean([r[4] for r in rows])

    valid_mi_maes = [r[5] for r in rows if not math.isnan(r[5])]
    valid_mi_hits = [r[6] for r in rows if not math.isnan(r[6])]
    mi_mae_all = np.mean(valid_mi_maes) if valid_mi_maes else float('nan')
    mi_hit_all = np.mean(valid_mi_hits) if valid_mi_hits else float('nan')

    print("=" * 110)
    print(f"OVERALL RESULTS ({len(rows)} folds):")
    print(f"  MACRO LEVEL : MAE={m_mae_all:.3f}   MASE={m_mae_all/nmae_all:.3f}   Hit%={m_hit_all*100:.1f}%")
    print(f"  MICRO LEVEL : MAE={mi_mae_all:.3f}   Hit%={mi_hit_all*100:.1f}%")
    print()
    print("INTERPRETATION:")
    print("  MACRO MAE   = Error in predicting the broad topic count (e.g., 'Percentage').")
    print("  MACRO Hit%  = Accuracy in predicting the heavy-weight broad topics.")
    print("  MICRO MAE   = Error in perfectly distributing the broad topic down into specific sub-topics.")
    print("  MICRO Hit%  = Did we correctly identify the exact #1 most frequent micro-topic for a given broad topic?")

if __name__ == '__main__':
    main()
