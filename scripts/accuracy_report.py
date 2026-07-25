"""
accuracy_report.py
Honest walk-forward (rolling-origin) accuracy for the DM topic forecaster.

For each section, for each year t (with >= MIN_HISTORY prior usable years):
  - train ONLY on years < t
  - predict topic distribution for year t
  - compare against what ACTUALLY appeared in year t

Metrics:
  MAE      = mean |predicted_count - actual_count| per topic (lower better)
  Hit%     = of the topics that were actually in the top-half by count,
             how many did we also rank in our predicted top-half (higher better)
  MASE     = MAE / MAE(naive last-year) — <1 means we beat "assume same as last year"
"""
import sys, collections, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from dirichlet_forecaster import (load_counts, section_years, share_matrix,
    marginalised_mean, tune_params, actual_vec, SEC_TARGET, MIN_HISTORY)

cnt, syt, years = load_counts()

print("WALK-FORWARD ACCURACY REPORT — AFCAT Topic Prediction (DM-Multinomial)")
print("=" * 78)
print(f"{'Year':>6} {'Section':20} {'MAE':>6} {'naiveMAE':>8} {'Hit%':>6}  Top-3 predicted -> actual")
print("-" * 78)

rows = []
for sec in sorted(SEC_TARGET):
    topics = sorted(cnt[sec])
    yrs = section_years(syt, sec)
    for i in range(MIN_HISTORY, len(yrs)):
        hist, ty = yrs[:i], yrs[i]
        g, lam = tune_params(cnt, syt, sec, topics, hist)
        S = share_matrix(cnt, syt, sec, hist, topics)
        pred = marginalised_mean(S, SEC_TARGET[sec], g, lam=lam)
        act = actual_vec(cnt, syt, sec, ty, topics)
        naive = actual_vec(cnt, syt, sec, hist[-1], topics)  # last-year-as-forecast
        mae = float(np.abs(pred - act).mean())
        nmae = float(np.abs(naive - act).mean())
        k = len(topics)
        pred_top = set(np.argsort(pred)[-max(k // 2, 1):])
        act_top = set(np.argsort(act)[-max(k // 2, 1):])
        hit = len(pred_top & act_top) / max(len(act_top), 1)
        top3 = sorted(range(k), key=lambda x: -pred[x])[:3]
        pairs = " | ".join(f"{topics[j]}:{pred[j]:.1f}->{act[j]:.1f}" for j in top3)
        rows.append((ty, sec, mae, nmae, hit, pairs))

for ty, sec, mae, nmae, hit, pairs in sorted(rows):
    print(f"{ty:>6} {sec:20} {mae:>6.3f} {nmae:>8.3f} {hit*100:>5.0f}%  {pairs}")

mae_all = np.mean([r[2] for r in rows])
nmae_all = np.mean([r[3] for r in rows])
hit_all = np.mean([r[4] for r in rows])
print("=" * 78)
print(f"OVERALL  MAE={mae_all:.3f}  naiveMAE={nmae_all:.3f}  "
      f"MASE={mae_all/nmae_all:.3f}  Hit%={hit_all*100:.1f}  n_folds={len(rows)}")
print()
print("PER-SECTION:")
for sec in sorted(SEC_TARGET):
    sr = [(r[2], r[3], r[4]) for r in rows if r[1] == sec]
    if sr:
        m, n, h = zip(*sr)
        avg_mae = np.mean(m)
        avg_qs = SEC_TARGET[sec] / len(cnt[sec])
        rel_err = (avg_mae / avg_qs) * 100
        print(f"  {sec:20} MAE={avg_mae:.3f}  MASE={avg_mae/np.mean(n):.3f}  "
              f"RelErr%={rel_err:.1f}%  Hit%={np.mean(h)*100:.1f}  folds={len(m)}")

print()
print("INTERPRETATION:")
print("  MAE      = on average we're off by this many questions per topic")
print("  RelErr%  = MAE relative to average topic size (Verbal & Reasoning perform best here)")
print("  MASE <1  = model beats 'same as last year' baseline")
print("  Hit%     = we correctly identify which topics are heavy-weight")
