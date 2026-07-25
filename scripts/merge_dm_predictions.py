"""
merge_dm_predictions.py
Replace ml_predictions in data.js with DM forecaster output.
Everything else (question_bank, rising_topics, declining_topics, mock_test) is untouched.
"""
import json, sys, re, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from dirichlet_forecaster import DirichletForecaster, load_counts, section_years, SEC_TARGET

ROOT = Path(__file__).resolve().parent.parent
DATA_JS = ROOT / "output" / "predictions_2026" / "data.js"

# ── 1. Run DM forecaster on full history (all topics, all years) ──────────────
fc = DirichletForecaster.from_repo()
dm = fc.predict()          # dict: section -> {topics: [{topic, expected_count_exact, ci90_low, ci90_high, share}]}

# ── 2. Build historical averages from Q.json for confidence/trend fields ──────
cnt, syt, years = load_counts()
hist_avg = {}   # (section, topic) -> avg per year
for sec in SEC_TARGET:
    for t in cnt[sec]:
        yrs = section_years(syt, sec)
        total = sum(cnt[sec][t][y] for y in yrs)
        hist_avg[(sec, t)] = round(total / max(len(yrs), 1), 2)

# ── 3. Convert DM output to the existing ml_predictions schema ────────────────
# Original schema fields: Section, Topic, Predicted_Questions, Confidence,
#   Trend, Historical_Avg, Is_Core, raw_count, topic_code, topic_name
CORE_THRESHOLD = 2.0   # topics expected >= this are "core"

new_preds = []
for sec, blk in dm.items():
    for r in blk["topics"]:
        t = r["topic"]
        exp = r["expected_count_exact"]
        ha  = hist_avg.get((sec, t), 0)
        # confidence: higher for topics with more history and higher expected count
        n_yrs = sum(1 for y in section_years(syt, sec) if cnt[sec][t][y] > 0)
        conf  = round(min(0.95, 0.45 + 0.35 * min(n_yrs / 10, 1.0) + 0.20 * min(exp / 4, 1.0)), 4)
        # trend: compare last-3-year avg vs earlier avg
        yrs = section_years(syt, sec)
        recent = [cnt[sec][t][y] for y in yrs[-3:]] if len(yrs) >= 3 else []
        earlier = [cnt[sec][t][y] for y in yrs[:-3]] if len(yrs) > 3 else []
        r_avg = sum(recent) / len(recent) if recent else 0
        e_avg = sum(earlier) / len(earlier) if earlier else r_avg
        trend = "Increasing" if r_avg > e_avg * 1.2 else ("Decreasing" if r_avg < e_avg * 0.8 else "Stable")
        new_preds.append({
            "Section": sec,
            "Topic": t,
            "Predicted_Questions": round(exp, 2),
            "Confidence": f"{round(conf * 100, 2)}%",
            "Trend": trend,
            "Historical_Avg": ha,
            "Is_Core": "✓" if exp >= CORE_THRESHOLD else "",
            "raw_count": exp,
            "topic_code": t,
            "topic_name": t,
            "ci90_low": r["ci90_low"],
            "ci90_high": r["ci90_high"],
        })

# ── 4. Load existing afcat_2026_predictions.json, replace only ml_predictions ───
PRED_JSON = ROOT / "output" / "predictions_2026" / "afcat_2026_predictions.json"
with open(PRED_JSON, "r", encoding="utf-8") as f:
    d = json.load(f)

d["ml_predictions"] = new_preds

# ── 5. Write back ─────────────────────────────────────────────────────────────
with open(PRED_JSON, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

# ── 6. Sanity check ───────────────────────────────────────────────────────────
by_sec = collections.defaultdict(list)
for r in new_preds: by_sec[r["Section"]].append(r)
print("DM predictions merged into afcat_2026_predictions.json:")
for s in sorted(by_sec):
    tot = sum(r["raw_count"] for r in by_sec[s])
    print(f"  {s}: {len(by_sec[s])} topics, sum={tot:.1f} (target {SEC_TARGET[s]})")
print("Done.")
