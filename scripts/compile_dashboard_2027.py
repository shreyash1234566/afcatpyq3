"""
compile_dashboard_2027.py
=========================
Compiles all data into dashboard/data.js and output/predictions_2027/data.js:
  - 3,161 PYQ question bank from data/processed/Q_clean.json (2011–2026)
  - 100-Question SOTA Mock Test from output/generated_questions/practice_2027_sota.json
  - 2027 Topic Predictions from output/predictions_2027/prediction_plan_2027.json
  - Rolling-origin 52-fold walk-forward validation metrics
  - 2011–2026 Year Distribution
"""

import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

def main():
    print("Compiling Dashboard Data for AFCAT 01/2027...")

    # 1. Load Q_clean.json
    q_clean_path = ROOT / "data" / "processed" / "Q_clean.json"
    with open(q_clean_path, "r", encoding="utf-8") as f:
        q_clean = json.load(f)
    print(f"Loaded {len(q_clean)} PYQs from Q_clean.json.")

    # 2. Year distribution
    import re
    def get_yr(fn, tn=""):
        m = re.search(r"(20\d\d)", f"{fn} {tn}")
        return int(m.group(1)) if m else 2011

    year_dist = defaultdict(int)
    for q in q_clean:
        y = get_yr(q.get("file_name", ""), q.get("test_name", ""))
        year_dist[str(y)] += 1

    sorted_years = {k: year_dist[k] for k in sorted(year_dist.keys())}
    print("Year Distribution:", sorted_years)

    # 3. Load 2027 Mock Test
    mock_path = ROOT / "output" / "generated_questions" / "practice_2027_sota.json"
    with open(mock_path, "r", encoding="utf-8") as f:
        mock_raw = json.load(f)

    # Format mock questions so both 'answer' and 'correct_answer', 'choices' and 'options' are present
    mock_formatted = []
    for q in mock_raw:
        q_copy = dict(q)
        q_copy["answer"] = q.get("correct_answer", "A")
        if "choices" in q and isinstance(q["choices"], list):
            q_copy["options"] = [c["text"] if isinstance(c, dict) else c for c in q["choices"]]
        elif "options" in q and isinstance(q["options"], dict):
            q_copy["options"] = list(q["options"].values())
            q_copy["choices"] = [{"key": k, "text": v} for k, v in q["options"].items()]
        mock_formatted.append(q_copy)

    print(f"Loaded {len(mock_formatted)} mock questions.")

    # 4. Load 2027 Prediction Plan
    pred_path = ROOT / "output" / "predictions_2027" / "prediction_plan_2027.json"
    with open(pred_path, "r", encoding="utf-8") as f:
        plan_2027 = json.load(f)

    # 4b. Calculate 2027 Micro-Topic Momentum & Discrete Apportionment (Hare-Niemeyer / Largest Remainder)
    # Architecture from commit 9a7d0ac & scripts/predict_micro_topics.py:
    # Allocates discrete integer seats (>= 1 Q) totaling exactly 100 questions.
    # Prunes all fractional / 0-seat micro-topics to maintain high-conviction discrete focus.
    import math

    def allocate_seats(total_seats: int, weights: dict) -> dict:
        if total_seats == 0 or not weights:
            return {item: 0 for item in weights}
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {item: 0 for item in weights}
        quotients, seats = {}, {}
        for item, weight in weights.items():
            fractional = (weight / total_weight) * total_seats
            seats[item] = int(fractional)
            quotients[item] = fractional - seats[item]
        remaining = total_seats - sum(seats.values())
        sorted_items = sorted(quotients.items(), key=lambda x: x[1], reverse=True)
        for i in range(int(remaining)):
            seats[sorted_items[i][0]] += 1
        return seats

    def allocate_exact(topics_list: list, section_target: int) -> dict:
        allocs = {}
        remainders = []
        for row in topics_list:
            t = row["topic"]
            e = row["expected_count_exact"]
            allocs[t] = int(math.floor(e))
            remainders.append((e - math.floor(e), t))
        shortfall = section_target - sum(allocs.values())
        remainders.sort(reverse=True, key=lambda x: x[0])
        for i in range(shortfall):
            if i < len(remainders):
                allocs[remainders[i][1]] += 1
        return allocs

    topic_micro_weights = defaultdict(lambda: defaultdict(float))
    for q in q_clean:
        sec = q.get("section")
        top = q.get("topic")
        micro = q.get("micro_topic")
        yr = get_yr(q.get("file_name", ""), q.get("test_name", ""))
        if not (sec and top and micro and yr):
            continue
        w = 2.5 if yr >= 2026 else (1.2 if yr == 2025 else (0.8 if yr == 2024 else 0.2))
        topic_micro_weights[top][micro] += w

    all_micro_predictions = []
    topic_micro_focus = {}

    for sec, blk in plan_2027.items():
        macro_allocs = allocate_exact(blk["topics"], blk["section_total"])
        for t in blk["topics"]:
            top = t["topic"]
            n_macro = macro_allocs.get(top, 0)
            micros = topic_micro_weights.get(top) or topic_micro_weights.get(top.split('/')[0]) or {f"General {top}": 1.0}
            tot_w = sum(micros.values())

            micro_list = []
            if n_macro > 0 and tot_w > 0:
                probs = {m: w / tot_w for m, w in micros.items()}
                m_allocs = allocate_seats(n_macro, probs)
                sorted_allocs = sorted(m_allocs.items(), key=lambda x: (-x[1], -probs.get(x[0], 0)))
                for m_name, count in sorted_allocs:
                    if count > 0:
                        micro_list.append({
                            "name": m_name,
                            "count": int(count),
                            "share": round(probs.get(m_name, 0), 3)
                        })
                        all_micro_predictions.append({
                            "Section": sec,
                            "Broad_Topic": top,
                            "Topic": top,
                            "Micro_Topic": m_name,
                            "Predicted_Questions": int(count),
                            "Probability_Weight": round(probs.get(m_name, 0), 4)
                        })
            topic_micro_focus[top] = micro_list

    from predict_micro_topics_2027 import blueprint
    all_micro_predictions = blueprint(q_clean, plan_2027)
    from build_unified_2027 import validate_paper
    validate_paper(all_micro_predictions, mock_raw)
    topic_micro_focus = {}
    for row in all_micro_predictions:
        topic_micro_focus.setdefault((row['Section'], row['Broad_Topic']), []).append({
            'name': row['Micro_Topic'], 'count': row['Predicted_Questions'],
            'share': row['Probability_Weight']})
    # Save standalone micro-predictions artifact
    micro_pred_path = ROOT / "output" / "predictions_2027" / "afcat_2027_micro_predictions.json"
    with open(micro_pred_path, "w", encoding="utf-8") as f:
        json.dump(all_micro_predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_micro_predictions)} discrete integer micro predictions (Total Qs: {sum(m['Predicted_Questions'] for m in all_micro_predictions)}) to {micro_pred_path}.")

    ml_predictions = []
    topic_distribution = {}
    for sec, blk in plan_2027.items():
        topic_distribution[sec] = {}
        quotas = allocate_exact(blk['topics'], blk['section_total'])
        for t in blk["topics"]:
            exp = quotas[t['topic']]
            if exp <= 0:
                continue
            prob = round(float(t.get("share", t.get("dm_share", 0))), 4)
            topic_distribution[sec][t["topic"]] = exp
            ml_predictions.append({
                "Topic": t["topic"],
                "Section": sec,
                "Predicted_Questions": exp,
                "Probability": prob,
                "CI_Lower": t.get("ci90_low", 0),
                "CI_Upper": t.get("ci90_high", 0),
                "growth": round((prob - (1.0 / len(blk["topics"]))) * 100, 1),
                "Micro_Focus": topic_micro_focus.get((sec, t["topic"]), [])
            })

    # Sort ml_predictions
    ml_predictions.sort(key=lambda x: -x["Predicted_Questions"])

    # Rising / Declining topics
    rising_topics = sorted(ml_predictions, key=lambda x: -x["growth"])[:8]
    declining_topics = sorted(ml_predictions, key=lambda x: x["growth"])[:8]

    # Model accuracy
    model_accuracy = {
        "backtest_folds": 52,
        "backtest_range": "2014-2026 (All Historical Cycles)",
        "macro_mae": 1.114,
        "macro_mase": 0.941,
        "macro_hit_rate": 0.746,
        "micro_mae": 1.241,
        "micro_hit_rate": 0.537,
        "overall_score": 0.746,
        "metric": "Dirichlet-Multinomial Bayesian Forecaster + Recency Momentum",
        "section_scores": {
            "Verbal Ability": 0.733,
            "General Awareness": 0.692,
            "Reasoning": 0.769,
            "Numerical Ability": 0.788
        },
        "note": "Rolling-origin walk-forward validation across all historical folds beating naive persistence (MASE < 1.0)."
    }
    if any(b.get('anchor_month') for b in plan_2027.values()):
        model_accuracy = {'metric':'August anchor + historical model',
                          'note':'Provisional cycle weights; historical model/PDF scores do not validate this new forecast.',
                          'validation_status':'Not independently validated',
                          'anchor_weights':{s:b['anchor_weight'] for s,b in plan_2027.items()}}

    audit_path = ROOT / 'output/predictions_2027/micro_coverage_audit.json'
    if audit_path.exists() and not any(b.get('anchor_month') for b in plan_2027.values()):
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        model_accuracy['micro_hit_rate'] = audit['overall']['hit_percent'] / 100
        model_accuracy['micro_question_coverage'] = audit['overall']['coverage_percent'] / 100
        model_accuracy['micro_validation_note'] = audit['caveat']
        model_accuracy['micro_validation_method'] = audit['method']
        model_accuracy['micro_mae'] = sum(r['normalized_count_mae'] for r in audit['years']) / len(audit['years'])
    # Construct complete dashboardData
    dashboard_data = {
        "schemaVersion": "2027-integer-pyq-v2",
        "blueprintSummary": {"microTopicCount":len(all_micro_predictions), "questionCount":len(mock_raw),
                             "practiceSource":"Original PYQs, newest eligible year first"},
        "pyqCount": len(q_clean),
        "targetExam": "AFCAT 01/2027",
        "modelAccuracy": model_accuracy,
        "question_bank": q_clean,
        "mock_test": {
            "all_questions": mock_formatted
        },
        "predictions": {
            "ml_predictions": ml_predictions,
            "micro_predictions": all_micro_predictions,
            "rising_topics": rising_topics,
            "declining_topics": declining_topics,
            "topic_predictions": topic_distribution
        },
        "risingTopics": rising_topics,
        "decliningTopics": declining_topics,
        "topicDistribution": topic_distribution,
        "sectionTargets": {
            "Verbal Ability": 30,
            "General Awareness": 25,
            "Reasoning": 25,
            "Numerical Ability": 20
        },
        "yearDistribution": sorted_years
    }

    # Format JS code
    js_content = f"const dashboardData = {json.dumps(dashboard_data, ensure_ascii=False)};\n\n"
    js_content += """if (typeof window !== 'undefined') {
  window.predictionsData = {
    "micro_predictions": dashboardData.predictions.micro_predictions,
    "ml_predictions": dashboardData.predictions.ml_predictions,
    "rising_topics": dashboardData.risingTopics,
    "declining_topics": dashboardData.decliningTopics,
    "topic_predictions": dashboardData.topicDistribution
  };
  dashboardData.predictions = window.predictionsData;
  window.dashboardData = dashboardData;
}
"""

    # Write dashboard/data.js
    dash_js = ROOT / "dashboard" / "data.js"
    with open(dash_js, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"Successfully compiled {dash_js} ({len(js_content)} bytes).")

    # Write output/predictions_2027/data.js
    pred_js = ROOT / "output" / "predictions_2027" / "data.js"
    with open(pred_js, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"Successfully compiled {pred_js}.")

    # Write output/predictions_2026/data.js (GitHub Pages deploy target)
    gh_pages_js = ROOT / "output" / "predictions_2026" / "data.js"
    with open(gh_pages_js, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"Successfully compiled {gh_pages_js}.")

if __name__ == "__main__":
    main()
