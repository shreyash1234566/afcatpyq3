"""
predict_micro_topics_2027.py
============================
Official Discrete Integer Micro-Topic Forecaster for AFCAT 01/2027.

Architecture (Commit 9a7d0ac & scripts/predict_micro_topics.py):
  1. Loads Dirichlet-Multinomial macro topic prediction plan (output/predictions_2027/prediction_plan_2027.json).
  2. Allocates exact macro-topic quotas (30 Verbal, 25 GA, 25 Reasoning, 20 Numerical) via Largest Remainder.
  3. Computes exponential recency momentum weights across all historical PYQs (2011–2026):
     w_2026 = 2.5, w_2025 = 1.2, w_2024 = 0.8, w_older = 0.2.
  4. Apportions discrete integer seats (>= 1 Q) to micro-topics using Largest Remainder (Hare-Niemeyer).
  5. Prunes all fractional / 0-seat micro-topics to produce a high-conviction discrete blueprint (~80-90 topics, exactly 100 Qs).
"""

import json
import math
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

def micro_weights(questions, cutoff):
    """Section-aware evidence using strictly pre-cutoff years."""
    weights = defaultdict(lambda: defaultdict(float))
    for q in questions:
        year = extract_year(q.get('file_name', ''), q.get('test_name', ''))
        sec, top, micro = q.get('section'), q.get('topic'), q.get('micro_topic')
        if not (sec and top and micro) or year >= cutoff:
            continue
        weight = {1: 2.5, 2: 1.2, 3: 0.8}.get(cutoff - year, 0.2)
        weights[(sec, top)][micro] += weight
    return weights

def micro_probs(weights, section, topic):
    micros = (weights.get((section, topic))
              or weights.get((section, topic.split('/')[0]))
              or {f'General {topic}': 1.0})
    total = sum(micros.values())
    return {m: w / total for m, w in micros.items()}

def blueprint(questions, plan, cutoff=2027):
    if plan and all(block.get('regime_year') == 2026 for block in plan.values()):
        from regime_anchor_2026 import anchor_blueprint
        filtered = [q for q in questions if extract_year(q.get('file_name', ''), q.get('test_name', '')) < cutoff]
        from audit_shift_coverage import paper_groups
        filtered = [q for group in paper_groups(filtered) if not group['invalid'] for q in group['rows']]
        return anchor_blueprint(filtered, plan)
    weights = micro_weights(questions, cutoff)
    rows = []
    for sec, block in plan.items():
        quotas = allocate_exact(block['topics'], block['section_total'])
        for topic, count in quotas.items():
            if count <= 0:
                continue
            probs = micro_probs(weights, sec, topic)
            for micro, seats in allocate_seats(count, probs).items():
                if seats > 0:
                    rows.append({'Section': sec, 'Broad_Topic': topic, 'Topic': topic,
                                 'Micro_Topic': micro, 'Predicted_Questions': seats,
                                 'Probability_Weight': round(probs[micro], 4)})
    assert sum(r['Predicted_Questions'] for r in rows) == sum(b['section_total'] for b in plan.values())
    return rows

def extract_year(filename: str, test_name: str = "") -> int:
    m = re.search(r"(20\d\d)", f"{filename} {test_name}")
    return int(m.group(1)) if m else 2011

def allocate_seats(total_seats: int, weights: dict) -> dict:
    """
    Allocate discrete integer seats (questions) proportionally using the Largest Remainder Method.
    Weights is a dictionary of {micro_topic: float_weight}.
    """
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
    for i in range(int(round(remaining))):
        seats[sorted_items[i][0]] += 1
    return seats

def allocate_exact(topics_list: list, section_target: int) -> dict:
    """Allocate exact integer seats to macro topics using Largest Remainder."""
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

def main():
    print("=" * 70)
    print(" AFCAT 01/2027 DISCRETE INTEGER MICRO-TOPIC FORECASTER")
    print("=" * 70)

    q_clean_path = ROOT / "data" / "processed" / "Q_clean.json"
    pred_path = ROOT / "output" / "predictions_2027" / "prediction_plan_2027.json"

    if not q_clean_path.exists():
        print(f"ERROR: {q_clean_path} not found.")
        return
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found.")
        return

    with open(q_clean_path, "r", encoding="utf-8") as f:
        qs = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        plan_2027 = json.load(f)

    # 1. Historical Micro-Topic Recency Momentum Weights (2011-2026)
    topic_micro_weights = defaultdict(lambda: defaultdict(float))
    for q in qs:
        sec = q.get("section")
        top = q.get("topic")
        micro = q.get("micro_topic")
        yr = extract_year(q.get("file_name", ""), q.get("test_name", ""))
        if not (sec and top and micro and yr):
            continue
        w = 2.5 if yr >= 2026 else (1.2 if yr == 2025 else (0.8 if yr == 2024 else 0.2))
        topic_micro_weights[top][micro] += w

    # 2. Discrete Apportionment
    final_micro_predictions = []
    section_order = ["Verbal Ability", "General Awareness", "Reasoning", "Numerical Ability"]

    for sec in section_order:
        blk = plan_2027.get(sec, {})
        if not blk:
            continue
        print(f"\n" + "=" * 60)
        print(f" {sec.upper()} SECTION ({blk['section_total']} Qs)")
        print("=" * 60)

        macro_allocs = allocate_exact(blk["topics"], blk["section_total"])
        for top, n_q in sorted(macro_allocs.items(), key=lambda x: -x[1]):
            if n_q < 1:
                continue
            micros = topic_micro_weights.get(top) or topic_micro_weights.get(top.split('/')[0]) or {f"General {top}": 1.0}
            tot_w = sum(micros.values())
            probs = {m: w / tot_w for m, w in micros.items()}

            m_allocs = allocate_seats(n_q, probs)
            sorted_m = sorted(m_allocs.items(), key=lambda x: (-x[1], -probs.get(x[0], 0)))

            print(f"\n   {top} (Macro: {n_q} Qs)")
            for m_name, count in sorted_m:
                if count > 0:
                    print(f"       |-- {count} Qs : {m_name}")
                    final_micro_predictions.append({
                        "Section": sec,
                        "Broad_Topic": top,
                        "Micro_Topic": m_name,
                        "Predicted_Questions": int(count),
                        "Probability_Weight": round(probs.get(m_name, 0), 4)
                    })

    final_micro_predictions = blueprint(qs, plan_2027)
    # 3. Save Results
    out_path = ROOT / "output" / "predictions_2027" / "afcat_2027_micro_predictions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_micro_predictions, f, indent=2, ensure_ascii=False)

    total_qs = sum(m["Predicted_Questions"] for m in final_micro_predictions)
    print("\n" + "=" * 70)
    print(f"[OK] Discrete micro predictions saved: {len(final_micro_predictions)} micro-topics, {total_qs}/100 questions.")
    print(f"     Path: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
