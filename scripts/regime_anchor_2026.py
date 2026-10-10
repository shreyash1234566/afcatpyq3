"""Explicit 70% observed 2026 / 30% pre-2026 model blend.

This ports the old generator's weighting rule, not a claim that 70% is optimal.
For an August test, the caller must pass only January/earlier observed papers.
For January 2027, all available 2026 papers can be anchors. Section quotas stay fixed.
"""
from collections import Counter, defaultdict
import re
from topic_normalization_map import TOPIC_NORMALIZATION
from predict_micro_topics_2027 import allocate_exact, allocate_seats, micro_weights, micro_probs

ANCHOR_WEIGHT = 0.70


def normalized(q):
    row = dict(q)
    row['topic'] = TOPIC_NORMALIZATION.get(q.get('topic'), q.get('topic'))
    return row


def year(q):
    m = re.search(r'20\d{2}', q.get('test_name') or q.get('file_name', ''))
    return int(m.group()) if m else None


def is_anchor(q, anchor_year=2026, anchor_month=None):
    name = (q.get('test_name') or q.get('file_name', '')).lower()
    return year(q) == anchor_year and (anchor_month is None or anchor_month.lower() in name)


def blend_plan(questions, historical_plan, weight=ANCHOR_WEIGHT, anchor_year=2026, anchor_month=None):
    values = weight.values() if isinstance(weight, dict) else [weight]
    if not all(0 <= w <= 1 for w in values):
        raise ValueError('Anchor weight must be in [0, 1]')
    anchors = [normalized(q) for q in questions if is_anchor(q, anchor_year, anchor_month)]
    plan = {}
    for sec, block in historical_plan.items():
        counts = Counter(q['topic'] for q in anchors if q.get('section') == sec)
        total = sum(counts.values())
        target = block['section_total']
        history = {r['topic']: r['expected_count_exact'] / target for r in block['topics']}
        section_weight = weight.get(sec, ANCHOR_WEIGHT) if isinstance(weight, dict) else weight
        w = section_weight if total else 0.0
        shares = {topic: w * counts[topic] / max(total, 1) + (1-w) * history.get(topic, 0)
                  for topic in sorted(counts.keys() | history.keys())}
        z = sum(shares.values())
        plan[sec] = {'section_total': target, 'regime_year': anchor_year, 'anchor_month': anchor_month, 'anchor_weight': w,
                     'historical_weight': 1-w, 'anchor_questions': total,
                     'interval_status': 'Hybrid intervals not calibrated; old DM intervals are not reused.',
                     'topics': [{'topic': t, 'section': sec, 'expected_count_exact': p/z*target,
                                 'share': p/z} for t, p in shares.items()]}
    return plan


def blended_micro_probs(questions, section, topic, weight=ANCHOR_WEIGHT, anchor_year=2026, anchor_month=None):
    rows = [normalized(q) for q in questions]
    older = [q for q in rows if year(q) is not None and year(q) <= anchor_year
             and not is_anchor(q, anchor_year, anchor_month)]
    weights = micro_weights(older, anchor_year + (1 if anchor_month else 0))
    historical = micro_probs(weights, section, topic)
    counts = Counter(q['micro_topic'] for q in rows if is_anchor(q, anchor_year, anchor_month)
                     and q.get('section') == section and q.get('topic') == topic and q.get('micro_topic'))
    total = sum(counts.values())
    w = weight if total else 0.0
    return {micro: w*counts[micro]/max(total, 1) + (1-w)*historical.get(micro, 0)
            for micro in sorted(counts.keys() | historical.keys())}


def anchor_blueprint(questions, plan):
    rows = []
    for sec, block in plan.items():
        quotas = allocate_exact(block['topics'], block['section_total'])
        for topic, quota in quotas.items():
            if quota <= 0:
                continue
            probs = blended_micro_probs(questions, sec, topic, block.get('anchor_weight', ANCHOR_WEIGHT),
                                        block.get('regime_year', 2026), block.get('anchor_month'))
            for micro, n in allocate_seats(quota, probs).items():
                if n:
                    rows.append({'Section': sec, 'Broad_Topic': topic, 'Topic': topic,
                                 'Micro_Topic': micro, 'Predicted_Questions': n,
                                 'Probability_Weight': probs[micro]})
    assert sum(r['Predicted_Questions'] for r in rows) == sum(b['section_total'] for b in plan.values())
    return rows
