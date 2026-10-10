"""Conservative paper-by-paper backtest and explicit 2026-topic-regime candidate.

No paper from the target day/month/year uncertainty interval enters training.
No aliases are fitted to test outcomes. Family scoring is explicitly coarser than
exact micro-label scoring and is not reported as exact-topic accuracy.
"""
import calendar
import collections
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'models'))
sys.path.insert(0, str(ROOT / 'scripts'))
from dirichlet_forecaster import (SEC_TARGET, MIN_HISTORY, REGIME_START_YEAR,
    section_years, share_matrix, tune_params, marginalised_mean)
from topic_normalization_map import TOPIC_NORMALIZATION
from predict_micro_topics_2027 import blueprint
from regime_anchor_2026 import blend_plan, anchor_blueprint

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_name) if name})


def dates(name):
    """Return date uncertainty bounds, never fabricate a day within a month."""
    match = re.search(r'20\d{2}', name)
    if not match:
        return None
    year = int(match.group())
    month_match = re.search(r'\b(' + '|'.join(MONTHS) + r')\s*(\d{1,2})?\b', name, re.I)
    if month_match:
        month = MONTHS[month_match[1].lower()]
        if month_match[2]:
            day = int(month_match[2])
            return dt.date(year, month, day), dt.date(year, month, day), 'day'
        return dt.date(year, month, 1), dt.date(year, month, calendar.monthrange(year, month)[1]), 'month'
    return dt.date(year, 1, 1), dt.date(year, 12, 31), 'year'


def canonical(question):
    q = dict(question)
    topic = q.get('topic')
    q['topic'] = TOPIC_NORMALIZATION.get(topic, topic)
    return q


def family(name):
    """Small, declared study-family mapping; never merge different subjects."""
    if re.match(r'^Synonym(?:\s*-|$)', name, re.I):
        return 'Synonym question family'
    if re.match(r'^Antonym(?:\s*-|$)', name, re.I):
        return 'Antonym question family'
    aliases = {'Sentence-Level Error Identification': 'Sentence Error Identification',
               'Sentence Rearrangement (PQRS)': 'Sentence Rearrangement',
               'Complex→Simple Sentence Transformation': 'Sentence Transformation (Complex to Simple)',
               'Detail-based RC': 'Detail-based Question'}
    return aliases.get(name, name)


def paper_groups(questions):
    grouped = collections.defaultdict(list)
    for q in questions:
        name = q.get('test_name') or q.get('file_name') or '<unknown>'
        # A generic year label may describe several distinct official papers.
        if re.fullmatch(r'AFCAT 20\d{2}', name):
            name = (q.get('file_name') or name).replace('_', ' ')
        grouped[name].append(q)
    papers = []
    for name, rows in sorted(grouped.items()):
        bounds = dates(name)
        seen = set()
        unique = []
        for q in rows:
            # Repeated stems (especially RC/figures) are not duplicate questions.
            key = (q.get('file_name'), q.get('question_number'), q.get('section'),
                   q.get('question_text'), json.dumps(q.get('choices'), sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique.append(q)
        numbers = [q.get('question_number') for q in unique]
        invalid = (not bounds or len(unique) > 100
                   or any(q.get('section') not in SEC_TARGET for q in unique))
        papers.append({'name': name, 'rows': unique, 'dates': bounds,
                       'invalid': invalid, 'duplicates_removed': len(rows) - len(unique),
                       'complete': not invalid and len(unique) == 100 and len(set(numbers)) == 100 and set(numbers) == set(range(1, 101))})
    return papers


def fit_macro(rows):
    cnt = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    totals = collections.defaultdict(collections.Counter)
    for raw in rows:
        q = canonical(raw)
        year = int(re.search(r'20\d{2}', q.get('test_name') or q['file_name']).group())
        cnt[q['section']][q['topic']][year] += 1
        totals[q['section']][year] += 1
    plan = {}
    for sec in SEC_TARGET:
        history = section_years(totals, sec)
        post = [y for y in history if y >= REGIME_START_YEAR]
        if len(post) >= MIN_HISTORY:
            history = post
        if len(history) < MIN_HISTORY:
            return None
        topics = sorted(cnt[sec])
        gamma, lam = tune_params(cnt, totals, sec, topics, history)
        S = share_matrix(cnt, totals, sec, history, topics)
        means = marginalised_mean(S, SEC_TARGET[sec], gamma, lam=lam)
        plan[sec] = {'section_total': SEC_TARGET[sec], 'topics': [
            {'topic': t, 'expected_count_exact': float(n)} for t, n in zip(topics, means)]}
    return plan


def regime_plan(rows, prior_plan):
    """2026 evidence dominates after 2+ papers; older forecast = one pseudo-paper.
    This is an explicit candidate, not a claim of an optimal/calibrated prior.
    """
    new = [canonical(q) for q in rows if '2026' in q.get('file_name', '')]
    if not new:
        return None, []
    plan = {}
    for sec, old in prior_plan.items():
        counts = collections.Counter(q['topic'] for q in new if q['section'] == sec)
        total = sum(counts.values())
        old_counts = {r['topic']: r['expected_count_exact'] for r in old['topics']}
        # One target-sized section worth of older pseudo-counts.
        merged = {t: counts[t] + old_counts.get(t, 0) for t in sorted(counts.keys() | old_counts.keys())}
        normalizer = sum(merged.values())
        plan[sec] = {'section_total': SEC_TARGET[sec], 'topics': [
            {'topic': t, 'expected_count_exact': n / normalizer * SEC_TARGET[sec]}
            for t, n in merged.items()], 'regime_year': 2026,
            'regime_evidence_share': total / (total + SEC_TARGET[sec])}
    return plan, new


def score(predictions, rows, families=False):
    predicted = collections.Counter()
    actual = collections.Counter()
    transform = family if families else lambda value: value
    for p in predictions:
        predicted[(p['Section'], TOPIC_NORMALIZATION.get(p['Broad_Topic'], p['Broad_Topic']),
                   transform(p['Micro_Topic']))] += p['Predicted_Questions']
    for raw in rows:
        q = canonical(raw)
        actual[(q['section'], q['topic'], transform(q.get('micro_topic') or '<unlabelled>'))] += 1
    hits = sum(actual[k] > 0 for k in predicted)
    covered = sum(n for k, n in actual.items() if k in predicted)
    exact = sum(actual[k] == predicted[k] for k in predicted)
    union = sorted(predicted.keys() | actual.keys())
    return {'predicted_slots': len(predicted), 'hit_slots': hits,
            'hit_percent': 100 * hits / len(predicted),
            'recall_percent': 100 * hits / len(actual),
            'covered_questions': covered, 'observed_questions': len(rows),
            'coverage_percent': 100 * covered / len(rows),
            'exact_count_percent': 100 * exact / len(predicted),
            'predicted_slot_mae': sum(abs(predicted[k] - actual[k]) for k in predicted) / len(predicted),
            'union_mae': sum(abs(predicted[k] - actual[k]) for k in union) / len(union),
            'slots': [{'section': k[0], 'topic': k[1], 'micro_topic': k[2],
                       'predicted': predicted[k], 'actual': actual[k]} for k in union]}


def main():
    source = ROOT / 'data/processed/Q_clean.json'
    raw = source.read_bytes()
    qs = json.loads(raw)
    papers = paper_groups(qs)
    valid = [p for p in papers if not p['invalid']]
    results = []
    for target in sorted((p for p in valid if p['complete']), key=lambda p: (p['dates'][0], p['name'])):
        earlier = [p for p in valid if p['dates'][1] < target['dates'][0]]
        training = [q for p in earlier for q in p['rows']]
        plan = fit_macro(training)
        if plan is None:
            continue
        year = target['dates'][0].year
        # Explicit prefix filtering includes earlier current-year papers.
        cutoff = max(int(re.search(r'20\d{2}', q.get('test_name') or q['file_name']).group()) for q in training) + 1
        literal = blueprint(training, plan, cutoff)
        normalized = blueprint([canonical(q) for q in training], plan, cutoff)
        row = {'paper': target['name'], 'date_precision': target['dates'][2],
               'training_papers': [p['name'] for p in earlier], 'training_questions': len(training),
               'section_counts': dict(collections.Counter(q['section'] for q in target['rows'])),
               'current_literal': score(literal, target['rows']),
               'canonical_macro_lookup': score(normalized, target['rows']),
               'study_family_coverage': score(normalized, target['rows'], families=True)}
        old_rows = [q for q in training if '2026' not in q['file_name']]
        pre_plan = fit_macro(old_rows)
        candidate = blend_plan(training, pre_plan) if pre_plan and any('2026' in q['file_name'] for q in training) else None
        if candidate:
            pred = anchor_blueprint(training, candidate)
            row['regime_2026_base'] = score(pred, target['rows'])
            row['regime_2026_family'] = score(pred, target['rows'], families=True)
        if year == 2026 and any('2026' in q['file_name'] for q in training):
            no_jan = [q for q in training if '2026' not in q['file_name']]
            pre_plan = fit_macro(no_jan)
            if pre_plan:
                row['without_january_2026'] = score(blueprint([canonical(q) for q in no_jan], pre_plan, cutoff), target['rows'])
        results.append(row)
        print(f"{row['paper']}: exact-label coverage {row['current_literal']['covered_questions']}/100; canonical lookup {row['canonical_macro_lookup']['covered_questions']}/100; study families {row['study_family_coverage']['covered_questions']}/100", flush=True)
    overview = []
    for method in ('current_literal', 'canonical_macro_lookup', 'study_family_coverage', 'regime_2026_base', 'regime_2026_family'):
        scores = [r[method] for r in results if method in r]
        if scores:
            overview.append({'method': method, 'papers': len(scores),
                             'coverage_percent': 100 * sum(s['covered_questions'] for s in scores) / sum(s['observed_questions'] for s in scores),
                             'hit_percent': 100 * sum(s['hit_slots'] for s in scores) / sum(s['predicted_slots'] for s in scores)})
    report = {'dataset_sha256': hashlib.sha256(raw).hexdigest(), 'results': results, 'summary': overview,
              'paper_inventory': [{'paper': p['name'], 'questions': len(p['rows']), 'complete': p['complete'],
                                   'excluded_invalid': p['invalid'], 'duplicates_removed': p['duplicates_removed'],
                                   'date_precision': p['dates'][2] if p['dates'] else None} for p in papers],
              'limitations': ['Only papers with 100 unique numbered questions are scored.',
                             'Same-day papers never train each other; unknown dates exclude overlapping intervals.',
                             'Date of exam is not proof of date of availability; labeling was retrospective.',
                             'Partial valid earlier papers can enter training; invalid merged papers are excluded.',
                             'Study-family scores are coarser and cannot replace exact micro-topic scores.',
                             'Historical subject counts differ from current fixed targets; count metrics are most comparable for 2026.',
                             'The 2026-base candidate has only two out-of-sample August shifts, not enough to establish January 2027 performance.']}
    out = ROOT / 'output/predictions_2027/shift_coverage_audit.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    # Export an explicitly non-deployed 2027 candidate, not overwrite the live plan.
    full = [q for p in valid for q in p['rows']]
    old = fit_macro([q for q in full if '2026' not in q['file_name']])
    candidate = blend_plan(full, old)
    if candidate:
        (out.parent / 'prediction_plan_2027_2026_base_candidate.json').write_text(json.dumps(candidate, indent=2), encoding='utf-8')
        (out.parent / 'micro_2027_2026_base_candidate.json').write_text(json.dumps(anchor_blueprint(full, candidate), indent=2), encoding='utf-8')
    lines = ['# Chronological paper-by-paper validation', '',
             '| Paper | Exact-label coverage | Canonical macro lookup | Study-family coverage |', '|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['paper']} | {r['current_literal']['covered_questions']}/100 | {r['canonical_macro_lookup']['covered_questions']}/100 | {r['study_family_coverage']['covered_questions']}/100 |")
    lines += ['', '## Aggregate metrics', '', '| Method | Papers | Topic precision | Question coverage |', '|---|---:|---:|---:|']
    for r in overview:
        lines.append(f"| {r['method']} | {r['papers']} | {r['hit_percent']:.1f}% | {r['coverage_percent']:.1f}% |")
    lines += ['', '## January-to-August 2026 comparison', '', '| Paper | Without January | With January (canonical lookup) | 2026-base candidate |', '|---|---:|---:|---:|']
    for r in results:
        if 'without_january_2026' in r:
            lines.append(f"| {r['paper']} | {r['without_january_2026']['covered_questions']}/100 | {r['canonical_macro_lookup']['covered_questions']}/100 | {r['regime_2026_base']['covered_questions']}/100 |")
    lines += ['', '## Limitations', ''] + ['- ' + s for s in report['limitations']]
    (out.parent / 'SHIFT_COVERAGE_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('SUMMARY:', overview)


if __name__ == '__main__':
    main()
