"""Previous August -> next January/February only. Never infer a cycle from Paper1 alone."""
import collections
import json
import re
from pathlib import Path
from audit_shift_coverage import ROOT, paper_groups, fit_macro, canonical, score
from predict_micro_topics_2027 import blueprint
from regime_anchor_2026 import blend_plan, anchor_blueprint, is_anchor


def macro_error(plan, actual, section):
    expected = {r['topic']: r['expected_count_exact'] for r in plan[section]['topics']}
    observed = collections.Counter(canonical(q)['topic'] for q in actual if q['section'] == section)
    keys = expected.keys() | observed.keys()
    return {'l1_count_error': sum(abs(expected.get(k, 0)-observed[k]) for k in keys),
            'union_mae': sum(abs(expected.get(k, 0)-observed[k]) for k in keys)/len(keys)}


def shares(rows, section, key):
    counts = collections.Counter(q.get(key) or '<unlabelled>' for q in rows if q['section'] == section)
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()} if total else {}


def changes(anchor, actual):
    result = {}
    for sec in ('Verbal Ability', 'General Awareness', 'Reasoning', 'Numerical Ability'):
        a = shares([canonical(q) for q in anchor], sec, 'topic')
        b = shares([canonical(q) for q in actual], sec, 'topic')
        result[sec] = {'total_variation': 0.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in a.keys()|b.keys()),
                       'topics': [{'topic': k, 'august_share': a.get(k,0), 'next_january_share': b.get(k,0),
                                   'change_percentage_points': 100*(b.get(k,0)-a.get(k,0))}
                                  for k in sorted(a.keys()|b.keys())]}
    return result


def main():
    qs = json.loads((ROOT/'data/processed/Q_clean.json').read_text(encoding='utf-8'))
    papers = paper_groups(qs)
    valid = [p for p in papers if not p['invalid']]
    results = []
    unavailable = []
    weight_trials = []
    for target in papers:
        if not re.search(r'\b(jan|feb)', target['name'], re.I) or not target['dates']:
            continue
        y = target['dates'][0].year
        anchors = [p for p in valid if p['dates'][0].year == y-1 and re.search(r'\baug', p['name'], re.I)]
        if not target['complete'] or not anchors:
            unavailable.append({'paper': target['name'], 'reason': 'No previous-August records' if not anchors else 'Target is partial or a conflicting merged paper'})
            continue
        earlier = [p for p in valid if p['dates'][1] < target['dates'][0]]
        training = [q for p in earlier for q in p['rows']]
        anchor_rows = [q for p in anchors for q in p['rows']]
        base = fit_macro(training)
        prior_rows = [q for q in training if not is_anchor(q, y-1, 'aug')]
        prior = fit_macro(prior_rows)
        if base is None or prior is None:
            unavailable.append({'paper': target['name'], 'reason': 'Insufficient prior history'})
            continue
        regular = blueprint([canonical(q) for q in training], base, cutoff=y)
        candidate = blend_plan(training, prior, anchor_year=y-1, anchor_month='aug')
        anchored = anchor_blueprint(training, candidate)
        # Exploratory sweep: score against this transition, never call it held-out
        # accuracy of the winning weight. An outer fold would need earlier pairs.
        for step in range(21):
            weight = step / 20
            trial_plan = blend_plan(training, prior, weight=weight, anchor_year=y-1, anchor_month='aug')
            predictions = anchor_blueprint(training, trial_plan)
            weight_trials.append({'paper': target['name'], 'august_weight': weight,
                                 'macro': {sec: macro_error(trial_plan, target['rows'], sec) for sec in prior},
                                 'micro': {sec: score([p for p in predictions if p['Section']==sec],
                                                  [q for q in target['rows'] if q['section']==sec]) for sec in prior}})
        result = {'paper': target['name'], 'anchor_papers': [p['name'] for p in anchors],
                  'anchor_questions': len(anchor_rows), 'target_questions': len(target['rows']),
                  'regular_exact': score(regular, target['rows']),
                  'august_70_30_exact': score(anchored, target['rows']),
                  'regular_family': score(regular, target['rows'], families=True),
                  'august_70_30_family': score(anchored, target['rows'], families=True),
                  'topic_changes': changes(anchor_rows, target['rows'])}
        results.append(result)
        print(target['name'], 'regular', result['regular_exact']['covered_questions'],
              'August-anchor', result['august_70_30_exact']['covered_questions'], 'out of 100', flush=True)
    out = ROOT/'output/predictions_2027'
    exploratory = {}
    if weight_trials:
        for sec in ('Verbal Ability','General Awareness','Reasoning','Numerical Ability'):
            curve = []
            for step in range(21):
                weight = step/20
                trials = [r for r in weight_trials if r['august_weight']==weight]
                curve.append({'august_weight':weight,
                              'macro_l1':sum(r['macro'][sec]['l1_count_error'] for r in trials)/len(trials),
                              'micro_coverage':sum(r['micro'][sec]['coverage_percent'] for r in trials)/len(trials),
                              'micro_hit':sum(r['micro'][sec]['hit_percent'] for r in trials)/len(trials)})
            best_macro = min(curve, key=lambda r:(r['macro_l1'],r['august_weight']))
            best_micro = max(curve, key=lambda r:(r['micro_coverage'],-r['august_weight']))
            exploratory[sec] = {'macro_count_optimum':best_macro, 'micro_coverage_optimum':best_micro,
                                'curve':curve, 'status':'Exploratory optimum on the same scored pair; not an out-of-sample accuracy result.'}
    report = {'transition': 'August of Y -> January/February of Y+1', 'reliable_test_pairs': len(results),
              'exploratory_weight_analysis': exploratory,
              'results': results, 'unavailable': unavailable,
              'conclusion': 'Too few reliable transitions to learn or certify a reusable cycle-change rule. The 70/30 blend is a historical configuration, not an estimated optimum.',
              'limitations': ['Anchor sets can be partial memory-based papers.',
                              'Exact micro-label scores and coarser study-family scores must remain separate.',
                              '2023 first-cycle filenames have no confirmed month; they are not silently treated as January.',
                              'Multiple shifts from the same cycle are dependent, not independent training years.']}
    jan_2026 = [q for p in valid if p['dates'][0].year == 2026 and re.search(r'\bjan', p['name'], re.I) for q in p['rows']]
    aug_2026 = [q for p in valid if p['dates'][0].year == 2026 and re.search(r'\baug', p['name'], re.I) for q in p['rows']]
    if jan_2026 and aug_2026:
        report['january_to_august_2026_topic_changes'] = changes(jan_2026, aug_2026)
    (out/'august_to_january_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# August -> next January/February validation', '',
             '| Target | Previous-August questions | Regular exact coverage | August-anchor exact coverage | Regular family coverage | Anchor family coverage |',
             '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['paper']} | {r['anchor_questions']} | {r['regular_exact']['covered_questions']}/100 | {r['august_70_30_exact']['covered_questions']}/100 | {r['regular_family']['covered_questions']}/100 | {r['august_70_30_family']['covered_questions']}/100 |")
        lines += ['', '## Observed macro changes for '+r['paper'], '', '| Subject | Topic | August share | January share | Change (percentage points) |', '|---|---|---:|---:|---:|']
        for sec, data in r['topic_changes'].items():
            for t in sorted(data['topics'], key=lambda v: -abs(v['change_percentage_points']))[:5]:
                lines.append(f"| {sec} | {t['topic']} | {100*t['august_share']:.1f}% | {100*t['next_january_share']:.1f}% | {t['change_percentage_points']:+.1f} |")
    lines += ['', '## Limitations', '', report['conclusion']] + ['- '+s for s in report['limitations']]
    lines += ['', '## Exploratory weight sweep (NOT held-out winning-model accuracy)', '',
              '| Subject | Best August weight for macro count error | Best August weight for micro coverage |',
              '|---|---:|---:|']
    for sec, row in exploratory.items():
        lines.append(f"| {sec} | {100*row['macro_count_optimum']['august_weight']:.0f}% | {100*row['micro_coverage_optimum']['august_weight']:.0f}% |")
    lines += ['', 'Choosing and reporting a winning weight on this one pair is tuning, not validation. Macro-count and micro-coverage objectives may choose different weights. No automatically learned fluctuation correction is deployed from one pair.']
    if report.get('january_to_august_2026_topic_changes') and results:
        lines += ['', '## Direction comparison: macro distribution change', '',
                  'Total variation = half the sum of absolute topic-share differences. This is distribution movement, not accuracy.',
                  '| Subject | Aug 2025 -> Jan 2026 | Jan 2026 -> Aug 2026 |', '|---|---:|---:|']
        first = next((r for r in results if '2026' in r['paper']), None)
        if first:
            for sec, data in report['january_to_august_2026_topic_changes'].items():
                lines.append(f"| {sec} | {100*first['topic_changes'][sec]['total_variation']:.1f}% | {100*data['total_variation']:.1f}% |")
    (out/'AUGUST_TO_JANUARY_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')
    # January 2027 candidate: latest AUGUST is the anchor, January stays in history.
    all_rows = [q for p in valid for q in p['rows']]
    prior = fit_macro([q for q in all_rows if not is_anchor(q, 2026, 'aug')])
    candidate = blend_plan(all_rows, prior, anchor_year=2026, anchor_month='aug')
    (out/'prediction_plan_2027_august_anchor_candidate.json').write_text(json.dumps(candidate, indent=2), encoding='utf-8')
    predictions = anchor_blueprint(all_rows, candidate)
    (out/'micro_2027_august_anchor_candidate.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')
    if exploratory:
        weights = {sec: r['macro_count_optimum']['august_weight'] for sec,r in exploratory.items()}
        diagnostic = blend_plan(all_rows, prior, weight=weights, anchor_year=2026, anchor_month='aug')
        for block in diagnostic.values():
            block['validation_status'] = 'EXPLORATORY: one cross-year transition; do not deploy as a validated improvement.'
        (out/'prediction_plan_2027_exploratory_subject_weights.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    print('Reliable transition pairs:',len(results),'| January 2027 candidate:',len(predictions),'micro slots,',sum(r['Predicted_Questions'] for r in predictions),'questions')


if __name__ == '__main__':
    main()
