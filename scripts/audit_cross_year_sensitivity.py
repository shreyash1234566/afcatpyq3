"""Provisional 2023-2026 cross-year sensitivity analysis, separated from confirmed audits.

The 2023 cycle dates and first 100 January-2025 rows are explicitly provisional;
neither is silently promoted to verified ground truth or written into Q_clean.
"""
import collections
import json
import re
from pathlib import Path
from audit_shift_coverage import ROOT, canonical, fit_macro, score
from audit_august_to_january import changes, macro_error
from regime_anchor_2026 import blend_plan, anchor_blueprint


def year(q):
    m = re.search(r'20\d{2}', q.get('file_name', ''))
    return int(m.group()) if m else None


def main():
    qs = json.loads((ROOT/'data/processed/Q_clean.json').read_text(encoding='utf-8'))
    jan25 = [q for q in qs if '2025_Jan' in q.get('file_name','')]
    block25 = jan25[:100]
    assert [q['question_number'] for q in block25] == list(range(1,101))
    # Audit-local recovery only; do not overwrite source questions/images.
    repaired = [q for q in qs if '2025_Jan' not in q.get('file_name','')] + block25
    pairs = [(2023, [q for q in repaired if year(q)==2022 and 'Aug' in q['file_name']],
                    [q for q in repaired if q['file_name']=='AFCAT_2023_01_Memory.pdf'],
                    '2023 cycle 01 is inferred from filename, not confirmed month'),
             (2024, [q for q in repaired if q['file_name']=='AFCAT_2023_02_Memory.pdf'],
                    [q for q in repaired if '2024 Feb 16' in q.get('test_name','')],
                    '2023 cycle 02 is inferred; February-16 paper is assembled from fragments'),
             (2025, [q for q in repaired if year(q)==2024 and 'Aug' in q['file_name']], block25,
                    'January target is first ordered 1-100 block from a 229-row merged file; August GA missing'),
             (2026, [q for q in repaired if year(q)==2025 and 'Aug' in q['file_name']],
                    [q for q in repaired if '2026 Jan' in q.get('test_name','')],
                    'January target complete; August memory-based sources partial')]
    results = []
    curves_by_section = collections.defaultdict(list)
    for y, anchor, target, caveat in pairs:
        if not anchor or not target:
            continue
        anchor_ids = {id(q) for q in anchor}
        training = [q for q in repaired if year(q) and year(q)<y and q.get('section') in
                    ('Verbal Ability','General Awareness','Reasoning','Numerical Ability')]
        prior = fit_macro([q for q in training if id(q) not in anchor_ids])
        if prior is None:
            continue
        section_counts = collections.Counter(q['section'] for q in anchor)
        movement = changes(anchor, target)
        for sec, value in movement.items():
            if not section_counts[sec]:
                value['total_variation'] = None
                value['topics'] = []
        curves = {}
        for sec in prior:
            if not section_counts[sec]:
                continue
            trials = []
            for step in range(21):
                w = step/20
                # Explicit anchor records allow filename-inferred cycles without pretending exact dates.
                staged = [dict(q, test_name=f'AFCAT {y-1} Aug inferred-cycle') if id(q) in anchor_ids else q for q in training]
                plan = blend_plan(staged, prior, weight=w, anchor_year=y-1, anchor_month='aug')
                error = macro_error(plan, target, sec)
                trials.append({'august_weight':w, **error})
            curves[sec] = trials
            curves_by_section[sec].append({'year':y,'trials':trials})
        # Outer-style diagnostic: weights selected only from earlier transitions.
        weights = {}
        for sec in prior:
            past = [c for c in curves_by_section[sec] if c['year']<y]
            if past and section_counts[sec]:
                index = min(range(21), key=lambda i:(sum(c['trials'][i]['l1_count_error'] for c in past), i))
                weights[sec] = index/20
            else:
                weights[sec] = 0.0
        staged = [dict(q, test_name=f'AFCAT {y-1} Aug inferred-cycle') if id(q) in anchor_ids else q for q in training]
        selected = blend_plan(staged, prior, weight=weights, anchor_year=y-1, anchor_month='aug')
        predicted = anchor_blueprint(staged, selected)
        row = {'target_year':y,'data_status':'PROVISIONAL sensitivity analysis','caveat':caveat,
               'anchor_questions':len(anchor),'target_questions':len(target),
               'anchor_subject_counts':dict(section_counts), 'topic_changes':movement,
               'weight_curves':curves, 'weights_from_earlier_transitions':weights,
               'micro_exact':score(predicted,target), 'micro_family':score(predicted,target,families=True)}
        results.append(row)
        print(y,'topic movement', {s:round(v['total_variation']*100,1) if v['total_variation'] is not None else None for s,v in movement.items()},flush=True)
    sensitivity = {}
    for sec, curves in curves_by_section.items():
        earlier = [c for c in curves if c['year']<2026]
        if earlier:
            optimum = min(range(21),key=lambda i:(sum(c['trials'][i]['l1_count_error'] for c in earlier),i))
            all_optimum = min(range(21),key=lambda i:(sum(c['trials'][i]['l1_count_error'] for c in curves),i))
            sensitivity[sec] = {'earlier_pair_count':len(earlier),'weight_without_2026':optimum/20,
                                'weight_with_2026':all_optimum/20,
                                'average_pair_optimum_without_2026':sum(min(c['trials'], key=lambda t:(t['l1_count_error'],t['august_weight']))['august_weight'] for c in earlier)/len(earlier),
                                'average_pair_optimum':sum(min(c['trials'], key=lambda t:(t['l1_count_error'],t['august_weight']))['august_weight'] for c in curves)/len(curves),
                                'pair_count':len(curves)}
    report = {'status':'PROVISIONAL. Do not replace confirmed audit or publish these as validated accuracy.',
              'results':results, 'weight_sensitivity':sensitivity,
              'older_year_review': [{'target_year':y, 'status':'Unavailable for cycle calibration: source month/cycle not established',
                                    'files':sorted({q['file_name'] for q in qs if year(q) in (y-1,y)})} for y in range(2016,2023)],
              'findings':['2026 may be unusual for particular subjects, not automatically for the whole exam.',
                          'The 2025 and 2023/2024 inputs need source-paper verification before weight calibration is trusted.',
                          'Missing GA in August 2024 is missing data, not zero probability.',
                          'No rows in Q_clean.json were edited or removed; all recovery is local to this sensitivity analysis.']}
    out = ROOT/'output/predictions_2027'
    (out/'cross_year_sensitivity.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines = ['# Provisional August -> first-cycle comparison, 2023-2026', '', report['status'], '',
             '| Target year | Verbal movement | GA movement | Reasoning movement | Numerical movement |',
             '|---|---:|---:|---:|---:|']
    for r in results:
        vals = [r['topic_changes'][s]['total_variation'] for s in ('Verbal Ability','General Awareness','Reasoning','Numerical Ability')]
        lines.append('| '+str(r['target_year'])+' | '+' | '.join(f'{100*v:.1f}%' if v is not None else 'Missing data' for v in vals)+' |')
    lines += ['', '## Weight sensitivity (exploratory macro-count objective)', '',
              '| Subject | Earlier transitions available | Best August weight excluding 2026 | Including 2026 |', '|---|---:|---:|---:|']
    for sec,r in sensitivity.items():
        lines.append(f"| {sec} | {r['earlier_pair_count']} | {100*r['weight_without_2026']:.0f}% | {100*r['weight_with_2026']:.0f}% |")
    lines += ['', '## Input caveats', '']+[f"- {r['target_year']}: {r['caveat']}" for r in results]
    lines += ['', '## Review back to 2016', '',
              '2016–2022 transitions cannot be established from the current year-only source labels. They are excluded from cycle averages, not treated as zero change.',
              '', '## Production average excluding 2026', '',
              '2026 is excluded from calibration at the user’s request, not statistically proven to be an outlier. August 2026 remains the latest forecast anchor.']
    for sec,r in sensitivity.items():
        lines.append(f"- {sec}: {100*r['average_pair_optimum_without_2026']:.1f}% August weight, based on {r['earlier_pair_count']} provisional transitions.")
    lines += ['', '## Interpretation', '']+['- '+s for s in report['findings']]
    (out/'CROSS_YEAR_SENSITIVITY.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Exploratory weight sensitivity:',sensitivity)


if __name__ == '__main__':
    main()
