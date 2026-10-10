"""Walk-forward evaluation of discrete predictions, not mock-to-plan agreement.
Year coverage counts observed questions, including previously unseen micro-topics.
Multiple shifts in a year are pooled; this is not a single-paper coverage guarantee.
"""
import collections
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'models'))
from dirichlet_forecaster import (SEC_TARGET, MIN_HISTORY, load_counts,
    section_years, share_matrix, tune_params, marginalised_mean)
from topic_normalization_map import TOPIC_NORMALIZATION
from predict_micro_topics_2027 import blueprint, extract_year


def evaluate(predictions, actual, section_totals):
    pred = collections.Counter()
    for r in predictions:
        pred[(r['Section'], r['Broad_Topic'], r['Micro_Topic'])] += r['Predicted_Questions']
    observed = collections.Counter()
    for q in actual:
        observed[(q.get('section'), TOPIC_NORMALIZATION.get(q.get('topic'), q.get('topic')),
                  q.get('micro_topic') or '<unlabelled>')] += 1
    keys = pred.keys() | observed.keys()
    covered = sum(n for key, n in observed.items() if key in pred)
    hits = sum(observed[k] > 0 for k in pred)
    exact = sum(pred[k] == observed[k] for k in pred)
    # Integer forecasts are single-paper quotas; observed yearly totals must be scaled.
    scaled = {k: n / section_totals[k[0]] * SEC_TARGET[k[0]] for k, n in observed.items()}
    return {'predicted_slots': len(pred), 'actual_slots': len(observed),
            'hit_slots': hits, 'hit_percent': 100 * hits / max(len(pred), 1),
            'recall_percent': 100 * hits / max(len(observed), 1),
            'covered_questions': covered, 'observed_questions': len(actual),
            'coverage_percent': 100 * covered / max(len(actual), 1),
            'missed_questions': len(actual) - covered,
            'exact_raw_count_percent': 100 * exact / max(len(pred), 1),
            'normalized_count_mae': float(np.mean([abs(pred[k] - scaled.get(k, 0)) for k in keys])),
            'slots': [{'section': k[0], 'topic': k[1], 'micro_topic': k[2],
                       'predicted': pred[k], 'actual_raw': observed[k],
                       'actual_scaled': scaled.get(k, 0)} for k in sorted(keys)]}


def main():
    qs = json.loads((ROOT / 'data/processed/Q_clean.json').read_text(encoding='utf-8'))
    cnt, totals, _ = load_counts(ROOT / 'data/processed/Q_clean.json')
    years = sorted({extract_year(q.get('file_name', ''), q.get('test_name', '')) for q in qs})
    results = []
    for year in years:
        training = [q for q in qs if extract_year(q.get('file_name', ''), q.get('test_name', '')) < year]
        plan = {}
        for sec in SEC_TARGET:
            history = [y for y in section_years(totals, sec) if y < year]
            if len(history) < MIN_HISTORY:
                continue
            topics = sorted(t for t in cnt[sec] if any(cnt[sec][t][y] for y in history))
            gamma, lam = tune_params(cnt, totals, sec, topics, history)
            S = share_matrix(cnt, totals, sec, history, topics)
            counts = marginalised_mean(S, SEC_TARGET[sec], gamma, lam=lam)
            plan[sec] = {'section_total': SEC_TARGET[sec], 'topics': [
                {'topic': t, 'expected_count_exact': float(n)} for t, n in zip(topics, counts)]}
        if len(plan) != len(SEC_TARGET):
            continue
        predictions = blueprint(training, plan, cutoff=year)
        actual = [q for q in qs if q.get('section') in SEC_TARGET and
                  extract_year(q.get('file_name', ''), q.get('test_name', '')) == year]
        section_totals = collections.Counter(q['section'] for q in actual)
        if not all(section_totals.values()) or len(section_totals) != 4:
            continue
        row = {'year': year, **evaluate(predictions, actual, section_totals)}
        row['sections'] = {sec: evaluate([r for r in predictions if r['Section'] == sec],
                             [q for q in actual if q['section'] == sec], section_totals) for sec in SEC_TARGET}
        results.append(row)
        print(f"{year}: hit {row['hit_percent']:.1f}% | questions covered {row['covered_questions']}/{len(actual)} ({row['coverage_percent']:.1f}%) | slots {row['predicted_slots']}", flush=True)
    covered = sum(r['covered_questions'] for r in results)
    total = sum(r['observed_questions'] for r in results)
    report = {'method': 'pre-year macro refit + shared discrete micro allocation',
              'caveat': 'Pooled observed PYQs, not guaranteed complete papers. Exact raw-count matches across multi-paper years are not single-exam accuracy. Labels may have been assigned retrospectively.',
              'years': results, 'overall': {'covered_questions': covered, 'observed_questions': total,
                  'coverage_percent': 100 * covered / max(total, 1),
                  'hit_percent': 100 * sum(r['hit_slots'] for r in results) / max(sum(r['predicted_slots'] for r in results), 1)}}
    path = ROOT / 'output/predictions_2027/micro_coverage_audit.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Micro-topic coverage audit', '',
             'This is an expanding-window, pre-year test of the current code. It is not a reproduction of the supplied PDF audits.',
             'Every predicted slot has an integer quota. Future years are excluded from evidence and parameter tuning.', '',
             '| Year | Hit precision | Questions covered | Coverage | Predicted slots |',
             '|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['year']} | {r['hit_percent']:.1f}% | {r['covered_questions']}/{r['observed_questions']} | {r['coverage_percent']:.1f}% | {r['predicted_slots']} |")
    lines += ['', '| Subject | Hit precision | Questions covered | Coverage |', '|---|---:|---:|---:|']
    for sec in SEC_TARGET:
        rs = [r['sections'][sec] for r in results]
        c = sum(r['covered_questions'] for r in rs)
        n = sum(r['observed_questions'] for r in rs)
        hits = sum(r['hit_slots'] for r in rs)
        slots = sum(r['predicted_slots'] for r in rs)
        lines.append(f'| {sec} | {100*hits/max(slots,1):.1f}% | {c}/{n} | {100*c/max(n,1):.1f}% |')
    lines += ['', '## Definitions and limitations', '',
              '- Hit precision: predicted nonzero micro-topic slots that appeared, divided by all predicted slots.',
              '- Question coverage: observed questions whose section, normalized broad topic, and exact micro-topic label match a predicted slot. Reading these topics is not a guarantee of answering correctly.',
              '- Recall: distinct observed slots that were predicted, divided by all distinct observed slots.',
              '- Normalized-count MAE: absolute count error over the union of predicted and observed slots, after scaling each observed section to the target exam quota.',
              '- Exact raw-count matches are included in JSON, but are not single-exam exact-count accuracy for years containing multiple papers.',
              '- Missing/new labels count as misses; no test-year micro-topic vocabulary is used in predictions.',
              '- Yearly totals pool papers, shifts, and partial memory-based records. They cannot be presented as coverage of one complete 100-question exam.',
              '- Older weights and topic labels are retrospective choices. This is a code backtest, not a prospective independent evaluation.',
              '- The supplied PDF audits use a particular frozen 2026 blueprint. These results use pre-year refits of the current algorithm; the two protocols and taxonomies need reconciliation before comparing accuracy.',
              '- A macro allocation is fitted solely to obtain pre-year quotas; the production macro model and 2027 plan were not changed.',
              '- No nominal calibration confidence band is claimed: folds and topics are dependent.', '']
    (ROOT / 'output/predictions_2027/MICRO_COVERAGE_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('OVERALL:', report['overall'])


if __name__ == '__main__':
    main()
