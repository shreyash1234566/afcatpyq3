"""One blueprint for forecasting, practice selection and dashboard compilation.

Weights are provisional averages of available transition optima, not held-out accuracy.
Practice questions are preserved originals, never numeric mutations with stale answers.
"""
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path
from audit_shift_coverage import fit_macro, paper_groups
from regime_anchor_2026 import blend_plan, is_anchor, normalized
from predict_micro_topics_2027 import blueprint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/predictions_2027'

def allocation(rows):
    return Counter({(r['Section'], r['Broad_Topic'], r['Micro_Topic']): r['Predicted_Questions'] for r in rows})

def validate_paper(rows, paper):
    actual = Counter((q['section'], q['topic'], q['micro_topic']) for q in paper)
    if actual != allocation(rows):
        raise ValueError(f'Blueprint mismatch: missing={allocation(rows)-actual}, extra={actual-allocation(rows)}')
    if len(paper) != 100 or {q['question_number'] for q in paper} != set(range(1,101)):
        raise ValueError('Paper must contain exactly 100 uniquely numbered questions')
    for q in paper:
        choices = q.get('choices', [])
        keys = {c.get('key') for c in choices if isinstance(c,dict) and c.get('text')}
        if keys != {'A','B','C','D'} or q.get('correct_answer') not in keys:
            raise ValueError('Missing choices/answer for question '+str(q['question_number']))
        if q.get('has_figure') and not (q.get('image_path') or q.get('figure_path')):
            raise ValueError('Missing figure for question '+str(q['question_number']))

def select_paper(rows, candidates, seed=202701):
    rng = random.Random(seed)
    used, paper, missing = set(), [], []
    for row in rows:
        key = (row['Section'],row['Broad_Topic'],row['Micro_Topic'])
        pool = [normalized(q) for q in candidates if (q.get('section'),normalized(q).get('topic'),q.get('micro_topic')) == key]
        rng.shuffle(pool)
        pool.sort(key=lambda q:int(re.search(r'20\d{2}',q.get('file_name','')).group()) if re.search(r'20\d{2}',q.get('file_name','')) else 0,reverse=True)
        for _ in range(row['Predicted_Questions']):
            chosen = None
            for q in pool:
                identity = (q.get('question_text'),json.dumps(q.get('choices'),sort_keys=True),q.get('passage'),q.get('image_path'))
                if identity in used or q.get('correct_answer',q.get('answer')) not in ('A','B','C','D'):
                    continue
                if len(q.get('choices',[])) != 4 or (q.get('has_figure') and not q.get('image_path')):
                    continue
                used.add(identity)
                chosen = copy.deepcopy(q)
                chosen['correct_answer'] = q.get('correct_answer',q.get('answer'))
                chosen['original_test_name'] = q.get('original_test_name',q.get('test_name'))
                chosen['original_question_number'] = q.get('question_number')
                chosen['test_name'] = 'AFCAT 01/2027 Blueprint Practice (curated)'
                chosen['question_number'] = len(paper)+1
                chosen['source'] = 'pyq-curated-original'
                paper.append(chosen)
                break
            if chosen is None:
                missing.append(key)
    if missing:
        raise ValueError('Cannot safely fill answered, exact-label slots: '+repr(Counter(missing)))
    validate_paper(rows,paper)
    return paper

def main():
    qs = json.loads((ROOT/'data/processed/Q_clean.json').read_text(encoding='utf-8'))
    report = json.loads((OUT/'cross_year_sensitivity.json').read_text(encoding='utf-8'))
    weights = {s:r['average_pair_optimum_without_2026'] for s,r in report['weight_sensitivity'].items()}
    usable = [q for g in paper_groups(qs) if not g['invalid'] for q in g['rows']]
    prior = fit_macro([q for q in usable if not is_anchor(q,2026,'aug')])
    plan = blend_plan(usable,prior,weight=weights,anchor_year=2026,anchor_month='aug')
    for block in plan.values():
        block['validation_status'] = 'Provisional average of pre-2026 transition optima; 2026 excluded by request, older cycle dates missing. Not validated forecast accuracy.'
        block['calibration_excluded_years'] = [2026]
    rows = blueprint(usable,plan)
    # Always save candidate, but never replace live artifacts with an incomplete paper.
    (OUT/'prediction_plan_2027_unified_candidate.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    # Supplemental solved keys are separate from untouched source data.
    solved = {
        ('AFCAT_2012_Official_Paper1.pdf',29):('D','Jim Corbett National Park is in Uttarakhand, formerly called Uttaranchal.'),
        ('AFCAT_2016_Official_Paper2.pdf',15):('A','Ascendancy is the correctly spelt word.'),
        ('AFCAT_2016_Official_Paper2.pdf',16):('B','Hegemony is the correctly spelt word.'),
        ('AFCAT_2016_Official_Paper2.pdf',17):('C','Perpetuate is the correctly spelt word.'),
        ('AFCAT_2016_Official_Paper2.pdf',18):('C','Apocalyptic is the correctly spelt word.'),
        ('AFCAT_2025_Aug23_Shift2_Memory.pdf',75):('C','A connoisseur is an expert judge of art and craftsmanship.'),
        ('AFCAT_2025_Aug23_Shift2_Memory.pdf',77):('D','Ingenious means exceptionally clever or inventive.'),
        ('AFCAT_2025_Aug23_Shift2_Memory.pdf',37):('D','Stomata are leaf pores that permit gas exchange.'),
        ('AFCAT_2025_Aug24_Shift1_Memory.pdf',1):('B','Total distance = 1265 km; time = 375/50 + 315/70 + 575/57.5 = 22 h; average = 57.5 km/h.'),
        ('AFCAT_2025_Aug24_Shift1_Memory.pdf',17):('C','The month has 5 Sundays. Average = (5*410 + 25*230)/30 = 260.'),
        ('AFCAT_2025_Aug24_Shift1_Memory.pdf',16):('A','Circumference ratio equals radius ratio 7:6. Difference = 45 m, so inner radius = 6*45 = 270 m.'),
        ('AFCAT_2025_Aug24_Shift1_Memory.pdf',20):('B','Downstream = 1.5*18 = 27 km/h; stream speed = (27-18)/2 = 4.5 km/h.')}
    candidates = []
    for q in usable:
        record = dict(q)
        key = (q.get('file_name'),q.get('question_number'))
        if key in solved:
            record['correct_answer'],record['explanation'] = solved[key]
            record['answer_provenance'] = 'Supplemental worked solution, not an official answer key'
        candidates.append(record)
    paper = select_paper(rows,candidates)
    for path, value in [(OUT/'prediction_plan_2027.json',plan), (OUT/'afcat_2027_micro_predictions.json',rows),
                        (ROOT/'output/generated_questions/practice_2027_sota.json',paper)]:
        path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Shared blueprint:',len(rows),'micro-topics, 100 questions; weights:',weights)

if __name__ == '__main__':
    main()
