import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_shift_coverage import dates, paper_groups, score
from regime_anchor_2026 import blend_plan, anchor_blueprint, blended_micro_probs
from regime_anchor_2026 import is_anchor
from predict_micro_topics_2027 import blueprint
from build_unified_2027 import validate_paper


class ShiftValidationTests(unittest.TestCase):
    def test_deploy_uses_unified_compiler_and_versioned_frontend(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root/'.github/workflows/deploy-dashboard.yml').read_text(encoding='utf-8')
        self.assertIn('python scripts/compile_dashboard_2027.py',workflow)
        self.assertNotIn('python scripts/generate_complete_data.py',workflow)
        for relative in ('dashboard/index.html','output/predictions_2026/index.html'):
            page = (root/relative).read_text(encoding='utf-8')
            self.assertIn('data.js?v=2027-integer-pyq-v2',page)
            self.assertIn('Number.isInteger(r.Predicted_Questions)',page)
    def test_live_blueprint_and_paper_match(self):
        import json
        root = Path(__file__).resolve().parents[1]
        qs = json.loads((root/'data/processed/Q_clean.json').read_text(encoding='utf-8'))
        plan = json.loads((root/'output/predictions_2027/prediction_plan_2027.json').read_text(encoding='utf-8'))
        rows = blueprint(qs,plan)
        paper = json.loads((root/'output/generated_questions/practice_2027_sota.json').read_text(encoding='utf-8'))
        validate_paper(rows,paper)
        self.assertLessEqual(len(rows),100)
        self.assertTrue(all(r['Predicted_Questions'] > 0 for r in rows))
        self.assertTrue(all(b.get('calibration_excluded_years') == [2026] for b in plan.values()))

    def test_dashboard_artifacts_match_paper_and_blueprint(self):
        import json
        from collections import Counter
        root = Path(__file__).resolve().parents[1]
        text = (root/'dashboard/data.js').read_text(encoding='utf-8')
        for relative in ('output/predictions_2027/data.js','output/predictions_2026/data.js'):
            self.assertEqual(text,(root/relative).read_text(encoding='utf-8'))
        data = json.loads(text.split('const dashboardData = ',1)[1].split(';\n\n',1)[0])
        rows = data['predictions']['micro_predictions']
        validate_paper(rows,data['mock_test']['all_questions'])
        micro_macro = Counter()
        for r in rows:
            micro_macro[(r['Section'],r['Broad_Topic'])] += r['Predicted_Questions']
        macro = Counter({(r['Section'],r['Topic']):r['Predicted_Questions'] for r in data['predictions']['ml_predictions']})
        self.assertEqual(macro,micro_macro)
        original = json.loads((root/'data/processed/Q_clean.json').read_text(encoding='utf-8'))
        self.assertEqual(original,data['question_bank'])

    def test_mock_is_original_pyq_not_generated_text(self):
        import json
        root = Path(__file__).resolve().parents[1]
        original = json.loads((root/'data/processed/Q_clean.json').read_text(encoding='utf-8'))
        paper = json.loads((root/'output/generated_questions/practice_2027_sota.json').read_text(encoding='utf-8'))
        for q in paper:
            matches = [r for r in original if r.get('file_name') == q['file_name']
                       and r.get('question_number') == q['original_question_number']
                       and r['question_text'] == q['question_text'] and r['choices'] == q['choices']]
            self.assertTrue(matches)
            self.assertEqual(q['source'],'pyq-curated-original')
            self.assertTrue(any(r.get('passage') == q.get('passage') and r.get('image_path') == q.get('image_path') for r in matches))
    def test_previous_august_anchor_excludes_january_same_year(self):
        jan = {'test_name': 'AFCAT 2026 Jan 31', 'file_name': '2026.pdf'}
        aug = {'test_name': 'AFCAT 2026 Aug Shift 1', 'file_name': '2026.pdf'}
        self.assertFalse(is_anchor(jan, 2026, 'aug'))
        self.assertTrue(is_anchor(aug, 2026, 'aug'))
    def test_uncertain_month_never_orders_same_month_shifts(self):
        one = dates('AFCAT 2026 Aug Shift 1')
        two = dates('AFCAT 2026 Aug Shift 2')
        self.assertFalse(one[1] < two[0])
        self.assertLess(dates('AFCAT 2026 Jan 31 Shift 1')[1], one[0])

    def test_separate_old_official_papers_and_repeated_stems(self):
        qs = [{'test_name': 'AFCAT 2012', 'file_name': f'paper{p}_2012.pdf',
               'question_number': n, 'section': 'Reasoning', 'topic': 'Verbal Analogy',
               'question_text': 'Same figure instruction', 'choices': []}
              for p in (1, 2) for n in range(1, 101)]
        groups = paper_groups(qs)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(g['complete'] for g in groups))

    def test_anchor_blend_is_70_30_and_sections_sum(self):
        plan = {'General Awareness': {'section_total': 25, 'topics': [
            {'topic': 'Sports', 'expected_count_exact': 25}]}}
        qs = [{'section': 'General Awareness', 'topic': 'Defence', 'micro_topic': 'Air Force Asset',
               'test_name': 'AFCAT 2026 Jan 31', 'file_name': '2026.pdf'} for _ in range(25)]
        candidate = blend_plan(qs, plan)
        counts = {r['topic']: r['expected_count_exact'] for r in candidate['General Awareness']['topics']}
        self.assertAlmostEqual(counts['Defence'], 17.5)
        self.assertAlmostEqual(counts['Sports'], 7.5)
        predicted = blueprint(qs, candidate)
        self.assertEqual(sum(r['Predicted_Questions'] for r in predicted), 25)
        self.assertEqual(predicted, anchor_blueprint(qs, candidate))

    def test_micro_subjects_never_mix(self):
        qs = [{'section': 'Reasoning', 'topic': 'Verbal Analogy', 'micro_topic': 'Reasoning only',
               'test_name': 'AFCAT 2026 Jan 31', 'file_name': '2026.pdf'}]
        probs = blended_micro_probs(qs, 'Verbal Ability', 'Verbal Analogy')
        self.assertNotIn('Reasoning only', probs)

    def test_subject_weights_are_supported_and_bounded(self):
        prior = {'Reasoning': {'section_total': 25, 'topics': [{'topic':'Verbal Analogy', 'expected_count_exact':25}]}}
        rows = [{'section':'Reasoning', 'topic':'Coding-Decoding', 'micro_topic':'Number Coding',
                 'file_name':'2026.pdf', 'test_name':'AFCAT 2026 Aug Shift 1'}]
        plan = blend_plan(rows, prior, weight={'Reasoning':0.05}, anchor_month='aug')
        self.assertEqual(plan['Reasoning']['anchor_weight'], 0.05)
        with self.assertRaises(ValueError):
            blend_plan(rows, prior, weight={'Reasoning':1.1})

    def test_coverage_counts_questions_not_quotas(self):
        pred = [{'Section': 'Reasoning', 'Broad_Topic': 'Verbal Analogy',
                 'Micro_Topic': 'Word Pair Analogy', 'Predicted_Questions': 1}]
        actual = [{'section': 'Reasoning', 'topic': 'Verbal Analogy',
                   'micro_topic': 'Word Pair Analogy'} for _ in range(3)]
        result = score(pred, actual)
        self.assertEqual(result['covered_questions'], 3)
        self.assertEqual(result['exact_count_percent'], 0)


if __name__ == '__main__':
    unittest.main()
