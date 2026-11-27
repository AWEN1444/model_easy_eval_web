import copy
import unittest
from comparison import compare_runs
from datasets import validate_suite


def make_run(passes=(True, True), state='finished', evaluator='exact'):
    suite = validate_suite({'id': 'suite', 'name': 'Suite', 'version': '1', 'questions': [
        {'id': 'a', 'category': 'math', 'question': '1+1', 'evaluator': evaluator, 'expected': '2'},
        {'id': 'b', 'category': 'logic', 'question': '2+2', 'evaluator': evaluator, 'expected': '4'},
    ]})
    results = [dict(question=q, status='success', passed=passed, score=100 if passed else 0, answer={'text': 'answer'})
               for q, passed in zip(suite['questions'], passes)]
    return {'id': 'run', 'protocol_version': '2', 'state': state, 'suite': suite,
            'model': {'id': 'v1', 'model': 'served-name', 'checkpoint': '/v1', 'temperature': 0, 'max_tokens': 4096},
            'judge': {'id': 'j1', 'model': 'judge-model', 'checkpoint': 'fixed-v1'}, 'rubric': 'fixed', 'results': results}


class ComparisonTest(unittest.TestCase):
    def test_different_checkpoint_same_model_name_is_comparable(self):
        old = make_run()
        new = copy.deepcopy(old)
        new['model'].update(id='v2', checkpoint='/v2', base_url='http://different')
        result = compare_runs(new, old)
        self.assertTrue(result['compatible'])
        self.assertEqual(result['verdict'], 'pass')

    def test_paired_alignment_uses_ids_not_completion_order(self):
        old = make_run((True, False))
        new = make_run((False, True))
        new['results'].reverse()
        result = compare_runs(new, old, {'min_pass_rate': 0})
        self.assertEqual(result['counts'], {'improved': 1, 'regressed': 1, 'unchanged': 0, 'unresolved': 0})
        self.assertEqual(result['regressed'][0]['id'], 'a')
        self.assertEqual(result['improved'][0]['id'], 'b')
        self.assertEqual(result['verdict'], 'fail')

    def test_failure_cannot_shrink_denominator_or_pass_gate(self):
        old = make_run()
        new = make_run()
        new['results'][1] = {'question': new['suite']['questions'][1], 'status': 'judge_error', 'error': 'timeout'}
        result = compare_runs(new, old)
        self.assertEqual(result['summary']['candidate']['pass_rate'], .5)
        self.assertEqual(result['summary']['candidate']['graded_pass_rate'], 1)
        self.assertEqual(result['counts']['unresolved'], 1)
        self.assertEqual(result['verdict'], 'insufficient')

    def test_missing_cancelled_or_running_runs_are_insufficient(self):
        for state in ('cancelled', 'running', 'interrupted'):
            self.assertEqual(compare_runs(make_run(state=state), make_run())['verdict'], 'insufficient')
        incomplete = make_run()
        incomplete['results'].pop()
        self.assertEqual(compare_runs(incomplete, make_run())['verdict'], 'insufficient')

    def test_protocol_mismatch_explained(self):
        for mutate in (
            lambda r: r['suite'].update(version='2'),
            lambda r: r['suite']['questions'][0].update(expected='different'),
            lambda r: r['model'].update(temperature=.5),
            lambda r: r.update(protocol_version='3'),
        ):
            changed = make_run()
            mutate(changed)
            result = compare_runs(changed, make_run())
            self.assertFalse(result['compatible'])
            self.assertEqual(result['verdict'], 'insufficient')
            self.assertTrue(result['reasons'])

    def test_judge_ignored_for_deterministic_suite(self):
        old, new = make_run(), make_run()
        new['judge'] = None
        new['rubric'] = 'different unused rubric'
        self.assertTrue(compare_runs(new, old)['compatible'])

    def test_judge_version_and_prompt_must_match_when_used(self):
        for change in ('judge', 'rubric'):
            old, new = make_run(evaluator='judge'), make_run(evaluator='judge')
            if change == 'judge':
                new['judge']['checkpoint'] = 'new-judge'
            else:
                new['rubric'] = 'new prompt'
            self.assertFalse(compare_runs(new, old)['compatible'])

    def test_duplicate_unknown_or_legacy_result_rejected(self):
        old = make_run()
        duplicate = make_run()
        duplicate['results'].append(duplicate['results'][0])
        self.assertFalse(compare_runs(duplicate, old)['compatible'])
        unknown = make_run()
        unknown['results'][0]['question'] = {'id': 'unknown'}
        self.assertFalse(compare_runs(unknown, old)['compatible'])
        self.assertEqual(compare_runs({'results': []}, old)['verdict'], 'insufficient')

    def test_critical_failure_blocks_even_permissive_thresholds(self):
        old, new = make_run(), make_run((False, True))
        for run in (old, new):
            run['suite']['questions'][0]['critical'] = True
        result = compare_runs(new, old, {key: value for key, value in [
            ('min_pass_rate', 0), ('max_regression_rate', 1), ('max_category_drop', 1), ('max_pass_rate_drop', 1)]})
        self.assertEqual(result['verdict'], 'fail')
        self.assertIn('存在未通过的关键题', result['explanations'])

    def test_score_changes_visible_without_false_pass_transition(self):
        old, new = make_run(), make_run()
        old['results'][0]['score'] = 81
        new['results'][0]['score'] = 99
        result = compare_runs(new, old)
        self.assertEqual(result['counts']['improved'], 0)
        self.assertEqual(result['unchanged'][0]['score_delta'], 18)

    def test_invalid_threshold_rejected(self):
        for invalid in ({'min_pass_rate': 80}, {'min_pass_rate': True}, {'misspelled': .5}):
            with self.assertRaises(ValueError):
                compare_runs(make_run(), make_run(), invalid)


if __name__ == '__main__':
    unittest.main()
