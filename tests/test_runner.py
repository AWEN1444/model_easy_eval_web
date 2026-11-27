import threading
import unittest
from unittest.mock import patch

from runner import run_evaluation, sanitize_config, summarize_results, summarize_run


def suite():
    return {'id': 'suite', 'name': 'Suite', 'version': '1', 'questions': [
        {'id': 'one', 'question': '1+1?', 'category': 'math', 'evaluator': 'exact', 'expected': '2'},
        {'id': 'two', 'question': '2+2?', 'category': 'math', 'evaluator': 'exact', 'expected': '4'},
        {'id': 'three', 'question': 'format?', 'category': 'format', 'evaluator': 'json', 'expected': {'key': 1}},
    ]}


class RunnerTest(unittest.TestCase):
    def test_summary_does_not_hide_failures(self):
        results = [
            {'question': {'category': 'a'}, 'status': 'success', 'score': 100, 'passed': True},
            {'question': {'category': 'a'}, 'status': 'success', 'score': 0, 'passed': False},
            {'question': {'category': 'a'}, 'status': 'generation_error'},
        ]
        result = summarize_results(results, total=4)
        self.assertEqual(result['total'], 4)
        self.assertEqual(result['completed'], 2)
        self.assertEqual(result['pass_rate'], .25)
        self.assertEqual(result['graded_pass_rate'], .5)
        self.assertEqual(result['errors'], 1)
        self.assertEqual(result['completion_rate'], .5)
        self.assertEqual(result['mean_score'], 50)

    @patch('engine.complete')
    def test_snapshots_and_progress_are_persistable(self, complete_mock):
        complete_mock.side_effect = [{'text': '2'}, {'text': 'wrong'}, {'text': '{"key":1}'}]
        updates = []
        model = {'id': 'checkpoint-10', 'name': 'Training 10', 'model': 'shared-api-name',
                 'key': 'SECRET', 'extra': {'top_p': .9}, 'checkpoint': '/models/10'}
        result = run_evaluation(suite(), model, workers=1, on_update=updates.append, metadata={'name': 'My experiment'})
        self.assertEqual(result['state'], 'finished')
        self.assertEqual(result['name'], 'My experiment')
        self.assertEqual(result['protocol_version'], '2')
        self.assertEqual(result['summary']['passed'], 2)
        self.assertEqual(result['summary']['failed'], 1)
        self.assertEqual(result['summary']['pass_rate'], 2 / 3)
        self.assertEqual(len(updates), 5)
        self.assertEqual(updates[0]['results'], [])
        self.assertEqual(result['suite']['questions'][2]['expected'], {'key': 1})
        self.assertNotIn('key', result['model'])
        self.assertNotIn('SECRET', str(result))
        self.assertEqual(model['key'], 'SECRET')

    def test_sanitize_config_nested_credentials(self):
        value = {'key': 'secret', 'api_key_env': 'MODEL_API_KEY',
                 'extra': {'password': 'secret', 'x-api-key': 'secret', 'provider_token': 'secret', 'top_p': .9},
                 'base_url': 'https://user:secret@example.com/v1?api_key=secret&client-secret=secret&provider-token=secret&region=eu#secret'}
        clean = sanitize_config(value)
        self.assertNotIn('secret', str(clean))
        self.assertEqual(clean['api_key_env'], 'MODEL_API_KEY')
        self.assertEqual(clean['base_url'], 'https://example.com/v1?region=eu')

    @patch('engine.complete')
    def test_cancellation_before_run_marks_all_cases(self, complete_mock):
        event = threading.Event()
        event.set()
        result = run_evaluation(suite(), {'id': 'm', 'model': 'm'}, cancel_event=event)
        self.assertEqual(result['state'], 'cancelled')
        self.assertEqual(result['summary']['cancelled'], 3)
        self.assertEqual(result['summary']['completion_rate'], 0)
        self.assertEqual(result['summary']['categories']['math']['total'], 2)
        complete_mock.assert_not_called()

    @patch('engine.complete', return_value={'text': '2'})
    def test_cancellation_keeps_full_denominators(self, complete_mock):
        event = threading.Event()
        def update(run):
            if run['results']:
                event.set()
        result = run_evaluation(suite(), {'id': 'm', 'model': 'm'}, workers=1,
                                cancel_event=event, on_update=update)
        self.assertEqual(result['state'], 'cancelled')
        self.assertEqual(result['summary']['total'], 3)
        self.assertEqual(len(result['results']), 3)
        self.assertEqual(result['summary']['categories']['math']['total'], 2)

    def test_partial_summary_uses_planned_category_denominators(self):
        run = {'suite': suite(), 'results': [
            {'question': suite()['questions'][0], 'status': 'success', 'score': 100, 'passed': True}]}
        summary = summarize_run(run)
        self.assertEqual(summary['pass_rate'], 1 / 3)
        self.assertEqual(summary['categories']['math']['pass_rate'], .5)
        self.assertEqual(summary['categories']['format']['total'], 1)
        self.assertEqual(summary['categories']['format']['completion_rate'], 0)

    @patch('engine.complete')
    def test_unknown_evaluator_fails_before_any_request(self, complete_mock):
        value = suite()
        value['questions'][1]['evaluator'] = 'never_registered'
        with self.assertRaisesRegex(ValueError, '未注册评测器'):
            run_evaluation(value, {'id': 'm', 'model': 'm'})
        complete_mock.assert_not_called()

    @patch('engine.complete')
    def test_missing_judge_fails_before_any_request(self, complete_mock):
        value = suite()
        value['questions'][1]['evaluator'] = 'judge'
        with self.assertRaisesRegex(ValueError, '配置评委模型'):
            run_evaluation(value, {'id': 'm', 'model': 'm'})
        complete_mock.assert_not_called()

    def test_duplicate_question_ids_rejected(self):
        value = suite()
        value['questions'][1]['id'] = 'one'
        with self.assertRaises(ValueError):
            run_evaluation(value, {'model': 'm'})


if __name__ == '__main__':
    unittest.main()
