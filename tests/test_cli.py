import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli import main
from datasets import get_builtin_suite
from storage import Store


class CliIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.db = self.path / 'eval.sqlite3'
        self.answers = {}
        for question in get_builtin_suite('general-quick')['questions']:
            expected = question['expected']
            self.answers[question['question']] = json.dumps(expected, ensure_ascii=False) if question['evaluator'] == 'json' else expected
        self.config = self.path / 'model.json'
        self.config.write_text(json.dumps({'id': 'v1', 'name': 'base', 'role': 'base', 'checkpoint': 'weights/v1',
                                          'model': 'same-api-model', 'base_url': 'http://localhost:9000/v1',
                                          'key': 'test-inline-secret'}))

    def tearDown(self):
        self.temp.cleanup()

    def call(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(['--db', str(self.db), *args])
        return code, out.getvalue(), err.getvalue()

    def complete(self, config, messages):
        return {'text': self.answers[messages[-1]['content']], 'seconds': .01, 'usage': {'total_tokens': 10}, 'finish_reason': 'stop'}

    def record_baseline(self):
        code, out, err = self.call('run', '--model-config', str(self.config), '--suite', 'general-quick',
                                   '--record-baseline', '--output', str(self.path / 'baseline'))
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertEqual(result['verdict'], 'baseline_recorded')
        self.assertEqual(result['summary']['pass_rate'], 1)
        return result['run_id']

    def test_train_baseline_candidate_compare_export(self):
        with patch('engine.complete', side_effect=self.complete):
            baseline_id = self.record_baseline()
            config = json.loads(self.config.read_text())
            config.update(id='v2', name='candidate', role='candidate', checkpoint='weights/v2')
            self.config.write_text(json.dumps(config))
            code, out, err = self.call('run', '--model-config', str(self.config), '--baseline-policy', 'base',
                                       '--output', str(self.path / 'candidate'))
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertEqual(result['verdict'], 'pass')
        report = json.loads((self.path / 'candidate' / 'report.json').read_text())
        self.assertEqual(report['comparisons'][0]['baseline_run_id'], baseline_id)
        self.assertEqual(report['comparisons'][0]['counts']['unchanged'], 20)
        self.assertNotIn('test-inline-secret', json.dumps(report))
        self.assertTrue((self.path / 'candidate' / 'results.csv').exists())
        self.assertTrue((self.path / 'candidate' / 'report.md').exists())
        store = Store(self.db)
        self.assertEqual(len(store.list_runs()), 2)
        self.assertEqual(len(store.list_models()), 2)

    def test_regression_fails_and_protocol_change_is_insufficient(self):
        with patch('engine.complete', side_effect=self.complete):
            baseline_id = self.record_baseline()
        with patch('engine.complete', return_value={'text': 'WRONG', 'seconds': .01}):
            code, out, err = self.call('run', '--model', 'v1', '--baseline', baseline_id,
                                       '--output', str(self.path / 'failed'))
        self.assertEqual(code, 2, err)
        self.assertEqual(json.loads(out)['verdict'], 'fail')
        config = json.loads(self.config.read_text())
        config.update(id='v3', system='changed evaluation condition')
        self.config.write_text(json.dumps(config))
        with patch('engine.complete', side_effect=self.complete):
            code, out, err = self.call('run', '--model-config', str(self.config), '--baseline', baseline_id,
                                       '--output', str(self.path / 'mismatch'))
        self.assertEqual(code, 3, err)
        self.assertEqual(json.loads(out)['verdict'], 'insufficient')

    def test_failures_never_improve_denominator_and_missing_baseline(self):
        with patch('engine.complete', side_effect=self.complete):
            self.record_baseline()
        config = json.loads(self.config.read_text())
        config.update(id='v2', name='candidate', checkpoint='weights/v2')
        self.config.write_text(json.dumps(config))
        with patch('engine.complete', side_effect=ValueError('secret endpoint must not print')):
            code, out, err = self.call('run', '--model-config', str(self.config), '--output', str(self.path / 'errors'))
        self.assertEqual(code, 3, err)
        result = json.loads(out)
        self.assertEqual(result['summary']['pass_rate'], 0)
        self.assertEqual(result['summary']['errors'], 20)
        report = json.loads((self.path / 'errors' / 'report.json').read_text())
        self.assertEqual(len(report['comparisons']), 1)
        self.assertNotIn('secret endpoint', json.dumps(report))
