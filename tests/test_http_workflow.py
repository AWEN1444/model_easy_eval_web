"""Full core suite over the actual HTTP adapter, with a deterministic fake service."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from comparison import compare_runs
from datasets import get_builtin_suite
from reporting import build_report, select_baseline, write_report
from runner import run_evaluation
from storage import Store

HTTP_CLIENT = httpx.Client


class HttpWorkflowTest(unittest.TestCase):
    def test_core80_http_retry_persistence_and_regression_gate(self):
        suite = get_builtin_suite('general-core')
        by_prompt = {q['question']: q for q in suite['questions']}
        requests = []
        state = {'transient': True, 'wrong': False}

        def response(request):
            requests.append(request)
            payload = json.loads(request.content)
            question = by_prompt[payload['messages'][-1]['content']]
            if state['transient']:
                state['transient'] = False
                return httpx.Response(503, json={'error': 'warming up'}, request=request)
            expected = question['expected']
            text = json.dumps(expected, ensure_ascii=False) if question['evaluator'] == 'json' else expected
            if state['wrong'] and question['id'] == 'math-01':
                text = '0'
            return httpx.Response(200, json={'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                                           'usage': {'prompt_tokens': 20, 'completion_tokens': 5}}, request=request)

        transport = httpx.MockTransport(response)
        def client(**kwargs):
            return HTTP_CLIENT(transport=transport, **kwargs)
        model = {'id': 'base-v1', 'name': 'base', 'role': 'base', 'model': 'shared-api-alias', 'checkpoint': 'weights/base',
                 'base_url': 'http://eval.invalid/v1', 'key': 'ephemeral-auth-value', 'temperature': 0, 'max_tokens': 2048}
        with tempfile.TemporaryDirectory() as directory, patch('engine.httpx.Client', side_effect=client):
            store = Store(Path(directory) / 'eval.sqlite3')
            baseline = run_evaluation(suite, model, workers=1, retries=1, on_update=store.save_run)
            state['wrong'] = True
            candidate = run_evaluation(suite, dict(model, id='sft-v2', checkpoint='weights/v2'), workers=4,
                                       retries=1, on_update=store.save_run)
            result = compare_runs(candidate, baseline)
            self.assertEqual(baseline['summary']['passed'], 80)
            self.assertEqual(candidate['summary']['passed'], 79)
            self.assertEqual(result['verdict'], 'fail')
            self.assertEqual([row['id'] for row in result['regressed']], ['math-01'])
            self.assertEqual(len(requests), 161)
            self.assertTrue(all(request.headers['Authorization'] == 'Bearer ephemeral-auth-value' for request in requests))
            self.assertTrue(all(request.url.path == '/v1/chat/completions' for request in requests))
            self.assertNotIn('key', store.get_run(candidate['id'])['model'])
            path = write_report(build_report(candidate, [baseline]), Path(directory) / 'report')
            self.assertNotIn('ephemeral-auth-value', (path / 'report.json').read_text())
            self.assertIn('math-01', (path / 'report.md').read_text())
            # Same-version re-runs must not displace the actual previous training version.
            rerun = dict(candidate, id='rerun', created='9999')
            chosen = select_baseline(rerun, [candidate, baseline], 'previous')
            self.assertEqual(chosen['id'], baseline['id'])
