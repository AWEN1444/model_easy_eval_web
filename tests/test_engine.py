import os
import threading
import unittest
from unittest.mock import patch

import httpx

from engine import complete, evaluate, parse_judgment, register_evaluator


class EngineTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_judgment('```json\n{"score":90,"reason":"ok"}\n```')['score'], 90)

    def test_invalid(self):
        for score in ['101', 'true', '-1', '"90"', 'NaN', 'Infinity']:
            with self.subTest(score=score), self.assertRaises(ValueError):
                parse_judgment('{"score":' + score + ',"reason":"x"}')

    @patch('engine.complete')
    def test_evaluate(self, complete_mock):
        complete_mock.side_effect = [{'text': 'answer'}, {'text': '{"score":80,"reason":"ok"}'}]
        result = evaluate({'question': 'q'}, {'model': 'm'}, {'model': 'j'}, 'rubric')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['score'], 80)
        self.assertTrue(result['passed'])

    @patch('engine.complete')
    def test_judge_error_keeps_generation(self, complete_mock):
        complete_mock.side_effect = [{'text': 'answer'}, {'text': 'bad'}]
        result = evaluate({'question': 'q'}, {'model': 'm'}, {'model': 'j'}, 'rubric')
        self.assertEqual(result['status'], 'judge_error')
        self.assertEqual(result['answer']['text'], 'answer')

    def verdict(self, evaluator, expected, answer, options=None):
        with patch('engine.complete', return_value={'text': answer}):
            return evaluate({'question': 'q', 'evaluator': evaluator, 'expected': expected,
                             'options': options or {}}, {'model': 'm'})

    def test_exact_wrong_is_graded_failure(self):
        result = self.verdict('exact', '42', '41')
        self.assertEqual(result['status'], 'success')
        self.assertFalse(result['passed'])
        self.assertEqual(result['score'], 0)

    def test_exact_whitespace_and_case(self):
        self.assertTrue(self.verdict('exact', 'Hello', ' hello \n', {'case_sensitive': False})['passed'])
        self.assertFalse(self.verdict('exact', 'Hello', ' Hello ', {'trim': False})['passed'])

    def test_json_structural_comparison(self):
        self.assertTrue(self.verdict('json', {'a': 1, 'b': [False, None]}, '{"b":[false,null],"a":1}')['passed'])
        self.assertFalse(self.verdict('json', {'a': 1}, '```json\n{"a":1}\n```')['passed'])
        self.assertTrue(self.verdict('json', {'a': 1}, '```json\n{"a":1}\n```', {'allow_fences': True})['passed'])

    def test_json_strict_types_and_invalid_json_are_failures(self):
        for expected, answer in [(True, '1'), (1, 'true'), (1, '1.0'), ([1, 2], '[2,1]'), (None, 'NaN'), (1, 'text'), ({'x': 1}, '{"x":0,"x":1}')]:
            with self.subTest(expected=expected, answer=answer):
                result = self.verdict('json', expected, answer)
                self.assertEqual(result['status'], 'success')
                self.assertFalse(result['passed'])

    def test_contains_checks_all_expected_fragments(self):
        self.assertTrue(self.verdict('contains', ['TCP', 'UDP'], 'tcp vs udp', {'case_sensitive': False})['passed'])
        self.assertFalse(self.verdict('contains', ['TCP', 'UDP'], 'TCP only')['passed'])

    def test_regex_and_python_syntax(self):
        self.assertTrue(self.verdict('regex', r'^id-\d+$', 'id-42')['passed'])
        self.assertFalse(self.verdict('regex', r'^id-\d+$', 'prefix id-42')['passed'])
        result = self.verdict('python_syntax', None, '```python\ndef f():\n return 1\n```')
        self.assertTrue(result['passed'])
        self.assertIn('未执行', result['reason'])
        self.assertFalse(self.verdict('python_syntax', None, 'def broken(')['passed'])

    @patch('engine.complete', return_value={'text': 'anything'})
    def test_unknown_or_invalid_evaluator(self, complete_mock):
        result = evaluate({'question': 'q', 'evaluator': 'missing'}, {'model': 'm'})
        self.assertEqual(result['status'], 'evaluation_error')
        register_evaluator('invalid_test', lambda q, a: {'score': 100, 'passed': 'yes', 'reason': ''})
        result = evaluate({'question': 'q', 'evaluator': 'invalid_test'}, {'model': 'm'})
        self.assertEqual(result['status'], 'evaluation_error')

    @patch('engine.complete', return_value={'text': '42'})
    def test_plugin_cannot_mutate_frozen_question_snapshot(self, complete_mock):
        def plugin(question, answer):
            question['question'] = 'changed'
            question['options']['nested'] = 'changed'
            return {'score': 100, 'passed': True, 'reason': 'ok'}
        register_evaluator('mutating_test_plugin', plugin)
        question = {'question': 'original', 'evaluator': 'mutating_test_plugin', 'options': {'nested': 'original'}}
        result = evaluate(question, {'model': 'm'})
        self.assertEqual(question['question'], 'original')
        self.assertEqual(result['question']['question'], 'original')
        self.assertEqual(result['question']['options']['nested'], 'original')

    @patch('engine.time.sleep')
    @patch('engine.complete')
    def test_retry_transient_failure_only(self, complete_mock, sleep):
        request = httpx.Request('POST', 'https://example.com')
        response = httpx.Response(429, request=request)
        complete_mock.side_effect = [httpx.HTTPStatusError('secret-response', request=request, response=response), {'text': '42'}]
        result = evaluate({'question': 'q', 'evaluator': 'exact', 'expected': '42'}, {'model': 'm'}, retries=1)
        self.assertTrue(result['passed'])
        self.assertEqual(result['answer']['attempts'], 2)
        self.assertEqual(sleep.call_count, 1)

    @patch('engine.complete')
    def test_permanent_error_is_not_retried_or_leaked(self, complete_mock):
        request = httpx.Request('POST', 'https://example.com?api_key=SECRET')
        response = httpx.Response(401, request=request)
        complete_mock.side_effect = httpx.HTTPStatusError('SECRET', request=request, response=response)
        result = evaluate({'question': 'q'}, {'model': 'm'}, retries=5)
        self.assertEqual(complete_mock.call_count, 1)
        self.assertEqual(result['error'], 'HTTP 401')
        self.assertNotIn('SECRET', str(result))

    @patch('engine.complete')
    def test_cancel_before_request(self, complete_mock):
        event = threading.Event()
        event.set()
        result = evaluate({'question': 'q'}, {'model': 'm'}, cancel_event=event)
        self.assertEqual(result['status'], 'cancelled')
        complete_mock.assert_not_called()

    @patch('engine.complete', return_value={'text': '42', 'finish_reason': 'length'})
    def test_truncated_answer_is_incomplete(self, complete_mock):
        result = evaluate({'question': 'q', 'evaluator': 'exact', 'expected': '42'}, {'model': 'm'})
        self.assertEqual(result['status'], 'generation_error')
        self.assertEqual(result['error_kind'], 'truncated')
        self.assertNotIn('score', result)

    @patch('engine.complete', return_value={'text': '42', 'finish_reason': 'content_filter'})
    def test_filtered_or_tool_output_is_incomplete(self, complete_mock):
        result = evaluate({'question': 'q', 'evaluator': 'exact', 'expected': '42'}, {'model': 'm'})
        self.assertEqual(result['status'], 'generation_error')
        self.assertEqual(result['error_kind'], 'content_filter')

    @patch.dict(os.environ, {'TEST_EVAL_KEY': 'test-runtime-secret'})
    @patch('engine.httpx.Client')
    def test_complete_uses_environment_credentials(self, client_mock):
        response = client_mock.return_value.__enter__.return_value.post.return_value
        response.json.return_value = {'choices': [{'message': {'content': '42'}, 'finish_reason': 'stop'}]}
        result = complete({'base_url': 'https://example.com/v1', 'model': 'm', 'api_key_env': 'TEST_EVAL_KEY'}, [])
        call = client_mock.return_value.__enter__.return_value.post.call_args
        self.assertEqual(call.kwargs['headers']['Authorization'], 'Bearer test-runtime-secret')
        self.assertEqual(result['text'], '42')
        self.assertNotIn('test-runtime-secret', str(result))


if __name__ == '__main__':
    unittest.main()
