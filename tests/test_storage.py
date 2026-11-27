import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from storage import Store, initialize, validate_model


def model_config():
    return {'id': 'checkpoint-1', 'name': '训练版本1', 'model': 'served-model',
            'base_url': 'http://localhost:8000/v1', 'checkpoint': 'weights/step-100',
            'api_key_env': 'TEST_MODEL_KEY', 'key': 'never-store-this'}


def suite_config():
    return {'id': 'test', 'name': '测试题库', 'version': '1.0', 'questions': [
        {'id': 'q1', 'category': '结构化输出', 'question': '返回指定JSON', 'evaluator': 'json', 'expected': {'key': 'legitimate'}}]}


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'test.sqlite3'
        self.store = Store(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_frozen_suite_and_reopen(self):
        suite = self.store.save_suite(suite_config())
        changed = copy.deepcopy(suite)
        changed['questions'][0]['question'] = 'changed'
        with self.assertRaises(ValueError):
            self.store.save_suite(changed)
        changed['version'] = '1.1'
        self.store.save_suite(changed)
        reopened = Store(self.path)
        self.assertEqual(reopened.get_suite('test', '1.0')['questions'][0]['expected'], {'key': 'legitimate'})
        self.assertEqual(len(reopened.list_suites()), 2)

    def test_checkpoint_identity_and_no_persistent_key(self):
        first = self.store.save_model(model_config())
        self.assertNotIn('key', first)
        self.assertEqual(first['api_key_env'], 'TEST_MODEL_KEY')
        changed = dict(first, checkpoint='weights/step-200')
        with self.assertRaises(ValueError):
            self.store.save_model(changed)
        changed['id'] = 'checkpoint-2'
        self.store.save_model(changed)
        self.assertEqual(len(self.store.list_models()), 2)
        self.assertNotIn(b'never-store-this', self.path.read_bytes())

    def test_run_config_redaction_preserves_json_expectation(self):
        run = {'id': 'run1', 'model': model_config(), 'judge': {'key': 'judge-secret'},
               'suite': suite_config(), 'results': [], 'metadata': {'nested': {'api_key': 'nested-secret'}}}
        self.store.save_run(run)
        saved = self.store.get_run('run1')
        self.assertEqual(saved['suite']['questions'][0]['expected']['key'], 'legitimate')
        encoded = json.dumps(saved)
        for secret in ('never-store-this', 'judge-secret', 'nested-secret'):
            self.assertNotIn(secret, encoded)

    def test_legacy_run_table_preserved(self):
        legacy = {'id': 'old', 'targets': [], 'results': [], 'state': 'finished'}
        with sqlite3.connect(self.path) as conn:
            conn.execute('INSERT INTO runs VALUES (?,?)', ('old', json.dumps(legacy)))
        initialize(Store(self.path))
        self.assertEqual(self.store.get_run('old'), legacy)
        self.assertGreaterEqual(len(self.store.list_suites()), 2)

    def test_reject_credentials_in_url_or_extra(self):
        for changes in ({'base_url': 'https://user:password@example.org/v1'},
                        {'base_url': 'https://example.org/v1?api_key=secret'},
                        {'base_url': 'https://example.org/v1?client_secret=secret'},
                        {'base_url': 'https://example.org/v1#access_token=secret'},
                        {'extra': {'temperature': 1, 'nested': {'api_key': 'secret'}}},
                        {'max_tokens': 10.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_model(dict(model_config(), **changes))

    def test_legacy_url_credentials_redacted_on_save(self):
        run = {'id': 'legacy-url', 'model': {'base_url': 'https://user:password@example.org/v1?client_secret=LEAK&safe=1'}}
        saved = self.store.save_run(run)
        self.assertEqual(saved['model']['base_url'], 'https://example.org/v1?safe=1')
