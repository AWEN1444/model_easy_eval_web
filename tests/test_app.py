"""Dashboard workflows against a real temporary SQLite database, with no API calls."""
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from jobs import JobManager
from datasets import import_questions
from runner import run_evaluation
from storage import Store

APP = Path(__file__).resolve().parents[1] / 'app.py'


def sample_model(model_id='v1'):
    return {'id': model_id, 'name': 'Checkpoint '+model_id, 'checkpoint': 'checkpoint-'+model_id,
            'model': 'same-served-model', 'base_url': 'http://localhost:8000/v1',
            'temperature': 0.0, 'max_tokens': 2048, 'timeout': 1, 'system': '', 'extra': {},
            'role': 'candidate', 'training_data_version': 'train-1', 'training_config': {},
            'api_key_env': ''}


def sample_suite(question_count=1):
    return {'id':'app-test', 'name':'UI 测试题库', 'version':'1',
            'questions':[{'id':'q'+str(i), 'category':'测试', 'question':'只回答 2',
                          'evaluator':'exact', 'expected':'2'} for i in range(question_count)]}


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp.name)/'eval.sqlite3')
        self.env = patch.dict(os.environ, {'EVAL_DB':self.database})
        self.env.start()
        self.store = Store()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def app(self):
        return AppTest.from_file(str(APP), default_timeout=30).run()

    def test_all_pages_load_and_builtin_suites_persist(self):
        app = self.app()
        self.assertFalse(app.exception)
        suites = self.store.list_suites()
        self.assertEqual(sorted(len(s['questions']) for s in suites), [20, 80])
        for page in ['题库管理', '模型版本', '新建评测', '历史实验', '实验对比', '概览']:
            app.sidebar.radio(key='navigation').set_value(page).run()
            self.assertFalse(app.exception, page)

    def test_register_model_and_save_edited_suite_version(self):
        app = self.app()
        app.sidebar.radio(key='navigation').set_value('模型版本').run()
        app.text_input(key='model_name').input('训练第 3 版')
        app.text_input(key='model_checkpoint').input('weights/checkpoint-1200')
        app.text_input(key='model_api_id').input('dsl-model')
        app.text_input(key='model_training_data').input('dsl-data-v3')
        next(button for button in app.button if button.label == '登记模型版本').click().run()
        self.assertFalse(app.exception)
        models = self.store.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['checkpoint'], 'weights/checkpoint-1200')
        self.assertNotIn('key', models[0])
        app.sidebar.radio(key='navigation').set_value('题库管理').run()
        source = app.selectbox(key='suite_source').value
        app.session_state['suite_draft_questions'] = sample_suite()['questions']
        app.session_state['suite_revision'] += 1
        app.run()
        app.text_input(key='suite_version').input('ui-edited-1')
        next(button for button in app.button if button.label == '保存题库新版本').click().run()
        self.assertFalse(app.exception)
        saved = self.store.get_suite(source['id'], 'ui-edited-1')
        self.assertEqual(len(saved['questions']), 1)
        self.assertEqual(saved['questions'][0]['expected'], '2')
        original = self.store.get_suite(source['id'], source['version'])
        self.assertGreater(len(original['questions']), 1)

    def test_suite_editor_preserves_imported_extension_metadata(self):
        import json
        suite = sample_suite()
        suite['metadata'] = {'language':'custom-dsl', 'compiler_version':'2026.1'}
        suite['sampling_plan'] = {'split':'holdout', 'owner':'research'}
        suite['questions'][0].update({'tags':['boundary', 'regression'], 'difficulty':'hard',
                                      'plugin':{'expected_ast':{'kind':'literal', 'value':2}}})
        suite['questions'] = import_questions(json.dumps(suite), 'custom.json')
        self.store.save_suite(suite)
        app = self.app()
        app.sidebar.radio(key='navigation').set_value('题库管理').run()
        app.selectbox(key='suite_source').set_value(self.store.get_suite('app-test', '1')).run()
        app.text_input(key='suite_version').input('2')
        app.text_input(key='suite_name').input('修改名称后的 DSL 题库')
        next(button for button in app.button if button.label == '保存题库新版本').click().run()
        self.assertFalse(app.exception)
        saved = self.store.get_suite('app-test', '2')
        self.assertEqual(saved['metadata'], suite['metadata'])
        self.assertEqual(saved['sampling_plan'], suite['sampling_plan'])
        self.assertEqual(saved['questions'][0]['tags'], ['boundary', 'regression'])
        self.assertEqual(saved['questions'][0]['difficulty'], 'hard')
        self.assertEqual(saved['questions'][0]['plugin'], suite['questions'][0]['plugin'])
        self.assertEqual(saved['questions'][0]['expected'], '2')
        self.assertEqual(saved['name'], '修改名称后的 DSL 题库')

    def test_launch_background_evaluation_from_web(self):
        self.store.save_model(sample_model())
        self.store.save_suite(sample_suite())
        app = self.app()
        app.sidebar.radio(key='navigation').set_value('新建评测').run()
        app.selectbox(key='eval_suite').set_value(self.store.get_suite('app-test', '1')).run()
        with patch('engine.complete', return_value={'text':'2', 'seconds':0.01, 'usage':{}}):
            next(button for button in app.button if button.label == '启动评测').click().run()
            run_id = app.session_state['active_run_id']
            deadline = time.monotonic()+5
            run = None
            while time.monotonic() < deadline:
                try:
                    run = self.store.get_run(run_id)
                except ValueError:
                    pass
                if run and run.get('state') == 'finished':
                    break
                time.sleep(0.02)
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(run['state'], 'finished')
        self.assertEqual(run['summary']['pass_rate'], 1.0)
        app.sidebar.radio(key='navigation').set_value('历史实验').run()
        self.assertFalse(app.exception)

    def test_compare_page_shows_new_regression(self):
        baseline_model, candidate_model = sample_model('base'), sample_model('new')
        baseline_model['role'] = 'base'
        self.store.save_model(baseline_model)
        self.store.save_model(candidate_model)
        suite = sample_suite(2)
        self.store.save_suite(suite)
        with patch('engine.complete', side_effect=[
                {'text':'2', 'seconds':0.01}, {'text':'2', 'seconds':0.01},
                {'text':'3', 'seconds':0.01}, {'text':'2', 'seconds':0.01}]):
            baseline = run_evaluation(suite, baseline_model, workers=1, retries=0,
                                      on_update=self.store.save_run, metadata={'name':'基础模型评测'})
            candidate = run_evaluation(suite, candidate_model, workers=1, retries=0,
                                       on_update=self.store.save_run, metadata={'name':'新训练评测'})
        app = self.app()
        app.sidebar.radio(key='navigation').set_value('实验对比').run()
        app.selectbox(key='compare_candidate').set_value(candidate).run()
        app.radio(key='baseline_mode').set_value('基础模型').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('未满足当前替换标准' in error.value for error in app.error))
        regression = next(metric for metric in app.metric if metric.label == '新增退步题')
        self.assertEqual(regression.value, '1')
        self.assertTrue(any('q0' in markdown.value for markdown in app.markdown))

    def test_cancel_active_evaluation_from_web(self):
        self.store.save_model(sample_model())
        self.store.save_suite(sample_suite(5))
        app = self.app()
        app.sidebar.radio(key='navigation').set_value('新建评测').run()
        app.selectbox(key='eval_suite').set_value(self.store.get_suite('app-test', '1')).run()
        app.slider(key='eval_workers').set_value(1).run()
        started, release = threading.Event(), threading.Event()

        def complete(*args, **kwargs):
            started.set()
            release.wait(3)
            return {'text':'2', 'seconds':0.01}

        with patch('engine.complete', side_effect=complete):
            try:
                next(button for button in app.button if button.label == '启动评测').click().run()
                run_id = app.session_state['active_run_id']
                self.assertTrue(started.wait(2))
                next(button for button in app.button if button.label == '取消当前评测').click().run()
            finally:
                release.set()
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                run = self.store.get_run(run_id)
                if run['state'] == 'cancelled':
                    break
                time.sleep(0.02)
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(run['state'], 'cancelled')
        self.assertTrue(any('cancelled' in warning.value for warning in app.warning))

    def test_job_cancellation_preserves_partial_results(self):
        manager = JobManager(max_jobs=1)
        started, release = threading.Event(), threading.Event()

        def complete(*args, **kwargs):
            started.set()
            release.wait(2)
            return {'text':'2', 'seconds':0.01, 'usage':{}}

        with patch('engine.complete', side_effect=complete):
            run_id = manager.start(self.store, sample_suite(5), sample_model(), workers=1, retries=0)
            self.assertTrue(started.wait(2))
            self.assertTrue(manager.cancel(run_id))
            release.set()
            deadline = time.monotonic()+5
            while not manager.status(run_id)['done'] and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertTrue(manager.status(run_id)['done'])
        run = self.store.get_run(run_id)
        self.assertEqual(run['state'], 'cancelled')
        self.assertEqual(len(run['results']), 5)
        self.assertLess(run['summary']['completion_rate'], 1.0)
        self.assertTrue(any(result['status'] == 'cancelled' for result in run['results']))


if __name__ == '__main__':
    unittest.main()
