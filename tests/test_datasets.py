import copy
import json
import unittest
from datasets import builtin_suites, get_builtin_suite, import_questions, suite_hash, validate_suite


class DatasetTest(unittest.TestCase):
    def test_builtin_coverage_and_quick_subset(self):
        core = get_builtin_suite('general-core')
        quick = get_builtin_suite('general-quick')
        self.assertEqual(len(core['questions']), 80)
        self.assertEqual(len(quick['questions']), 20)
        self.assertEqual(len({q['category'] for q in core['questions']}), 8)
        by_id = {q['id']: q for q in core['questions']}
        self.assertTrue(all(q == by_id[q['id']] for q in quick['questions']))
        self.assertNotEqual(suite_hash(core), suite_hash(quick))
        self.assertEqual(len(builtin_suites()), 2)

    def test_reference_values_independently_checked(self):
        by_id = {q['id']: q['expected'] for q in get_builtin_suite('general-core')['questions']}
        # Derive answers independently rather than asserting the generator's literals.
        self.assertEqual(by_id['math-01'], str(17 + 28 * 3))
        self.assertEqual(by_id['math-03'], format(120 * .8 * 1.1, '.1f'))
        self.assertEqual(by_id['math-10'], str(sum(n % 2 == 0 for n in range(-3, 8))))
        self.assertEqual(by_id['code-02'], list(range(2, 8, 2)))
        self.assertEqual(by_id['code-04'], [7 // 2, 7 % 2])
        a = [[0]] * 2
        a[0][0] = 1
        self.assertEqual(by_id['code-10'], a)
        self.assertEqual(by_id['text-05'], str(len('模型eval')))
        self.assertEqual(by_id['text-02'], 'a,,b,'.split(','))
        self.assertEqual(by_id['api-06']['offset'], (3 - 1) * 20)

    def test_hash_ignores_row_order_display_label_but_tracks_rules(self):
        suite = get_builtin_suite('general-quick')
        changed = copy.deepcopy(suite)
        changed['name'] = 'another display title'
        changed['questions'].reverse()
        self.assertEqual(suite_hash(suite), suite_hash(changed))
        changed['questions'][0]['expected'] = 'changed'
        self.assertNotEqual(suite_hash(suite), suite_hash(changed))

    def test_legacy_jsonl_has_stable_ids_and_judge_default(self):
        text = '{"category":"code","question":"Explain recursion"}\n'
        first = import_questions(text, 'legacy.jsonl')
        self.assertEqual(first, import_questions(text, 'legacy.jsonl'))
        self.assertEqual(first[0]['evaluator'], 'judge')
        self.assertTrue(first[0]['id'])

    def test_csv_preserves_json_expected_options_and_critical(self):
        text = 'id,question,evaluator,expected,options,critical\na,Return false,json,false,"{""allow_fences"": false}",true\n'
        question = import_questions(text, 'q.csv')[0]
        self.assertIs(question['expected'], False)
        self.assertIs(question['critical'], True)
        self.assertEqual(question['options'], {'allow_fences': False})

    def test_json_suite_round_trip(self):
        suite = get_builtin_suite('general-quick')
        imported = import_questions(json.dumps(suite).encode(), 'suite.json')
        self.assertEqual(imported, suite['questions'])

    def test_duplicate_ids_and_empty_questions_rejected(self):
        suite = get_builtin_suite('general-quick')
        suite['questions'][1]['id'] = suite['questions'][0]['id']
        with self.assertRaises(ValueError):
            validate_suite(suite)
        with self.assertRaises(ValueError):
            import_questions('[{"question":"   "}]', 'empty.json')

    def test_invalid_rules_rejected(self):
        for question in [
            {'question': 'x', 'evaluator': 'json'},
            {'question': 'x', 'evaluator': 'exact', 'expected': False},
            {'question': 'x', 'evaluator': 'contains', 'expected': []},
            {'question': 'x', 'evaluator': 'regex', 'expected': '['},
            {'question': 'x', 'evaluator': 'regex', 'expected': ''},
            {'question': 'x', 'options': {'pass_score': float('nan')}},
            {'question': 'x', 'options': []},
            {'question': 'x', 'options': {'allow_fences': 'false'}},
        ]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                validate_suite({'id': 'x', 'name': 'x', 'version': '1', 'questions': [question]})

    def test_json_null_expected_is_valid(self):
        question = import_questions('[{"question":"Return null","evaluator":"json","expected":null}]', 'q.json')[0]
        self.assertIsNone(question['expected'])

    def test_custom_evaluator_preserves_extension_metadata(self):
        question = import_questions('[{"question":"Compile DSL","evaluator":"dsl_compile","source":"private","options":{"dialect":"v2"}}]', 'custom.json')[0]
        self.assertEqual(question['evaluator'], 'dsl_compile')
        self.assertEqual(question['source'], 'private')
        self.assertEqual(question['options'], {'dialect': 'v2'})


if __name__ == '__main__':
    unittest.main()
