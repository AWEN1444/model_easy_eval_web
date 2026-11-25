import unittest
from unittest.mock import patch
from engine import parse_judgment,evaluate

class EngineTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_judgment('```json\n{"score":90,"reason":"ok"}\n```')['score'],90)
    def test_invalid(self):
        for score in ['101','true','-1','"90"']:
            with self.assertRaises(ValueError): parse_judgment('{"score":'+score+',"reason":"x"}')
    @patch('engine.complete')
    def test_evaluate(self,complete):
        complete.side_effect=[{'text':'answer'},{'text':'{"score":80,"reason":"ok"}'}]
        r=evaluate({'question':'q'},{'model':'m'},{'model':'j'},'rubric')
        self.assertEqual(r['status'],'success'); self.assertEqual(r['score'],80)
    @patch('engine.complete')
    def test_judge_error(self,complete):
        complete.side_effect=[{'text':'answer'},{'text':'bad'}]
        r=evaluate({'question':'q'},{'model':'m'},{'model':'j'},'rubric')
        self.assertEqual(r['status'],'judge_error'); self.assertEqual(r['answer']['text'],'answer')
