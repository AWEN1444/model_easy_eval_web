"""Versioned evaluation suites and backwards-compatible dataset import."""
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

DATASET_DIR = Path(__file__).resolve().parent / 'datasets'
BUILTIN_EVALUATORS = {'exact', 'json', 'contains', 'regex', 'python_syntax', 'judge'}


def _check_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('JSON 数据不得包含 NaN 或 Infinity')
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError('对象字段名必须为字符串')
            _check_json(item)
    elif isinstance(value, list):
        for item in value:
            _check_json(item)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError('数据必须可以表示为 JSON')


def _question(raw):
    if not isinstance(raw, dict):
        raise ValueError('每道题必须为对象')
    q = copy.deepcopy(raw)
    if not isinstance(q.get('question'), str) or not q['question'].strip():
        raise ValueError('每道题的 question 必须为非空字符串')
    for field, default in [('category', '未分类'), ('reference', ''), ('criteria', '')]:
        value = q.get(field, default)
        if value is None:
            value = default
        if not isinstance(value, str):
            raise ValueError('%s 必须为字符串' % field)
        q[field] = value or default
    q['evaluator'] = q.get('evaluator') or 'judge'
    if not isinstance(q['evaluator'], str) or not re.fullmatch(r'[a-z][a-z0-9_]*', q['evaluator']):
        raise ValueError('无效的 evaluator 名称')
    # Custom evaluator names are intentionally allowed; the runner checks registration.
    q['options'] = q.get('options', {})
    if not isinstance(q['options'], dict):
        raise ValueError('options 必须为 JSON 对象')
    threshold = q['options'].get('pass_score', 80)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError('pass_score 必须为 0–100 数字')
    for option in ('case_sensitive', 'allow_fences', 'trim'):
        if option in q['options'] and not isinstance(q['options'][option], bool):
            raise ValueError('%s 必须为布尔值' % option)
    q['critical'] = q.get('critical', False)
    if not isinstance(q['critical'], bool):
        raise ValueError('critical 必须为布尔值')
    evaluator = q['evaluator']
    if evaluator in {'exact', 'regex'} and not isinstance(q.get('expected'), str):
        raise ValueError('%s 题必须提供字符串 expected' % evaluator)
    if evaluator == 'regex' and not q['expected']:
        raise ValueError('regex expected 不得为空')
    if evaluator == 'json' and 'expected' not in q:
        raise ValueError('json 题必须提供 expected，可为 null')
    if evaluator == 'contains' and (not isinstance(q.get('expected'), list) or not q['expected'] or any(not isinstance(s, str) or not s for s in q['expected'])):
        raise ValueError('contains 题 expected 必须为非空字符串数组')
    if evaluator == 'regex':
        try:
            re.compile(q['expected'])
        except re.error as error:
            raise ValueError('无效的正则表达式') from error
    _check_json(q)
    if not q.get('id'):
        identity = json.dumps({k: v for k, v in q.items() if k != 'id'}, sort_keys=True, ensure_ascii=False, allow_nan=False)
        q['id'] = 'q-' + hashlib.sha256(identity.encode()).hexdigest()[:16]
    if not isinstance(q['id'], str) or not q['id'].strip():
        raise ValueError('题目 id 必须为非空字符串')
    return q


def validate_suite(data):
    """Return a detached, normalized suite; reject ambiguous IDs and invalid rules."""
    if not isinstance(data, dict):
        raise ValueError('评测集必须为 JSON 对象')
    suite = copy.deepcopy(data)
    for field in ('id', 'name', 'version'):
        if not isinstance(suite.get(field), str) or not suite[field].strip():
            raise ValueError('评测集 %s 必须为非空字符串' % field)
    questions = suite.get('questions')
    if not isinstance(questions, list) or not questions:
        raise ValueError('评测集必须至少包含一道题')
    if len(questions) > 10000:
        raise ValueError('单套评测集最多支持 10000 道题')
    suite['questions'] = [_question(q) for q in questions]
    ids = [q['id'] for q in suite['questions']]
    if len(set(ids)) != len(ids):
        raise ValueError('题目 id 重复，请为每道题提供唯一且稳定的 id')
    _check_json(suite)
    return suite


def import_questions(raw, filename):
    """Import CSV, JSON array / suite object, or JSONL; preserve scoring rules."""
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8-sig')
    elif isinstance(raw, str):
        raw = raw.lstrip('\ufeff')
    else:
        raise ValueError('导入内容必须为文本或 UTF-8 字节')
    extension = Path(filename).suffix.lower()
    if extension == '.csv':
        questions = list(csv.DictReader(io.StringIO(raw)))
        for q in questions:
            q['evaluator'] = q.get('evaluator') or 'judge'
            for field in ('options', 'tags'):
                if q.get(field):
                    q[field] = json.loads(q[field])
                else:
                    q.pop(field, None)
            if q.get('critical'):
                critical = q['critical'].strip().lower()
                if critical not in ('true', 'false', '1', '0'):
                    raise ValueError('CSV critical 必须为 true/false')
                q['critical'] = critical in ('true', '1')
            else:
                q.pop('critical', None)
            if q['evaluator'] in ('json', 'contains') and 'expected' in q:
                q['expected'] = json.loads(q['expected'])
    elif extension == '.jsonl':
        questions = [json.loads(line) for line in raw.splitlines() if line.strip()]
    elif extension == '.json':
        parsed = json.loads(raw)
        questions = parsed.get('questions') if isinstance(parsed, dict) else parsed
    else:
        raise ValueError('仅支持 CSV / JSON / JSONL')
    if not isinstance(questions, list):
        raise ValueError('题库必须为数组，或包含 questions 数组的评测集对象')
    return validate_suite({'id': 'import', 'name': '导入题库', 'version': '1', 'questions': questions})['questions']


def suite_hash(suite):
    """Hash a normalized suite protocol, independent of display labels and row order."""
    normalized = validate_suite(suite)
    protocol = {
        'id': normalized['id'],
        'version': normalized['version'],
        'questions': sorted(normalized['questions'], key=lambda q: q['id']),
    }
    canonical = json.dumps(protocol, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def builtin_suites():
    return [validate_suite(json.loads(path.read_text(encoding='utf-8'))) for path in sorted(DATASET_DIR.glob('*.json'))]


def get_builtin_suite(suite_id):
    for suite in builtin_suites():
        if suite['id'] == suite_id:
            return suite
    raise ValueError('内置评测集不存在: %s' % suite_id)
