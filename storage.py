"""SQLite persistence for immutable evaluation assets and run snapshots."""
import copy
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_SECRET_FIELDS = {'key', 'api_key', 'apikey', 'authorization', 'password', 'secret', 'token', 'access_token', 'headers'}


def _secret_field(key):
    normalized = str(key).lower().replace('-', '_')
    return normalized in _SECRET_FIELDS or normalized.endswith(('_api_key', '_password', '_secret', '_token'))


def redact_config(value):
    """Remove configuration credentials without altering question/answer data."""
    if isinstance(value, dict):
        return {k: redact_config(v) for k, v in value.items()
                if not _secret_field(k)}
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    if isinstance(value, str) and value.startswith(('http://', 'https://')):
        try:
            url = urlsplit(value)
            query = [(key, item) for key, item in parse_qsl(url.query, keep_blank_values=True) if not _secret_field(key)]
            return urlunsplit((url.scheme, url.netloc.rsplit('@', 1)[-1], url.path, urlencode(query), ''))
        except ValueError:
            return '[invalid URL]'
    return value


def sanitize_run(run):
    data = copy.deepcopy(run)
    for key in ('model', 'judge', 'metadata'):
        if data.get(key) is not None:
            data[key] = redact_config(data[key])
    if 'targets' in data:
        data['targets'] = redact_config(data['targets'])
    return data


def validate_model(model):
    if not isinstance(model, dict):
        raise ValueError('模型配置必须为 JSON 对象')
    data = copy.deepcopy(model)
    for key in ('name', 'model', 'base_url'):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError('模型版本必须填写 name、model 和 base_url')
        data[key] = data[key].strip()
    try:
        url = urlsplit(data['base_url'])
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError()
        if any(char.isspace() for char in data['base_url']):
            raise ValueError()
        url.port
    except ValueError:
        raise ValueError('base_url 必须为 http(s) 地址，且不得包含凭据、查询参数或片段')
    data.setdefault('id', uuid.uuid4().hex)
    if not isinstance(data['id'], str) or not data['id'].strip():
        raise ValueError('模型版本 id 不能为空')
    data.setdefault('checkpoint', '')
    data.setdefault('training_data_version', '')
    data.setdefault('training_config', '')
    data.setdefault('role', 'candidate')
    if data['role'] not in ('candidate', 'base', 'champion'):
        raise ValueError('role 必须为 candidate、base 或 champion')
    data.setdefault('api_key_env', '')
    if not isinstance(data['api_key_env'], str):
        raise ValueError('api_key_env 必须为环境变量名')
    data.setdefault('temperature', 0.0)
    data.setdefault('max_tokens', 2048)
    data.setdefault('timeout', 120)
    data.setdefault('system', '')
    data.setdefault('extra', {})
    if not isinstance(data['extra'], dict):
        raise ValueError('extra 必须为 JSON 对象')
    forbidden = _SECRET_FIELDS | {'model', 'messages', 'stream'}
    if any(k.lower() in forbidden for k in data['extra']):
        raise ValueError('extra 不得覆盖核心字段或包含密钥')
    if redact_config(data['extra']) != data['extra']:
        raise ValueError('extra 不得包含密钥')
    for key, low, high in (('temperature', 0, 2), ('max_tokens', 1, 1000000), ('timeout', 1, 3600)):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError('%s 超出允许范围' % key)
    if not isinstance(data['max_tokens'], int):
        raise ValueError('max_tokens 必须为整数')
    if not isinstance(data['system'], str):
        raise ValueError('system 必须为字符串')
    return redact_config(data)


class Store:
    def __init__(self, db_path=None):
        self.path = str(db_path or os.environ.get('EVAL_DB', 'eval.sqlite3'))
        Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.path = str(Path(self.path).expanduser())
        with self.connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS suite_versions (
                    id TEXT NOT NULL, version TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    created TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (id, version));
                CREATE TABLE IF NOT EXISTS model_versions (
                    id TEXT PRIMARY KEY, created TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            ''')

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute('PRAGMA busy_timeout=30000')
        return connection

    def save_suite(self, suite):
        from datasets import validate_suite, suite_hash
        data = validate_suite(suite)
        fingerprint = suite_hash(data)
        with self.connect() as connection:
            old = connection.execute('SELECT fingerprint, data FROM suite_versions WHERE id=? AND version=?',
                                     (data['id'], data['version'])).fetchone()
            if old:
                if old[0] != fingerprint:
                    raise ValueError('此题库版本已冻结；修改题目后请使用新的版本号')
                return json.loads(old[1])
            connection.execute('INSERT INTO suite_versions VALUES (?,?,?,?,?)',
                               (data['id'], data['version'], fingerprint, datetime.now(timezone.utc).isoformat(),
                                json.dumps(data, ensure_ascii=False, allow_nan=False)))
        return data

    def list_suites(self):
        with self.connect() as connection:
            return [json.loads(row[0]) for row in connection.execute('SELECT data FROM suite_versions ORDER BY created DESC')]

    def get_suite(self, suite_id, version=None):
        with self.connect() as connection:
            if version is None:
                row = connection.execute('SELECT data FROM suite_versions WHERE id=? ORDER BY created DESC LIMIT 1', (suite_id,)).fetchone()
            else:
                row = connection.execute('SELECT data FROM suite_versions WHERE id=? AND version=?', (suite_id, version)).fetchone()
        if row is None:
            raise ValueError('未找到题库：%s%s' % (suite_id, '@' + version if version else ''))
        return json.loads(row[0])

    def save_model(self, model):
        data = validate_model(model)
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self.connect() as connection:
            old = connection.execute('SELECT data FROM model_versions WHERE id=?', (data['id'],)).fetchone()
            if old:
                if json.loads(old[0]) != data:
                    raise ValueError('模型版本已冻结；更换权重或配置请登记新的版本 id')
                return json.loads(old[0])
            connection.execute('INSERT INTO model_versions VALUES (?,?,?)',
                               (data['id'], datetime.now(timezone.utc).isoformat(), encoded))
        return data

    def list_models(self):
        with self.connect() as connection:
            return [json.loads(row[0]) for row in connection.execute('SELECT data FROM model_versions ORDER BY created DESC')]

    def get_model(self, model_id):
        with self.connect() as connection:
            row = connection.execute('SELECT data FROM model_versions WHERE id=?', (model_id,)).fetchone()
        if row is None:
            raise ValueError('未找到模型版本：' + model_id)
        return json.loads(row[0])

    def save_run(self, run):
        data = sanitize_run(run)
        if not isinstance(data.get('id'), str) or not data['id']:
            raise ValueError('实验必须有 id')
        with self.connect() as connection:
            connection.execute('INSERT OR REPLACE INTO runs (id,data) VALUES (?,?)',
                               (data['id'], json.dumps(data, ensure_ascii=False, allow_nan=False)))
        return data

    def list_runs(self, limit=100):
        with self.connect() as connection:
            rows = connection.execute('SELECT data FROM runs ORDER BY rowid DESC LIMIT ?', (int(limit),)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_run(self, run_id):
        with self.connect() as connection:
            row = connection.execute('SELECT data FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError('未找到实验：' + run_id)
        return json.loads(row[0])


def initialize(store):
    from datasets import builtin_suites
    for suite in builtin_suites():
        try:
            store.get_suite(suite['id'], suite['version'])
        except ValueError:
            store.save_suite(suite)
