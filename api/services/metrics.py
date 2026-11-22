"""SQLite inference metrics; never persist citation text or credentials."""
from __future__ import annotations

import csv
import io
import logging
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = Path(os.getenv('EVIDENCITE_METRICS_DB', '/workspace/var/metrics.sqlite3'))

DDL = '''CREATE TABLE IF NOT EXISTS inference_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT,
    configuration TEXT,
    status TEXT NOT NULL,
    predicted_label TEXT,
    confidence REAL,
    inference_ms REAL,
    server_ms REAL NOT NULL,
    client_ms REAL,
    input_tokens INTEGER,
    cached_input_tokens INTEGER,
    output_tokens INTEGER,
    thoughts_tokens INTEGER,
    cost_usd REAL,
    cost_note TEXT NOT NULL,
    error_type TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0
)'''

FIELDS = (
    'id', 'created_at', 'model_id', 'model_name', 'configuration', 'status',
    'predicted_label', 'confidence', 'inference_ms', 'server_ms', 'client_ms',
    'input_tokens', 'cached_input_tokens', 'output_tokens', 'thoughts_tokens',
    'cost_usd', 'cost_note', 'error_type', 'is_demo'
)


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH), timeout=5)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA busy_timeout=5000')
    db.execute(DDL)
    db.commit()
    return db


def nonnegative(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def cost_estimate(model_name, data):
    # Prices must be provided from verified tariffs. Never infer them.
    if model_name != 'gemini-3.5-flash-lite':
        return None, 'Sin tarifa API; infraestructura no cuantificada'
    try:
        rate_input = float(os.environ['GEMINI_35_INPUT_USD_PER_1M'])
        rate_output = float(os.environ['GEMINI_35_OUTPUT_USD_PER_1M'])
        rate_cached = float(os.environ.get('GEMINI_35_CACHED_USD_PER_1M', str(rate_input)))
        if not all(math.isfinite(r) and r >= 0 for r in (rate_input, rate_output, rate_cached)):
            raise ValueError('Invalid price')
        if data.get('input_tokens') is None or data.get('output_tokens') is None:
            return None, 'Tokens insuficientes para estimar costo'
        input_tokens = int(data['input_tokens'])
        cached = max(0, min(input_tokens, int(data.get('cached_input_tokens') or 0)))
        output_tokens = int(data['output_tokens'])
        thoughts = int(data.get('thoughts_tokens') or 0)
        thinking_rate_raw = os.environ.get('GEMINI_35_THINKING_USD_PER_1M')
        if thoughts > 0 and thinking_rate_raw is None:
            return None, 'Falta tarifa de tokens de razonamiento'
        thinking_rate = float(thinking_rate_raw) if thinking_rate_raw is not None else 0
        if not math.isfinite(thinking_rate) or thinking_rate < 0:
            raise ValueError('Invalid price')
        amount = ((input_tokens-cached)*rate_input + cached*rate_cached +
                  output_tokens*rate_output + thoughts*thinking_rate) / 1_000_000
        return amount, 'Estimado según tarifas configuradas'
    except (KeyError, ValueError, TypeError, OverflowError):
        return None, 'Tarifa Gemini pendiente de configurar'


def log_inference(model_id, result, server_ms, error=None):
    """Best effort: metric failures must never break inference."""
    run_id = str(uuid.uuid4())
    data = (result if isinstance(result, dict) else
            result.model_dump(mode='json') if hasattr(result, 'model_dump') else {})
    mode = model_id.rsplit('-', 1)[-1].upper()
    if mode not in ('A', 'B'):
        mode = 'A/B'
    model_name = data.get('model_name') or model_id
    cost, cost_note = cost_estimate(model_name, data)
    try:
        values = (
            run_id, datetime.now(timezone.utc).isoformat(), model_id, model_name,
            mode, 'error' if error else 'ok', data.get('predicted_label'),
            nonnegative(data.get('confidence')), nonnegative(data.get('latency_ms')),
            max(float(server_ms), 0), None,
            data.get('input_tokens'), data.get('cached_input_tokens'),
            data.get('output_tokens'), data.get('thoughts_tokens'),
            cost, cost_note, type(error).__name__ if error else None,
            int(bool(data.get('is_demo', False))),
        )
        with connect() as db:
            db.execute(
                'INSERT INTO inference_runs (' + ','.join(FIELDS) +
                ') VALUES (' + ','.join('?' for _ in FIELDS) + ')', values
            )
        return run_id
    except Exception as exc:
        logger.warning('Could not save inference metric: %s', exc)
        return None


def update_client_time(run_id, client_ms):
    with connect() as db:
        result = db.execute(
            'UPDATE inference_runs SET client_ms=? WHERE id=?',
            (client_ms, str(uuid.UUID(run_id)))
        )
        return result.rowcount > 0


def get_runs(limit=500):
    with connect() as db:
        rows = db.execute(
            'SELECT ' + ','.join(FIELDS) +
            ' FROM inference_runs ORDER BY created_at DESC LIMIT ?',
            (max(1, min(int(limit), 5000)),)
        ).fetchall()
    return [dict(row) for row in rows]


def make_csv():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(get_runs(limit=5000))
    return output.getvalue()
