"""
Persistence layer for session logs / telemetry.

Uses a local SQLite file so the prototype works with zero cloud setup.
Schema mirrors what you'd send to BigQuery in production -- swap `log_event`
and `fetch_history` for BigQuery client calls (see README.md) without
touching any calling code.
"""

import sqlite3
import time
from contextlib import contextmanager

DB_PATH = "telemetry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    user_text TEXT NOT NULL,
    fast_top TEXT,
    deep_top TEXT,
    ensemble_top TEXT,
    mixed_labels TEXT,
    fast_scores TEXT,
    deep_scores TEXT,
    ensemble_scores TEXT,
    fast_latency_ms REAL,
    deep_latency_ms REAL,
    response_source TEXT
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_event(user_text, fast, deep, ensemble, response_source):
    import json

    with _conn() as conn:
        conn.execute(
            """INSERT INTO events
               (ts, user_text, fast_top, deep_top, ensemble_top, mixed_labels,
                fast_scores, deep_scores, ensemble_scores,
                fast_latency_ms, deep_latency_ms, response_source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(),
                user_text,
                fast.top_label,
                deep.top_label,
                ensemble.top_label,
                json.dumps(ensemble.mixed_labels),
                json.dumps(fast.scores),
                json.dumps(deep.scores),
                json.dumps(ensemble.scores),
                fast.latency_ms,
                deep.latency_ms,
                response_source,
            ),
        )


def fetch_history():
    import pandas as pd

    with _conn() as conn:
        df = pd.read_sql_query("SELECT * FROM events ORDER BY ts ASC", conn)
    return df


def clear_history():
    with _conn() as conn:
        conn.execute("DELETE FROM events")
