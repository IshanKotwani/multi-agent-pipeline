import sqlite3
import uuid
from datetime import datetime

DEBUG_DB = "debug.db"

def init_debug_db():
    conn = sqlite3.connect(DEBUG_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS debug_logs (
            id TEXT PRIMARY KEY,
            task_id TEXT,
            agent_name TEXT,
            prompt_input TEXT,
            raw_output TEXT,
            error TEXT,
            attempt INTEGER,
            latency_ms INTEGER,
            model TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_debug(task_id: str, agent_name: str, prompt_input: str,
              raw_output: str = None, error: str = None,
              attempt: int = 0, latency_ms: int = 0, model: str = ""):
    init_debug_db()
    conn = sqlite3.connect(DEBUG_DB)
    conn.execute("""
        INSERT INTO debug_logs VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        str(uuid.uuid4()),
        task_id,
        agent_name,
        prompt_input[:2000],  # cap input length
        raw_output[:2000] if raw_output else None,
        error,
        attempt,
        latency_ms,
        model,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def get_debug_logs(task_id: str = None) -> list[dict]:
    init_debug_db()
    conn = sqlite3.connect(DEBUG_DB)
    conn.row_factory = sqlite3.Row
    if task_id:
        rows = conn.execute(
            "SELECT * FROM debug_logs WHERE task_id=? ORDER BY timestamp ASC",
            (task_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM debug_logs ORDER BY timestamp DESC LIMIT 100"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_failed_calls() -> list[dict]:
    init_debug_db()
    conn = sqlite3.connect(DEBUG_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM debug_logs
        WHERE error IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 50
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_recent_task_ids() -> list[str]:
    init_debug_db()
    conn = sqlite3.connect(DEBUG_DB)
    rows = conn.execute("""
        SELECT DISTINCT task_id, MIN(timestamp) as first_seen
        FROM debug_logs
        GROUP BY task_id
        ORDER BY first_seen DESC
        LIMIT 20
    """).fetchall()
    conn.close()
    return [r[0] for r in rows]