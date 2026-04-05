import sqlite3
import uuid
from datetime import datetime

DB_PATH = "pipeline.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_logs (
            id TEXT PRIMARY KEY,
            task_id TEXT,
            agent_name TEXT,
            status TEXT,
            passed INTEGER,
            model TEXT,
            latency_ms INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

PRICING = {
    "anthropic/claude-haiku-4-5":  {"input": 0.80,  "output": 4.00},
    "anthropic/claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},
}

def log_agent_call(task_id: str, agent_name: str, result: dict, status: str = "success", passed: int = -1):
    model = result.get("model", "anthropic/claude-haiku-4-5")
    pricing = PRICING.get(model, PRICING["anthropic/claude-haiku-4-5"])
    cost = (
        result["input_tokens"]  / 1_000_000 * pricing["input"] +
        result["output_tokens"] / 1_000_000 * pricing["output"]
    )
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO agent_logs VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        str(uuid.uuid4()),
        task_id,
        agent_name,
        status,
        passed,
        model,
        result["latency_ms"],
        result["input_tokens"],
        result["output_tokens"],
        round(cost, 6),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def get_all_logs():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM agent_logs ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]