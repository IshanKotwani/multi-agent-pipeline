import json
import os
import sqlite3
import uuid
from datetime import datetime

EVAL_DB = "evals.db"

# Fixed eval set — ground truth topics each research output must cover
EVAL_CASES = [
    {
        "id": "eval_001",
        "topic": "what is inflation",
        "must_contain": ["price", "monetary", "central bank", "interest rate"],
        "min_words": 100
    },
    {
        "id": "eval_002",
        "topic": "NYMEX natural gas futures",
        "must_contain": ["henry hub", "contract", "settlement", "mmbtu"],
        "min_words": 150
    },
    {
        "id": "eval_003",
        "topic": "LLM agent observability",
        "must_contain": ["latency", "token", "cost", "monitoring"],
        "min_words": 100
    }
]

def init_eval_db():
    conn = sqlite3.connect(EVAL_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id TEXT PRIMARY KEY,
            prompt_version TEXT,
            eval_id TEXT,
            topic TEXT,
            validation_passed INTEGER,
            keyword_score REAL,
            length_score REAL,
            overall_score REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def score_output(research_text: str, eval_case: dict) -> dict:
    """Score a research output against the eval case criteria."""
    text_lower = research_text.lower()

    # Keyword coverage score
    hits = sum(1 for kw in eval_case["must_contain"] if kw.lower() in text_lower)
    keyword_score = hits / len(eval_case["must_contain"])

    # Length score
    word_count = len(research_text.split())
    length_score = min(1.0, word_count / eval_case["min_words"])

    # Overall
    overall_score = round((keyword_score * 0.6 + length_score * 0.4), 3)

    return {
        "keyword_score": round(keyword_score, 3),
        "length_score": round(length_score, 3),
        "overall_score": overall_score,
        "keywords_hit": hits,
        "keywords_total": len(eval_case["must_contain"]),
        "word_count": word_count
    }

def log_eval_result(prompt_version: str, eval_case: dict, research_text: str, validation_passed: bool):
    init_eval_db()
    scores = score_output(research_text, eval_case)
    conn = sqlite3.connect(EVAL_DB)
    conn.execute("""
        INSERT INTO eval_runs VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        str(uuid.uuid4()),
        prompt_version,
        eval_case["id"],
        eval_case["topic"],
        1 if validation_passed else 0,
        scores["keyword_score"],
        scores["length_score"],
        scores["overall_score"],
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()
    return scores

def get_eval_results() -> list[dict]:
    init_eval_db()
    conn = sqlite3.connect(EVAL_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM eval_runs ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_version_comparison() -> list[dict]:
    """Compare average scores across prompt versions."""
    init_eval_db()
    conn = sqlite3.connect(EVAL_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT prompt_version,
               COUNT(*) as runs,
               AVG(overall_score) as avg_score,
               AVG(keyword_score) as avg_keyword,
               AVG(length_score) as avg_length,
               SUM(validation_passed) * 100.0 / COUNT(*) as pass_rate
        FROM eval_runs
        GROUP BY prompt_version
        ORDER BY timestamp DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]