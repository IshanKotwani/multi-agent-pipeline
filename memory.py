import hashlib
import subprocess
import json
import sys
import os

DB_PATH = "./chroma_db"

def _make_id(topic: str) -> str:
    return hashlib.md5(topic.lower().strip().encode()).hexdigest()

def store_research(topic: str, research_text: str, summary_text: str):
    """Store research by calling a subprocess to avoid ChromaDB threading issues."""
    try:
        data = {
            "action": "store",
            "topic": topic,
            "research": research_text,
            "summary": summary_text[:500]
        }
        subprocess.run(
            [sys.executable, "memory_worker.py"],
            input=json.dumps(data),
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"[Memory] Stored: '{topic}'")
    except Exception as e:
        print(f"[Memory] Store failed: {e}")

def retrieve_similar(topic: str, n_results: int = 2) -> list[dict]:
    """Retrieve similar research via subprocess."""
    try:
        data = {"action": "retrieve", "topic": topic, "n_results": n_results}
        result = subprocess.run(
            [sys.executable, "memory_worker.py"],
            input=json.dumps(data),
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.stdout.strip():
            items = json.loads(result.stdout.strip())
            for item in items:
                print(f"[Memory] Found: '{item['topic']}' similarity={item['similarity']}")
            return items
        return []
    except Exception as e:
        print(f"[Memory] Retrieve failed: {e}")
        return []