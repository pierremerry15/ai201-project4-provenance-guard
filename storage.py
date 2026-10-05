"""
Persistence (planning.md §10).

- Content store (SQLite): the CURRENT state of each submission. Status changes on appeal.
- Audit log (JSONL): append-only HISTORY of every decision and appeal. Never rewritten.
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

import config

_log_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Content store
# ---------------------------------------------------------------------------

def _connect():
    os.makedirs(os.path.dirname(config.DB_FILE) or ".", exist_ok=True)
    conn = sqlite3.connect(config.DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS content (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL,
                text TEXT NOT NULL,
                attribution TEXT NOT NULL,
                confidence REAL NOT NULL,
                ai_score REAL NOT NULL,
                llm_score REAL,
                stylometric_score REAL NOT NULL,
                signals_json TEXT NOT NULL,
                gate_note TEXT,
                label TEXT NOT NULL,
                status TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                appeal_reasoning TEXT,
                appealed_at TEXT
            )"""
        )


def save_content(rec: dict):
    with _connect() as conn:
        conn.execute(
            """INSERT INTO content (content_id, creator_id, text, attribution, confidence, ai_score,
                   llm_score, stylometric_score, signals_json, gate_note, label, status, submitted_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rec["content_id"], rec["creator_id"], rec["text"], rec["attribution"],
             rec["confidence"], rec["ai_score"], rec["llm_score"], rec["stylometric_score"],
             json.dumps(rec["signals"]), rec["gate_note"], rec["label"], rec["status"],
             rec["timestamp"]),
        )


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["signals"] = json.loads(d.pop("signals_json"))
    return d


def get_content(content_id: str):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM content WHERE content_id = ?", (content_id,)).fetchone()
    return _row_to_dict(row) if row else None


def mark_under_review(content_id: str, reasoning: str, appealed_at: str) -> bool:
    """Atomically move classified -> under_review. Returns False if it was already under review."""
    with _connect() as conn:
        cur = conn.execute(
            """UPDATE content SET status = 'under_review', appeal_reasoning = ?, appealed_at = ?
               WHERE content_id = ? AND status != 'under_review'""",
            (reasoning, appealed_at, content_id),
        )
        return cur.rowcount == 1


def appeal_queue() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM content WHERE status = 'under_review' ORDER BY appealed_at ASC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def append_audit(entry: dict):
    """Append one JSON object as one line. Creates logs/ if missing."""
    os.makedirs(os.path.dirname(config.AUDIT_LOG_FILE) or ".", exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _log_lock, open(config.AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_audit(limit: int = 50) -> list[dict]:
    """Most recent `limit` entries, newest first. Skips corrupted lines instead of failing."""
    if not os.path.exists(config.AUDIT_LOG_FILE):
        return []
    entries = []
    with open(config.AUDIT_LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(entries))[:limit]
