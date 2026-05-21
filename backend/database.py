"""
Gestionnaire SQLite — stockage profils, historique, métriques
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import SQLITE_PATH


class DatabaseManager:
    def __init__(self, db_path: Path = SQLITE_PATH):
        self.db_path = str(db_path)
        self._init_db()

    # ── Initialisation ────────────────────────────────────────────────────────
    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    prenom       TEXT,
                    profil_json  TEXT,
                    created_at   TEXT,
                    updated_at   TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT,
                    question        TEXT,
                    reponse         TEXT,
                    satisfaction    INTEGER,
                    pertinent       INTEGER,
                    personnalise    INTEGER,
                    commentaire     TEXT,
                    created_at      TEXT
                );

                CREATE TABLE IF NOT EXISTS metriques_ragas (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id          TEXT,
                    question            TEXT,
                    faithfulness        REAL,
                    answer_relevancy    REAL,
                    context_precision   REAL,
                    context_recall      REAL,
                    answer_correctness  REAL,
                    created_at          TEXT
                );
            """)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ── Sessions ──────────────────────────────────────────────────────────────
    def save_session(self, session_id: str, profil_dict: dict):
        now = datetime.now().isoformat()
        prenom = profil_dict.get("prenom", "")
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, prenom, profil_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    profil_json=excluded.profil_json,
                    updated_at=excluded.updated_at
            """, (session_id, prenom, json.dumps(profil_dict, ensure_ascii=False), now, now))

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT profil_json FROM sessions WHERE session_id=?",
                (session_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    # ── Messages ──────────────────────────────────────────────────────────────
    def save_message(self, session_id: str, role: str, content: str):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now)
            )

    def get_history(self, session_id: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
                (session_id,)
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    # ── Évaluations ───────────────────────────────────────────────────────────
    def save_evaluation(self, session_id: str, question: str, reponse: str,
                        satisfaction: int, pertinent: bool, personnalise: bool,
                        commentaire: str = ""):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO evaluations
                (session_id, question, reponse, satisfaction, pertinent, personnalise, commentaire, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, question, reponse, satisfaction,
                  int(pertinent), int(personnalise), commentaire, now))

    def get_stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(satisfaction) as avg_sat,
                    SUM(pertinent)*1.0/COUNT(*) as pct_pertinent,
                    SUM(personnalise)*1.0/COUNT(*) as pct_personnalise
                FROM evaluations
            """).fetchone()
            sessions_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            messages_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return {
            "total_evaluations": row[0] or 0,
            "avg_satisfaction": round(row[1] or 0, 2),
            "pct_pertinent": round((row[2] or 0) * 100, 1),
            "pct_personnalise": round((row[3] or 0) * 100, 1),
            "total_sessions": sessions_count,
            "total_messages": messages_count,
        }

    def get_ragas_stats(self) -> dict:
        with self._conn() as conn:
            try:
                row = conn.execute("""
                    SELECT AVG(faithfulness), AVG(answer_relevancy),
                        AVG(context_precision),
                        AVG(context_recall), AVG(answer_correctness)
                    FROM metriques_ragas
                """).fetchone()
            except Exception:
                row = (0, 0, 0, 0, 0)
        return {
            "faithfulness":        round(row[0] or 0, 3),
            "answer_relevancy":    round(row[1] or 0, 3),
            "context_precision":   round(row[2] or 0, 3),
            "context_recall":      round(row[3] or 0, 3),
            "answer_correctness":  round(row[4] or 0, 3),
        }
    def save_ragas_metric(self, session_id: str, question: str,
                      faithfulness: float, answer_relevancy: float,
                      context_precision: float,
                      context_recall: float = 0.0,
                      answer_correctness: float = 0.0):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            try:
                conn.execute("""
                    INSERT INTO metriques_ragas
                    (session_id, question, faithfulness, answer_relevancy,
                    context_precision, context_recall, answer_correctness, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, question, faithfulness, answer_relevancy,
                    context_precision, context_recall, answer_correctness, now))
            except Exception:
                conn.execute("""
                    INSERT INTO metriques_ragas
                    (session_id, question, faithfulness, answer_relevancy,
                    context_precision, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, question, faithfulness,
                    answer_relevancy, context_precision, now))