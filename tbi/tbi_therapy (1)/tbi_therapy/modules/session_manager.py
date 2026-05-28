"""SQLite-based session manager for patient data and therapy history."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from config import DATABASE_PATH


# ---------------------------------------------------------------
# Connection
# ---------------------------------------------------------------
@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------
# Schema
# ---------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    tbi_date TEXT,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT,
    average_accuracy REAL,
    dominant_severity TEXT,
    notes TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    target_text TEXT NOT NULL,
    predicted_phonemes TEXT,
    reference_phonemes TEXT,
    phoneme_acc REAL,
    severity TEXT,
    severity_score INTEGER,
    speech_rate REAL,
    pause_ratio REAL,
    fluency REAL,
    duration_sec REAL,
    lip_closure REAL,
    facial_asymmetry REAL,
    jaw_drop REAL,
    rl_action TEXT,
    rl_next_difficulty TEXT,
    full_data_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_id);
"""


def init_db():
    with get_db() as conn:
        conn.executescript(SCHEMA)
    print(f"[session_manager] initialized DB at {DATABASE_PATH}")


# ---------------------------------------------------------------
# Patients
# ---------------------------------------------------------------
def create_patient(name: str, age: Optional[int] = None,
                   tbi_date: Optional[str] = None, notes: str = "") -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO patients (name, age, tbi_date, notes) VALUES (?, ?, ?, ?)",
            (name, age, tbi_date, notes),
        )
        return cur.lastrowid


def list_patients() -> list:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_patient(patient_id: int) -> Optional[dict]:
    with get_db() as conn:
        r = conn.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
        return dict(r) if r else None


# ---------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------
def start_session(patient_id: int) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (patient_id) VALUES (?)", (patient_id,)
        )
        return cur.lastrowid


def end_session(session_id: int, average_accuracy: float,
                dominant_severity: str, notes: str = ""):
    with get_db() as conn:
        conn.execute(
            """UPDATE sessions
                  SET ended_at=?, average_accuracy=?, dominant_severity=?, notes=?
                WHERE id=?""",
            (datetime.utcnow().isoformat(), average_accuracy,
             dominant_severity, notes, session_id),
        )


def list_sessions(patient_id: int) -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE patient_id=? ORDER BY started_at DESC",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------
# Attempts (individual utterances)
# ---------------------------------------------------------------
def record_attempt(session_id: int, target_text: str,
                   analysis: dict, severity_info: dict,
                   facial_info: Optional[dict] = None,
                   rl_action: Optional[str] = None,
                   rl_next: Optional[str] = None) -> int:
    facial_info = facial_info or {}

    audio_duration = analysis.get("duration_sec", 0)
    pauses = analysis.get("pauses", 0)
    pause_ratio = (pauses / (audio_duration * 16000)) if audio_duration > 0 else 0

    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO attempts (
                session_id, target_text,
                predicted_phonemes, reference_phonemes,
                phoneme_acc, severity, severity_score,
                speech_rate, pause_ratio, fluency, duration_sec,
                lip_closure, facial_asymmetry, jaw_drop,
                rl_action, rl_next_difficulty, full_data_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, target_text,
                " ".join(analysis.get("predicted_phonemes", [])),
                " ".join(analysis.get("reference_phonemes", [])),
                analysis.get("phoneme_acc", 0),
                severity_info.get("severity"),
                severity_info.get("score"),
                analysis.get("speech_rate", 0),
                pause_ratio,
                analysis.get("fluency", 0),
                audio_duration,
                facial_info.get("lip_closure"),
                facial_info.get("facial_asymmetry"),
                facial_info.get("jaw_drop_ratio"),
                rl_action,
                rl_next,
                json.dumps({"analysis": analysis,
                            "severity": severity_info,
                            "facial": facial_info}),
            ),
        )
        return cur.lastrowid


def get_recent_accuracy_history(patient_id: int, n: int = 10) -> list:
    """Return the last N phoneme accuracies for this patient (for trend)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT a.phoneme_acc, a.timestamp
                 FROM attempts a
                 JOIN sessions s ON a.session_id = s.id
                WHERE s.patient_id = ?
                ORDER BY a.timestamp DESC
                LIMIT ?""",
            (patient_id, n),
        ).fetchall()
        return [dict(r) for r in rows]


def get_session_attempts(session_id: int) -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM attempts WHERE session_id=? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def session_count_for_patient(patient_id: int) -> int:
    with get_db() as conn:
        r = conn.execute(
            "SELECT COUNT(*) AS c FROM sessions WHERE patient_id=?",
            (patient_id,),
        ).fetchone()
        return r["c"] if r else 0
