"""
FocusMate storage module (SQLite).

All SQL for FocusMate lives in this file. Other modules never write SQL.
They call the functions below and get plain Python dictionaries back.

Tables:
    sessions        One row per study session, including the summary numbers.
    focus_readings  About one row per second while a session runs.
    break_events    Created at startup, but nothing writes to it in the
                    showcase. Planned event_type values for later:
                    warning_shown, break_prompted, break_taken, break_skipped.

Rules followed in this file:
    - Every function opens its own sqlite3 connection and closes it before
      returning, because Streamlit reruns the script often and a connection
      left open can cause "database is locked" errors.
    - Every query uses ? placeholders. Values are never put into SQL text
      with string formatting.
    - Times are stored as ISO text, for example 2026-11-01T10:30:00.
    - Only numbers and labels are stored. No webcam images, ever.
"""

import os
import sqlite3

# Location of the database file. Every function reads this variable at the
# moment it runs, so a test can point it at a temporary file first.
DB_PATH = "data/focusmate.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name            TEXT NOT NULL DEFAULT 'default',
    start_time           TEXT NOT NULL,
    end_time             TEXT,
    planned_duration_sec INTEGER,
    actual_duration_sec  INTEGER,
    status               TEXT NOT NULL DEFAULT 'active',
    avg_ear              REAL,
    total_blinks         INTEGER,
    face_missing_sec     INTEGER,
    looking_away_sec     INTEGER,
    avg_focus_preview    REAL
);

CREATE TABLE IF NOT EXISTS focus_readings (
    reading_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         INTEGER NOT NULL,
    timestamp          TEXT NOT NULL,
    elapsed_sec        REAL,
    face_detected      INTEGER NOT NULL,
    ear                REAL,
    blinks_in_interval INTEGER DEFAULT 0,
    head_direction     TEXT DEFAULT 'unknown',
    focus_preview      REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS break_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    details     TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_readings_session ON focus_readings(session_id);
"""


# ---------------------------------------------------------------------------
# Small private helpers
# ---------------------------------------------------------------------------

def _connect():
    """Open a new connection to DB_PATH that returns rows usable like dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # SQLite only checks FOREIGN KEY rules when this is switched on.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# Public functions (the interface other modules use)
# ---------------------------------------------------------------------------

def init_db():
    """Create the database file, its folder and all tables if they are missing.

    Safe to call many times. Existing tables and data are left as they are.
    """
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)

    conn = _connect()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
