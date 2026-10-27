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

import math
import os
import sqlite3
import sys
from datetime import datetime

# Location of the database file. Every function reads this variable at the
# moment it runs, so a test can point it at a temporary file first.
DB_PATH = "data/focusmate.db"

# Head direction labels allowed by Contract A and Contract B.
VALID_DIRECTIONS = ("center", "left", "right", "up", "down", "unknown")

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


def _now_iso():
    """Return the current local time as ISO text, to the second."""
    return datetime.now().isoformat(timespec="seconds")


def _to_float(value):
    """Return value as a float, or None if it is missing or not a valid number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _to_int(value, default):
    """Return value as an int, or the given default if it cannot be converted."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def start_session(user_name, planned_duration_sec):
    """Create a new session row with status 'active' and return its session_id.

    An empty or missing user_name is stored as 'default'.
    """
    name = str(user_name).strip() if user_name is not None else ""
    if not name:
        name = "default"
    planned = _to_int(planned_duration_sec, None)

    conn = _connect()
    try:
        with conn:  # commits on success, rolls back on error
            cursor = conn.execute(
                "INSERT INTO sessions (user_name, start_time, planned_duration_sec, status) "
                "VALUES (?, ?, ?, ?)",
                (name, _now_iso(), planned, "active"),
            )
        return cursor.lastrowid
    finally:
        conn.close()


def save_reading(session_id, reading):
    """Store one reading (Contract B dictionary) for a running session.

    The timestamp is added here, so the caller does not send it. Bad or
    missing values are stored as empty (NULL) instead of crashing, and an
    unknown head direction is stored as 'unknown'. This runs about once per
    second inside the live loop, so a database error is reported as a
    warning and the loop keeps going.
    """
    if not isinstance(reading, dict):
        reading = {}

    face_detected = 1 if reading.get("face_detected") else 0
    direction = reading.get("head_direction")
    if direction not in VALID_DIRECTIONS:
        direction = "unknown"

    values = (
        session_id,
        _now_iso(),
        _to_float(reading.get("elapsed_sec")),
        face_detected,
        _to_float(reading.get("ear")),
        _to_int(reading.get("blinks_in_interval"), 0),
        direction,
        _to_float(reading.get("focus_preview")),
    )

    try:
        conn = _connect()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO focus_readings (session_id, timestamp, elapsed_sec, "
                    "face_detected, ear, blinks_in_interval, head_direction, focus_preview) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
        finally:
            conn.close()
    except sqlite3.Error as error:
        print(f"[storage] Warning: reading not saved for session {session_id}: {error}",
              file=sys.stderr)
