"""
Tests for the FocusMate storage module (src/storage/database.py).

Run from the project root:
    python src/storage/test_database.py

Before anything runs, database.DB_PATH is pointed at a file inside a new
temporary folder. The real data/focusmate.db is never created, read or
changed. The temporary folder (and the test database in it) is deleted at
the end, so no manual cleanup is needed.

Optional: the same tests also run with pytest
    pytest src/storage/test_database.py
"""

import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback

try:
    from src.storage import database  # when run as a module from the project root
except ImportError:
    import database  # when run directly: python src/storage/test_database.py

# Exact key names of Contract C (session summary dictionary).
CONTRACT_C_KEYS = {
    "session_id", "user_name", "start_time", "end_time", "planned_duration_sec",
    "actual_duration_sec", "status", "avg_ear", "total_blinks", "face_missing_sec",
    "looking_away_sec", "avg_focus_preview",
}

_ORIGINAL_DB_PATH = database.DB_PATH
_TEMP_DIR = None


# ---------------------------------------------------------------------------
# Setup and teardown (pytest also calls these automatically)
# ---------------------------------------------------------------------------

def setup_module(module=None):
    """Point database.DB_PATH at a temporary file and create the tables."""
    global _TEMP_DIR
    _TEMP_DIR = tempfile.mkdtemp(prefix="focusmate_test_")
    # Use a "data" subfolder so the test also checks that init_db creates it.
    database.DB_PATH = os.path.join(_TEMP_DIR, "data", "focusmate.db")
    database.init_db()


def teardown_module(module=None):
    """Restore the real DB_PATH and delete the temporary folder."""
    database.DB_PATH = _ORIGINAL_DB_PATH
    if _TEMP_DIR:
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Fake data helpers
# ---------------------------------------------------------------------------

def _make_fake_readings():
    """Build 60 Contract B readings (one per second) with known patterns.

    Seconds 10 to 14: no face (face_detected False, ear None).
    Seconds 30 to 37: face found but looking left or right.
    Every 10th second with a face: drooping eyes (ear 0.20).
    Every 4th second with a face: one blink.
    Second 50: focus_preview is None, to check that empty values are ignored.
    """
    readings = []
    for second in range(60):
        if 10 <= second < 15:
            readings.append({
                "face_detected": False, "ear": None, "blinks_in_interval": 0,
                "head_direction": "unknown", "focus_preview": 0.0,
                "elapsed_sec": float(second),
            })
            continue

        if 30 <= second < 38:
            direction = "left" if second % 2 == 0 else "right"
        else:
            direction = "center"
        preview = 100.0 if direction == "center" else 60.0
        if second == 50:
            preview = None

        readings.append({
            "face_detected": True,
            "ear": 0.20 if second % 10 == 0 else 0.28,
            "blinks_in_interval": 1 if second % 4 == 0 else 0,
            "head_direction": direction,
            "focus_preview": preview,
            "elapsed_sec": float(second),
        })
    return readings


def _expected_summary(readings):
    """Calculate the expected summary numbers in plain Python, for comparison."""
    face_ears = [r["ear"] for r in readings if r["face_detected"] and r["ear"] is not None]
    previews = [r["focus_preview"] for r in readings if r["focus_preview"] is not None]
    return {
        "avg_ear": sum(face_ears) / len(face_ears),
        "total_blinks": sum(r["blinks_in_interval"] for r in readings),
        "face_missing_sec": sum(1 for r in readings if not r["face_detected"]),
        "looking_away_sec": sum(
            1 for r in readings if r["head_direction"] not in ("center", "unknown")),
        "avg_focus_preview": sum(previews) / len(previews),
    }


def _close(a, b):
    """True if two floats are equal apart from tiny rounding differences."""
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_uses_temporary_database():
    """The tests must never point at the real data/focusmate.db."""
    assert database.DB_PATH != _ORIGINAL_DB_PATH
    assert os.path.abspath(database.DB_PATH).startswith(os.path.abspath(_TEMP_DIR))
    assert os.path.exists(database.DB_PATH), "init_db should create the file and data/ folder"


def test_init_db_twice_and_tables_exist():
    """Calling init_db() again must not fail, and all tables plus the index exist."""
    database.init_db()
    database.init_db()
    conn = sqlite3.connect(database.DB_PATH)
    try:
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN (?, ?)", ("table", "index"))}
    finally:
        conn.close()
    for name in ("sessions", "focus_readings", "break_events", "idx_readings_session"):
        assert name in names, f"missing {name}"


def test_full_session_with_60_readings():
    """Create a session, save 60 readings, end it and check every summary number."""
    session_id = database.start_session("tester", 60)
    assert isinstance(session_id, int)

    before = database.get_session_summary(session_id)
    assert before["status"] == "active"
    assert before["end_time"] is None
    assert before["planned_duration_sec"] == 60

    readings = _make_fake_readings()
    for reading in readings:
        database.save_reading(session_id, reading)

    stored = database.get_readings(session_id)
    assert len(stored) == 60
    assert stored[10]["face_detected"] == 0 and stored[10]["ear"] is None
    assert stored[0]["face_detected"] == 1
    assert stored[0]["timestamp"], "save_reading should add a timestamp"

    database.end_session(session_id)
    summary = database.get_session_summary(session_id)
    expected = _expected_summary(readings)

    assert set(summary.keys()) == CONTRACT_C_KEYS
    assert summary["status"] == "ended"
    assert summary["end_time"] is not None
    assert summary["actual_duration_sec"] >= 0
    assert summary["user_name"] == "tester"
    assert _close(summary["avg_ear"], expected["avg_ear"])
    assert summary["total_blinks"] == expected["total_blinks"]
    assert summary["face_missing_sec"] == expected["face_missing_sec"] == 5
    assert summary["looking_away_sec"] == expected["looking_away_sec"] == 8
    assert _close(summary["avg_focus_preview"], expected["avg_focus_preview"])

    print("    Summary after 60 fake readings:")
    for key, value in summary.items():
        print(f"      {key:<22} {value}")


def test_end_session_without_readings():
    """With no readings, averages are None and counts are 0."""
    session_id = database.start_session("empty", 300)
    database.end_session(session_id)
    summary = database.get_session_summary(session_id)
    assert summary["status"] == "ended"
    assert summary["avg_ear"] is None
    assert summary["avg_focus_preview"] is None
    assert summary["total_blinks"] == 0
    assert summary["face_missing_sec"] == 0
    assert summary["looking_away_sec"] == 0


def test_end_session_twice_keeps_first_result():
    """A second end_session call (for example a double click) changes nothing."""
    session_id = database.start_session("double", 60)
    database.end_session(session_id)
    first = database.get_session_summary(session_id)
    time.sleep(1.1)  # make sure a second call would get a different end_time
    database.end_session(session_id)
    second = database.get_session_summary(session_id)
    assert first == second


def test_missing_session_is_handled():
    """Unknown session ids give None or empty results, never an error."""
    assert database.get_session_summary(999999) is None
    assert database.get_readings(999999) == []
    database.end_session(999999)
    database.delete_session(999999)


def test_bad_input_does_not_crash():
    """Bad values are stored as empty or 'unknown' instead of crashing."""
    session_id = database.start_session("   ", 60)
    assert database.get_session_summary(session_id)["user_name"] == "default"

    database.save_reading(session_id, {
        "face_detected": True, "ear": "not a number", "blinks_in_interval": None,
        "head_direction": "sideways", "focus_preview": float("nan"),
    })
    database.save_reading(session_id, None)

    stored = database.get_readings(session_id)
    assert len(stored) == 2
    assert stored[0]["ear"] is None
    assert stored[0]["blinks_in_interval"] == 0
    assert stored[0]["head_direction"] == "unknown"
    assert stored[0]["focus_preview"] is None
    assert stored[1]["face_detected"] == 0


def test_list_sessions_newest_first():
    """list_sessions returns Contract C dictionaries, newest first, and respects limit."""
    new_ids = [database.start_session(f"order{n}", 60) for n in range(3)]
    sessions = database.list_sessions(limit=100)
    ids = [s["session_id"] for s in sessions]
    assert ids == sorted(ids, reverse=True)
    assert ids[:3] == list(reversed(new_ids))
    assert all(set(s.keys()) == CONTRACT_C_KEYS for s in sessions)
    assert len(database.list_sessions(limit=2)) == 2


def test_delete_session_removes_readings():
    """delete_session removes the session row and all of its readings."""
    session_id = database.start_session("to_delete", 60)
    for reading in _make_fake_readings()[:5]:
        database.save_reading(session_id, reading)
    database.delete_session(session_id)
    assert database.get_session_summary(session_id) is None
    assert database.get_readings(session_id) == []


def test_data_is_kept_on_disk():
    """Saved sessions are in the file itself, so they survive an app restart."""
    session_id = database.start_session("persist", 60)
    database.end_session(session_id)
    # A completely separate connection, like a fresh app start would make.
    conn = sqlite3.connect(database.DB_PATH)
    try:
        row = conn.execute(
            "SELECT user_name, status FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row == ("persist", "ended")


def test_recreate_after_file_deleted():
    """Deleting the .db file and calling init_db() again recreates everything."""
    os.remove(database.DB_PATH)
    assert not os.path.exists(database.DB_PATH)
    database.init_db()
    assert os.path.exists(database.DB_PATH)
    session_id = database.start_session("fresh", 60)
    assert database.get_session_summary(session_id)["status"] == "active"


# Run order. test_recreate_after_file_deleted wipes the data, so it runs last.
TESTS = [
    test_uses_temporary_database,
    test_init_db_twice_and_tables_exist,
    test_full_session_with_60_readings,
    test_end_session_without_readings,
    test_end_session_twice_keeps_first_result,
    test_missing_session_is_handled,
    test_bad_input_does_not_crash,
    test_list_sessions_newest_first,
    test_delete_session_removes_readings,
    test_data_is_kept_on_disk,
    test_recreate_after_file_deleted,
]


def main():
    """Run every test, print PASS or FAIL for each, and clean up."""
    real_db_existed_before = os.path.exists(_ORIGINAL_DB_PATH)
    passed, failed = 0, 0

    setup_module()
    print(f"Using temporary database: {database.DB_PATH}\n")
    try:
        for test in TESTS:
            try:
                test()
                print(f"PASS  {test.__name__}")
                passed += 1
            except Exception:
                print(f"FAIL  {test.__name__}")
                traceback.print_exc()
                failed += 1
    finally:
        temp_dir = _TEMP_DIR
        teardown_module()

    print(f"\n{passed} passed, {failed} failed")
    print(f"Temporary folder removed: {not os.path.exists(temp_dir)}")
    real_db_unchanged = os.path.exists(_ORIGINAL_DB_PATH) == real_db_existed_before
    print(f"Real {_ORIGINAL_DB_PATH} untouched: {real_db_unchanged}")
    return 0 if failed == 0 and real_db_unchanged else 1


if __name__ == "__main__":
    sys.exit(main())
