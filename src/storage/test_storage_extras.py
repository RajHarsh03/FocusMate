"""
Tests for the FocusMate storage extras: demo seeder, chart, analytics and
CSV export (seed_demo_data.py, charts.py, analytics.py, export.py).

Run from the project root (needs matplotlib installed):
    python src/storage/test_storage_extras.py

Before anything runs, database.DB_PATH is pointed at a file inside a new
temporary folder, so the real data/focusmate.db is never created, read or
changed. The temporary folder is deleted at the end.

Optional: the same tests also run with pytest
    pytest src/storage/test_storage_extras.py
"""

import csv
import io
import math
import os
import shutil
import sys
import tempfile
import traceback
from datetime import date, datetime

try:
    from src.storage import database, seed_demo_data, charts, analytics, export
except ImportError:
    import database
    import seed_demo_data
    import charts
    import analytics
    import export

from matplotlib.figure import Figure

CONTRACT_C_KEYS = {
    "session_id", "user_name", "start_time", "end_time", "planned_duration_sec",
    "actual_duration_sec", "status", "avg_ear", "total_blinks", "face_missing_sec",
    "looking_away_sec", "avg_focus_preview",
}
OVERALL_KEYS = {"total_sessions", "total_study_sec", "avg_focus", "best_session", "total_blinks"}
DAILY_KEYS = {"date", "sessions", "total_study_sec", "avg_focus"}
DEMO_READINGS = 25 * 60

_ORIGINAL_DB_PATH = database.DB_PATH
_TEMP_DIR = None


# ---------------------------------------------------------------------------
# Setup and teardown (pytest also calls these automatically)
# ---------------------------------------------------------------------------

def setup_module(module=None):
    """Point database.DB_PATH at a temporary file and create the tables."""
    global _TEMP_DIR
    _TEMP_DIR = tempfile.mkdtemp(prefix="focusmate_extras_test_")
    database.DB_PATH = os.path.join(_TEMP_DIR, "data", "focusmate.db")
    database.init_db()


def teardown_module(module=None):
    """Restore the real DB_PATH and delete the temporary folder."""
    database.DB_PATH = _ORIGINAL_DB_PATH
    if _TEMP_DIR:
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count(sql, params=()):
    """Run a COUNT query on the temporary database and return the number."""
    conn = database._connect()
    try:
        return conn.execute(sql, params).fetchone()[0]
    finally:
        conn.close()


def _demo_sessions():
    """All demo sessions as Contract C dictionaries."""
    return [s for s in database.list_sessions(limit=1000) if s["user_name"] == "demo"]


def _make_real_session(name="harsh"):
    """Create one short ended session through the normal functions."""
    session_id = database.start_session(name, 60)
    for second in range(5):
        database.save_reading(session_id, {
            "face_detected": True, "ear": 0.29, "blinks_in_interval": 1,
            "head_direction": "center", "focus_preview": 100.0,
            "elapsed_sec": float(second + 1),
        })
    database.end_session(session_id)
    return session_id


def _close(a, b, tol=1e-9):
    """True if two floats are equal apart from small rounding differences."""
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# ---------------------------------------------------------------------------
# Tests (run in this order; each one documents what state it expects)
# ---------------------------------------------------------------------------

def test_uses_temporary_database():
    """All modules share one database module, pointed at the temporary file."""
    assert database.DB_PATH != _ORIGINAL_DB_PATH
    assert os.path.abspath(database.DB_PATH).startswith(os.path.abspath(_TEMP_DIR))
    for module in (seed_demo_data, charts, analytics, export):
        assert module.database is database, f"{module.__name__} uses another database module"


def test_overall_stats_without_data():
    """With no ended sessions, counts are 0 and the rest is None."""
    stats = analytics.get_overall_stats()
    assert set(stats.keys()) == OVERALL_KEYS
    assert stats == {"total_sessions": 0, "total_study_sec": 0, "avg_focus": None,
                     "best_session": None, "total_blinks": 0}
    # A session that is still running does not count.
    database.start_session("still_running", 60)
    assert analytics.get_overall_stats()["total_sessions"] == 0
    assert analytics.get_daily_summary() == []


def test_seed_creates_sessions_and_keeps_real_ones():
    """Seeding creates 6 realistic demo sessions; clear_demo_data keeps real sessions."""
    real_id = _make_real_session()
    ids = seed_demo_data.seed_demo_sessions()
    assert len(ids) == 6

    demo = _demo_sessions()
    assert sorted(s["session_id"] for s in demo) == sorted(ids)

    today = date.today()
    days = set()
    focus_values = []
    for s in demo:
        start = datetime.fromisoformat(s["start_time"])
        end = datetime.fromisoformat(s["end_time"])
        assert s["status"] == "ended"
        assert start.date() < today, "demo sessions must lie in the past"
        assert 9 <= start.hour < 21, "start must be between 9:00 and 21:00"
        assert s["planned_duration_sec"] == s["actual_duration_sec"] == DEMO_READINGS
        assert (end - start).total_seconds() == DEMO_READINGS
        days.add(start.date())
        focus_values.append(s["avg_focus_preview"])

        readings = database.get_readings(s["session_id"])
        assert len(readings) == DEMO_READINGS, "one reading per second"
        assert readings[0]["timestamp"] > s["start_time"]
        assert readings[-1]["timestamp"] <= s["end_time"]

        blinks_per_min = s["total_blinks"] / (DEMO_READINGS / 60)
        assert 15 <= blinks_per_min <= 20, f"blinks per minute {blinks_per_min}"

        directions = [r["head_direction"] for r in readings if r["face_detected"]]
        assert set(directions) <= {"center", "left", "right", "down"}
        assert directions.count("center") > 0
        for r in readings:
            if r["face_detected"]:
                assert 0.18 <= r["ear"] <= 0.33
            else:
                # Same as the real preview function (Spec 4.2): no face gives 0.
                assert r["ear"] is None and r["focus_preview"] == 0.0
                assert r["head_direction"] == "unknown"

        # The summary must equal the end_session calculation on the same readings.
        present = [r for r in readings if r["face_detected"]]
        previews = [r["focus_preview"] for r in readings if r["focus_preview"] is not None]
        assert _close(s["avg_ear"], sum(r["ear"] for r in present) / len(present))
        assert s["total_blinks"] == sum(r["blinks_in_interval"] for r in readings)
        assert s["face_missing_sec"] == len(readings) - len(present)
        assert s["looking_away_sec"] == sum(
            1 for r in readings if r["head_direction"] not in ("center", "unknown"))
        assert _close(s["avg_focus_preview"], sum(previews) / len(previews))

    assert len(days) == 6, "6 sessions on 6 different days"
    # Different focus profiles: one strong, one weak, clear spread overall.
    assert max(focus_values) >= 80 and min(focus_values) <= 50
    assert any(s["face_missing_sec"] >= 300 for s in demo), "one session with long face-missing"

    # Clearing removes only demo data.
    removed = seed_demo_data.clear_demo_data()
    assert removed == 6
    assert _demo_sessions() == []
    assert database.get_session_summary(real_id)["user_name"] == "harsh"
    assert len(database.get_readings(real_id)) == 5
    orphans = _count(
        "SELECT COUNT(*) FROM focus_readings WHERE session_id NOT IN "
        "(SELECT session_id FROM sessions)")
    assert orphans == 0


def test_seed_twice_does_not_duplicate():
    """Seeding twice still gives exactly 6 demo sessions and 6 x 1500 readings."""
    seed_demo_data.seed_demo_sessions()
    seed_demo_data.seed_demo_sessions()
    demo_ids = [s["session_id"] for s in _demo_sessions()]
    assert len(demo_ids) == 6
    demo_readings = _count(
        "SELECT COUNT(*) FROM focus_readings WHERE session_id IN "
        "(SELECT session_id FROM sessions WHERE user_name = ?)", ("demo",))
    assert demo_readings == 6 * DEMO_READINGS


def test_seed_custom_count():
    """count controls the number of demo sessions, and 0 creates none."""
    assert len(seed_demo_data.seed_demo_sessions(count=3)) == 3
    assert len(_demo_sessions()) == 3
    assert seed_demo_data.seed_demo_sessions(count=0) == []
    assert _demo_sessions() == []
    seed_demo_data.seed_demo_sessions()  # leave 6 demo sessions for the next tests


def test_focus_chart():
    """make_focus_chart returns a Figure with the right axes, or None without readings."""
    demo = _demo_sessions()
    # The face-missing profile has the most missing seconds.
    gappy = max(demo, key=lambda s: s["face_missing_sec"])
    fig = charts.make_focus_chart(gappy["session_id"])
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_ylim() == (0.0, 100.0)
    assert ax.get_xlabel() == "Time (minutes)"
    assert str(gappy["session_id"]) in ax.get_title()
    assert len(ax.lines) == 2, "raw line and moving average"

    raw_y = ax.lines[0].get_ydata()
    smooth_y = ax.lines[1].get_ydata()
    missing = gappy["face_missing_sec"]
    assert sum(1 for v in raw_y if math.isnan(v)) == missing, "line must break at gaps"
    assert sum(1 for v in smooth_y if math.isnan(v)) == missing
    assert max(ax.lines[0].get_xdata()) == DEMO_READINGS / 60

    # Works without a screen: render to PNG in memory.
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    assert buffer.getvalue()[:8] == b"\x89PNG\r\n\x1a\n"

    empty_id = database.start_session("no_readings", 60)
    assert charts.make_focus_chart(empty_id) is None
    assert charts.make_focus_chart(999999) is None


def test_overall_stats_with_data():
    """Totals, average and best session match the ended sessions in the database."""
    ended = [s for s in database.list_sessions(limit=1000) if s["status"] == "ended"]
    stats = analytics.get_overall_stats()
    assert set(stats.keys()) == OVERALL_KEYS
    assert stats["total_sessions"] == len(ended)
    assert stats["total_study_sec"] == sum(s["actual_duration_sec"] for s in ended)
    assert stats["total_blinks"] == sum(s["total_blinks"] for s in ended)
    focus = [s["avg_focus_preview"] for s in ended if s["avg_focus_preview"] is not None]
    assert _close(stats["avg_focus"], sum(focus) / len(focus))
    best = stats["best_session"]
    assert set(best.keys()) == CONTRACT_C_KEYS
    assert best["avg_focus_preview"] == max(focus)


def test_daily_summary_groups_by_date():
    """One row per date, newest first; two sessions on one date are grouped."""
    seed_demo_data.seed_demo_sessions(count=8)  # 8 sessions over 6 days
    days = analytics.get_daily_summary(days=7)
    assert all(set(d.keys()) == DAILY_KEYS for d in days)
    dates = [d["date"] for d in days]
    assert dates == sorted(dates, reverse=True)
    assert len(dates) == len(set(dates))

    ended = [s for s in database.list_sessions(limit=1000) if s["status"] == "ended"]
    assert sum(d["sessions"] for d in days) == len(ended)
    assert sum(1 for d in days if d["sessions"] == 2) == 2, "two dates with 2 demo sessions"

    for d in days:
        same_day = [s for s in ended if s["start_time"][:10] == d["date"]]
        assert d["sessions"] == len(same_day)
        assert d["total_study_sec"] == sum(s["actual_duration_sec"] for s in same_day)

    # days=1 covers only today: just the real session made earlier.
    today_only = analytics.get_daily_summary(days=1)
    assert [d["date"] for d in today_only] == [date.today().isoformat()]
    assert analytics.get_daily_summary(days=0) == []
    seed_demo_data.seed_demo_sessions()  # back to 6 demo sessions


def test_export_session_csv():
    """CSV has a header and one row per reading; None becomes an empty cell."""
    demo = _demo_sessions()
    gappy = max(demo, key=lambda s: s["face_missing_sec"])
    sid = gappy["session_id"]

    path = export.export_session_csv(sid)
    expected_default = os.path.join(os.path.dirname(database.DB_PATH), "exports",
                                    f"session_{sid}.csv")
    assert path == expected_default
    assert os.path.exists(path)

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == export.READING_COLUMNS
    assert len(rows) - 1 == len(database.get_readings(sid)) == DEMO_READINGS
    face_col = rows[0].index("face_detected")
    ear_col = rows[0].index("ear")
    missing_rows = [r for r in rows[1:] if r[face_col] == "0"]
    assert len(missing_rows) == gappy["face_missing_sec"]
    assert all(r[ear_col] == "" for r in missing_rows), "None must be an empty cell"

    custom = os.path.join(_TEMP_DIR, "nested", "folder", "out.csv")
    assert export.export_session_csv(sid, custom) == custom
    assert os.path.exists(custom)

    empty_id = database.start_session("export_empty", 60)
    empty_path = export.export_session_csv(empty_id)
    with open(empty_path, newline="", encoding="utf-8") as handle:
        assert len(list(csv.reader(handle))) == 1, "header only"

    assert export.export_session_csv(999999) is None
    assert not os.path.exists(os.path.join(os.path.dirname(database.DB_PATH), "exports",
                                           "session_999999.csv"))


TESTS = [
    test_uses_temporary_database,
    test_overall_stats_without_data,
    test_seed_creates_sessions_and_keeps_real_ones,
    test_seed_twice_does_not_duplicate,
    test_seed_custom_count,
    test_focus_chart,
    test_overall_stats_with_data,
    test_daily_summary_groups_by_date,
    test_export_session_csv,
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
