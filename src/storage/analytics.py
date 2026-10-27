"""
FocusMate analytics.

Simple numbers across many sessions, for the dashboard's analytics area.
Only finished sessions (status "ended") are counted, so a session that is
still running, or one that was never stopped, does not distort the results.
"""

from datetime import date, timedelta

try:
    from src.storage import database  # when imported from the project root
except ImportError:
    import database  # when run directly: python src/storage/analytics.py


def get_overall_stats():
    """Return totals over all ended sessions as a dictionary.

    Keys:
        total_sessions   number of ended sessions
        total_study_sec  sum of actual_duration_sec
        avg_focus        average of avg_focus_preview over ended sessions, or None
        best_session     Contract C dictionary of the ended session with the
                         highest avg_focus_preview (newest wins a tie), or None
        total_blinks     sum of total_blinks
    With no ended sessions, the counts are 0 and the rest is None.
    """
    conn = database._connect()
    try:
        totals = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(actual_duration_sec), 0), "
            "AVG(avg_focus_preview), COALESCE(SUM(total_blinks), 0) "
            "FROM sessions WHERE status = ?",
            ("ended",),
        ).fetchone()
        best = conn.execute(
            "SELECT session_id, user_name, start_time, end_time, planned_duration_sec, "
            "actual_duration_sec, status, avg_ear, total_blinks, face_missing_sec, "
            "looking_away_sec, avg_focus_preview "
            "FROM sessions WHERE status = ? AND avg_focus_preview IS NOT NULL "
            "ORDER BY avg_focus_preview DESC, start_time DESC, session_id DESC LIMIT 1",
            ("ended",),
        ).fetchone()
    finally:
        conn.close()

    return {
        "total_sessions": totals[0],
        "total_study_sec": totals[1],
        "avg_focus": totals[2],
        "best_session": dict(best) if best is not None else None,
        "total_blinks": totals[3],
    }


def get_daily_summary(days=7):
    """Return one dictionary per date with ended sessions, newest date first.

    Covers today and the days before it, `days` dates in total (days=7 means
    today plus the 6 days before). The date comes from start_time.
    Keys: date (YYYY-MM-DD), sessions, total_study_sec, avg_focus
    (average of avg_focus_preview for that date, or None).
    """
    days = database._to_int(days, 7)
    if days < 1:
        return []
    first_day = (date.today() - timedelta(days=days - 1)).isoformat()

    conn = database._connect()
    try:
        rows = conn.execute(
            "SELECT substr(start_time, 1, 10) AS day, COUNT(*) AS sessions, "
            "COALESCE(SUM(actual_duration_sec), 0) AS total_study_sec, "
            "AVG(avg_focus_preview) AS avg_focus "
            "FROM sessions WHERE status = ? AND substr(start_time, 1, 10) >= ? "
            "GROUP BY day ORDER BY day DESC",
            ("ended", first_day),
        ).fetchall()
    finally:
        conn.close()

    return [
        {"date": row["day"], "sessions": row["sessions"],
         "total_study_sec": row["total_study_sec"], "avg_focus": row["avg_focus"]}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Small demo: python src/storage/analytics.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    database.init_db()
    stats = get_overall_stats()
    print("Overall:")
    for key, value in stats.items():
        if key == "best_session" and value is not None:
            value = f"session {value['session_id']} ({value['avg_focus_preview']:.1f})"
        print(f"  {key:<16} {value}")
    print("\nLast 7 days:")
    for day in get_daily_summary():
        focus = "none" if day["avg_focus"] is None else f"{day['avg_focus']:.1f}"
        print(f"  {day['date']}  sessions={day['sessions']}  "
              f"minutes={day['total_study_sec'] // 60}  avg_focus={focus}")
