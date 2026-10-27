"""
FocusMate demo data seeder.

Fills the database with fake but realistic study sessions, so the past
sessions screen, the analytics and the charts are not empty at the showcase.

Every demo session:
    - has user_name "demo", so it can be removed again with clear_demo_data()
    - lies in the past (one of the last 6 days, start between 9:00 and 21:00)
    - is planned for 25 minutes and has one reading per second, like the app
    - follows Contract B for its readings and is summarised with the same
      calculation as end_session()

Demo readings are made with a fixed random seed, so every run gives the
same data. Only numbers and labels are stored, never images.
"""

import random
from datetime import date, datetime, timedelta

try:
    from src.storage import database  # when imported from the project root
except ImportError:
    import database  # when run directly: python src/storage/seed_demo_data.py

DEMO_USER = "demo"
DEMO_DURATION_SEC = 25 * 60      # 25 minute sessions, one reading per second
DEMO_DAYS = 6                    # sessions are spread over the last 6 days
RANDOM_SEED = 2026               # fixed seed, so the demo data is repeatable

# Focus profiles. Each value controls one part of the fake behaviour:
#   away       share of face-present time spent looking away
#   away_len   length range of one looking-away moment, in seconds
#   missing    share of time with no face in the frame
#   miss_len   length range of one no-face period, in seconds
#   closed     share of face-present seconds with eyes mostly closed (EAR < 0.21)
#   droop      share of face-present seconds with drooping eyes (0.21 to 0.24)
# "later" (fatigue profile only) replaces the values for the second half.
PROFILES = [
    {"name": "very focused", "away": 0.12, "away_len": (3, 8), "missing": 0.01,
     "miss_len": (3, 6), "closed": 0.08, "droop": 0.30},
    {"name": "medium", "away": 0.42, "away_len": (4, 15), "missing": 0.04,
     "miss_len": (3, 10), "closed": 0.22, "droop": 0.45},
    {"name": "weak", "away": 0.72, "away_len": (8, 30), "missing": 0.08,
     "miss_len": (5, 15), "closed": 0.58, "droop": 0.36},
    {"name": "fatigue", "away": 0.12, "away_len": (3, 8), "missing": 0.02,
     "miss_len": (3, 8), "closed": 0.08, "droop": 0.30,
     "later": {"away": 0.55, "away_len": (6, 20), "closed": 0.50, "droop": 0.40}},
    {"name": "face missing", "away": 0.25, "away_len": (4, 12), "missing": 0.40,
     "miss_len": (40, 120), "closed": 0.12, "droop": 0.30},
    {"name": "average", "away": 0.30, "away_len": (4, 12), "missing": 0.04,
     "miss_len": (3, 10), "closed": 0.15, "droop": 0.38},
]


# ---------------------------------------------------------------------------
# Fake reading generation (no database work here)
# ---------------------------------------------------------------------------

def _start_chance(share, length_range):
    """Chance per second of starting an episode, so it fills about `share` of the time."""
    mean_length = sum(length_range) / 2
    if share <= 0:
        return 0.0
    return share / (mean_length * (1 - share))


def _focus_preview(face_detected, ear, head_direction):
    """Preview focus value, using the same rules as Spec Section 4.2.

    A local copy, because the real compute_focus_preview() belongs to
    Muskan's module. Returns 0 when there is no face, exactly like the real
    function, so demo sessions average the same way as real sessions.
    """
    if not face_detected:
        return 0.0
    value = 100.0
    if head_direction != "center":
        value -= 40
    if ear < 0.21:
        value -= 40
    elif ear < 0.24:
        value -= 15
    return max(0.0, min(100.0, value))


def _make_readings(profile, rng):
    """Build one session of Contract B readings (one per second) for a profile."""
    total = DEMO_DURATION_SEC
    readings = []
    away_left = 0          # seconds left in the current looking-away moment
    away_direction = "center"
    missing_left = 0       # seconds left in the current no-face period

    for second in range(total):
        settings = dict(profile)
        if "later" in profile and second >= total // 2:
            settings.update(profile["later"])

        # No-face periods.
        if missing_left == 0 and rng.random() < _start_chance(
                settings["missing"], settings["miss_len"]):
            missing_left = rng.randint(*settings["miss_len"])
        if missing_left > 0:
            missing_left -= 1
            away_left = 0
            readings.append({
                "face_detected": False, "ear": None, "blinks_in_interval": 0,
                "head_direction": "unknown",
                "focus_preview": _focus_preview(False, None, "unknown"),
                "elapsed_sec": float(second + 1),
            })
            continue

        # Looking-away moments (left, right or down for a few seconds).
        if away_left == 0 and rng.random() < _start_chance(
                settings["away"], settings["away_len"]):
            away_left = rng.randint(*settings["away_len"])
            away_direction = rng.choice(["left", "right", "down"])
        if away_left > 0:
            away_left -= 1
            direction = away_direction
        else:
            direction = "center"

        # Eye openness for this second.
        pick = rng.random()
        if pick < settings["closed"]:
            ear = rng.uniform(0.18, 0.209)
        elif pick < settings["closed"] + settings["droop"]:
            ear = rng.uniform(0.21, 0.239)
        else:
            ear = rng.uniform(0.25, 0.33)
        ear = round(ear, 3)

        readings.append({
            "face_detected": True, "ear": ear, "blinks_in_interval": 0,
            "head_direction": direction,
            "focus_preview": _focus_preview(True, ear, direction),
            "elapsed_sec": float(second + 1),
        })

    # Blinks: 15 to 20 per minute over the whole session, spread over
    # seconds where a face is visible (a blink cannot be seen without one).
    blink_target = round(rng.uniform(15, 20) * total / 60)
    visible = [i for i, r in enumerate(readings) if r["face_detected"]]
    for _ in range(blink_target):
        if not visible:
            break
        readings[rng.choice(visible)]["blinks_in_interval"] += 1

    return readings


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def clear_demo_data():
    """Delete all sessions with user_name "demo", with their readings and break events.

    Real sessions are never touched. Returns the number of demo sessions removed.
    """
    conn = database._connect()
    try:
        with conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE user_name = ?", (DEMO_USER,)
            ).fetchone()[0]
            conn.execute(
                "DELETE FROM focus_readings WHERE session_id IN "
                "(SELECT session_id FROM sessions WHERE user_name = ?)", (DEMO_USER,))
            conn.execute(
                "DELETE FROM break_events WHERE session_id IN "
                "(SELECT session_id FROM sessions WHERE user_name = ?)", (DEMO_USER,))
            conn.execute("DELETE FROM sessions WHERE user_name = ?", (DEMO_USER,))
        return count
    finally:
        conn.close()


def seed_demo_sessions(count=6):
    """Create `count` finished demo sessions in the past and return their session_ids.

    Old demo data is cleared first, so seeding twice never duplicates
    anything. The tables are created if they are missing. Sessions cycle
    through the focus profiles in PROFILES and through the last 6 days.
    """
    count = max(0, database._to_int(count, 6))
    database.init_db()
    clear_demo_data()

    rng = random.Random(RANDOM_SEED)
    today = date.today()
    created = []

    conn = database._connect()
    try:
        with conn:  # one transaction: either all demo sessions are saved or none
            for index in range(count):
                profile = PROFILES[index % len(PROFILES)]
                day = today - timedelta(days=(index % DEMO_DAYS) + 1)
                # Start between 9:00 and 20:59, a few minutes apart on repeat days.
                start = datetime(day.year, day.month, day.day,
                                 rng.randint(9, 20), rng.randint(0, 59))
                end = start + timedelta(seconds=DEMO_DURATION_SEC)

                cursor = conn.execute(
                    "INSERT INTO sessions (user_name, start_time, planned_duration_sec, status) "
                    "VALUES (?, ?, ?, ?)",
                    (DEMO_USER, start.isoformat(timespec="seconds"),
                     DEMO_DURATION_SEC, "active"),
                )
                session_id = cursor.lastrowid

                rows = []
                for reading in _make_readings(profile, rng):
                    stamp = start + timedelta(seconds=reading["elapsed_sec"])
                    rows.append((
                        session_id,
                        stamp.isoformat(timespec="seconds"),
                        reading["elapsed_sec"],
                        1 if reading["face_detected"] else 0,
                        reading["ear"],
                        reading["blinks_in_interval"],
                        reading["head_direction"],
                        reading["focus_preview"],
                    ))
                conn.executemany(
                    "INSERT INTO focus_readings (session_id, timestamp, elapsed_sec, "
                    "face_detected, ear, blinks_in_interval, head_direction, focus_preview) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )

                # Same summary calculation as end_session(), but with the past end time.
                stats = database._calculate_summary(conn, session_id)
                conn.execute(
                    "UPDATE sessions SET end_time = ?, actual_duration_sec = ?, status = ?, "
                    "avg_ear = ?, total_blinks = ?, face_missing_sec = ?, "
                    "looking_away_sec = ?, avg_focus_preview = ? "
                    "WHERE session_id = ?",
                    (
                        end.isoformat(timespec="seconds"),
                        DEMO_DURATION_SEC,
                        "ended",
                        stats["avg_ear"],
                        stats["total_blinks"],
                        stats["face_missing_sec"],
                        stats["looking_away_sec"],
                        stats["avg_focus_preview"],
                        session_id,
                    ),
                )
                created.append(session_id)
    finally:
        conn.close()

    return created


# ---------------------------------------------------------------------------
# Run directly to prepare the real database for a demo:
#     python src/storage/seed_demo_data.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    database.init_db()
    removed = clear_demo_data()
    print(f"Removed {removed} old demo session(s).")
    new_ids = seed_demo_sessions()
    print(f"Created {len(new_ids)} demo session(s) in {database.DB_PATH}\n")

    header = f"{'id':>4}  {'start':<16}  {'min':>4}  {'avg focus':>9}  " \
             f"{'avg EAR':>7}  {'blinks':>6}  {'no face s':>9}  {'away s':>6}"
    print(header)
    print("-" * len(header))
    for new_id in new_ids:
        s = database.get_session_summary(new_id)
        print(f"{s['session_id']:>4}  {s['start_time'][:16].replace('T', ' '):<16}  "
              f"{s['actual_duration_sec'] // 60:>4}  {s['avg_focus_preview']:>9.1f}  "
              f"{s['avg_ear']:>7.3f}  {s['total_blinks']:>6}  "
              f"{s['face_missing_sec']:>9}  {s['looking_away_sec']:>6}")
