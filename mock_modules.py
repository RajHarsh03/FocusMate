"""
FocusMate mock modules (for building and testing the dashboard).

This file has fake versions of every function the dashboard (app.py) calls,
with the SAME names, the SAME arguments and the SAME return formats as the
real modules:

    Vision   (Qaem)   : create_landmarker, open_camera, process_frame, close_camera
    Detection (Muskan): compute_ear, BlinkDetector, compute_focus_preview
    Timer    (Muskan) : SessionTimer
    Storage  (Harsh)  : init_db, start_session, save_reading, end_session,
                        get_session_summary, get_readings, list_sessions,
                        delete_session, make_focus_chart, get_overall_stats,
                        get_daily_summary, export_session_csv

Why it exists: the dashboard can be built and demonstrated before the other
modules are joined in. In app.py, USE_MOCK = True uses this file and
USE_MOCK = False uses the real modules. Nothing in app.py changes otherwise.

How the fake data behaves:
    - The fake camera draws a simple moving picture. No webcam is used.
    - process_frame() follows a fixed 20 second script, so the dashboard shows
      every case: looking at the screen, blinking, looking away, drooping eyes
      and no face.
    - Storage is kept in memory only. It disappears when the app stops, and
      the real database file (data/focusmate.db) is never touched.

Nothing here is the real logic. The numbers are made up to look realistic.
"""

import csv
import os
import tempfile
import time
from datetime import date, datetime, timedelta

import numpy as np
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Settings you can change to test the dashboard
# ---------------------------------------------------------------------------

# Set to True to test the "camera cannot open" message in the dashboard.
MOCK_CAMERA_FAILS = False

# The fake camera waits this long per frame, so it behaves like a webcam
# running at about 30 frames per second.
MOCK_FRAME_DELAY_SEC = 1 / 30

# Head direction labels allowed by Contract A and Contract B.
VALID_DIRECTIONS = ("center", "left", "right", "up", "down", "unknown")

# The fixed script that process_frame() follows, repeated every 20 seconds.
# Each entry: (start second, end second, what is happening)
_SCRIPT = [
    (0, 8, "focused"),       # looking at the screen, normal blinks
    (8, 11, "looking_left"),
    (11, 14, "drooping"),    # eyes half closed
    (14, 16, "no_face"),     # user left the frame
    (16, 20, "focused"),
]
_SCRIPT_LENGTH_SEC = 20

# The EAR value of the most recent fake frame. process_frame() sets it and
# compute_ear() reads it, so the two fake functions agree with each other.
_current_fake_ear = None


# ---------------------------------------------------------------------------
# Vision (fake version of src/vision/vision.py)
# ---------------------------------------------------------------------------

class _FakeLandmarker:
    """Stands in for the MediaPipe FaceLandmarker object."""

    def close(self):
        """Matches the real object's close() so cleanup code works the same."""


class _FakeCamera:
    """Stands in for cv2.VideoCapture. read() returns a drawn test picture."""

    def __init__(self):
        self._opened = True
        self._frame_number = 0

    def isOpened(self):  # same name as the OpenCV method
        """Return True until release() is called."""
        return self._opened

    def read(self):
        """Return (True, frame) like OpenCV, or (False, None) after release."""
        if not self._opened:
            return False, None
        time.sleep(MOCK_FRAME_DELAY_SEC)
        self._frame_number += 1

        # A plain dark picture (480 high, 640 wide, BGR) with a moving dot.
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        x = 320 + int(200 * np.sin(self._frame_number / 20))
        frame[230:250, x - 10:x + 10] = (0, 200, 255)
        frame[20:24, 20:620] = (200, 200, 200)
        return True, frame

    def release(self):
        """Close the fake camera."""
        self._opened = False


def create_landmarker(model_path="models/face_landmarker.task"):
    """Return a fake landmarker. The model file is not needed in mock mode."""
    return _FakeLandmarker()


def open_camera(index=0):
    """Return a fake camera, or None when MOCK_CAMERA_FAILS is True."""
    if MOCK_CAMERA_FAILS:
        return None
    return _FakeCamera()


def close_camera(cap):
    """Release the camera, same as the real function."""
    if cap is not None:
        cap.release()


def _script_step(now):
    """Return what the fake user is doing at this moment of the 20 second script."""
    second = now % _SCRIPT_LENGTH_SEC
    for start, end, step in _SCRIPT:
        if start <= second < end:
            return step, second
    return "focused", second


def process_frame(frame_bgr, landmarker):
    """Return a Contract A dictionary that follows the fixed fake script.

    Contract A keys: face_detected, landmarks (478 (x, y) tuples or None),
    head_direction, yaw, pitch.
    """
    global _current_fake_ear

    if frame_bgr is None:
        _current_fake_ear = None
        return {"face_detected": False, "landmarks": None,
                "head_direction": "unknown", "yaw": None, "pitch": None}

    step, second = _script_step(time.monotonic())

    if step == "no_face":
        _current_fake_ear = None
        return {"face_detected": False, "landmarks": None,
                "head_direction": "unknown", "yaw": None, "pitch": None}

    # Open eyes are around 0.29. A short dip below 0.21 near the start of
    # every 3 second block is a blink (lasts about 0.2 seconds).
    if step == "drooping":
        ear = 0.225
    elif (second % 3) < 0.2:
        ear = 0.15
    else:
        ear = 0.29
    _current_fake_ear = ear

    if step == "looking_left":
        direction, yaw, pitch = "left", 28.0, -2.0
    else:
        direction, yaw, pitch = "center", 3.0, -1.5

    # 478 fake landmark points in pixel coordinates (the values do not matter
    # in mock mode, only the format does).
    landmarks = [(320 + (i % 30), 240 + (i // 30)) for i in range(478)]

    return {"face_detected": True, "landmarks": landmarks,
            "head_direction": direction, "yaw": yaw, "pitch": pitch}


# ---------------------------------------------------------------------------
# Detection (fake versions of src/detection/ear_blink.py and focus_preview.py)
# ---------------------------------------------------------------------------

def compute_ear(landmarks):
    """Return the fake EAR of the current frame, or None if there is no face."""
    if landmarks is None:
        return None
    return _current_fake_ear


class BlinkDetector:
    """Fake blink counter with the same methods and attribute as the real one.

    Counts one blink when the EAR stays below the threshold for at least
    min_frames frames and then comes back up.
    """

    def __init__(self, ear_threshold=0.21, min_frames=2):
        """Create the counter with the same default settings as the real one."""
        self.ear_threshold = ear_threshold
        self.min_frames = min_frames
        self.total_blinks = 0
        self._closed_frames = 0

    def reset(self):
        """Set the blink count back to 0."""
        self.total_blinks = 0
        self._closed_frames = 0

    def update(self, ear):
        """Return True on the frame a blink finishes, otherwise False."""
        if ear is None:
            self._closed_frames = 0
            return False
        if ear <= self.ear_threshold:
            self._closed_frames += 1
            return False
        blinked = self._closed_frames >= self.min_frames
        self._closed_frames = 0
        if blinked:
            self.total_blinks += 1
        return blinked


def compute_focus_preview(face_detected, ear, head_direction):
    """Return a fake preview focus value from 0 to 100 (0 when there is no face)."""
    if not face_detected:
        return 0.0
    value = 100.0
    if head_direction != "center":
        value -= 40.0
    if ear is not None and ear < 0.21:
        value -= 40.0
    elif ear is not None and ear < 0.24:
        value -= 15.0
    return max(0.0, min(100.0, value))


# ---------------------------------------------------------------------------
# Timer (fake version of src/timer/session_timer.py)
# ---------------------------------------------------------------------------

class SessionTimer:
    """Fake countdown timer with the same methods as the real SessionTimer."""

    def __init__(self, planned_duration_sec):
        """Create a timer for the planned number of seconds."""
        self.planned_duration_sec = max(0.0, float(planned_duration_sec))
        self._start = None
        self._stop = None

    def start(self):
        """Start the timer. A second call while running does nothing."""
        if self._start is not None and self._stop is None:
            return
        self._start = time.monotonic()
        self._stop = None

    def stop(self):
        """Stop the timer so elapsed_sec() stays fixed."""
        if self._start is not None and self._stop is None:
            self._stop = time.monotonic()

    def elapsed_sec(self):
        """Seconds since start (0.0 before start, frozen after stop)."""
        if self._start is None:
            return 0.0
        end = self._stop if self._stop is not None else time.monotonic()
        return max(0.0, end - self._start)

    def remaining_sec(self):
        """Seconds left, never below 0."""
        return max(0.0, self.planned_duration_sec - self.elapsed_sec())

    def is_running(self):
        """True while started and not stopped."""
        return self._start is not None and self._stop is None

    def is_finished(self):
        """True once the planned duration has passed."""
        return self.elapsed_sec() >= self.planned_duration_sec

    def format_time(self, seconds):
        """Return seconds as MM:SS."""
        minutes, secs = divmod(max(0, int(seconds)), 60)
        return f"{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Storage (fake, in-memory versions of src/storage/*.py)
# ---------------------------------------------------------------------------

# session_id -> Contract C dictionary
_sessions = {}
# session_id -> list of reading dictionaries (same keys as get_readings())
_readings = {}
_next_session_id = 1
_next_reading_id = 1

# Contract C keys, in the same order the real storage returns them.
_SUMMARY_KEYS = [
    "session_id", "user_name", "start_time", "end_time", "planned_duration_sec",
    "actual_duration_sec", "status", "avg_ear", "total_blinks",
    "face_missing_sec", "looking_away_sec", "avg_focus_preview",
]

_READING_COLUMNS = [
    "reading_id", "session_id", "timestamp", "elapsed_sec", "face_detected",
    "ear", "blinks_in_interval", "head_direction", "focus_preview",
]


def _now_iso():
    """Current local time as ISO text, to the second."""
    return datetime.now().isoformat(timespec="seconds")


def _number_or_none(value):
    """Return value as a float, or None if it is missing or not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def init_db():
    """Nothing to create in mock mode (data is kept in memory)."""


def start_session(user_name, planned_duration_sec):
    """Create a fake session with status 'active' and return its session_id."""
    global _next_session_id
    name = str(user_name).strip() if user_name is not None else ""
    session_id = _next_session_id
    _next_session_id += 1
    _sessions[session_id] = {
        "session_id": session_id,
        "user_name": name or "default",
        "start_time": _now_iso(),
        "end_time": None,
        "planned_duration_sec": int(planned_duration_sec),
        "actual_duration_sec": None,
        "status": "active",
        "avg_ear": None,
        "total_blinks": None,
        "face_missing_sec": None,
        "looking_away_sec": None,
        "avg_focus_preview": None,
    }
    _readings[session_id] = []
    return session_id


def save_reading(session_id, reading):
    """Store one Contract B reading in memory. Bad values become None."""
    global _next_reading_id
    if session_id not in _readings:
        return
    if not isinstance(reading, dict):
        reading = {}
    direction = reading.get("head_direction")
    _readings[session_id].append({
        "reading_id": _next_reading_id,
        "session_id": session_id,
        "timestamp": _now_iso(),
        "elapsed_sec": _number_or_none(reading.get("elapsed_sec")),
        "face_detected": 1 if reading.get("face_detected") else 0,
        "ear": _number_or_none(reading.get("ear")),
        "blinks_in_interval": int(reading.get("blinks_in_interval") or 0),
        "head_direction": direction if direction in VALID_DIRECTIONS else "unknown",
        "focus_preview": _number_or_none(reading.get("focus_preview")),
    })
    _next_reading_id += 1


def _average(values):
    """Average of the values that are not None, or None if there are none."""
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def end_session(session_id):
    """End a fake session and fill its summary numbers (spec Section 6.3)."""
    session = _sessions.get(session_id)
    if session is None or session["status"] == "ended":
        return
    rows = _readings.get(session_id, [])
    end_time = _now_iso()
    started = datetime.fromisoformat(session["start_time"])
    ended = datetime.fromisoformat(end_time)
    session.update({
        "end_time": end_time,
        "actual_duration_sec": max(0, int((ended - started).total_seconds())),
        "status": "ended",
        "avg_ear": _average([r["ear"] for r in rows if r["face_detected"] == 1]),
        "total_blinks": sum(r["blinks_in_interval"] for r in rows),
        "face_missing_sec": sum(1 for r in rows if r["face_detected"] == 0),
        "looking_away_sec": sum(1 for r in rows
                                if r["head_direction"] not in ("center", "unknown")),
        "avg_focus_preview": _average([r["focus_preview"] for r in rows]),
    })


def get_session_summary(session_id):
    """Return a copy of the Contract C dictionary, or None if not found."""
    session = _sessions.get(session_id)
    return {key: session[key] for key in _SUMMARY_KEYS} if session else None


def get_readings(session_id):
    """Return the readings of one session, oldest first."""
    return [dict(r) for r in _readings.get(session_id, [])]


def list_sessions(limit=20):
    """Return up to `limit` Contract C dictionaries, newest first."""
    newest_first = sorted(_sessions.values(),
                          key=lambda s: (s["start_time"], s["session_id"]),
                          reverse=True)
    return [get_session_summary(s["session_id"]) for s in newest_first[:int(limit)]]


def delete_session(session_id):
    """Remove one fake session and its readings."""
    _sessions.pop(session_id, None)
    _readings.pop(session_id, None)


def make_focus_chart(session_id):
    """Return a matplotlib Figure of focus over time, or None with no readings."""
    rows = _readings.get(session_id, [])
    if not rows:
        return None
    minutes = [(r["elapsed_sec"] or 0) / 60.0 for r in rows]
    focus = [r["focus_preview"] if r["face_detected"] == 1 else np.nan for r in rows]

    fig = Figure(figsize=(8, 3.6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(minutes, focus, color="tab:blue", linewidth=1.5, label="Focus (mock data)")
    ax.set_title(f"Focus over time: session {session_id} (mock data)")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Focus (preview, 0 to 100)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def _ended_sessions():
    """All fake sessions with status 'ended' (analytics only count these)."""
    return [s for s in _sessions.values() if s["status"] == "ended"]


def get_overall_stats():
    """Return the same keys as the real get_overall_stats()."""
    ended = _ended_sessions()
    with_focus = [s for s in ended if s["avg_focus_preview"] is not None]
    best = max(with_focus, key=lambda s: (s["avg_focus_preview"], s["start_time"]),
               default=None)
    return {
        "total_sessions": len(ended),
        "total_study_sec": sum(s["actual_duration_sec"] or 0 for s in ended),
        "avg_focus": _average([s["avg_focus_preview"] for s in ended]),
        "best_session": get_session_summary(best["session_id"]) if best else None,
        "total_blinks": sum(s["total_blinks"] or 0 for s in ended),
    }


def get_daily_summary(days=7):
    """Return one dictionary per date with ended sessions, newest date first."""
    first_day = (date.today() - timedelta(days=int(days) - 1)).isoformat()
    by_day = {}
    for s in _ended_sessions():
        day = s["start_time"][:10]
        if day >= first_day:
            by_day.setdefault(day, []).append(s)
    return [
        {"date": day,
         "sessions": len(items),
         "total_study_sec": sum(s["actual_duration_sec"] or 0 for s in items),
         "avg_focus": _average([s["avg_focus_preview"] for s in items])}
        for day, items in sorted(by_day.items(), reverse=True)
    ]


def export_session_csv(session_id, output_path=None):
    """Write the fake readings to a CSV file and return its path (None if missing).

    In mock mode the file goes to the system temp folder, not data/exports/.
    """
    if session_id not in _sessions:
        return None
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "focusmate_mock_exports",
                                   f"session_{int(session_id)}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_READING_COLUMNS)
        for row in _readings[session_id]:
            writer.writerow([row[c] for c in _READING_COLUMNS])
    return output_path


# ---------------------------------------------------------------------------
# Demo and self-check: python mock_modules.py
# Runs a 6 second fake session through every function and checks the formats.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CONTRACT_A_KEYS = {"face_detected", "landmarks", "head_direction", "yaw", "pitch"}

    init_db()
    landmarker = create_landmarker()
    cap = open_camera()
    timer = SessionTimer(6)
    blinks = BlinkDetector()
    session_id = start_session("default", 6)
    timer.start()

    last_save = time.monotonic()
    blinks_since_save = 0
    seen_directions = set()
    while not timer.is_finished():
        ok, frame = cap.read()
        vision = process_frame(frame, landmarker)
        assert set(vision) == CONTRACT_A_KEYS, "Contract A keys do not match"
        assert vision["head_direction"] in VALID_DIRECTIONS
        if vision["face_detected"]:
            assert len(vision["landmarks"]) == 478
        seen_directions.add(vision["head_direction"])

        ear = compute_ear(vision["landmarks"])
        if blinks.update(ear):
            blinks_since_save += 1
        preview = compute_focus_preview(vision["face_detected"], ear,
                                        vision["head_direction"])
        assert 0.0 <= preview <= 100.0

        if time.monotonic() - last_save >= 1.0:
            save_reading(session_id, {
                "face_detected": vision["face_detected"], "ear": ear,
                "blinks_in_interval": blinks_since_save,
                "head_direction": vision["head_direction"],
                "focus_preview": preview, "elapsed_sec": timer.elapsed_sec(),
            })
            blinks_since_save = 0
            last_save = time.monotonic()

    timer.stop()
    end_session(session_id)
    close_camera(cap)

    summary = get_session_summary(session_id)
    assert list(summary) == _SUMMARY_KEYS, "Contract C keys do not match"
    print(f"Directions seen: {sorted(seen_directions)}")
    print(f"Readings saved: {len(get_readings(session_id))} (about 6 expected)")
    print("Summary (Contract C):")
    for key, value in summary.items():
        print(f"  {key:<22} {value}")
    print(f"Chart is a Figure: {isinstance(make_focus_chart(session_id), Figure)}")
    print(f"Chart for a missing session: {make_focus_chart(999)}")
    print(f"Overall stats keys: {sorted(get_overall_stats())}")
    print(f"Daily summary: {get_daily_summary()}")
    print(f"CSV written to: {export_session_csv(session_id)}")
    print(f"CSV for a missing session: {export_session_csv(999)}")
    print("All mock checks passed.")
