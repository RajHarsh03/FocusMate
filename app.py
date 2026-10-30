"""
FocusMate dashboard (Streamlit). Owner: Ayush Kumar (2360060).

Run from the project root (the folder that contains this file):
    streamlit run app.py

What this file does:
    - Screen A: setup and live session (name, duration, Start and Stop, live
      camera, big timer, "Focus (preview)" meter, status row).
    - Screen B: end-of-session summary (numbers from the database, the focus
      over time chart and a CSV download).
    - Screen C: past sessions and simple analytics.
    - The main loop from FocusMate-Module-Specs.md Section 7, which joins the
      vision, detection, timer and storage modules together.

What this file does NOT do (on purpose):
    - No SQL. It only calls the storage functions.
    - No EAR, blink, head pose or timer math. It only calls the other modules.
    - No images are saved anywhere. Frames are only shown on screen.

USE_MOCK switch:
    True  -> use mock_modules.py (fake camera and fake data, works today
             without the other members' code or a webcam).
    False -> use the real modules from src/ (needs the vision, detection and
             timer modules, plus models/face_landmarker.task).
"""

import os
import time
from datetime import datetime
from types import SimpleNamespace

import cv2
import streamlit as st

USE_MOCK = False

# Study length choices. 1 minute is there for quick demos.
DURATION_CHOICES = {
    "1 minute": 60,
    "5 minutes": 5 * 60,
    "15 minutes": 15 * 60,
    "25 minutes": 25 * 60,
}
DEFAULT_USER_NAME = "default"
# clear_demo_data() deletes every session saved under this name, so real
# users must never be saved with it.
RESERVED_NAME = "demo"
# Readings are saved to the database once per second, not on every frame.
SAVE_INTERVAL_SEC = 1.0


# ---------------------------------------------------------------------------
# Loading the modules (mock or real)
# ---------------------------------------------------------------------------

def load_modules(use_mock):
    """Return (modules, error_text). All functions are reached as modules.<name>.

    The dashboard code below never needs to know whether the functions are
    fake or real, because both versions have the same names and formats.
    """
    try:
        if use_mock:
            import mock_modules as mock
            return SimpleNamespace(
                create_landmarker=mock.create_landmarker,
                open_camera=mock.open_camera,
                process_frame=mock.process_frame,
                close_camera=mock.close_camera,
                compute_ear=mock.compute_ear,
                BlinkDetector=mock.BlinkDetector,
                compute_focus_preview=mock.compute_focus_preview,
                SessionTimer=mock.SessionTimer,
                init_db=mock.init_db,
                start_session=mock.start_session,
                save_reading=mock.save_reading,
                end_session=mock.end_session,
                get_session_summary=mock.get_session_summary,
                list_sessions=mock.list_sessions,
                make_focus_chart=mock.make_focus_chart,
                get_overall_stats=mock.get_overall_stats,
                get_daily_summary=mock.get_daily_summary,
                export_session_csv=mock.export_session_csv,
            ), None

        from src.vision import vision
        from src.detection import ear_blink, focus_preview
        from src.timer import session_timer
        from src.storage import analytics, charts, database, export
        return SimpleNamespace(
            create_landmarker=vision.create_landmarker,
            open_camera=vision.open_camera,
            process_frame=vision.process_frame,
            close_camera=vision.close_camera,
            compute_ear=ear_blink.compute_ear,
            BlinkDetector=ear_blink.BlinkDetector,
            compute_focus_preview=focus_preview.compute_focus_preview,
            SessionTimer=session_timer.SessionTimer,
            init_db=database.init_db,
            start_session=database.start_session,
            save_reading=database.save_reading,
            end_session=database.end_session,
            get_session_summary=database.get_session_summary,
            list_sessions=database.list_sessions,
            make_focus_chart=charts.make_focus_chart,
            get_overall_stats=analytics.get_overall_stats,
            get_daily_summary=analytics.get_daily_summary,
            export_session_csv=export.export_session_csv,
        ), None
    except ImportError as error:
        return None, (
            f"Could not load the real modules ({error}). The dashboard needs "
            "src/vision/vision.py, src/detection/ear_blink.py, "
            "src/detection/focus_preview.py, src/timer/session_timer.py and "
            "src/storage/. Set USE_MOCK = True at the top of app.py to run "
            "with fake data instead."
        )


fm, MODULES_ERROR = load_modules(USE_MOCK)


@st.cache_resource(show_spinner="Loading the face landmark model...")
def get_landmarker():
    """Create the face landmarker once and reuse it on every rerun.

    If loading fails, the error is raised (and not cached), so the next
    Start press tries again, for example after the model file is added.
    """
    return fm.create_landmarker()


# ---------------------------------------------------------------------------
# Session state and small helpers
# ---------------------------------------------------------------------------

def init_state():
    """Create every st.session_state value the dashboard uses, once."""
    defaults = {
        "running": False,          # True while a session is live
        "session_id": None,        # id of the current or last session
        "timer": None,             # SessionTimer of the current session
        "blink_detector": None,    # BlinkDetector of the current session
        "last_save_time": 0.0,     # time.monotonic() of the last saved reading
        "blinks_since_save": 0,    # blinks counted since the last saved reading
        "cap": None,               # the open camera, or None
        "summary_id": None,        # session whose summary is shown (Screen B)
        "message": None,           # (kind, text) shown on the live tab
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clean_user_name(raw_name):
    """Return (name to save, warning text or None) for the typed name."""
    name = (raw_name or "").strip()
    if not name:
        return DEFAULT_USER_NAME, None
    if name.lower() == RESERVED_NAME:
        return DEFAULT_USER_NAME, (
            f"The name '{RESERVED_NAME}' is kept for demo data. "
            f"This session will be saved as '{DEFAULT_USER_NAME}'."
        )
    return name, None


def format_duration(seconds):
    """Turn seconds into short text such as '12 min 05 s', or 'No data'."""
    if seconds is None:
        return "No data"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def format_number(value, digits):
    """Round a number for display, or return 'No data' if it is missing."""
    return "No data" if value is None else f"{value:.{digits}f}"


def format_start_time(iso_text):
    """Turn '2026-11-01T10:30:00' into '01 Nov 2026, 10:30'."""
    try:
        return datetime.fromisoformat(iso_text).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return "Unknown"


# ---------------------------------------------------------------------------
# Start and Stop (run as button callbacks, before the page is redrawn)
# ---------------------------------------------------------------------------

def handle_start():
    """Start button: open the camera, create the session and start the timer."""
    state = st.session_state
    if state.running:
        return
    state.message = None

    try:
        get_landmarker()
    except Exception as error:  # model file missing or broken
        state.message = ("error",
                         "The face landmark model could not be loaded "
                         f"({error}). Check that models/face_landmarker.task "
                         "exists (see the Readme for the download command).")
        return

    cap = fm.open_camera()
    if cap is None:
        state.message = ("error",
                         "The webcam could not be opened. Close other apps that "
                         "use the camera (Zoom, Teams, the Camera app) and press "
                         "Start again.")
        return

    user_name, _ = clean_user_name(state.user_name)
    planned_sec = DURATION_CHOICES[state.duration_label]

    state.cap = cap
    state.timer = fm.SessionTimer(planned_sec)
    state.blink_detector = fm.BlinkDetector()
    state.blink_detector.reset()
    state.session_id = fm.start_session(user_name, planned_sec)
    state.timer.start()
    state.last_save_time = time.monotonic()
    state.blinks_since_save = 0
    state.summary_id = None
    state.running = True


def finish_session():
    """End the running session: stop the timer, save the summary, free the camera."""
    state = st.session_state
    if state.timer is not None:
        state.timer.stop()
    if state.session_id is not None:
        fm.end_session(state.session_id)
    fm.close_camera(state.cap)
    state.cap = None
    state.running = False
    state.summary_id = state.session_id


def handle_stop():
    """Stop button: end the session early."""
    if st.session_state.running:
        finish_session()


# ---------------------------------------------------------------------------
# Screen A: setup and live session
# ---------------------------------------------------------------------------

def render_setup():
    """Name box, duration choice and the Start and Stop buttons."""
    running = st.session_state.running
    col_name, col_duration = st.columns(2)
    with col_name:
        st.text_input("Your name (optional)", value=DEFAULT_USER_NAME,
                      key="user_name", disabled=running, max_chars=40)
    with col_duration:
        st.selectbox("Session length", list(DURATION_CHOICES),
                     key="duration_label", disabled=running)

    _, warning = clean_user_name(st.session_state.user_name)
    if warning and not running:
        st.warning(warning)

    col_start, col_stop, _ = st.columns([1, 1, 4])
    with col_start:
        st.button("Start", type="primary", on_click=handle_start,
                  disabled=running, width="stretch")
    with col_stop:
        st.button("Stop", on_click=handle_stop, disabled=not running,
                  width="stretch")


def make_live_layout():
    """Create the empty boxes that the live loop fills on every frame."""
    col_video, col_info = st.columns([3, 2], gap="large")
    with col_video:
        video_box = st.empty()
    with col_info:
        status_box = st.empty()
        timer_box = st.empty()
        meter_box = st.empty()
        details_box = st.empty()
    return SimpleNamespace(video=video_box, status=status_box, timer=timer_box,
                           meter=meter_box, details=details_box)


def show_live_values(boxes, frame_bgr, vision, ear, preview):
    """Draw one frame and the current numbers into the live boxes."""
    state = st.session_state
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes.video.image(frame_rgb, channels="RGB", width="stretch")
    boxes.status.markdown("**Session status:** :green[Active]")
    boxes.timer.metric("Time left", state.timer.format_time(state.timer.remaining_sec()))
    boxes.meter.progress(int(round(preview)),
                         text=f"Focus (preview): {preview:.0f} / 100")
    with boxes.details.container():
        col_face, col_head = st.columns(2)
        col_face.metric("Face", "Detected" if vision["face_detected"] else "Not detected")
        col_head.metric("Head direction", vision["head_direction"].capitalize())
        col_ear, col_blinks = st.columns(2)
        col_ear.metric("Eye openness (EAR)", format_number(ear, 3))
        col_blinks.metric("Blinks", state.blink_detector.total_blinks)


def run_live_loop(boxes):
    """The main loop (spec Section 7). Runs until Stop or the timer ends.

    Pressing Stop makes Streamlit stop this loop and rerun the script, and
    handle_stop() then ends the session. When the timer runs out, the loop
    ends the session itself and reruns the page to show the summary.
    """
    state = st.session_state
    landmarker = get_landmarker()

    while state.running:
        if state.timer.is_finished():
            finish_session()
            st.rerun()

        ok, frame = state.cap.read()
        if not ok or frame is None:
            state.message = ("error",
                             "The webcam stopped sending pictures, so the session "
                             "was ended. Your readings up to now are saved.")
            finish_session()
            st.rerun()

        frame = cv2.flip(frame, 1)  # mirror view, like looking in a mirror

        vision = fm.process_frame(frame, landmarker)                 # Contract A
        ear = fm.compute_ear(vision["landmarks"])                     # None if no face
        if state.blink_detector.update(ear):
            state.blinks_since_save += 1
        preview = fm.compute_focus_preview(vision["face_detected"], ear,
                                           vision["head_direction"])

        show_live_values(boxes, frame, vision, ear, preview)

        now = time.monotonic()
        if now - state.last_save_time >= SAVE_INTERVAL_SEC:
            fm.save_reading(state.session_id, {                       # Contract B
                "face_detected": vision["face_detected"],
                "ear": ear,
                "blinks_in_interval": state.blinks_since_save,
                "head_direction": vision["head_direction"],
                "focus_preview": preview,
                "elapsed_sec": state.timer.elapsed_sec(),
            })
            state.blinks_since_save = 0
            state.last_save_time = now


# ---------------------------------------------------------------------------
# Screen B: end-of-session summary
# ---------------------------------------------------------------------------

def render_summary(session_id):
    """Show the Contract C numbers, the focus chart and a CSV download."""
    summary = fm.get_session_summary(session_id)
    if summary is None:
        st.info("The summary of this session could not be found.")
        return

    st.subheader("Session summary")
    st.caption(f"Session {summary['session_id']} | {summary['user_name']} | "
               f"started {format_start_time(summary['start_time'])} | "
               f"planned {format_duration(summary['planned_duration_sec'])}")

    row_one = st.columns(3)
    row_one[0].metric("Total time studied", format_duration(summary["actual_duration_sec"]))
    row_one[1].metric("Average EAR", format_number(summary["avg_ear"], 3))
    row_one[2].metric("Total blinks", summary["total_blinks"] or 0)
    row_two = st.columns(3)
    row_two[0].metric("Time with no face", format_duration(summary["face_missing_sec"]))
    row_two[1].metric("Time looking away", format_duration(summary["looking_away_sec"]))
    row_two[2].metric("Average focus (preview)",
                      format_number(summary["avg_focus_preview"], 1))

    fig = fm.make_focus_chart(session_id)
    if fig is None:
        st.info("No readings were saved for this session, so there is no chart.")
    else:
        st.pyplot(fig)

    csv_path = fm.export_session_csv(session_id)
    if csv_path is not None and os.path.exists(csv_path):
        with open(csv_path, "rb") as handle:
            st.download_button("Download readings (CSV)", data=handle.read(),
                               file_name=os.path.basename(csv_path),
                               mime="text/csv")

    st.caption("To study again, choose a length and press Start above.")


# ---------------------------------------------------------------------------
# Screen C: past sessions and analytics
# ---------------------------------------------------------------------------

def render_past_sessions():
    """Show overall numbers, the last 7 days and a table of recent sessions."""
    stats = fm.get_overall_stats()
    st.subheader("All finished sessions")
    cols = st.columns(4)
    cols[0].metric("Sessions", stats["total_sessions"])
    cols[1].metric("Total study time", format_duration(stats["total_study_sec"]))
    cols[2].metric("Average focus (preview)", format_number(stats["avg_focus"], 1))
    cols[3].metric("Total blinks", stats["total_blinks"])
    best = stats["best_session"]
    if best is not None:
        st.caption(f"Best session: {format_start_time(best['start_time'])}, "
                   f"average focus {best['avg_focus_preview']:.1f}")

    st.subheader("Last 7 days")
    days = fm.get_daily_summary(7)
    if days:
        st.dataframe([
            {"Date": day["date"],
             "Sessions": day["sessions"],
             "Study time": format_duration(day["total_study_sec"]),
             "Average focus": format_number(day["avg_focus"], 1)}
            for day in days
        ], hide_index=True)
    else:
        st.info("No finished sessions in the last 7 days.")

    st.subheader("Recent sessions")
    sessions = fm.list_sessions(limit=20)
    if sessions:
        st.dataframe([
            {"Date": format_start_time(s["start_time"]),
             "Name": s["user_name"],
             "Duration": format_duration(s["actual_duration_sec"]),
             "Average focus": format_number(s["avg_focus_preview"], 1),
             "Status": s["status"].capitalize()}
            for s in sessions
        ], hide_index=True)
    else:
        st.info("No sessions yet. Start one on the Live session tab.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main():
    """Build the page. Streamlit runs this from the top on every interaction."""
    st.set_page_config(page_title="FocusMate", layout="wide")
    st.title("FocusMate")
    st.caption("AI-powered focus and productivity monitor. Everything runs on "
               "this laptop. No video is saved or uploaded.")

    if MODULES_ERROR:
        st.error(MODULES_ERROR)
        st.stop()

    if USE_MOCK:
        st.info("Mock mode: the camera and all numbers are fake test data, "
                "and nothing is saved to the database.")
    elif os.path.abspath(os.getcwd()) != os.path.dirname(os.path.abspath(__file__)):
        st.warning("Start the app from the project root (the folder with "
                   "app.py), otherwise a second, empty database is created.")

    init_state()
    fm.init_db()

    live_tab, past_tab = st.tabs(["Live session", "Past sessions"])

    # The past sessions tab is drawn first, because the live loop below keeps
    # running and nothing after it would be drawn until the session ends.
    with past_tab:
        render_past_sessions()

    with live_tab:
        render_setup()
        state = st.session_state
        if state.message:
            kind, text = state.message
            (st.error if kind == "error" else st.info)(text)

        if state.running:
            run_live_loop(make_live_layout())
        elif state.summary_id is not None:
            render_summary(state.summary_id)
        else:
            st.info("Choose a session length and press Start. Allow camera "
                    "access if Windows asks.")


if __name__ == "__main__":
    main()
