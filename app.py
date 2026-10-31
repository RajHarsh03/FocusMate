"""FocusMate Streamlit dashboard. Owner: Ayush Kumar (2360060).

Run from the project root:
    streamlit run app.py

The dashboard joins the vision, detection, timer and storage modules. This file
contains presentation and orchestration only. It does not contain SQL, EAR,
head-pose, blink or focus-scoring logic.
"""

import os
import time
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import altair as alt
import cv2
import streamlit as st

from ui import components as ui

USE_MOCK = False
USE_ADAPTIVE_SESSION_FLOW = True

# A dismissed break suggestion waits this long before focus-state checks resume.
CONTINUE_COOLDOWN_SEC = 30.0
# Missing face shows an inline notice immediately and one toast after this delay.
FACE_MISSING_TOAST_SEC = 5.0

DURATION_OPTIONS = ("5 min", "15 min", "25 min", "45 min", "Custom")
DURATION_SECONDS = {
    "5 min": 5 * 60,
    "15 min": 15 * 60,
    "25 min": 25 * 60,
    "45 min": 45 * 60,
}
DEFAULT_USER_NAME = "default"
RESERVED_NAME = "demo"
SAVE_INTERVAL_SEC = 1.0


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def load_modules(use_mock):
    """Return one namespace containing either fake or real module functions."""
    try:
        # The full scorer and state machine work with real or mocked signals.
        from src.detection.focus_score import FocusScorer
        from src.timer.session_state import SessionState, SessionStateMachine

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
                FocusScorer=FocusScorer,
                SessionState=SessionState,
                SessionStateMachine=SessionStateMachine,
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
            FocusScorer=FocusScorer,
            SessionState=SessionState,
            SessionStateMachine=SessionStateMachine,
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
            "the vision, detection, timer and storage modules. Set USE_MOCK "
            "to True at the top of app.py to run with fake data."
        )


fm, MODULES_ERROR = load_modules(USE_MOCK)


@st.cache_resource(show_spinner="Loading the face landmark model...")
def get_landmarker():
    """Create the face landmarker once and reuse it on Streamlit reruns."""
    return fm.create_landmarker()


# ---------------------------------------------------------------------------
# Session state and formatting
# ---------------------------------------------------------------------------

def init_state():
    """Create every session-state value once."""
    defaults = {
        "running": False,
        "session_id": None,
        "timer": None,
        "blink_detector": None,
        "focus_scorer": None,
        "session_machine": None,
        "adaptive_state": "active",
        "break_prompt_visible": False,
        "continue_cooldown_until": 0.0,
        "face_missing_since": None,
        "face_missing_toast_sent": False,
        "pending_toast": None,
        "last_save_time": 0.0,
        "blinks_since_save": 0,
        "cap": None,
        "summary_id": None,
        "message": None,
        "user_name": "",
        "duration_choice": "25 min",
        "custom_minutes": 10,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clean_user_name(raw_name):
    """Return the stored name and an optional warning for the typed name."""
    name = (raw_name or "").strip()
    if not name:
        return DEFAULT_USER_NAME, None
    if name.lower() == RESERVED_NAME:
        return DEFAULT_USER_NAME, (
            f"The name '{RESERVED_NAME}' is reserved for showcase data. "
            f"This session will be stored without a display name."
        )
    return name, None


def selected_duration_sec():
    """Return the duration selected in the setup toolbar, in seconds."""
    choice = st.session_state.duration_choice
    if choice == "Custom":
        return max(1, int(st.session_state.custom_minutes)) * 60
    return DURATION_SECONDS[choice]


def format_duration(seconds):
    """Turn seconds into short, human-readable text."""
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def format_number(value, digits=1):
    """Round a number for display, or return a quiet placeholder."""
    return "-" if value is None else f"{value:.{digits}f}"


def format_start_time(iso_text):
    """Turn ISO time into a full, readable date and time."""
    try:
        return datetime.fromisoformat(iso_text).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return "Unknown"


def relative_start_time(iso_text):
    """Return a recent time such as 'Today, 21:40' or '29 Oct, 15:09'."""
    try:
        started = datetime.fromisoformat(iso_text)
    except (TypeError, ValueError):
        return "Unknown"
    difference = (date.today() - started.date()).days
    if difference == 0:
        prefix = "Today"
    elif difference == 1:
        prefix = "Yesterday"
    else:
        prefix = started.strftime("%d %b")
    return f"{prefix}, {started.strftime('%H:%M')}"


# ---------------------------------------------------------------------------
# Session start and stop callbacks
# ---------------------------------------------------------------------------

def handle_start():
    """Open the camera, create a database session and start the timer."""
    state = st.session_state
    if state.running:
        return
    state.message = None

    try:
        get_landmarker()
    except Exception as error:
        state.message = (
            "error",
            "The face landmark model could not be loaded "
            f"({error}). Check models/face_landmarker.task.",
        )
        return

    cap = fm.open_camera()
    if cap is None:
        state.message = (
            "error",
            "The webcam could not be opened. Close other camera apps and try again.",
        )
        return

    user_name, _ = clean_user_name(state.user_name)
    planned_sec = selected_duration_sec()

    state.cap = cap
    state.timer = fm.SessionTimer(planned_sec)
    state.blink_detector = fm.BlinkDetector()
    state.blink_detector.reset()
    state.focus_scorer = fm.FocusScorer() if USE_ADAPTIVE_SESSION_FLOW else None
    if state.focus_scorer is not None:
        state.focus_scorer.reset()
        state.session_machine = fm.SessionStateMachine(timer=state.timer)
        # The state machine starts the shared timer. SessionTimer.start() is
        # idempotent, so the stable fallback can still start it directly.
        state.session_machine.start()
    else:
        state.session_machine = None
        state.timer.start()
    state.session_id = fm.start_session(user_name, planned_sec)
    state.last_save_time = time.monotonic()
    state.blinks_since_save = 0
    state.adaptive_state = "active"
    state.break_prompt_visible = False
    state.continue_cooldown_until = 0.0
    state.face_missing_since = None
    state.face_missing_toast_sent = False
    state.pending_toast = None
    state.summary_id = None
    state.running = True


def finish_session():
    """Stop the timer, finish storage and release the camera."""
    state = st.session_state
    if state.session_machine is not None:
        state.session_machine.stop()
    elif state.timer is not None:
        state.timer.stop()
    if state.session_id is not None:
        fm.end_session(state.session_id)
    fm.close_camera(state.cap)
    state.cap = None
    state.running = False
    state.break_prompt_visible = False
    state.adaptive_state = "ended"
    state.summary_id = state.session_id


def handle_stop():
    """End a running session early."""
    if st.session_state.running:
        finish_session()


def handle_continue_session():
    """Dismiss a break suggestion and pause state checks for 30 seconds."""
    state = st.session_state
    if not state.running or state.timer is None:
        return
    state.break_prompt_visible = False
    state.continue_cooldown_until = state.timer.elapsed_sec() + CONTINUE_COOLDOWN_SEC
    state.session_machine = fm.SessionStateMachine(timer=state.timer)
    state.session_machine.start()
    state.adaptive_state = "active"
    state.pending_toast = "Session continued. Focus checks resume in 30 seconds."


def handle_take_break():
    """End the current session after the user accepts the break suggestion."""
    if st.session_state.running:
        finish_session()
        st.session_state.message = (
            "info",
            "Session saved. Take a short break and return when you feel ready.",
        )


# ---------------------------------------------------------------------------
# Idle session setup
# ---------------------------------------------------------------------------

def render_setup():
    """Render a single horizontal session toolbar."""
    ui.section_header(
        "Plan your session",
        "Set a duration, then begin when you are ready.",
    )
    with st.container(border=True):
        st.markdown('<div class="fm-toolbar-anchor"></div>', unsafe_allow_html=True)
        name_col, duration_col, action_col = st.columns(
            [1.35, 2.25, 0.9], gap="medium", vertical_alignment="bottom"
        )
        with name_col:
            st.text_input(
                "Name",
                key="user_name",
                placeholder="Optional",
                max_chars=40,
            )
        with duration_col:
            st.segmented_control(
                "Duration",
                DURATION_OPTIONS,
                key="duration_choice",
                selection_mode="single",
                width="stretch",
            )
        with action_col:
            st.button(
                "Start session",
                type="primary",
                on_click=handle_start,
                width="stretch",
            )

        if st.session_state.duration_choice == "Custom":
            st.number_input(
                "Custom duration (minutes)",
                min_value=1,
                max_value=180,
                step=1,
                key="custom_minutes",
                width=220,
            )

        _, warning = clean_user_name(st.session_state.user_name)
        if warning:
            st.warning(warning)


# ---------------------------------------------------------------------------
# Live session
# ---------------------------------------------------------------------------

def make_live_layout():
    """Create stable placeholders for the camera and live signals."""
    ui.section_header(
        "Live session",
        "Your signals update locally in real time. Nothing from the camera is recorded.",
    )
    camera_col, signal_col = st.columns([1.38, 1], gap="large", vertical_alignment="top")

    with camera_col:
        with st.container(border=True):
            camera_status = st.empty()
            camera_image = st.empty()

    with signal_col:
        with st.container(border=True):
            st.markdown('<div class="fm-signal-anchor"></div>', unsafe_allow_html=True)
            focus_col, timer_col = st.columns([1.1, 0.9], vertical_alignment="center")
            focus_box = focus_col.empty()
            timer_box = timer_col.empty()
            notice_box = st.empty()
            metrics_box = st.empty()
            st.write("")
            if st.session_state.break_prompt_visible:
                ui.break_prompt()
                continue_col, break_col = st.columns(2)
                with continue_col:
                    st.button(
                        "Continue session",
                        key="continue_after_prompt",
                        on_click=handle_continue_session,
                        type="primary",
                        width="stretch",
                    )
                with break_col:
                    st.button(
                        "End and take a break",
                        key="accept_break_prompt",
                        on_click=handle_take_break,
                        type="secondary",
                        width="stretch",
                    )
            else:
                st.button(
                    "End session",
                    key="live_stop",
                    on_click=handle_stop,
                    type="secondary",
                    width="stretch",
                )

    return SimpleNamespace(
        camera_status=camera_status,
        camera_image=camera_image,
        focus=focus_box,
        timer=timer_box,
        notice=notice_box,
        metrics=metrics_box,
    )


def show_live_values(boxes, frame_bgr, vision, ear, focus_score, session_state):
    """Update stable live placeholders with the full score and state."""
    state = st.session_state
    face_detected = bool(vision["face_detected"])
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    elapsed_text = state.timer.format_time(state.timer.elapsed_sec())

    ui.camera_panel(
        frame_rgb,
        elapsed_text,
        face_detected,
        boxes.camera_status,
        boxes.camera_image,
    )
    ui.focus_ring(
        focus_score,
        face_detected,
        boxes.focus,
        state=session_state,
    )
    ui.timer_display(
        state.timer.format_time(state.timer.remaining_sec()),
        elapsed_text,
        boxes.timer,
    )

    if not face_detected:
        ui.inline_notice(
            "Face not detected. Move gently into the camera frame.",
            boxes.notice,
        )
    elif session_state == "break_prompt":
        ui.inline_notice(
            "Your focus has stayed low. Choose whether to continue or take a break.",
            boxes.notice,
            tone="warning",
        )
    elif session_state == "warning":
        ui.inline_notice(
            "Focus is dropping. Look towards the screen and adjust your posture.",
            boxes.notice,
            tone="warning",
        )
    elif state.timer.elapsed_sec() < state.continue_cooldown_until:
        seconds_left = int(state.continue_cooldown_until - state.timer.elapsed_sec())
        ui.inline_notice(
            f"Continuing by choice. Focus checks resume in {seconds_left} seconds.",
            boxes.notice,
        )
    else:
        boxes.notice.empty()

    direction_labels = {
        "center": "Centered",
        "left": "Left",
        "right": "Right",
        "up": "Up",
        "down": "Down",
        "unknown": "-",
    }
    direction = direction_labels.get(vision["head_direction"], "-")
    state_labels = {
        "active": "Active",
        "warning": "Warning",
        "break_prompt": "Break suggested",
        "ended": "Ended",
    }
    ui.metric_grid(
        [
            {
                "label": "Session state",
                "value": state_labels.get(session_state, "Active"),
                "caption": "Based on sustained focus",
                "muted": False,
            },
            {
                "label": "Head direction",
                "value": direction,
                "caption": "Relative to the screen" if face_detected else "Waiting for face",
                "muted": direction == "-",
            },
            {
                "label": "Eye openness (EAR)",
                "value": format_number(ear, 3),
                "caption": "Current eye signal" if ear is not None else "Waiting for face",
                "muted": ear is None,
            },
            {
                "label": "Blinks",
                "value": state.blink_detector.total_blinks,
                "caption": "This session",
                "muted": False,
            },
        ],
        boxes.metrics,
    )


def _track_face_presence(face_detected, elapsed):
    """Track face gaps, show one toast and reset stale decision dwell on return."""
    state = st.session_state
    if face_detected:
        if state.face_missing_since is not None:
            missing_for = elapsed - state.face_missing_since
            if missing_for >= 1.0 and state.session_machine is not None:
                # A face gap is missing evidence, not low focus. Start decision
                # dwell fresh so time spent off-camera cannot trigger a warning.
                state.session_machine = fm.SessionStateMachine(timer=state.timer)
                state.session_machine.start()
                state.adaptive_state = "active"
                state.break_prompt_visible = False
        state.face_missing_since = None
        state.face_missing_toast_sent = False
        return

    if state.face_missing_since is None:
        state.face_missing_since = elapsed
        return

    missing_for = elapsed - state.face_missing_since
    if missing_for >= FACE_MISSING_TOAST_SEC and not state.face_missing_toast_sent:
        st.toast("We cannot see you. Move back into the camera frame.")
        state.face_missing_toast_sent = True


def _handle_state_transition(status):
    """React once when the state machine changes state."""
    if not status.state_changed:
        return

    state = st.session_state
    state.adaptive_state = status.state.value

    if status.state is fm.SessionState.WARNING:
        if status.previous_state is fm.SessionState.BREAK_PROMPT:
            state.break_prompt_visible = False
            state.pending_toast = "Focus is recovering."
            st.rerun()
        st.toast("Focus is dropping. Adjust your posture and look towards the screen.")

    elif status.state is fm.SessionState.BREAK_PROMPT:
        state.break_prompt_visible = True
        state.pending_toast = "Your focus has stayed low. A short break may help."
        st.rerun()

    elif status.state is fm.SessionState.ACTIVE:
        state.break_prompt_visible = False
        st.toast("Focus recovered. Session is active again.")

    elif status.state is fm.SessionState.ENDED:
        finish_session()
        st.rerun()


def run_live_loop(boxes):
    """Run scoring, state decisions, storage and presentation until session end."""
    state = st.session_state
    landmarker = get_landmarker()

    while state.running:
        if state.timer.is_finished():
            finish_session()
            st.rerun()

        ok, frame = state.cap.read()
        if not ok or frame is None:
            state.message = (
                "error",
                "The webcam stopped sending pictures. The readings saved so far are safe.",
            )
            finish_session()
            st.rerun()

        # Detect head direction on the original camera frame. Mirror only the
        # displayed image, otherwise physical left and right become reversed.
        vision = fm.process_frame(frame, landmarker)
        display_frame = cv2.flip(frame, 1)
        face_detected = bool(vision["face_detected"])
        ear = fm.compute_ear(vision["landmarks"])
        if state.blink_detector.update(ear):
            state.blinks_since_save += 1

        if USE_ADAPTIVE_SESSION_FLOW:
            if face_detected:
                focus_value = state.focus_scorer.update(
                    True,
                    ear,
                    vision["head_direction"],
                    vision.get("yaw"),
                    vision.get("pitch"),
                )
            else:
                # Keep the persisted no-face score consistent with Contract B,
                # without pulling the EMA down and causing a false warning later.
                focus_value = 0.0
            elapsed = state.timer.elapsed_sec()
            _track_face_presence(face_detected, elapsed)
            in_cooldown = elapsed < state.continue_cooldown_until
            # A missing face shows its own notice but does not count as evidence
            # of low focus. A dismissed prompt gets a short decision cooldown.
            decision_score = None if (not face_detected or in_cooldown) else focus_value
            status = state.session_machine.update(decision_score)
            state.adaptive_state = status.state.value
            _handle_state_transition(status)
        else:
            focus_value = fm.compute_focus_preview(
                face_detected, ear, vision["head_direction"]
            )
            state.adaptive_state = "active"

        show_live_values(
            boxes,
            display_frame,
            vision,
            ear,
            focus_value,
            state.adaptive_state,
        )

        now = time.monotonic()
        if now - state.last_save_time >= SAVE_INTERVAL_SEC:
            fm.save_reading(
                state.session_id,
                {
                    "face_detected": face_detected,
                    "ear": ear,
                    "blinks_in_interval": state.blinks_since_save,
                    "head_direction": vision["head_direction"],
                    # Contract B keeps this historical key name. New adaptive
                    # sessions store the smoothed full score in this field.
                    "focus_preview": focus_value,
                    "elapsed_sec": state.timer.elapsed_sec(),
                },
            )
            state.blinks_since_save = 0
            state.last_save_time = now


# ---------------------------------------------------------------------------
# Completed session
# ---------------------------------------------------------------------------

def style_session_figure(figure):
    """Apply the active UI theme to the Matplotlib session chart."""
    theme_type = st.context.theme.get("type") or "dark"
    if theme_type == "light":
        text_color = "#5F6672"
        border_color = "#D9DDE5"
        accent = "#4F6FE8"
    else:
        text_color = "#A7ADB7"
        border_color = "#30343C"
        accent = "#6C8CFF"

    figure.patch.set_alpha(0)
    for axis in figure.axes:
        axis.set_facecolor("none")
        axis.title.set_color(text_color)
        axis.xaxis.label.set_color(text_color)
        axis.yaxis.label.set_color(text_color)
        axis.tick_params(colors=text_color)
        for spine in axis.spines.values():
            spine.set_color(border_color)
        axis.grid(True, color=border_color, alpha=0.55, linewidth=0.7)
        for index, line in enumerate(axis.lines):
            line.set_color(accent)
            line.set_alpha(0.25 if index == 0 else 1.0)
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("none")
            legend.get_frame().set_edgecolor(border_color)
            for label in legend.get_texts():
                label.set_color(text_color)
    return figure


def render_summary(session_id):
    """Show one completed session, its chart and CSV download."""
    summary = fm.get_session_summary(session_id)
    if summary is None:
        ui.empty_state(
            "Summary unavailable",
            "The completed session could not be found.",
            compact=True,
        )
        return

    ui.section_header(
        "Session complete",
        f"Started {format_start_time(summary['start_time'])} | "
        f"Planned for {format_duration(summary['planned_duration_sec'])}",
        kicker="Saved locally",
    )

    focus_col, time_col, blink_col, away_col = st.columns([1.25, 1, 1, 1])
    with focus_col:
        ui.stat_card(
            "Average focus",
            format_number(summary["avg_focus_preview"], 0),
            "Session score",
            hero=True,
        )
    with time_col:
        ui.stat_card("Study time", format_duration(summary["actual_duration_sec"]))
    with blink_col:
        ui.stat_card("Blinks", summary["total_blinks"] or 0)
    with away_col:
        ui.stat_card("Looking away", format_duration(summary["looking_away_sec"]))

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.caption(f"Average EAR: {format_number(summary['avg_ear'], 3)}")
    with detail_cols[1]:
        st.caption(f"Face outside frame: {format_duration(summary['face_missing_sec'])}")

    st.write("")
    chart_title_col, chart_action_col = st.columns(
        [4, 1.25], gap="medium", vertical_alignment="bottom"
    )
    with chart_title_col:
        ui.section_header(
            "Focus over time",
            "Gaps show moments when a face was not available.",
        )

    csv_path = fm.export_session_csv(session_id)
    if csv_path is not None and os.path.exists(csv_path):
        with chart_action_col:
            with open(csv_path, "rb") as handle:
                st.download_button(
                    "Download session data",
                    data=handle.read(),
                    file_name=os.path.basename(csv_path),
                    mime="text/csv",
                    width="stretch",
                )

    fig = fm.make_focus_chart(session_id)
    if fig is None:
        ui.empty_state(
            "No focus readings",
            "This session ended before enough readings were saved.",
            compact=True,
        )
    else:
        st.pyplot(style_session_figure(fig), width="stretch")


# ---------------------------------------------------------------------------
# History and analytics
# ---------------------------------------------------------------------------

def _session_day(session):
    """Return a session start date, or None for invalid data."""
    try:
        return datetime.fromisoformat(session["start_time"]).date()
    except (TypeError, ValueError):
        return None


def _period_sessions(sessions, first_offset, last_offset):
    """Return ended sessions whose dates are within day offsets from today."""
    newest = date.today() - timedelta(days=first_offset)
    oldest = date.today() - timedelta(days=last_offset)
    return [
        session
        for session in sessions
        if session["status"] == "ended"
        and _session_day(session) is not None
        and oldest <= _session_day(session) <= newest
    ]


def _sum_value(sessions, key):
    """Sum non-empty numeric values from session dictionaries."""
    return sum((session.get(key) or 0) for session in sessions)


def _average_value(sessions, key):
    """Average non-empty numeric values, or return None."""
    values = [session.get(key) for session in sessions if session.get(key) is not None]
    return sum(values) / len(values) if values else None


def _delta_text(current, previous, unit=""):
    """Describe the difference from the previous seven-day period."""
    if previous is None or current is None:
        return "No previous period"
    difference = current - previous
    sign = "+" if difference > 0 else ""
    if isinstance(current, int) and isinstance(previous, int):
        number = f"{difference:d}"
    else:
        number = f"{difference:.1f}"
    suffix = f" {unit}" if unit else ""
    return f"{sign}{number}{suffix} vs previous 7 days"


def _daily_series(sessions, key, average=False):
    """Return seven values, oldest to newest, filling days without data with 0."""
    values = []
    for offset in range(6, -1, -1):
        day = date.today() - timedelta(days=offset)
        on_day = [session for session in sessions if _session_day(session) == day]
        value = _average_value(on_day, key) if average else _sum_value(on_day, key)
        values.append(0 if value is None else value)
    return values


def render_history_chart(daily_rows):
    """Show seven distinct day slots with visible focus bars."""
    if not daily_rows:
        ui.empty_state(
            "No focus trend yet",
            "Complete a session to begin building your seven-day view.",
            compact=True,
        )
        return

    rows_by_date = {row["date"]: row for row in daily_rows}
    values = []
    date_order = []
    for offset in range(6, -1, -1):
        day = date.today() - timedelta(days=offset)
        date_label = day.strftime("%d %b")
        date_order.append(date_label)
        row = rows_by_date.get(day.isoformat())
        focus = round(row["avg_focus"], 1) if row and row["avg_focus"] is not None else 0
        values.append(
            {
                "date_label": date_label,
                "focus": focus,
                "focus_label": f"{focus:.0f}" if row else "",
                "availability": "Session recorded" if row else "No session",
                "sessions": row["sessions"] if row else 0,
                "study_minutes": round(row["total_study_sec"] / 60, 1) if row else 0,
                "track": 100,
            }
        )

    theme_type = st.context.theme.get("type") or "dark"
    axis_color = "#5F6672" if theme_type == "light" else "#A7ADB7"
    track_color = "#E7EAF0" if theme_type == "light" else "#242831"
    bar_color = "#4F6FE8" if theme_type == "light" else "#6C8CFF"

    data = alt.Data(values=values)
    x_axis = alt.X(
        "date_label:N",
        title=None,
        sort=date_order,
        axis=alt.Axis(labelAngle=0),
    )
    y_scale = alt.Scale(domain=[0, 100])

    tracks = alt.Chart(data).mark_bar(
        color=track_color,
        cornerRadiusTopLeft=5,
        cornerRadiusTopRight=5,
        size=54,
    ).encode(
        x=x_axis,
        y=alt.Y("track:Q", title="Average focus", scale=y_scale),
    )
    bars = alt.Chart(data).mark_bar(
        color=bar_color,
        cornerRadiusTopLeft=5,
        cornerRadiusTopRight=5,
        size=54,
    ).encode(
        x=x_axis,
        y=alt.Y("focus:Q", title="Average focus", scale=y_scale),
        tooltip=[
            alt.Tooltip("date_label:N", title="Date"),
            alt.Tooltip("availability:N", title="Activity"),
            alt.Tooltip("focus:Q", title="Average focus", format=".1f"),
            alt.Tooltip("sessions:Q", title="Sessions"),
            alt.Tooltip("study_minutes:Q", title="Study minutes", format=".1f"),
        ],
    )
    labels = alt.Chart(data).mark_text(
        color=axis_color,
        dy=-10,
        fontSize=12,
        fontWeight=600,
    ).encode(
        x=x_axis,
        y=alt.Y("focus:Q", scale=y_scale),
        text="focus_label:N",
    )

    chart = (
        (tracks + bars + labels)
        .properties(height=245, background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axis(
            grid=False,
            domain=False,
            tickSize=0,
            labelPadding=9,
            titlePadding=14,
            labelColor=axis_color,
            titleColor=axis_color,
        )
    )
    st.altair_chart(chart, width="stretch", theme=None)


def render_recent_sessions(sessions):
    """Show recent sessions with relative time and a focus progress column."""
    if not sessions:
        ui.empty_state(
            "No sessions yet",
            "Your completed sessions will appear here automatically.",
            compact=True,
        )
        return

    rows = []
    for session in sessions[:20]:
        rows.append(
            {
                "When": relative_start_time(session["start_time"]),
                "Duration": format_duration(session["actual_duration_sec"]),
                "Focus": session["avg_focus_preview"],
                "Status": "Complete" if session["status"] == "ended" else "In progress",
            }
        )

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        row_height=42,
        column_config={
            "When": st.column_config.TextColumn("When", width="medium"),
            "Duration": st.column_config.TextColumn("Duration", width="small"),
            "Focus": st.column_config.ProgressColumn(
                "Focus", min_value=0, max_value=100, format="%.0f", width="medium"
            ),
            "Status": st.column_config.TextColumn("Status", width="small"),
        },
    )


def render_past_sessions():
    """Render the focus history with hierarchy, deltas, chart and session list."""
    stats = fm.get_overall_stats()
    all_sessions = fm.list_sessions(limit=200)
    ended = [session for session in all_sessions if session["status"] == "ended"]

    ui.section_header(
        "Focus history",
        "A clear view of completed sessions and recent study patterns.",
    )

    if stats["total_sessions"] == 0:
        ui.empty_state(
            "Your history starts with one session",
            "Complete a focus session and this page will show trends, totals and recent activity.",
            "Measures focus, study time, blinks and attention changes",
        )
        return

    current = _period_sessions(ended, 0, 6)
    previous = _period_sessions(ended, 7, 13)
    current_focus = _average_value(current, "avg_focus_preview")
    previous_focus = _average_value(previous, "avg_focus_preview")

    focus_series = _daily_series(ended, "avg_focus_preview", average=True)
    session_series = [
        len(_period_sessions(ended, offset, offset)) for offset in range(6, -1, -1)
    ]
    time_series = _daily_series(ended, "actual_duration_sec")
    blink_series = _daily_series(ended, "total_blinks")

    focus_col, sessions_col, time_col, blinks_col = st.columns([1.35, 1, 1, 1])
    with focus_col:
        ui.stat_card(
            "Average focus",
            format_number(stats["avg_focus"], 0),
            _delta_text(current_focus, previous_focus, "pts"),
            focus_series,
            hero=True,
        )
    with sessions_col:
        ui.stat_card(
            "Sessions",
            stats["total_sessions"],
            _delta_text(len(current), len(previous)),
            session_series,
        )
    with time_col:
        ui.stat_card(
            "Study time",
            format_duration(stats["total_study_sec"]),
            _delta_text(_sum_value(current, "actual_duration_sec") // 60,
                        _sum_value(previous, "actual_duration_sec") // 60,
                        "min"),
            time_series,
        )
    with blinks_col:
        ui.stat_card(
            "Blinks",
            stats["total_blinks"],
            _delta_text(_sum_value(current, "total_blinks"),
                        _sum_value(previous, "total_blinks")),
            blink_series,
        )

    best = stats["best_session"]
    if best is not None:
        st.markdown(
            '<div class="fm-best-session">Best session: '
            f"{relative_start_time(best['start_time'])} with a focus score of "
            f"{best['avg_focus_preview']:.0f}</div>",
            unsafe_allow_html=True,
        )

    ui.section_header("Last 7 days", "Daily average focus across completed sessions.")
    render_history_chart(fm.get_daily_summary(7))

    st.write("")
    ui.section_header("Recent sessions", "Latest completed and in-progress sessions.")
    render_recent_sessions(all_sessions)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main():
    """Build the page. Streamlit reruns this on every interaction."""
    st.set_page_config(page_title="FocusMate", layout="wide")
    ui.load_css()
    ui.sync_browser_theme()
    ui.hero()

    if MODULES_ERROR:
        st.error(MODULES_ERROR)
        st.stop()

    if USE_MOCK:
        st.info("Mock mode is active. Camera signals are simulated and not saved.")
    elif os.path.abspath(os.getcwd()) != os.path.dirname(os.path.abspath(__file__)):
        st.warning("Start the app from the project root so it uses the correct database.")

    init_state()
    fm.init_db()

    if st.session_state.pending_toast:
        st.toast(st.session_state.pending_toast)
        st.session_state.pending_toast = None

    live_tab, history_tab = st.tabs(["Live session", "History"])

    # Draw history first because the live loop holds the script while running.
    with history_tab:
        render_past_sessions()

    with live_tab:
        state = st.session_state
        if state.message:
            kind, message = state.message
            (st.error if kind == "error" else st.info)(message)

        if state.running:
            run_live_loop(make_live_layout())
        else:
            render_setup()
            if state.summary_id is not None:
                st.write("")
                render_summary(state.summary_id)
            else:
                ui.empty_state(
                    "Ready when you are",
                    "A session measures your eye openness, head direction and blink pattern in real time.",
                    "Camera stays on-device | Readings are stored locally",
                )


if __name__ == "__main__":
    main()
