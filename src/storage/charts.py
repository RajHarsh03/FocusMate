"""
FocusMate charts.

make_focus_chart() draws the focus preview of one session over time and
returns a matplotlib Figure. The dashboard shows it with st.pyplot(fig).

Only the Figure class is used (no pyplot), so this works without a screen
and does not keep old charts in memory between Streamlit reruns.
"""

from datetime import datetime

import numpy as np
from matplotlib.figure import Figure

try:
    from src.storage import database  # when imported from the project root
except ImportError:
    import database  # when run directly: python src/storage/charts.py

SMOOTH_WINDOW_SEC = 30   # moving average over about 30 readings (30 seconds)


def _moving_average(values, window):
    """Centered moving average that skips empty (NaN) values.

    Positions that are empty themselves stay empty, so the smooth line
    breaks at the same gaps as the raw line.
    """
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    sums = np.concatenate(([0.0], np.cumsum(filled)))
    counts = np.concatenate(([0], np.cumsum(valid)))

    n = len(values)
    half = window // 2
    starts = np.clip(np.arange(n) - half, 0, n)
    ends = np.clip(np.arange(n) + (window - half), 0, n)
    window_sums = sums[ends] - sums[starts]
    window_counts = counts[ends] - counts[starts]

    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = window_sums / window_counts
    smooth[~valid] = np.nan
    return smooth


def make_focus_chart(session_id):
    """Return a Figure of focus preview over time for one session, or None.

    X axis: time in minutes. Y axis: focus preview from 0 to 100.
    The raw readings are drawn as a light line and a 30 second moving
    average on top. The line is broken where the face was missing or the
    focus value is empty, instead of drawing a false line across the gap.
    Returns None if the session has no readings (or does not exist).
    """
    readings = database.get_readings(session_id)
    if not readings:
        return None

    minutes = np.array([
        (r["elapsed_sec"] if r["elapsed_sec"] is not None else i + 1) / 60.0
        for i, r in enumerate(readings)
    ])
    focus = np.array([
        r["focus_preview"]
        if r["focus_preview"] is not None and r["face_detected"] == 1 else np.nan
        for r in readings
    ], dtype=float)

    window = max(1, min(SMOOTH_WINDOW_SEC, len(readings) // 5))
    smooth = _moving_average(focus, window)

    fig = Figure(figsize=(8, 3.6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(minutes, focus, color="tab:blue", alpha=0.25, linewidth=0.8,
            label="Focus (raw)")
    ax.plot(minutes, smooth, color="tab:blue", linewidth=2.0,
            label=f"Focus (moving average, {window} s)")

    summary = database.get_session_summary(session_id)
    started = summary["start_time"] if summary else None
    try:
        when = datetime.fromisoformat(started).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        when = "unknown date"
    ax.set_title(f"Focus over time: session {session_id} ({when})")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Focus (preview, 0 to 100)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(minutes.max(), 1 / 60))
    ax.grid(True, alpha=0.3)

    if np.isnan(focus).all():
        ax.text(0.5, 0.5, "No focus data (face not detected)",
                transform=ax.transAxes, ha="center", va="center", color="gray")
    else:
        ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Small demo: python src/storage/charts.py
# Saves a chart of the newest session to data/focus_chart_demo.png
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    database.init_db()
    latest = database.list_sessions(limit=1)
    if not latest:
        print("No sessions yet. Run: python src/storage/seed_demo_data.py")
    else:
        sid = latest[0]["session_id"]
        figure = make_focus_chart(sid)
        if figure is None:
            print(f"Session {sid} has no readings.")
        else:
            out = os.path.join(os.path.dirname(database.DB_PATH) or ".", "focus_chart_demo.png")
            figure.savefig(out)
            print(f"Chart of session {sid} saved to {out}")
