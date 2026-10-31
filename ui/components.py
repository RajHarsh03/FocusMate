"""Reusable UI components for FocusMate.

These helpers only render presentation. They do not calculate focus, operate
the camera, change session state or write to the database.
"""

import html
import math

import streamlit as st


ICONS = {
    "focus": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="7.5" stroke="currentColor" stroke-width="1.5"/><circle cx="12" cy="12" r="2.5" fill="currentColor"/></svg>""",
    "lock": """<svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="5" y="10" width="14" height="10" rx="2" stroke="currentColor" stroke-width="1.6"/><path d="M8 10V7.5a4 4 0 0 1 8 0V10" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>""",
    "empty": """<svg width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true"><circle cx="16" cy="16" r="10" stroke="currentColor" stroke-width="1.5"/><path d="M7 16h3m12 0h3M16 7v3m0 12v3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="16" cy="16" r="2.5" fill="currentColor"/></svg>""",
}

THEME_TOKENS = {
    "dark": {
        "bg-0": "#0C0D0F",
        "bg-1": "#121418",
        "bg-2": "#171A20",
        "bg-3": "#1D2027",
        "border-subtle": "rgba(255, 255, 255, 0.10)",
        "border-strong": "rgba(255, 255, 255, 0.18)",
        "text-primary": "#F2F4F7",
        "text-secondary": "#A7ADB7",
        "text-tertiary": "#737B87",
        "accent": "#6C8CFF",
        "accent-soft": "rgba(108, 140, 255, 0.13)",
    },
    "light": {
        "bg-0": "#F7F8FA",
        "bg-1": "#FFFFFF",
        "bg-2": "#F0F2F5",
        "bg-3": "#EAEDF2",
        "border-subtle": "rgba(20, 24, 32, 0.10)",
        "border-strong": "rgba(20, 24, 32, 0.18)",
        "text-primary": "#17191E",
        "text-secondary": "#5F6672",
        "text-tertiary": "#858D99",
        "accent": "#4F6FE8",
        "accent-soft": "rgba(79, 111, 232, 0.11)",
    },
}


def _safe(value):
    """Escape dynamic text before it is placed in HTML."""
    return html.escape(str(value), quote=True)


def _write(markup, target=None):
    """Render HTML into Streamlit or into an existing placeholder."""
    renderer = target.markdown if target is not None else st.markdown
    renderer(markup, unsafe_allow_html=True)


def load_css(path="styles.css"):
    """Load CSS with explicit tokens for the active Streamlit theme."""
    theme_type = st.context.theme.get("type") or "dark"
    if theme_type not in THEME_TOKENS:
        theme_type = "dark"
    tokens = THEME_TOKENS[theme_type]
    token_css = ":root {" + "".join(
        f"--fm-{name}: {value};" for name, value in tokens.items()
    ) + "}"
    with open(path, encoding="utf-8") as style_file:
        st.markdown(
            f"<style>{style_file.read()}\n{token_css}</style>",
            unsafe_allow_html=True,
        )


def sync_browser_theme():
    """Keep custom CSS tokens aligned with Streamlit's theme menu."""
    st.html(
        """
        <script>
        (() => {
          const host = window;
          const root = document.documentElement;
          const key = `stActiveTheme-${location.pathname}-v2`;
          const systemDark = matchMedia("(prefers-color-scheme: dark)");

          function selectedTheme() {
            try {
              return JSON.parse(localStorage.getItem(key) || '"System"');
            } catch (error) {
              return "System";
            }
          }

          function applyTheme() {
            const selected = selectedTheme();
            const resolved = selected === "Light"
              ? "light"
              : selected === "Dark"
                ? "dark"
                : systemDark.matches ? "dark" : "light";
            root.dataset.fmTheme = resolved;
          }

          applyTheme();
          if (host.__focusmateThemeTimer) {
            clearInterval(host.__focusmateThemeTimer);
          }
          host.__focusmateThemeTimer = setInterval(applyTheme, 150);
          systemDark.addEventListener("change", applyTheme);
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def hero():
    """Render the compact product heading and privacy statement."""
    _write(
        f"""
        <header class="fm-hero">
          <div class="fm-brand-row">
            <span class="fm-mark">{ICONS['focus']}</span>
            <h1>FocusMate</h1>
          </div>
          <p class="fm-value-prop">A calmer way to understand your focus while you study.</p>
          <div class="fm-privacy-row">{ICONS['lock']}<span>Processed on-device, nothing recorded</span></div>
        </header>
        """
    )


def section_header(title, description=None, kicker=None, target=None):
    """Render a consistent section title with optional supporting text."""
    kicker_html = f'<div class="fm-section-kicker">{_safe(kicker)}</div>' if kicker else ""
    description_html = f"<p>{_safe(description)}</p>" if description else ""
    _write(
        f'<div class="fm-section-heading">{kicker_html}<h2>{_safe(title)}</h2>{description_html}</div>',
        target,
    )


def _status_pill_html(text, tone="neutral", detail=None):
    """Return status-pill HTML for standalone or composed components."""
    detail_html = f"<span>{_safe(detail)}</span>" if detail else ""
    return (
        f'<span class="fm-status-pill" data-tone="{_safe(tone)}">'
        f'<span class="fm-status-dot"></span><span>{_safe(text)}</span>{detail_html}</span>'
    )


def status_pill(text, tone="neutral", detail=None, target=None):
    """Return status HTML and render it only when a target is supplied."""
    markup = _status_pill_html(text, tone, detail)
    if target is not None:
        _write(markup, target)
    return markup


def _sparkline(values):
    """Return a tiny accessible SVG line for a short numeric series."""
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(clean) < 2:
        return ""
    low, high = min(clean), max(clean)
    span = high - low or 1.0
    width, height = 76.0, 26.0
    points = []
    for index, value in enumerate(clean):
        x = index * width / (len(clean) - 1)
        y = height - 2 - ((value - low) / span) * (height - 4)
        points.append(f"{x:.1f},{y:.1f}")
    return (
        '<svg class="fm-sparkline" viewBox="0 0 76 26" fill="none" aria-hidden="true">'
        f'<polyline points="{" ".join(points)}" stroke="currentColor" stroke-width="1.6" '
        'stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>'
    )


def stat_card(label, value, delta=None, sparkline=None, hero=False, target=None):
    """Render a history statistic with optional period delta and sparkline."""
    classes = "fm-stat-card is-hero" if hero else "fm-stat-card"
    delta_html = f'<div class="fm-stat-delta">{_safe(delta)}</div>' if delta else ""
    spark_html = _sparkline(sparkline or [])
    _write(
        f'<div class="{classes}"><div class="fm-stat-label">{_safe(label)}</div>'
        f'<div class="fm-stat-value">{_safe(value)}</div>{delta_html}{spark_html}</div>',
        target,
    )


def _metric_tile_html(label, value, caption=None, muted=False):
    """Return one compact live metric tile as HTML."""
    muted_class = " is-muted" if muted else ""
    caption_html = f'<div class="fm-metric-caption">{_safe(caption)}</div>' if caption else ""
    return (
        f'<div class="fm-metric-tile{muted_class}">'
        f'<div class="fm-metric-label">{_safe(label)}</div>'
        f'<div class="fm-metric-value">{_safe(value)}</div>{caption_html}</div>'
    )


def metric_tile(label, value, caption=None, muted=False, target=None):
    """Render one compact live metric tile with a label, value and caption."""
    markup = _metric_tile_html(label, value, caption, muted)
    _write(markup, target)
    return markup


def metric_grid(items, target=None):
    """Render metric tile dictionaries in a stable two-column grid."""
    tiles = "".join(_metric_tile_html(**item) for item in items)
    _write(f'<div class="fm-metric-grid">{tiles}</div>', target)


def focus_ring(score, face_detected=True, target=None, state="active"):
    """Render the full focus score and its current session state."""
    numeric = max(0.0, min(100.0, float(score or 0.0)))
    circumference = 2 * math.pi * 58
    offset = circumference * (1 - numeric / 100.0)
    state_class = f" is-{state}" if state in ("warning", "break_prompt") else ""
    muted_class = " is-muted" if not face_detected else ""
    display = "-" if not face_detected else str(int(round(numeric)))
    if not face_detected:
        label = "Waiting for face"
    elif state == "break_prompt":
        label = "Break suggested"
    elif state == "warning":
        label = "Focus dropping"
    elif numeric >= 70:
        label = "Focused"
    elif numeric >= 40:
        label = "Attention drifting"
    else:
        label = "Low attention"
    _write(
        f"""
        <div class="fm-focus-ring-wrap{muted_class}{state_class}">
          <svg class="fm-focus-ring" viewBox="0 0 168 168" role="img" aria-label="Focus score {display}">
            <circle class="fm-ring-track" cx="84" cy="84" r="58"/>
            <circle class="fm-ring-value" cx="84" cy="84" r="58"
              stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"/>
            <text class="fm-ring-score" x="84" y="84" text-anchor="middle" dominant-baseline="middle">{display}</text>
            <text class="fm-ring-unit" x="84" y="112" text-anchor="middle">FOCUS</text>
          </svg>
          <div class="fm-ring-label">{_safe(label)}</div>
        </div>
        """,
        target,
    )


def timer_display(remaining_text, elapsed_text, target=None):
    """Render the countdown using tabular numerals so it never shifts width."""
    _write(
        f"""
        <div class="fm-timer">
          <div class="fm-timer-label">Time remaining</div>
          <div class="fm-timer-value">{_safe(remaining_text)}</div>
          <div class="fm-timer-caption">{_safe(elapsed_text)} elapsed</div>
        </div>
        """,
        target,
    )


def camera_panel(frame_rgb, elapsed_text, face_detected, status_target, image_target):
    """Update the camera status bar and edge-to-edge image in stable placeholders."""
    face_text = "Face detected" if face_detected else "Waiting for face"
    tone = "active" if face_detected else "neutral"
    status = _status_pill_html("Live", "active")
    face = _status_pill_html(face_text, tone)
    status_target.markdown(
        f'<div class="fm-camera-status"><div class="fm-camera-status-left">{status}'
        f'<span>{_safe(elapsed_text)}</span></div>{face}</div>',
        unsafe_allow_html=True,
    )
    image_target.image(frame_rgb, channels="RGB", width="stretch")


def inline_notice(text, target=None, tone="neutral"):
    """Render one calm inline notice for a live signal or state."""
    _write(
        f'<div class="fm-inline-notice is-{_safe(tone)}">'
        f'<span class="fm-status-dot"></span><span>{_safe(text)}</span></div>',
        target,
    )


def break_prompt(target=None):
    """Render the persistent advisory shown after sustained low focus."""
    _write(
        """
        <div class="fm-break-prompt">
          <div class="fm-break-prompt-label">Break suggestion</div>
          <div class="fm-break-prompt-title">A short break may help</div>
          <div class="fm-break-prompt-copy">
            Your focus has stayed low for a while. You can continue, or end this
            session and return after a short break.
          </div>
        </div>
        """,
        target,
    )


def empty_state(title, description, measures=None, compact=False, target=None):
    """Render a small useful empty state with a minimal inline SVG."""
    measure_html = f'<div class="fm-empty-measures">{_safe(measures)}</div>' if measures else ""
    style = ' style="margin-top:16px;margin-bottom:16px"' if compact else ""
    _write(
        f"""
        <div class="fm-empty-state"{style}>
          <div class="fm-empty-icon">{ICONS['empty']}</div>
          <div><h3>{_safe(title)}</h3><p>{_safe(description)}</p>{measure_html}</div>
        </div>
        """,
        target,
    )
