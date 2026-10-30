"""
FocusMate full Focus Score with temporal smoothing (Week 3).

IMPORTANT: this is NOT the preview meter. The preview meter
(src/detection/focus_preview.py, compute_focus_preview) is a separate, frozen
showcase contract with fixed point subtractions. This module is the Week 3
"full" score, which the eight-week plan describes as a score built from EAR and
head position and then smoothed over time.

What makes the full score different from the preview:
    - The eye signal is GRADED, not a flat step. EAR is mapped continuously:
      fully open eyes score 100 on the eye component, fully closed score 0, and
      values in between slide smoothly. The preview, by contrast, only ever
      subtracts a fixed 40 or 15.
    - The head signal is GRADED too. When the vision module supplies yaw/pitch
      angles (Contract A), the head component falls off smoothly as the head
      turns further from centre. When only the head_direction label is
      available, each label maps to a graded component value.
    - The two components are combined with configurable WEIGHTS, then smoothed
      with an exponential moving average (EMA). The preview has no weights and
      no smoothing.

Honesty about what this is:
    This score is a HEURISTIC prototype signal, not a scientifically validated
    measure of a person's true cognitive focus. The mappings, weights, EMA alpha
    and thresholds are prototype defaults chosen so the demo behaves sensibly.
    They are all configurable and should be stated as heuristic in the report.

Missing-data policy (unchanged and deliberate):
    - No face -> the configured no-face score (0 by default), matching the
      preview meter and the storage summary so averages line up.
    - A missing EAR while a face is present does NOT force the eye component to
      0, because "cannot measure" is not evidence of closed eyes. Instead the
      eye component is treated as neutral (a configurable value).
    - An "unknown" head direction is likewise treated as neutral, not as
      definite looking-away. A lost measurement is never scored as distraction.

This module does no webcam or database work, so it is fully unit testable.

Spec references: FocusMate-Module-Specs.md (Sections 2.3, 4.2 for the preview
contract this builds alongside); eight-week plan Week 3 (EAR + head position,
smoothed).
"""

from dataclasses import dataclass, field

# Head-direction labels from Contract A. Kept here so the mapping below is
# explicit rather than relying on magic strings scattered around.
VALID_DIRECTIONS = ("center", "left", "right", "up", "down", "unknown")


@dataclass
class FocusScoreConfig:
    """All tunable numbers for the full Focus Score, in one place.

    Every value here is a prototype default, not a scientifically validated
    constant. Change them freely; nothing else in the module hard-codes a number.
    """

    # -- blend weights (eye vs head). They are normalised at use time, so they
    #    do not have to sum to exactly 1.0.
    ear_weight: float = 0.6          # how much eye openness matters
    head_weight: float = 0.4         # how much head position matters

    # -- eye-openness component (graded mapping of EAR to 0..100) -----------
    ear_closed: float = 0.18         # at/below this, eye component is 0
    ear_open: float = 0.28           # at/above this, eye component is 100
    ear_neutral_component: float = 70.0   # used when EAR is missing (face present)

    # -- head component, label fallback (used when no yaw/pitch available) --
    #    Graded: centred is best, up/down are mild, left/right are stronger
    #    deviations, and "unknown" is neutral rather than a full penalty.
    head_label_components: dict = field(default_factory=lambda: {
        "center": 100.0,
        "up": 60.0,
        "down": 60.0,
        "left": 20.0,
        "right": 20.0,
        "unknown": 70.0,   # neutral: we do not know, so do not assume distraction
    })

    # -- head component, angle mapping (used when yaw/pitch are provided) ---
    #    The component falls linearly from 100 at 0 degrees off-centre to 0 at
    #    head_angle_max degrees, using the larger of |yaw| and |pitch|.
    head_angle_max: float = 35.0     # degrees off-centre that score 0 on head

    # -- overall bounds and smoothing ---------------------------------------
    no_face_score: float = 0.0       # score when no face is detected
    min_score: float = 0.0           # lower clamp
    max_score: float = 100.0         # upper clamp
    alpha: float = 0.3               # EMA weight for the newest raw score (prototype default)

    def __post_init__(self):
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in the range (0, 1]")
        if self.ear_weight < 0 or self.head_weight < 0:
            raise ValueError("weights must not be negative")
        if (self.ear_weight + self.head_weight) <= 0:
            raise ValueError("at least one of ear_weight/head_weight must be positive")
        if self.ear_open <= self.ear_closed:
            raise ValueError("ear_open must be greater than ear_closed")
        if self.head_angle_max <= 0:
            raise ValueError("head_angle_max must be positive")


@dataclass
class FocusComponents:
    """A snapshot of one frame's scoring, for the dashboard and for debugging."""

    face_detected: bool
    ear: float | None
    head_direction: str
    eye_component: float        # graded eye-openness contribution, 0..100
    head_component: float       # graded head-position contribution, 0..100
    ear_missing: bool           # eye component came from the neutral fallback
    head_from_angles: bool      # head component used yaw/pitch, not the label
    raw_score: float
    smoothed_score: float


@dataclass
class FocusScorer:
    """Turns per-frame face signals into a smoothed full Focus Score.

    Call update(...) once per frame. It returns the smoothed score and remembers
    it, so the next frame is smoothed against it. Read `raw_score`,
    `smoothed_score` and `components` for the dashboard, and call `reset()` at
    the start of a new session.
    """

    config: FocusScoreConfig = field(default_factory=FocusScoreConfig)

    def __post_init__(self):
        self._raw_score = None
        self._smoothed_score = None
        self._components = None

    # -- read-only views for the dashboard ---------------------------------

    @property
    def raw_score(self):
        """The most recent unsmoothed score, or None before the first update."""
        return self._raw_score

    @property
    def smoothed_score(self):
        """The most recent smoothed score, or None before the first update."""
        return self._smoothed_score

    @property
    def components(self):
        """A FocusComponents snapshot of the last frame, or None before the first update."""
        return self._components

    # -- graded component helpers ------------------------------------------

    def eye_component(self, face_detected, ear):
        """Return the graded eye-openness component (0..100) and whether EAR was missing.

        Fully open eyes (>= ear_open) give 100, fully closed (<= ear_closed)
        give 0, and values between slide linearly. A missing EAR while a face is
        present returns the configured neutral value, not 0.
        """
        config = self.config
        if ear is None:
            return config.ear_neutral_component, True
        if ear <= config.ear_closed:
            return 0.0, False
        if ear >= config.ear_open:
            return 100.0, False
        span = config.ear_open - config.ear_closed
        return 100.0 * (ear - config.ear_closed) / span, False

    def head_component(self, head_direction, yaw=None, pitch=None):
        """Return the graded head-position component (0..100) and whether angles were used.

        If yaw/pitch are given, the component falls linearly from 100 at centre
        to 0 at head_angle_max degrees off-centre (using the larger absolute
        angle). Otherwise the head_direction label is mapped through
        head_label_components, with "unknown" treated as neutral.
        """
        config = self.config
        if yaw is not None or pitch is not None:
            off = max(abs(yaw or 0.0), abs(pitch or 0.0))
            fraction = min(1.0, off / config.head_angle_max)
            return 100.0 * (1.0 - fraction), True

        label = head_direction if head_direction in config.head_label_components else "unknown"
        return float(config.head_label_components[label]), False

    # -- the main operations ------------------------------------------------

    def reset(self):
        """Forget all history. The next update starts smoothing fresh."""
        self._raw_score = None
        self._smoothed_score = None
        self._components = None

    def compute_raw(self, face_detected, ear, head_direction, yaw=None, pitch=None):
        """Return the unsmoothed 0..100 full score for one frame, without storing it.

        The score is a weighted blend of the graded eye and head components.
        No face returns the configured no-face score.
        """
        config = self.config
        if not face_detected:
            return self._clamp(config.no_face_score), None

        eye, ear_missing = self.eye_component(face_detected, ear)
        head, head_from_angles = self.head_component(head_direction, yaw, pitch)

        total_weight = config.ear_weight + config.head_weight
        blended = (config.ear_weight * eye + config.head_weight * head) / total_weight
        raw = self._clamp(blended)

        detail = {
            "eye_component": eye,
            "head_component": head,
            "ear_missing": ear_missing,
            "head_from_angles": head_from_angles,
        }
        return raw, detail

    def update(self, face_detected, ear, head_direction, yaw=None, pitch=None):
        """Score one frame, smooth it against history, store everything, and return it.

        Args:
            face_detected: True if a face was found this frame.
            ear: the frame's EAR (float), or None if unavailable.
            head_direction: "center", "left", "right", "up", "down" or "unknown".
            yaw, pitch: optional head angles in degrees (Contract A). Used for a
                graded head component when present; ignored when None.

        Returns:
            float smoothed focus score (0 to 100).
        """
        direction = head_direction if head_direction in VALID_DIRECTIONS else "unknown"
        raw, detail = self.compute_raw(face_detected, ear, direction, yaw, pitch)

        if self._smoothed_score is None:
            # First valid frame: nothing to smooth against, so start at raw.
            smoothed = raw
        else:
            alpha = self.config.alpha
            smoothed = alpha * raw + (1.0 - alpha) * self._smoothed_score

        self._raw_score = raw
        self._smoothed_score = smoothed

        if detail is None:   # no face
            self._components = FocusComponents(
                face_detected=False, ear=ear, head_direction=direction,
                eye_component=0.0, head_component=0.0,
                ear_missing=(ear is None), head_from_angles=False,
                raw_score=raw, smoothed_score=smoothed,
            )
        else:
            self._components = FocusComponents(
                face_detected=True, ear=ear, head_direction=direction,
                eye_component=detail["eye_component"],
                head_component=detail["head_component"],
                ear_missing=detail["ear_missing"],
                head_from_angles=detail["head_from_angles"],
                raw_score=raw, smoothed_score=smoothed,
            )
        return smoothed

    # -- small helpers ------------------------------------------------------

    def _clamp(self, value):
        """Keep a score within the configured min and max."""
        return max(self.config.min_score, min(self.config.max_score, float(value)))


# ---------------------------------------------------------------------------
# Small demo: python src/detection/focus_score.py
# Shows raw vs smoothed over a short frame stream, no webcam needed.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scorer = FocusScorer()
    # Steady focus, a drooping spell, a glance away, a no-face gap, then back.
    # Watch the smoothed value move more gently than the raw value, and note the
    # graded response: EAR 0.23 is partway down, not a flat step.
    frames = [
        (True, 0.30, "center"),
        (True, 0.30, "center"),
        (True, 0.23, "center"),   # drooping: graded, not a fixed -15
        (True, 0.30, "left"),     # looking away
        (True, 0.30, "center"),
        (False, None, "unknown"),  # no face -> 0
        (True, 0.30, "center"),
    ]
    print(f"{'face':<5} {'ear':<5} {'dir':<7} {'eye':>6} {'head':>6} {'raw':>7} {'smoothed':>9}")
    for face, ear_value, direction in frames:
        smoothed = scorer.update(face, ear_value, direction)
        c = scorer.components
        print(f"{face!s:<5} {ear_value!s:<5} {direction:<7} "
              f"{c.eye_component:6.1f} {c.head_component:6.1f} "
              f"{scorer.raw_score:7.2f} {smoothed:9.2f}")
