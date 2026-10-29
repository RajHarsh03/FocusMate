"""
FocusMate session state flow (Week 4).

A small state machine that drives one study session through four states:

    ACTIVE  ->  WARNING  ->  BREAK_PROMPT  ->  ENDED

How states change:
    - ACTIVE: the normal state while focus is fine.
    - WARNING: entered when the smoothed focus score stays below the warning
      threshold for a sustained time (dwell). A gentle "you are slipping" state.
    - BREAK_PROMPT: entered when focus stays below the lower break threshold for
      a longer sustained time, or when WARNING persists that long. This is where
      the dashboard would suggest a break.
    - ENDED: the session is over. Reached ONLY when the planned duration expires
      or stop() is called. Low focus by itself never ends a session.

Anti-flicker:
    Frame-by-frame scores are noisy, so a raw dip should not flip the state.
    Two mechanisms prevent rapid switching:
      1. Dwell: a worsening condition must hold for a minimum time before the
         state escalates (3 s to reach WARNING, 5 s to reach BREAK_PROMPT).
      2. Hysteresis: to recover back toward ACTIVE, the score must climb above a
         higher recovery threshold (60), not merely back over the entry one, and
         that recovery must also hold for the dwell time.

Determinism:
    update() takes the current focus score and either an explicit elapsed time
    or reads a SessionTimer. Nothing here sleeps or reads the wall clock
    directly, so tests advance time by hand and get exact, repeatable results.

This module does no webcam or database work.

Spec reference: FocusMate-Module-Specs.md (session flow is Spec 4.4, postponed
for the showcase and implemented here as approved Week 4 work).
"""

from dataclasses import dataclass, field
from enum import Enum

try:
    from src.timer.session_timer import SessionTimer  # from the project root
except ImportError:
    from session_timer import SessionTimer  # direct: python src/timer/session_state.py


class SessionState(Enum):
    """The four session states, in order of increasing concern."""

    ACTIVE = "active"
    WARNING = "warning"
    BREAK_PROMPT = "break_prompt"
    ENDED = "ended"


@dataclass
class SessionStateConfig:
    """Tunable thresholds and timing for the state flow.

    All values are prototype defaults, not scientifically validated numbers.
    Scores are on the 0 to 100 focus scale.
    """

    warning_threshold: float = 50.0       # below this, focus is "low"
    break_threshold: float = 30.0         # below this, focus is "very low"
    recovery_threshold: float = 60.0      # at/above this, focus has "recovered"
    warning_dwell_sec: float = 3.0        # low must hold this long to warn
    break_dwell_sec: float = 5.0          # very low (or warning) this long -> break
    recovery_dwell_sec: float = 3.0       # recovery must hold this long to step back

    def __post_init__(self):
        if self.break_threshold > self.warning_threshold:
            raise ValueError("break_threshold must be <= warning_threshold")
        if self.recovery_threshold < self.warning_threshold:
            raise ValueError("recovery_threshold must be >= warning_threshold")
        for name in ("warning_dwell_sec", "break_dwell_sec", "recovery_dwell_sec"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass
class SessionStatus:
    """A snapshot of the session for the dashboard, returned by update()."""

    state: SessionState
    elapsed_sec: float
    remaining_sec: float
    smoothed_score: float | None     # the score used for this decision, or None
    state_changed: bool              # did the state change on this update?
    previous_state: SessionState | None
    time_in_state_sec: float         # how long the current state has been held


class SessionStateMachine:
    """Drives one session through ACTIVE / WARNING / BREAK_PROMPT / ENDED.

    Wiring for the dashboard's loop:
        machine = SessionStateMachine()
        machine.start()
        ...
        status = machine.update(smoothed_score)   # once per frame/second
        ...
        machine.stop()                            # user pressed Stop

    The score passed to update() is expected to be the smoothed focus score from
    FocusScorer. Pass None when there is no reading (no face / not computed yet):
    missing readings are held safely and never escalate the state on their own.
    """

    def __init__(self, planned_duration_sec=1500, config=None, timer=None):
        """Create the machine.

        Args:
            planned_duration_sec: session length used to build a SessionTimer
                when one is not supplied.
            config: a SessionStateConfig, or None for defaults.
            timer: a SessionTimer to reuse (tests inject one with a fake clock).
                When None, a timer is created from planned_duration_sec.
        """
        self.config = config if config is not None else SessionStateConfig()
        self.timer = timer if timer is not None else SessionTimer(planned_duration_sec)

        self._state = SessionState.ACTIVE
        self._previous_state = None
        self._last_score = None
        # When the current state was entered, and the current elapsed time, both
        # measured on the timer's clock so everything stays deterministic.
        self._state_entered_at = 0.0
        self._last_elapsed = 0.0
        # How long the "escalate" or "recover" condition has held continuously.
        self._bad_since = None     # elapsed time when the current bad run began
        self._good_since = None    # elapsed time when the current recovery began

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        """Start the session and its timer in the ACTIVE state."""
        self.timer.start()
        self._state = SessionState.ACTIVE
        self._previous_state = None
        self._state_entered_at = self.timer.elapsed_sec()
        self._last_elapsed = self._state_entered_at
        self._bad_since = None
        self._good_since = None

    def stop(self):
        """Stop the session now. Moves to ENDED unless already ended."""
        self.timer.stop()
        self._last_elapsed = self.timer.elapsed_sec()
        if self._state is not SessionState.ENDED:
            self._change_state(SessionState.ENDED)

    def reset(self, planned_duration_sec=None):
        """Return to a fresh, not-started ACTIVE session.

        Args:
            planned_duration_sec: optional new duration. When given, a new timer
                is built with the same clock as the old one, so an injected test
                clock is preserved.
        """
        if planned_duration_sec is not None:
            self.timer = SessionTimer(planned_duration_sec, clock=self.timer._clock)
        else:
            self.timer = SessionTimer(self.timer.planned_duration_sec,
                                      clock=self.timer._clock)
        self._state = SessionState.ACTIVE
        self._previous_state = None
        self._last_score = None
        self._state_entered_at = 0.0
        self._last_elapsed = 0.0
        self._bad_since = None
        self._good_since = None

    # -- the per-frame step -------------------------------------------------

    def update(self, smoothed_score):
        """Advance the machine by one reading and return a SessionStatus.

        Args:
            smoothed_score: the current smoothed focus score (0 to 100), or None
                if there is no reading this frame.

        Returns:
            SessionStatus describing the (possibly unchanged) state.
        """
        elapsed = self.timer.elapsed_sec()
        self._last_elapsed = elapsed
        self._last_score = smoothed_score
        changed = False

        # Time-based end always wins and cannot be undone.
        if self._state is not SessionState.ENDED and self.timer.is_finished():
            changed = self._change_state(SessionState.ENDED)
            return self._status(changed, elapsed)

        if self._state is SessionState.ENDED:
            return self._status(False, elapsed)

        # A missing reading is held: do not advance the bad/good runs on it, so a
        # gap in readings never escalates or recovers the state by itself.
        if smoothed_score is None:
            return self._status(False, elapsed)

        self._track_runs(smoothed_score, elapsed)

        config = self.config
        if self._state is SessionState.ACTIVE:
            if self._bad_held(config.break_threshold, config.break_dwell_sec, elapsed):
                changed = self._change_state(SessionState.BREAK_PROMPT)
            elif self._bad_held(config.warning_threshold, config.warning_dwell_sec, elapsed):
                changed = self._change_state(SessionState.WARNING)

        elif self._state is SessionState.WARNING:
            # Escalate if focus stays low long enough, whether it is merely low
            # or very low: sustained warning should lead to a break prompt.
            if self._bad_held(config.warning_threshold, config.break_dwell_sec, elapsed):
                changed = self._change_state(SessionState.BREAK_PROMPT)
            elif self._good_held(elapsed):
                changed = self._change_state(SessionState.ACTIVE)

        elif self._state is SessionState.BREAK_PROMPT:
            # From a break prompt, only a sustained recovery steps back, one level
            # at a time, to WARNING. It never jumps straight to ACTIVE.
            if self._good_held(elapsed):
                changed = self._change_state(SessionState.WARNING)

        return self._status(changed, elapsed)

    # -- run tracking and dwell checks -------------------------------------

    def _track_runs(self, score, elapsed):
        """Update how long the score has been continuously bad and continuously good.

        "Bad" means below the warning threshold; "good" means at or above the
        recovery threshold. A score in the gap between the two thresholds is
        neither: it breaks a bad run (the user is no longer below warning, so a
        warning must not keep building) and it also breaks a recovery run (a
        half-recovery does not count as recovered). This two-sided reset is the
        hysteresis that stops rapid flicker around a single threshold.
        """
        if score < self.config.warning_threshold:
            if self._bad_since is None:
                self._bad_since = elapsed
            self._good_since = None
        elif score >= self.config.recovery_threshold:
            if self._good_since is None:
                self._good_since = elapsed
            self._bad_since = None
        else:
            # In the hysteresis band: neither clearly bad nor clearly recovered.
            self._bad_since = None
            self._good_since = None

    def _bad_held(self, threshold, dwell_sec, elapsed):
        """True if the score has been below `threshold` continuously for `dwell_sec`.

        Uses the tracked bad run plus a direct check against the given threshold,
        so the same run can serve both the warning and the break checks.
        """
        if self._bad_since is None or self._last_score is None:
            return False
        if self._last_score >= threshold:
            return False
        return (elapsed - self._bad_since) >= dwell_sec

    def _good_held(self, elapsed):
        """True if the score has been at or above the recovery threshold for the dwell."""
        if self._good_since is None:
            return False
        return (elapsed - self._good_since) >= self.config.recovery_dwell_sec

    # -- helpers ------------------------------------------------------------

    def _change_state(self, new_state):
        """Switch to new_state, record the change, and reset the dwell runs."""
        if new_state is self._state:
            return False
        self._previous_state = self._state
        self._state = new_state
        self._state_entered_at = self._last_elapsed
        # Starting fresh in a new state: clear the runs so dwell is measured from
        # entry, which stops immediate re-triggering.
        self._bad_since = None
        self._good_since = None
        return True

    def _status(self, changed, elapsed):
        """Build the SessionStatus snapshot returned to the caller."""
        return SessionStatus(
            state=self._state,
            elapsed_sec=elapsed,
            remaining_sec=self.timer.remaining_sec(),
            smoothed_score=self._last_score,
            state_changed=changed,
            previous_state=self._previous_state if changed else None,
            time_in_state_sec=max(0.0, elapsed - self._state_entered_at),
        )

    # -- read-only views ----------------------------------------------------

    @property
    def state(self):
        """The current SessionState."""
        return self._state

    def is_ended(self):
        """True once the session has reached ENDED."""
        return self._state is SessionState.ENDED


# ---------------------------------------------------------------------------
# Small demo: python src/timer/session_state.py
# Drives the machine with a fake clock and a scripted score, no webcam needed.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

    clock = FakeClock()
    timer = SessionTimer(planned_duration_sec=60, clock=clock)
    machine = SessionStateMachine(timer=timer)
    machine.start()

    # One second per step. Focus is fine, then drops low, then very low, then
    # recovers. Watch WARNING and BREAK_PROMPT appear only after the dwell times.
    scripted = ([90] * 2 + [40] * 6 + [20] * 6 + [95] * 6)
    for score in scripted:
        status = machine.update(float(score))
        marker = "  <-- changed" if status.state_changed else ""
        print(f"t={status.elapsed_sec:5.1f}s score={score:>3} "
              f"state={status.state.value:<12}{marker}")
        clock.advance(1)

    machine.stop()
    print(f"after stop: state={machine.state.value}")
