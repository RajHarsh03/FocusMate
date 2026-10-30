"""
Tests for the FocusMate timer modules (Muskan).

Covers:
    - session_timer.py  SessionTimer (elapsed, remaining, start/stop rules)
    - session_state.py  the Active / Warning / Break Prompt / Ended flow

Every test injects a fake clock, so there are no real sleeps and the results
are exact and repeatable. No webcam is needed.

Run from the project root:
    python src/timer/test_timer.py

Optional, the same tests also run with pytest:
    pytest src/timer/test_timer.py
"""

import math
import sys
import traceback

try:
    from src.timer.session_timer import SessionTimer
    from src.timer.session_state import (
        SessionState, SessionStateConfig, SessionStateMachine,
    )
except ImportError:
    from session_timer import SessionTimer
    from session_state import (
        SessionState, SessionStateConfig, SessionStateMachine,
    )


class FakeClock:
    """A controllable clock: starts at 0 and only moves when advance() is called."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _close(a, b, tol=1e-9):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# ---------------------------------------------------------------------------
# SessionTimer tests
# ---------------------------------------------------------------------------

def test_elapsed_zero_before_start():
    timer = SessionTimer(60, clock=FakeClock())
    assert timer.elapsed_sec() == 0.0
    assert timer.is_running() is False


def test_elapsed_and_remaining_while_running():
    clock = FakeClock()
    timer = SessionTimer(60, clock=clock)
    timer.start()
    clock.advance(25)
    assert _close(timer.elapsed_sec(), 25.0)
    assert _close(timer.remaining_sec(), 35.0)
    assert timer.is_running() is True
    assert timer.is_finished() is False


def test_start_twice_does_not_reset():
    clock = FakeClock()
    timer = SessionTimer(60, clock=clock)
    timer.start()
    clock.advance(10)
    timer.start()           # should be ignored
    clock.advance(5)
    assert _close(timer.elapsed_sec(), 15.0)


def test_stop_freezes_elapsed():
    clock = FakeClock()
    timer = SessionTimer(60, clock=clock)
    timer.start()
    clock.advance(20)
    timer.stop()
    clock.advance(100)      # time keeps moving
    assert _close(timer.elapsed_sec(), 20.0)
    assert timer.is_running() is False


def test_remaining_never_negative_and_finished():
    clock = FakeClock()
    timer = SessionTimer(30, clock=clock)
    timer.start()
    clock.advance(45)       # past the end
    assert timer.remaining_sec() == 0.0
    assert timer.is_finished() is True


def test_format_time():
    timer = SessionTimer(0, clock=FakeClock())
    assert timer.format_time(0) == "00:00"
    assert timer.format_time(9) == "00:09"
    assert timer.format_time(75) == "01:15"
    assert timer.format_time(600) == "10:00"
    assert timer.format_time(-5) == "00:00"   # negative clamped


def test_restart_after_stop_begins_new_run():
    clock = FakeClock()
    timer = SessionTimer(60, clock=clock)
    timer.start()
    clock.advance(10)
    timer.stop()
    clock.advance(5)
    timer.start()           # fresh run
    clock.advance(3)
    assert _close(timer.elapsed_sec(), 3.0)


# ---------------------------------------------------------------------------
# Session state machine helpers
# ---------------------------------------------------------------------------

def _machine(planned=1000, config=None):
    """Build a machine with a fake clock we can advance; returns (machine, clock)."""
    clock = FakeClock()
    timer = SessionTimer(planned, clock=clock)
    machine = SessionStateMachine(config=config, timer=timer)
    machine.start()
    return machine, clock


def _feed(machine, clock, score, seconds, step=1.0):
    """Feed a constant score across `seconds` of elapsed time.

    Updates happen at the current clock time, then the clock advances by `step`,
    so a call that begins at t covers updates at t, t+step, ... up to t+seconds.
    The machine measures dwell from the first bad/good frame it sees, so feeding
    `seconds` equal to a dwell drives exactly that transition. Returns the last
    SessionStatus.
    """
    status = None
    ticks = int(round(seconds / step)) + 1   # inclusive of both endpoints
    for _ in range(ticks):
        status = machine.update(score)
        clock.advance(step)
    return status


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------

def test_starts_active():
    machine, _ = _machine()
    assert machine.state is SessionState.ACTIVE
    assert machine.is_ended() is False


def test_good_focus_stays_active():
    machine, clock = _machine()
    status = _feed(machine, clock, score=90.0, seconds=30)
    assert machine.state is SessionState.ACTIVE
    assert status.state_changed is False


def test_low_focus_enters_warning_after_dwell():
    """Below the warning threshold for the warning dwell -> WARNING, not before."""
    config = SessionStateConfig(warning_threshold=50, warning_dwell_sec=3, break_dwell_sec=5)
    machine, clock = _machine(config=config)

    # Two seconds of low focus: not long enough yet.
    _feed(machine, clock, score=40.0, seconds=2)
    assert machine.state is SessionState.ACTIVE
    # One more second reaches the 3 second dwell.
    status = machine.update(40.0)
    assert machine.state is SessionState.WARNING
    assert status.state_changed is True
    assert status.previous_state is SessionState.ACTIVE


def test_very_low_focus_from_active_can_reach_break_prompt():
    """Sustained very-low focus escalates ACTIVE straight to BREAK_PROMPT at break dwell."""
    config = SessionStateConfig(warning_threshold=50, break_threshold=30,
                                warning_dwell_sec=3, break_dwell_sec=5)
    machine, clock = _machine(config=config)
    # Score 20 is below both thresholds. It warns after the 3 s warning dwell,
    # then escalates to a break prompt after the 5 s break dwell measured from
    # entering warning. We check the whole progression tick by tick.
    states = []
    for _ in range(10):   # t = 0..9
        status = machine.update(20.0)
        states.append(status.state)
        clock.advance(1)
    assert SessionState.WARNING in states
    assert states[-1] is SessionState.BREAK_PROMPT


def test_warning_escalates_to_break_on_sustained_low():
    """Sustained low focus (not necessarily below break threshold) escalates to break."""
    config = SessionStateConfig(warning_threshold=50, break_threshold=30,
                                warning_dwell_sec=3, break_dwell_sec=5)
    machine, clock = _machine(config=config)
    # 45 is below warning (50) but above break (30). It should still reach the
    # break prompt: warning after the warning dwell (3 s), then break after the
    # break dwell (5 s) measured from entering warning, so about 8 s of low focus.
    status = _feed(machine, clock, score=45.0, seconds=9)
    assert machine.state is SessionState.BREAK_PROMPT


def test_recovery_steps_back_one_level_with_hysteresis():
    """From BREAK_PROMPT, sustained recovery goes to WARNING, then ACTIVE, not straight to ACTIVE."""
    config = SessionStateConfig(warning_threshold=50, break_threshold=30,
                                recovery_threshold=60, warning_dwell_sec=3,
                                break_dwell_sec=5, recovery_dwell_sec=3)
    machine, clock = _machine(config=config)

    _feed(machine, clock, score=20.0, seconds=9)      # drive to break prompt
    assert machine.state is SessionState.BREAK_PROMPT

    # Recover: high score held for the recovery dwell steps back to WARNING.
    _feed(machine, clock, score=90.0, seconds=3)
    assert machine.state is SessionState.WARNING
    # A further sustained recovery steps WARNING back to ACTIVE.
    _feed(machine, clock, score=90.0, seconds=3)
    assert machine.state is SessionState.ACTIVE


def test_hysteresis_band_does_not_recover():
    """A score between warning and recovery thresholds must not step the state back."""
    config = SessionStateConfig(warning_threshold=50, recovery_threshold=60,
                                warning_dwell_sec=3, break_dwell_sec=5, recovery_dwell_sec=3)
    machine, clock = _machine(config=config)
    _feed(machine, clock, score=40.0, seconds=3)      # -> WARNING
    assert machine.state is SessionState.WARNING
    # 55 is above warning but below recovery (60): not a real recovery.
    _feed(machine, clock, score=55.0, seconds=10)
    assert machine.state is SessionState.WARNING


def test_no_rapid_flicker_on_noisy_scores():
    """Alternating good/bad scores around the threshold must not flip the state each frame."""
    config = SessionStateConfig(warning_threshold=50, recovery_threshold=60,
                                warning_dwell_sec=3, break_dwell_sec=5, recovery_dwell_sec=3)
    machine, clock = _machine(config=config)
    changes = 0
    for i in range(20):
        score = 45.0 if i % 2 == 0 else 55.0   # noisy around the threshold
        status = machine.update(score)
        if status.state_changed:
            changes += 1
        clock.advance(1)
    # The bad run keeps resetting, so it should never dwell long enough to warn.
    assert machine.state is SessionState.ACTIVE
    assert changes == 0


def test_timer_expiry_ends_session():
    """Reaching the planned duration moves to ENDED regardless of focus."""
    machine, clock = _machine(planned=10)
    status = _feed(machine, clock, score=90.0, seconds=10)
    assert machine.state is SessionState.ENDED
    assert machine.is_ended() is True
    # Further updates stay ended.
    status = machine.update(90.0)
    assert machine.state is SessionState.ENDED
    assert status.state_changed is False


def test_low_focus_alone_never_ends_session():
    """No matter how low or how long the focus stays, the session does not end on its own."""
    machine, clock = _machine(planned=10_000)
    _feed(machine, clock, score=0.0, seconds=120)
    assert machine.state is SessionState.BREAK_PROMPT
    assert machine.is_ended() is False


def test_explicit_stop_ends_session():
    """stop() moves any non-ended state to ENDED immediately."""
    machine, clock = _machine()
    _feed(machine, clock, score=40.0, seconds=3)   # WARNING
    machine.stop()
    assert machine.state is SessionState.ENDED
    assert machine.is_ended() is True


def test_missing_readings_are_held_safely():
    """None scores (no reading) neither escalate nor recover the state."""
    config = SessionStateConfig(warning_threshold=50, warning_dwell_sec=3, break_dwell_sec=5)
    machine, clock = _machine(config=config)
    # Low for 2 s, then a gap of None readings, then low again.
    _feed(machine, clock, score=40.0, seconds=2)
    _feed(machine, clock, score=None, seconds=10)   # readings missing
    assert machine.state is SessionState.ACTIVE     # gap did not escalate
    # The bad run continues from where real readings resume.
    _feed(machine, clock, score=40.0, seconds=1)
    assert machine.state is SessionState.WARNING


def test_reset_returns_to_fresh_active():
    """reset() clears state and history and restarts cleanly."""
    machine, clock = _machine(planned=100)
    _feed(machine, clock, score=20.0, seconds=9)   # drive to break prompt
    assert machine.state is SessionState.BREAK_PROMPT
    machine.reset()
    assert machine.state is SessionState.ACTIVE
    assert machine.is_ended() is False
    # The new session runs on a fresh clock-zero timer.
    machine.start()
    status = machine.update(90.0)
    assert status.elapsed_sec == 0.0
    assert status.state is SessionState.ACTIVE


def test_status_reports_fields_for_dashboard():
    """SessionStatus carries the fields the dashboard needs."""
    machine, clock = _machine(planned=100)
    status = machine.update(90.0)
    assert status.state is SessionState.ACTIVE
    assert status.remaining_sec == 100.0
    assert status.smoothed_score == 90.0
    assert status.time_in_state_sec >= 0.0
    # On a change, previous_state is populated.
    changed_status = _feed(machine, clock, score=40.0, seconds=3)
    assert changed_status.state_changed is True
    assert changed_status.previous_state is SessionState.ACTIVE


def test_invalid_config_rejected():
    """Nonsensical threshold orderings raise instead of silently misbehaving."""
    for kwargs in (
        {"warning_threshold": 30, "break_threshold": 50},   # break > warning
        {"warning_threshold": 50, "recovery_threshold": 40},  # recovery < warning
        {"warning_dwell_sec": -1},
    ):
        try:
            SessionStateConfig(**kwargs)
            assert False, f"config {kwargs} should have raised"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_elapsed_zero_before_start,
    test_elapsed_and_remaining_while_running,
    test_start_twice_does_not_reset,
    test_stop_freezes_elapsed,
    test_remaining_never_negative_and_finished,
    test_format_time,
    test_restart_after_stop_begins_new_run,
    test_starts_active,
    test_good_focus_stays_active,
    test_low_focus_enters_warning_after_dwell,
    test_very_low_focus_from_active_can_reach_break_prompt,
    test_warning_escalates_to_break_on_sustained_low,
    test_recovery_steps_back_one_level_with_hysteresis,
    test_hysteresis_band_does_not_recover,
    test_no_rapid_flicker_on_noisy_scores,
    test_timer_expiry_ends_session,
    test_low_focus_alone_never_ends_session,
    test_explicit_stop_ends_session,
    test_missing_readings_are_held_safely,
    test_reset_returns_to_fresh_active,
    test_status_reports_fields_for_dashboard,
    test_invalid_config_rejected,
]


def main():
    passed, failed = 0, 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
