"""
FocusMate session timer.

A small countdown timer for a study session. It reports how long the session
has run and how long is left, and whether the planned duration is finished.

It uses time.monotonic() by default, so changing the computer's clock during a
session does not affect it. A clock function can be injected instead, which lets
tests run instantly and deterministically without real sleeps.

Spec reference: FocusMate-Module-Specs.md, Section 4.3.
"""

import time


class SessionTimer:
    """Counts elapsed and remaining time for one study session.

    Typical use:
        timer = SessionTimer(planned_duration_sec=1500)
        timer.start()
        ...                      # elapsed_sec() grows while running
        timer.stop()             # elapsed_sec() then stays fixed

    Rules (from the spec):
        - elapsed_sec() before start() returns 0.0.
        - Calling start() twice does not reset a running timer.
        - After stop(), elapsed_sec() stays fixed at the final value.
    """

    def __init__(self, planned_duration_sec, clock=time.monotonic):
        """Create a timer.

        Args:
            planned_duration_sec: how long the session is planned to run, in
                seconds. Negative values are treated as 0.
            clock: a function returning a steadily increasing number of seconds.
                Defaults to time.monotonic. Tests can pass a controllable clock.
        """
        self.planned_duration_sec = max(0.0, float(planned_duration_sec))
        self._clock = clock
        self._start_time = None   # clock value when start() was called
        self._stop_time = None    # clock value when stop() was called

    def start(self):
        """Start the timer. Calling it again while running has no effect."""
        if self._start_time is not None and self._stop_time is None:
            # Already running: do not restart.
            return
        if self._start_time is not None and self._stop_time is not None:
            # Was stopped: a fresh start begins a new run from now.
            self._stop_time = None
        self._start_time = self._clock()

    def stop(self):
        """Stop the timer and freeze the elapsed time. No effect if not running."""
        if self._start_time is None or self._stop_time is not None:
            return
        self._stop_time = self._clock()

    def elapsed_sec(self):
        """Return seconds since start. 0.0 before start; frozen after stop."""
        if self._start_time is None:
            return 0.0
        end = self._stop_time if self._stop_time is not None else self._clock()
        return max(0.0, end - self._start_time)

    def remaining_sec(self):
        """Return seconds left until the planned duration, never below 0."""
        return max(0.0, self.planned_duration_sec - self.elapsed_sec())

    def is_running(self):
        """True only while started and not yet stopped."""
        return self._start_time is not None and self._stop_time is None

    def is_finished(self):
        """True once elapsed time has reached the planned duration."""
        return self.elapsed_sec() >= self.planned_duration_sec

    def format_time(self, seconds):
        """Format a number of seconds as MM:SS (minutes can exceed 99)."""
        seconds = max(0, int(seconds))
        minutes, secs = divmod(seconds, 60)
        return f"{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Small demo: python src/timer/session_timer.py
# Uses a fake clock so it prints instantly, with no real waiting.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    class FakeClock:
        """A clock the demo can advance by hand."""

        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

    clock = FakeClock()
    timer = SessionTimer(planned_duration_sec=10, clock=clock)

    print(f"before start: elapsed={timer.elapsed_sec():.0f} "
          f"running={timer.is_running()}")
    timer.start()
    clock.advance(4)
    print(f"after 4s: elapsed={timer.format_time(timer.elapsed_sec())} "
          f"remaining={timer.format_time(timer.remaining_sec())} "
          f"finished={timer.is_finished()}")
    clock.advance(6)
    print(f"after 10s: elapsed={timer.format_time(timer.elapsed_sec())} "
          f"remaining={timer.format_time(timer.remaining_sec())} "
          f"finished={timer.is_finished()}")
    timer.stop()
    clock.advance(100)  # time keeps moving, but elapsed is frozen
    print(f"after stop: elapsed={timer.format_time(timer.elapsed_sec())} (frozen)")
