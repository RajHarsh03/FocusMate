"""
FocusMate eye signals: Eye Aspect Ratio (EAR) and blink detection.

This module turns the face landmarks produced by the vision module into two
simple numbers the rest of the app can use:

    - EAR (Eye Aspect Ratio): how open the eyes are. Open eyes give about
      0.25 to 0.35; closed eyes drop below about 0.2.
    - A blink count, from a small stateful detector.

It only does the math on landmark points. It never opens a webcam, runs
MediaPipe, touches the database, or draws anything. That keeps it fully
testable with fake landmarks, no camera needed.

Input format (Contract A, produced by the vision module, Qaem):
    landmarks is a list of 478 (x, y) tuples in pixel coordinates, or None
    when no face was found.

Spec reference: FocusMate-Module-Specs.md, Section 4.1.
"""

import math

# MediaPipe Face Landmarker (Tasks API, 478-point model) eye landmark indices,
# listed in the EAR point order p1, p2, p3, p4, p5, p6.
#   p1, p4 are the two horizontal corners of the eye.
#   p2, p3, p5, p6 are the upper and lower lid points.
# These come straight from the spec and must not be changed without updating it.
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Default blink detector settings (see BlinkDetector for the meaning).
DEFAULT_EAR_THRESHOLD = 0.21
DEFAULT_MIN_FRAMES = 2


def _distance(point_a, point_b):
    """Straight-line (Euclidean) distance between two (x, y) points."""
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def eye_aspect_ratio(landmarks, eye_indices):
    """Return the EAR for one eye.

    EAR = ( |p2 - p6| + |p3 - p5| ) / ( 2 * |p1 - p4| )

    The two vertical distances (upper to lower lid) are divided by twice the
    horizontal distance (corner to corner). The value stays fairly steady while
    the eye is open and drops sharply when it closes.

    Args:
        landmarks: list of (x, y) tuples (pixel coordinates).
        eye_indices: list of six landmark indices in the order p1..p6.

    Returns:
        float EAR for that eye, or None if the points are missing or the eye
        width is zero (which would otherwise divide by zero).
    """
    if landmarks is None or eye_indices is None:
        return None
    if len(eye_indices) != 6:
        return None

    try:
        p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_indices)
    except (IndexError, TypeError):
        # An index past the end of the list, or landmarks that are not indexable.
        return None

    try:
        vertical = _distance(p2, p6) + _distance(p3, p5)
        horizontal = _distance(p1, p4)
    except (TypeError, IndexError):
        # A point that is not a usable (x, y) pair.
        return None

    if horizontal == 0:
        return None
    return vertical / (2.0 * horizontal)


def compute_ear(landmarks):
    """Return the average EAR of both eyes, or None when it cannot be computed.

    This is the main EAR entry point for the rest of the app. It reads the two
    eyes using RIGHT_EYE and LEFT_EYE, then averages whatever eyes gave a valid
    value. If neither eye can be measured (no landmarks, or bad data), it
    returns None instead of guessing.

    Args:
        landmarks: list of 478 (x, y) tuples, or None if no face was found.

    Returns:
        float average EAR, or None.
    """
    if landmarks is None:
        return None

    right = eye_aspect_ratio(landmarks, RIGHT_EYE)
    left = eye_aspect_ratio(landmarks, LEFT_EYE)

    valid = [value for value in (right, left) if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


class BlinkDetector:
    """Counts blinks from a stream of EAR values, one value per frame.

    A blink is counted as: the EAR stays below `ear_threshold` for at least
    `min_frames` frames in a row, and then rises back above the threshold. The
    blink is reported on the exact frame it finishes (the frame the eye reopens),
    so holding the eyes closed for a long time counts as a single blink, not many.

    Call update(ear) once per frame. Pass None (no face) and it is treated as
    "cannot tell": any run in progress is dropped and no blink is counted.
    """

    def __init__(self, ear_threshold=DEFAULT_EAR_THRESHOLD, min_frames=DEFAULT_MIN_FRAMES):
        """Create a detector.

        Args:
            ear_threshold: EAR at or below which the eye is treated as closed.
            min_frames: how many closed frames in a row make a real blink,
                        which filters out one-frame noise.
        """
        self.ear_threshold = float(ear_threshold)
        self.min_frames = max(1, int(min_frames))
        self.total_blinks = 0
        # Number of consecutive closed frames seen so far in the current run.
        self._closed_frames = 0

    def reset(self):
        """Clear the blink count and any run in progress (call at session start)."""
        self.total_blinks = 0
        self._closed_frames = 0

    def update(self, ear):
        """Feed one frame's EAR and return True only on the frame a blink completes.

        Args:
            ear: the frame's EAR (float), or None when there is no face.

        Returns:
            True on the exact frame a blink finishes, otherwise False.
        """
        if ear is None:
            # No face this frame. Drop any run in progress so a blink is never
            # stitched together across a gap where we could not see the eyes.
            self._closed_frames = 0
            return False

        if ear <= self.ear_threshold:
            # Eye is closed this frame: extend the current run.
            self._closed_frames += 1
            return False

        # Eye is open this frame. If the run just before it was long enough,
        # that was a blink finishing right now.
        blinked = self._closed_frames >= self.min_frames
        self._closed_frames = 0
        if blinked:
            self.total_blinks += 1
        return blinked


# ---------------------------------------------------------------------------
# Small demo with fake landmarks: python src/detection/ear_blink.py
# No webcam needed. It builds simple eye shapes by hand to show EAR and blinks.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def _fake_landmarks(openness):
        """Build 478 fake points where both eyes have the given vertical opening.

        openness is the gap in pixels between the upper and lower lids. A wide
        gap means open eyes (high EAR); a tiny gap means closed eyes (low EAR).
        Every eye point is placed so the six-point EAR formula sees it.
        """
        points = [(0.0, 0.0)] * 478
        half = openness / 2.0
        for indices, corner_x in ((RIGHT_EYE, 100.0), (LEFT_EYE, 300.0)):
            p1, p2, p3, p4, p5, p6 = indices
            width = 30.0  # fixed corner-to-corner width
            points[p1] = (corner_x, 0.0)             # left corner
            points[p4] = (corner_x + width, 0.0)     # right corner
            points[p2] = (corner_x + 10.0, half)     # upper lid
            points[p3] = (corner_x + 20.0, half)     # upper lid
            points[p6] = (corner_x + 10.0, -half)    # lower lid
            points[p5] = (corner_x + 20.0, -half)    # lower lid
        return points

    open_eyes = _fake_landmarks(openness=18.0)
    closed_eyes = _fake_landmarks(openness=3.0)

    print(f"EAR (open eyes)   = {compute_ear(open_eyes):.3f}")
    print(f"EAR (closed eyes) = {compute_ear(closed_eyes):.3f}")
    print(f"EAR (no face)     = {compute_ear(None)}")

    # Simulate a frame stream: open, a 3-frame blink, open again, then a long
    # 5-frame closure (still only one blink), then a no-face gap.
    detector = BlinkDetector()
    open_ear = compute_ear(open_eyes)
    closed_ear = compute_ear(closed_eyes)
    stream = [open_ear, closed_ear, closed_ear, closed_ear, open_ear,
              open_ear, closed_ear, closed_ear, closed_ear, closed_ear,
              closed_ear, open_ear, None, open_ear]
    for ear_value in stream:
        if detector.update(ear_value):
            print("  blink detected")
    print(f"Total blinks = {detector.total_blinks} (expected 2)")
