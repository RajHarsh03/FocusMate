"""
FocusMate preview focus meter.

This is the simple, easy-to-explain focus meter used in the showcase demo.
It is NOT the final Focus Score. It takes three plain inputs (whether a face
was found, the EAR, and the head direction label) and returns a number from
0 to 100.

Starting rules (Spec Section 4.2):
    - No face detected               -> 0
    - Face found, start from         -> 100
    - Head direction is not "center" -> subtract 40
    - EAR below 0.21 (eyes closed)   -> subtract 40
    - EAR 0.21 to 0.24 (drooping)    -> subtract 15
Then clamp the result to the 0 to 100 range.

Spec reference: FocusMate-Module-Specs.md, Section 4.2.
"""

# Defaults kept in one place so the demo numbers are easy to find and tweak.
FACE_BASE_SCORE = 100.0
AWAY_PENALTY = 40.0        # head direction is not "center"
EYES_CLOSED_PENALTY = 40.0  # EAR below EYES_CLOSED_EAR
EYES_DROOP_PENALTY = 15.0   # EAR between EYES_CLOSED_EAR and EYES_DROOP_EAR
EYES_CLOSED_EAR = 0.21
EYES_DROOP_EAR = 0.24


def compute_focus_preview(face_detected, ear, head_direction):
    """Return a preview focus value from 0 to 100.

    Args:
        face_detected: True if the vision module found a face this frame.
        ear: the frame's EAR (float), or None if there is no face or it could
             not be measured.
        head_direction: one of "center", "left", "right", "up", "down",
                        "unknown".

    Returns:
        float between 0 and 100. No face always gives 0, matching the demo
        seeder and the storage summary so previews average the same way.
    """
    if not face_detected:
        return 0.0

    value = FACE_BASE_SCORE

    # Documented rule (Spec 4.2 and the reference copy in
    # src/storage/seed_demo_data.py): any head direction that is not "center"
    # costs points. This keeps the preview averaging identical to the seeded
    # demo data. In practice a present face is labelled center/left/right/up/
    # down; "unknown" only arrives with no face, which already returned 0 above.
    if head_direction != "center":
        value -= AWAY_PENALTY

    # Eye openness costs points only when we actually have a reading. A missing
    # EAR (None) applies no penalty, so a lost measurement is not scored as
    # distraction.
    if ear is not None:
        if ear < EYES_CLOSED_EAR:
            value -= EYES_CLOSED_PENALTY
        elif ear < EYES_DROOP_EAR:
            value -= EYES_DROOP_PENALTY

    return max(0.0, min(100.0, value))


# ---------------------------------------------------------------------------
# Small demo: python src/detection/focus_preview.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        (False, None, "unknown"),   # no face -> 0
        (True, 0.30, "center"),     # ideal -> 100
        (True, 0.30, "left"),       # looking away -> 60
        (True, 0.18, "center"),     # eyes closed -> 60
        (True, 0.22, "center"),     # drooping -> 85
        (True, 0.18, "left"),       # away + closed -> 20
        (True, None, "center"),     # face but no EAR -> 100 (no eye penalty)
    ]
    for face, ear_value, direction in cases:
        score = compute_focus_preview(face, ear_value, direction)
        print(f"face={face!s:<5} ear={ear_value!s:<5} dir={direction:<7} -> {score:5.1f}")
