"""
Tests for the FocusMate detection modules (Muskan).

Covers:
    - ear_blink.py     EAR calculation and blink detection
    - focus_preview.py the showcase preview meter
    - focus_score.py   the preview-based focus score with EMA smoothing

All tests use fake landmarks and simulated values. No webcam is needed.

Run from the project root:
    python src/detection/test_detection.py

Optional, the same tests also run with pytest:
    pytest src/detection/test_detection.py
"""

import math
import sys
import traceback

try:
    from src.detection import ear_blink, focus_preview, focus_score
except ImportError:
    import ear_blink
    import focus_preview
    import focus_score


# ---------------------------------------------------------------------------
# Fake landmark helpers (no webcam)
# ---------------------------------------------------------------------------

def _eye_points(landmarks, target_ear, corner_x, width=30.0):
    """Place the six points of one eye so its EAR equals target_ear.

    The two corners (p1, p4) sit `width` apart on the centre line. The upper lid
    points (p2, p3) sit at +half and the lower lid points (p5, p6) at -half, so
    each vertical distance |p2 - p6| and |p3 - p5| is 2 * half. The EAR formula
    then gives:
        EAR = (2*half + 2*half) / (2*width) = 2*half / width
    so to hit target_ear we set half = target_ear * width / 2.
    """
    p1, p2, p3, p4, p5, p6 = (ear_blink.RIGHT_EYE if corner_x < 200 else ear_blink.LEFT_EYE)
    half = target_ear * width / 2.0
    landmarks[p1] = (corner_x, 0.0)
    landmarks[p4] = (corner_x + width, 0.0)
    landmarks[p2] = (corner_x + 10.0, half)
    landmarks[p3] = (corner_x + 20.0, half)
    landmarks[p6] = (corner_x + 10.0, -half)
    landmarks[p5] = (corner_x + 20.0, -half)


def make_landmarks(target_ear):
    """Return 478 fake (x, y) points whose EAR equals target_ear for both eyes."""
    points = [(0.0, 0.0)] * 478
    _eye_points(points, target_ear, corner_x=100.0)   # right eye
    _eye_points(points, target_ear, corner_x=300.0)   # left eye
    return points


def _close(a, b, tol=1e-6):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# ---------------------------------------------------------------------------
# EAR tests
# ---------------------------------------------------------------------------

def test_make_landmarks_produces_target_ear():
    """The fake-landmark helper hits the EAR it is asked for, so other tests are sound."""
    for target in (0.10, 0.20, 0.25, 0.30, 0.35):
        ear = ear_blink.compute_ear(make_landmarks(target))
        assert _close(ear, target), f"wanted {target}, got {ear}"


def test_ear_open_and_closed_ranges():
    """Open eyes give roughly 0.25 to 0.35; closed eyes drop below about 0.2."""
    open_ear = ear_blink.compute_ear(make_landmarks(0.30))
    closed_ear = ear_blink.compute_ear(make_landmarks(0.12))
    assert 0.25 <= open_ear <= 0.35, open_ear
    assert closed_ear < 0.2, closed_ear


def test_single_eye_ear_matches_formula():
    """eye_aspect_ratio reproduces the documented EAR formula on known points."""
    landmarks = make_landmarks(0.27)
    one_eye = ear_blink.eye_aspect_ratio(landmarks, ear_blink.RIGHT_EYE)
    assert _close(one_eye, 0.27)


def test_compute_ear_averages_both_eyes():
    """compute_ear is the average of the two eyes when they differ."""
    landmarks = [(0.0, 0.0)] * 478
    _eye_points(landmarks, 0.30, corner_x=100.0)  # right EAR 0.30
    _eye_points(landmarks, 0.20, corner_x=300.0)  # left EAR 0.20
    assert _close(ear_blink.compute_ear(landmarks), 0.25)


def test_ear_handles_missing_and_invalid_input():
    """None, wrong index counts and zero-width eyes return None, never crash."""
    assert ear_blink.compute_ear(None) is None
    assert ear_blink.eye_aspect_ratio(None, ear_blink.LEFT_EYE) is None
    assert ear_blink.eye_aspect_ratio(make_landmarks(0.3), [1, 2, 3]) is None  # not 6

    # Index past the end of a short list.
    assert ear_blink.eye_aspect_ratio([(0.0, 0.0)] * 10, ear_blink.LEFT_EYE) is None

    # Zero-width eye (both corners at the same point) would divide by zero.
    zero_width = [(0.0, 0.0)] * 478
    for i in ear_blink.RIGHT_EYE:
        zero_width[i] = (5.0, 0.0)
    assert ear_blink.eye_aspect_ratio(zero_width, ear_blink.RIGHT_EYE) is None


def test_compute_ear_one_bad_eye_still_returns_other():
    """If one eye is unusable, compute_ear falls back to the eye that works."""
    landmarks = make_landmarks(0.30)            # both eyes valid at 0.30
    for i in ear_blink.LEFT_EYE:                # wreck the left eye (zero width)
        landmarks[i] = (7.0, 7.0)
    result = ear_blink.compute_ear(landmarks)
    assert result is not None and _close(result, 0.30)


# ---------------------------------------------------------------------------
# Blink detection tests
# ---------------------------------------------------------------------------

def test_single_blink_counts_once():
    """A short dip below threshold then back up is exactly one blink."""
    d = ear_blink.BlinkDetector(ear_threshold=0.21, min_frames=2)
    stream = [0.30, 0.15, 0.15, 0.30]   # open, closed x2, open
    results = [d.update(e) for e in stream]
    assert results == [False, False, False, True]
    assert d.total_blinks == 1


def test_long_closure_is_one_blink():
    """Holding the eyes closed for many frames counts as a single blink."""
    d = ear_blink.BlinkDetector(ear_threshold=0.21, min_frames=2)
    stream = [0.30] + [0.10] * 8 + [0.30]
    blinks = sum(1 for e in stream if d.update(e))
    assert blinks == 1
    assert d.total_blinks == 1


def test_too_short_dip_is_not_a_blink():
    """A one-frame dip (below min_frames) does not count."""
    d = ear_blink.BlinkDetector(ear_threshold=0.21, min_frames=2)
    stream = [0.30, 0.15, 0.30]    # only 1 closed frame
    assert not any(d.update(e) for e in stream)
    assert d.total_blinks == 0


def test_threshold_boundary_counts_as_closed():
    """EAR exactly at the threshold is treated as closed (<=)."""
    d = ear_blink.BlinkDetector(ear_threshold=0.21, min_frames=2)
    stream = [0.30, 0.21, 0.21, 0.30]
    assert sum(1 for e in stream if d.update(e)) == 1


def test_blink_none_breaks_run_and_counts_nothing():
    """A None (no face) in the middle of a closure cancels the blink."""
    d = ear_blink.BlinkDetector(ear_threshold=0.21, min_frames=2)
    stream = [0.30, 0.15, None, 0.15, 0.30]   # closure interrupted by no-face
    assert not any(d.update(e) for e in stream)
    assert d.total_blinks == 0


def test_blink_reset_clears_everything():
    """reset() zeroes the count and drops any run in progress."""
    d = ear_blink.BlinkDetector()
    for e in [0.30, 0.10, 0.10, 0.30]:
        d.update(e)
    assert d.total_blinks == 1
    d.update(0.10)   # start a new closure
    d.reset()
    assert d.total_blinks == 0
    # The interrupted closure must not resurface as a blink after reset.
    assert d.update(0.30) is False
    assert d.total_blinks == 0


def test_multiple_blinks_counted():
    """Several separated blinks each count once."""
    d = ear_blink.BlinkDetector(ear_threshold=0.21, min_frames=2)
    stream = [0.30, 0.10, 0.10, 0.30, 0.30, 0.10, 0.10, 0.10, 0.30]
    total = sum(1 for e in stream if d.update(e))
    assert total == 2 and d.total_blinks == 2


# ---------------------------------------------------------------------------
# Focus preview tests (Spec 4.2 behaviour)
# ---------------------------------------------------------------------------

def test_preview_no_face_is_zero():
    assert focus_preview.compute_focus_preview(False, None, "unknown") == 0.0
    assert focus_preview.compute_focus_preview(False, 0.30, "center") == 0.0


def test_preview_ideal_is_full():
    assert focus_preview.compute_focus_preview(True, 0.30, "center") == 100.0


def test_preview_penalties():
    assert focus_preview.compute_focus_preview(True, 0.30, "left") == 60.0     # away
    assert focus_preview.compute_focus_preview(True, 0.18, "center") == 60.0   # closed
    assert focus_preview.compute_focus_preview(True, 0.22, "center") == 85.0   # droop
    assert focus_preview.compute_focus_preview(True, 0.18, "left") == 20.0     # both


def test_preview_unknown_direction_penalised_per_contract():
    """Per the documented preview contract, any non-center label subtracts 40.

    The reference copy in src/storage/seed_demo_data.py uses
    `if head_direction != "center": value -= 40`, so a present face with an
    "unknown" label is treated the same way. (In real data "unknown" only
    occurs with no face, which already returns 0, so this is an edge case kept
    only to match the contract exactly.)
    """
    assert focus_preview.compute_focus_preview(True, 0.30, "unknown") == 60.0


def test_preview_missing_ear_applies_no_eye_penalty():
    """A face with no EAR reading keeps the eye score (no silent distraction)."""
    assert focus_preview.compute_focus_preview(True, None, "center") == 100.0


def test_preview_always_clamped_0_to_100():
    for face in (True, False):
        for ear in (None, 0.0, 0.18, 0.22, 0.30, 1.0):
            for direction in ("center", "left", "right", "up", "down", "unknown"):
                value = focus_preview.compute_focus_preview(face, ear, direction)
                assert 0.0 <= value <= 100.0


# ---------------------------------------------------------------------------
# Focus score tests (raw rules + EMA smoothing + components)
# ---------------------------------------------------------------------------

def _raw(scorer, *args, **kwargs):
    """Convenience: the raw score from compute_raw, dropping the detail dict."""
    return scorer.compute_raw(*args, **kwargs)[0]


def test_full_score_is_distinct_from_preview():
    """The full score must NOT just reproduce the preview's fixed subtractions.

    For a clearly drooping EAR (0.23) with a centred head, the preview subtracts
    a flat 15 (giving 85). The full score grades the eye signal, so with the
    default weights it should land on a different value, proving it is a real
    separate score and not the preview renamed.
    """
    preview_value = focus_preview.compute_focus_preview(True, 0.23, "center")
    assert preview_value == 85.0
    scorer = focus_score.FocusScorer()
    full_value = _raw(scorer, True, 0.23, "center")
    assert full_value != preview_value


def test_eye_component_is_graded():
    """The eye component slides between 0 and 100 instead of stepping."""
    scorer = focus_score.FocusScorer()           # ear_closed 0.18, ear_open 0.28
    assert scorer.eye_component(True, 0.28)[0] == 100.0   # fully open
    assert scorer.eye_component(True, 0.18)[0] == 0.0     # fully closed
    mid, missing = scorer.eye_component(True, 0.23)       # halfway
    assert _close(mid, 50.0) and missing is False
    # A value below closed clamps to 0; above open clamps to 100.
    assert scorer.eye_component(True, 0.10)[0] == 0.0
    assert scorer.eye_component(True, 0.40)[0] == 100.0


def test_eye_component_missing_ear_is_neutral_not_zero():
    """A missing EAR gives the neutral component, never a distraction-implying 0."""
    config = focus_score.FocusScoreConfig(ear_neutral_component=70.0)
    scorer = focus_score.FocusScorer(config=config)
    value, missing = scorer.eye_component(True, None)
    assert value == 70.0 and missing is True


def test_head_component_label_fallback_is_graded():
    """Without angles, each head label maps to a graded component value."""
    scorer = focus_score.FocusScorer()
    assert scorer.head_component("center")[0] == 100.0
    assert scorer.head_component("up")[0] == 60.0
    assert scorer.head_component("left")[0] == 20.0
    # unknown is neutral, not a full penalty, and does not come from angles.
    value, from_angles = scorer.head_component("unknown")
    assert value == 70.0 and from_angles is False


def test_head_component_uses_angles_when_present():
    """When yaw/pitch are given, the head component falls off with the angle."""
    scorer = focus_score.FocusScorer()           # head_angle_max 35
    centred, from_angles = scorer.head_component("center", yaw=0.0, pitch=0.0)
    assert centred == 100.0 and from_angles is True
    half = scorer.head_component("center", yaw=17.5, pitch=0.0)[0]
    assert _close(half, 50.0)
    # Beyond the max angle clamps to 0, using the larger of |yaw|, |pitch|.
    assert scorer.head_component("center", yaw=10.0, pitch=50.0)[0] == 0.0


def test_full_score_blends_eye_and_head_with_weights():
    """The raw score is the weighted blend of the two graded components."""
    config = focus_score.FocusScoreConfig(ear_weight=0.6, head_weight=0.4)
    scorer = focus_score.FocusScorer(config=config)
    # Eyes fully open (eye=100), head looking left (head=20).
    # blend = (0.6*100 + 0.4*20) / 1.0 = 68
    assert _close(_raw(scorer, True, 0.28, "left"), 68.0)


def test_full_score_no_face_is_zero():
    scorer = focus_score.FocusScorer()
    assert _raw(scorer, False, None, "unknown") == 0.0


def test_first_update_has_no_smoothing_lag():
    """The first frame starts the smoothed value at the raw value."""
    scorer = focus_score.FocusScorer()
    smoothed = scorer.update(True, 0.28, "center")
    assert smoothed == 100.0
    assert scorer.raw_score == 100.0
    assert scorer.smoothed_score == 100.0


def test_ema_smoothing_moves_gradually():
    """A sudden raw drop moves the smoothed score only partway (by alpha)."""
    config = focus_score.FocusScoreConfig(alpha=0.3, ear_weight=0.6, head_weight=0.4)
    scorer = focus_score.FocusScorer(config=config)
    scorer.update(True, 0.28, "center")          # raw 100, smoothed 100
    smoothed = scorer.update(True, 0.28, "left")  # raw 68 (see blend test)
    # smoothed = 0.3*68 + 0.7*100 = 90.4
    assert _close(smoothed, 90.4)
    assert _close(scorer.raw_score, 68.0)


def test_smoothing_is_less_jumpy_than_raw():
    """Over a noisy stream, the smoothed series varies less than the raw series."""
    scorer = focus_score.FocusScorer()
    raw_values, smooth_values = [], []
    stream = [(True, 0.28, "center"), (True, 0.28, "left"), (True, 0.28, "center"),
              (True, 0.18, "center"), (True, 0.28, "center"), (True, 0.28, "left")]
    for face, ear, direction in stream:
        smooth_values.append(scorer.update(face, ear, direction))
        raw_values.append(scorer.raw_score)

    def spread(series):
        return sum(abs(series[i] - series[i - 1]) for i in range(1, len(series)))

    assert spread(smooth_values) < spread(raw_values)


def test_alpha_one_means_no_smoothing():
    """alpha = 1 makes the smoothed value equal the raw value every frame."""
    scorer = focus_score.FocusScorer(config=focus_score.FocusScoreConfig(alpha=1.0))
    scorer.update(True, 0.28, "center")
    assert _close(scorer.update(True, 0.28, "left"), 68.0)


def test_invalid_config_rejected():
    for kwargs in ({"alpha": 0.0}, {"alpha": -0.1}, {"alpha": 1.5},
                   {"ear_open": 0.1, "ear_closed": 0.2},   # open <= closed
                   {"ear_weight": -1.0}, {"head_angle_max": 0.0}):
        try:
            focus_score.FocusScoreConfig(**kwargs)
            assert False, f"config {kwargs} should have raised"
        except ValueError:
            pass


def test_configurable_weights_and_thresholds():
    """Custom weights and EAR thresholds change the score as expected."""
    # Head-only weighting: eyes are ignored, so the score equals the head component.
    config = focus_score.FocusScoreConfig(ear_weight=0.0, head_weight=1.0)
    scorer = focus_score.FocusScorer(config=config)
    assert _close(_raw(scorer, True, 0.18, "center"), 100.0)   # closed eyes, but head centred
    assert _close(_raw(scorer, True, 0.28, "left"), 20.0)      # open eyes, but looking left


def test_components_snapshot_exposed_for_dashboard():
    """components reports the graded pieces and the data-source flags."""
    scorer = focus_score.FocusScorer()
    scorer.update(True, 0.18, "left")
    c = scorer.components
    assert c.face_detected is True
    assert c.head_direction == "left"
    assert c.eye_component == 0.0        # 0.18 == ear_closed -> 0
    assert c.head_component == 20.0      # left label
    assert c.ear_missing is False
    assert c.head_from_angles is False

    # Missing EAR flags the neutral fallback.
    scorer.update(True, None, "center")
    assert scorer.components.ear_missing is True

    # Supplying angles flips head_from_angles.
    scorer.update(True, 0.28, "center", yaw=0.0, pitch=0.0)
    assert scorer.components.head_from_angles is True


def test_reset_clears_smoothing_history():
    """After reset, the next frame starts fresh (no lag from old history)."""
    scorer = focus_score.FocusScorer()
    scorer.update(True, 0.28, "center")
    scorer.update(True, 0.18, "left")
    scorer.reset()
    assert scorer.raw_score is None and scorer.smoothed_score is None and scorer.components is None
    assert _close(scorer.update(True, 0.28, "left"), 68.0)   # starts at raw again


def test_none_head_direction_treated_as_unknown():
    """A None head direction must not crash and is treated as unknown (neutral)."""
    scorer = focus_score.FocusScorer()
    scorer.update(True, 0.28, None)
    assert scorer.components.head_direction == "unknown"
    assert scorer.components.head_component == 70.0   # neutral, not a full penalty


def test_scores_always_clamped():
    """Scores stay within 0 to 100 for all inputs."""
    scorer = focus_score.FocusScorer()
    for face in (True, False):
        for ear in (None, 0.0, 0.18, 0.23, 0.28, 1.0):
            for direction in ("center", "left", "right", "up", "down", "unknown"):
                value = _raw(scorer, face, ear, direction)
                assert 0.0 <= value <= 100.0


# ---------------------------------------------------------------------------
# Runner (mirrors the storage test style)
# ---------------------------------------------------------------------------

TESTS = [
    test_make_landmarks_produces_target_ear,
    test_ear_open_and_closed_ranges,
    test_single_eye_ear_matches_formula,
    test_compute_ear_averages_both_eyes,
    test_ear_handles_missing_and_invalid_input,
    test_compute_ear_one_bad_eye_still_returns_other,
    test_single_blink_counts_once,
    test_long_closure_is_one_blink,
    test_too_short_dip_is_not_a_blink,
    test_threshold_boundary_counts_as_closed,
    test_blink_none_breaks_run_and_counts_nothing,
    test_blink_reset_clears_everything,
    test_multiple_blinks_counted,
    test_preview_no_face_is_zero,
    test_preview_ideal_is_full,
    test_preview_penalties,
    test_preview_unknown_direction_penalised_per_contract,
    test_preview_missing_ear_applies_no_eye_penalty,
    test_preview_always_clamped_0_to_100,
    test_full_score_is_distinct_from_preview,
    test_eye_component_is_graded,
    test_eye_component_missing_ear_is_neutral_not_zero,
    test_head_component_label_fallback_is_graded,
    test_head_component_uses_angles_when_present,
    test_full_score_blends_eye_and_head_with_weights,
    test_full_score_no_face_is_zero,
    test_first_update_has_no_smoothing_lag,
    test_ema_smoothing_moves_gradually,
    test_smoothing_is_less_jumpy_than_raw,
    test_alpha_one_means_no_smoothing,
    test_invalid_config_rejected,
    test_configurable_weights_and_thresholds,
    test_components_snapshot_exposed_for_dashboard,
    test_reset_clears_smoothing_history,
    test_none_head_direction_treated_as_unknown,
    test_scores_always_clamped,
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
