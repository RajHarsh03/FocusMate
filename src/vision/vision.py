"""
FocusMate - Vision Module
Qaem Hussain Rizvi (2360045) | Branch: feat/vision

Job: open the webcam, run MediaPipe Face Landmarker on each frame,
estimate head direction (yaw and pitch), and return one clean
Contract A dictionary to the rest of the team.

Contract A output of process_frame():
    {
        "face_detected": True,
        "landmarks": [(x, y), ...],   # 478 pixel-coordinate tuples
        "head_direction": "center",   # center | left | right | up | down | unknown
        "yaw": 4.2,                   # float degrees, or None
        "pitch": -3.1                 # float degrees, or None
    }

When no face is found:
    {
        "face_detected": False,
        "landmarks": None,
        "head_direction": "unknown",
        "yaw": None,
        "pitch": None
    }

Head direction convention (user perspective, mirrored frame):
    yaw  > 0  -> user turned LEFT,  yaw  < 0  -> user turned RIGHT
    pitch > 0 -> user tilted DOWN,  pitch < 0 -> user tilted UP

Head pose is estimated from MediaPipe's facial transformation matrix.
Axes confirmed by live testing:
    axis_x  (arctan2 R[2,1], R[2,2]) -> up / down   -> stored as pitch
    axis_y  (arcsin  -R[2,0])        -> left / right -> stored as yaw
    axis_z  (arctan2  R[1,0], R[0,0])-> left / right (same plane, not used)
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


# Head direction thresholds (degrees).
YAW_THRESHOLD   = 15.0   # beyond +/-15 deg -> left or right
PITCH_THRESHOLD = 15.0   # beyond +/-15 deg -> up or down

# The no-face dictionary, returned whenever no face is visible.
NO_FACE = {
    "face_detected": False,
    "landmarks": None,
    "head_direction": "unknown",
    "yaw": None,
    "pitch": None,
}


def create_landmarker(model_path="models/face_landmarker.task"):
    """Load the MediaPipe Face Landmarker once at startup.

    Args:
        model_path: Path to the face_landmarker.task file.

    Returns:
        A FaceLandmarker object ready to process frames.
    """
    options = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return FaceLandmarker.create_from_options(options)


def open_camera(index=0):
    """Open the webcam and set it to 640x480.

    Args:
        index: Camera index (0 is usually the built-in webcam).

    Returns:
        A cv2.VideoCapture object, or None if the camera could not open.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def close_camera(cap):
    """Release the webcam so other apps can use it.

    Args:
        cap: The cv2.VideoCapture object returned by open_camera().
    """
    if cap is not None:
        cap.release()


def process_frame(frame_bgr, landmarker):
    """Run the face landmarker on one frame and return Contract A.

    This is the single function the rest of the team calls. It never crashes
    on bad input - if anything goes wrong it returns the no-face dictionary.

    Args:
        frame_bgr: One BGR frame from OpenCV (numpy array).
        landmarker: The FaceLandmarker object from create_landmarker().

    Returns:
        A Contract A dictionary (see module docstring for the exact keys).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return dict(NO_FACE)

    frame_height, frame_width = frame_bgr.shape[:2]

    # MediaPipe needs an RGB image wrapped in its own Image type.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    try:
        result = landmarker.detect(mp_image)
    except Exception:
        return dict(NO_FACE)

    # No face found in the frame.
    if not result.face_landmarks:
        return dict(NO_FACE)

    # Convert the 478 normalized landmarks to pixel (x, y) tuples.
    raw_landmarks = result.face_landmarks[0]
    pixel_landmarks = [
        (int(lm.x * frame_width), int(lm.y * frame_height))
        for lm in raw_landmarks
    ]

    # Estimate head direction from the facial transformation matrix.
    yaw, pitch = _get_yaw_pitch(result)
    direction = _angles_to_direction(yaw, pitch)

    return {
        "face_detected": True,
        "landmarks": pixel_landmarks,
        "head_direction": direction,
        "yaw": yaw,
        "pitch": pitch,
    }


def _get_yaw_pitch(landmarker_result):
    """Extract yaw and pitch from MediaPipe's facial transformation matrix.

    The matrix is a 4x4 rigid body transform. We take its 3x3 rotation block
    and decompose it into three Euler angles using the standard ZYX convention:
        axis_x = arctan2(R[2,1], R[2,2])  -> up/down   (stored as pitch)
        axis_y = arcsin(-R[2,0])           -> left/right (stored as yaw)

    Both axes confirmed stable by live testing on this setup.

    Args:
        landmarker_result: The object returned by landmarker.detect().

    Returns:
        (yaw, pitch) as floats in degrees, or (None, None) on failure.
    """
    try:
        matrix = landmarker_result.facial_transformation_matrixes[0].data
        r = np.array(matrix).reshape(4, 4)[:3, :3]

        # axis_y is left/right -> called yaw in Contract A.
        yaw_rad   = float(np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)))
        # axis_x is up/down -> called pitch in Contract A.
        pitch_rad = float(np.arctan2(r[2, 1], r[2, 2]))

        return float(np.degrees(yaw_rad)), float(np.degrees(pitch_rad))
    except Exception:
        return None, None


def _angles_to_direction(yaw, pitch):
    """Convert yaw and pitch to a head direction label.

    Convention (user perspective, mirrored frame):
        yaw  > +YAW_THRESHOLD    -> "left"
        yaw  < -YAW_THRESHOLD    -> "right"
        pitch > +PITCH_THRESHOLD -> "down"
        pitch < -PITCH_THRESHOLD -> "up"

    Args:
        yaw:   Float degrees, or None.
        pitch: Float degrees, or None.

    Returns:
        One of: "center", "left", "right", "up", "down", "unknown"
    """
    if yaw is None or pitch is None:
        return "unknown"

    if pitch > PITCH_THRESHOLD:
        return "down"
    if pitch < -PITCH_THRESHOLD:
        return "up"
    if yaw > YAW_THRESHOLD:
        return "left"
    if yaw < -YAW_THRESHOLD:
        return "right"
    return "center"



# ---------------------------------------------------------------------------
# Demo block: python src/vision/vision.py
# Opens the webcam, prints Contract A once per second, quits on 'q'.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("Loading face landmarker ...")
    try:
        landmarker = create_landmarker()
    except Exception as error:
        print(f"Could not load model: {error}")
        raise SystemExit(1)

    cap = open_camera()
    if cap is None:
        print("Could not open webcam. Check that it is connected and not used by another app.")
        raise SystemExit(1)

    print("Webcam open. Press 'q' to quit.")
    print("-" * 60)

    last_print_time = 0.0
    frame_count = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed. Stopping.")
            break

        output = process_frame(frame, landmarker)
        frame_count += 1

        # Print the result once per second so the terminal is readable.
        now = time.monotonic()
        elapsed = now - last_print_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            landmark_count = len(output["landmarks"]) if output["landmarks"] else 0
            print(
                f"face_detected={output['face_detected']!s:<5}  "
                f"head_direction={output['head_direction']:<8}  "
                f"yaw={str(round(output['yaw'], 1)) if output['yaw'] is not None else 'None':<7}  "
                f"pitch={str(round(output['pitch'], 1)) if output['pitch'] is not None else 'None':<7}  "
                f"landmarks={landmark_count}  "
                f"fps={fps:.1f}"
            )
            frame_count = 0
            last_print_time = now

        # Show a live preview window with direction and FPS overlaid.
        display = cv2.flip(frame, 1)
        label = output["head_direction"].upper()
        color = (0, 200, 0) if output["face_detected"] else (0, 0, 200)
        cv2.putText(display, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(display, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        cv2.imshow("FocusMate - Vision Demo (press q to quit)", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    close_camera(cap)
    cv2.destroyAllWindows()
    print("Camera closed.")
