import cv2
import numpy as np
from typing import Tuple

try:
    import mediapipe as mp
except Exception as e:
    mp = None


class HolisticWrapper:
    """Encapsulates MediaPipe Holistic detection lifecycle."""

    def __init__(self):
        if mp is None:
            raise RuntimeError("mediapipe is not installed or failed to import")
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        return results

    def draw_landmarks(self, image_bgr: np.ndarray, results) -> np.ndarray:
        image_bgr.flags.writeable = True
        # Draw pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )
        # Draw face
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image_bgr,
                results.face_landmarks,
                self.mp_holistic.FACE_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            )
        # Draw hands
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image_bgr,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=2),
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image_bgr,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=2),
            )
        return image_bgr


def extract_keypoints(results) -> np.ndarray:
    """Extracts 1662-dim feature vector from holistic results.

    Order: face(468x3), pose(33x4), left hand(21x3), right hand(21x3).
    Missing parts are zeros.
    """
    # Face: 468 * (x,y,z)
    face = np.zeros(468 * 3)
    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        face = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks]).flatten()

    # Pose: 33 * (x,y,z,visibility)
    pose = np.zeros(33 * 4)
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks]).flatten()

    # Left hand: 21 * (x,y,z)
    lh = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        lh_landmarks = results.left_hand_landmarks.landmark
        lh = np.array([[lm.x, lm.y, lm.z] for lm in lh_landmarks]).flatten()

    # Right hand: 21 * (x,y,z)
    rh = np.zeros(21 * 3)
    if results.right_hand_landmarks:
        rh_landmarks = results.right_hand_landmarks.landmark
        rh = np.array([[lm.x, lm.y, lm.z] for lm in rh_landmarks]).flatten()

    return np.concatenate([face, pose, lh, rh])


def overlay_prediction(
    image_bgr: np.ndarray,
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 230, 170),
) -> np.ndarray:
    """Draws prediction and confidence bar on the frame."""
    h, w = image_bgr.shape[:2]
    bar_w = int(w * 0.4)
    bar_h = 15
    x0, y0 = 20, 20
    conf_w = int(bar_w * float(confidence))
    cv2.rectangle(image_bgr, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    cv2.rectangle(image_bgr, (x0, y0), (x0 + conf_w, y0 + bar_h), color, -1)
    cv2.putText(
        image_bgr,
        f"{label} ({confidence:.2f})",
        (x0, y0 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return image_bgr

