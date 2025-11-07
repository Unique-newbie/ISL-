from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List

try:  # pragma: no cover - optional dependency path
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - fallback stub path
    cv2 = None
    mp = None
    np = None

logger = logging.getLogger(__name__)


@dataclass
class PoseEstimate:
    landmarks: List[float]
    confidence: float
    nmm_flags: Dict[str, float] = field(default_factory=dict)
    occluded: bool = False
    hint: str | None = None
    synthetic: bool = False


class PoseEstimator:
    """Wrapper around MediaPipe Holistic with a fast fallback stub."""

    def __init__(self) -> None:
        self._use_mediapipe = bool(mp and cv2 and np)
        self._holistic = None
        self._features_len = 1662
        if self._use_mediapipe:
            self._holistic = mp.solutions.holistic.Holistic(  # pragma: no cover - heavy path
                model_complexity=1,
                enable_segmentation=False,
                refine_face_landmarks=True,
                smooth_landmarks=True,
            )
            logger.info("MediaPipe Holistic enabled for pose estimation")
        else:
            logger.info("MediaPipe unavailable, running pose in stub mode")

    def estimate(self, image_bytes: bytes, seed_hint: int) -> PoseEstimate:
        if self._use_mediapipe:
            try:
                np_buff = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_buff, cv2.IMREAD_COLOR)
                if frame is not None:
                    results = self._holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    landmarks = self._collect_landmarks(results)
                    occluded = not bool(results.left_hand_landmarks and results.right_hand_landmarks)
                    hint = None
                    if occluded:
                        hint = "Keep both hands inside the frame"
                    elif not results.pose_landmarks:
                        hint = "Step back so your upper body is visible"

                    nmm = self._estimate_nmm(results)
                    return PoseEstimate(
                        landmarks=list(landmarks),
                        confidence=0.82 if landmarks.size else 0.5,
                        nmm_flags=nmm,
                        occluded=occluded,
                        hint=hint,
                        synthetic=False,
                    )
            except Exception:  # noqa: BLE001
                logger.debug("MediaPipe processing failed; falling back to stub", exc_info=True)

        return self._simulate(image_bytes, seed_hint)

    def _simulate(self, image_bytes: bytes, seed_hint: int) -> PoseEstimate:
        seed = (sum(image_bytes[:32]) + seed_hint) % 10_000
        rng = random.Random(seed)
        landmarks = [rng.random() for _ in range(self._features_len)]
        base_conf = 0.65 + (seed % 20) / 100
        occlusion = (seed % 7) == 0
        hint = "Improve lighting or face camera" if occlusion else None
        nmm = {
            "brow_raise": 0.2 + (seed % 3) * 0.2,
            "headshake": 1.0 if (seed % 11) == 0 else 0.0,
            "chin_tilt": 0.1 * (seed % 5),
        }
        return PoseEstimate(
            landmarks=landmarks,
            confidence=min(base_conf, 0.98),
            nmm_flags=nmm,
            occluded=occlusion,
            hint=hint,
            synthetic=True,
        )

    @staticmethod
    def _collect_landmarks(results: "mp.solutions.holistic.Holistic"):
        face = np.zeros(468 * 3, dtype=np.float32)
        pose = np.zeros(33 * 4, dtype=np.float32)
        lh = np.zeros(21 * 3, dtype=np.float32)
        rh = np.zeros(21 * 3, dtype=np.float32)

        if results.face_landmarks:
            face_landmarks = results.face_landmarks.landmark
            face = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32).flatten()

        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark
            pose = (
                np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks], dtype=np.float32)
                .flatten()
            )

        if results.left_hand_landmarks:
            lh_landmarks = results.left_hand_landmarks.landmark
            lh = np.array([[lm.x, lm.y, lm.z] for lm in lh_landmarks], dtype=np.float32).flatten()

        if results.right_hand_landmarks:
            rh_landmarks = results.right_hand_landmarks.landmark
            rh = np.array([[lm.x, lm.y, lm.z] for lm in rh_landmarks], dtype=np.float32).flatten()

        return np.concatenate([face, pose, lh, rh])

    @staticmethod
    def _estimate_nmm(results: "mp.solutions.holistic.Holistic") -> Dict[str, float]:
        nmm: Dict[str, float] = {"brow_raise": 0.0, "headshake": 0.0}
        if results.face_landmarks:
            # Rough proxy: compare y positions of eyebrows vs eyes.
            face = results.face_landmarks.landmark
            left_brow = face[65].y if len(face) > 65 else 0.5
            left_eye = face[159].y if len(face) > 159 else 0.5
            brow_raise = max(0.0, min(1.0, (left_eye - left_brow) * 6))
            nmm["brow_raise"] = brow_raise
        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[0]
            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]
            head_tilt = abs(left_shoulder.z - right_shoulder.z)
            nmm["headshake"] = min(1.0, head_tilt * 5)
        return nmm
