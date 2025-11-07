from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .pose import PoseEstimate
from .predictor import SignPredictor


@dataclass
class GlossObservation:
    gloss: str
    confidence: float
    timestamp: int
    nmm: Dict[str, float]


FALLBACK_SEQUENCE: List[Dict[str, float | str]] = [
    {"gloss": "HELLO", "brow_raise": 0.3, "headshake": 0.0},
    {"gloss": "YES", "brow_raise": 0.1, "headshake": 0.0},
    {"gloss": "NO", "brow_raise": 0.1, "headshake": 0.8},
    {"gloss": "THANK-YOU", "brow_raise": 0.2, "headshake": 0.0},
    {"gloss": "GOOD-MORNING", "brow_raise": 0.4, "headshake": 0.0},
]


class SignRecognizer:
    """Streaming recognizer backed by the pretrained SignLink LSTM."""

    def __init__(self) -> None:
        self.predictors: Dict[str, SignPredictor] = {}

    def _get_predictor(self, window_id: str) -> SignPredictor:
        if window_id not in self.predictors:
            self.predictors[window_id] = SignPredictor()
        return self.predictors[window_id]

    def _fallback_observation(self, frame_seq: int, timestamp: int, pose: PoseEstimate) -> GlossObservation:
        pattern = FALLBACK_SEQUENCE[(frame_seq // 3) % len(FALLBACK_SEQUENCE)]
        gloss = str(pattern["gloss"])
        nmm = {k: float(v) for k, v in pattern.items() if k != "gloss"}
        nmm.update(pose.nmm_flags)
        nmm.setdefault("occlusion", 1.0 if pose.occluded else 0.0)
        return GlossObservation(
            gloss=gloss,
            confidence=0.75,
            timestamp=timestamp,
            nmm=nmm,
        )

    def recognize(
        self,
        pose: PoseEstimate,
        window_id: str,
        frame_seq: int,
        timestamp: int,
    ) -> GlossObservation:
        if pose.synthetic:
            return self._fallback_observation(frame_seq, timestamp, pose)

        features = np.array(pose.landmarks or [], dtype=np.float32)
        predictor = self._get_predictor(window_id)
        label, conf, _ = predictor.update(features)
        gloss = label.replace(" ", "-").upper() if label else "DETECTING"

        nmm = dict(pose.nmm_flags)
        nmm.setdefault("occlusion", 1.0 if pose.occluded else 0.0)

        return GlossObservation(
            gloss=gloss,
            confidence=conf if conf > 0 else pose.confidence,
            timestamp=timestamp,
            nmm=nmm,
        )
