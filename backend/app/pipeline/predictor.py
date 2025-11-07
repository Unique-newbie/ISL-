from __future__ import annotations

import json
import os
from collections import Counter, deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import tensorflow as tf

ACTION_LABELS = ["hello", "yes", "no", "thank you", "good morning"]
SEQ_LEN = 30
FEATURES_LEN = 1662
MODEL_PATH = Path("models/signlink_lstm.h5")
LABELS_PATH = Path("models/labels.json")


def _load_labels() -> List[str]:
    if LABELS_PATH.exists():
        try:
            return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return ACTION_LABELS


class SignPredictor:
    """Streaming sequence classifier backed by the existing SignLink LSTM."""

    def __init__(self, threshold: float = 0.7, frame_skip: int = 5) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing pretrained model at {MODEL_PATH}. "
                "Train the model (see README) or download the open-source weights."
            )
        self.labels = _load_labels()
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.threshold = threshold
        self.frame_skip = frame_skip
        self.seq: Deque[np.ndarray] = deque(maxlen=SEQ_LEN)
        self.frame_counter = 0
        self.last_label: Optional[str] = None
        self.vote_window: Deque[Tuple[str, float]] = deque(maxlen=10)

    def reset(self) -> None:
        self.seq.clear()
        self.vote_window.clear()
        self.last_label = None
        self.frame_counter = 0

    def update(self, features: np.ndarray) -> Tuple[str, float, bool]:
        """Feed a single frame worth of features, return (label, conf, is_new)."""
        if features.size != FEATURES_LEN:
            padded = np.zeros(FEATURES_LEN, dtype=np.float32)
            padded[: min(FEATURES_LEN, features.size)] = features[:FEATURES_LEN]
            features = padded
        self.seq.append(features.astype(np.float32))
        self.frame_counter += 1

        if len(self.seq) < SEQ_LEN or self.frame_counter % self.frame_skip != 0:
            return self.last_label or "detecting", 0.0, False

        sample = np.expand_dims(np.array(self.seq, dtype=np.float32), axis=0)
        probs = self.model.predict(sample, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.labels[idx]

        self.vote_window.append((label, conf))
        voted_label, voted_conf = self._majority_vote()

        if voted_conf >= self.threshold:
            changed = voted_label != self.last_label
            self.last_label = voted_label
            return voted_label, voted_conf, changed

        return "detecting", voted_conf, False

    def _majority_vote(self) -> Tuple[str, float]:
        if not self.vote_window:
            return "detecting", 0.0
        labels = [item[0] for item in self.vote_window]
        winner, _ = Counter(labels).most_common(1)[0]
        confidences = [c for l, c in self.vote_window if l == winner]
        return winner, float(np.mean(confidences))
