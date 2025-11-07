import os
import json
from collections import deque, Counter
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.utils.constants import (
    ACTIONS,
    MODEL_PATH,
    LABELS_PATH,
    SEQ_LEN,
    FEATURES_LEN,
)


def _load_labels() -> List[str]:
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return ACTIONS


class SignPredictor:
    def __init__(self, threshold: float = 0.7, frame_skip: int = 5):
        self.threshold = threshold
        self.frame_skip = frame_skip
        self.seq = deque(maxlen=SEQ_LEN)
        self.labels = _load_labels()
        self.model = self._ensure_model()
        self.frame_count = 0
        self.last_pred: Optional[str] = None
        self.smooth_q = deque(maxlen=10)

    def _ensure_model(self):
        if not os.path.exists(MODEL_PATH):
            # Build and randomly initialize a compatible model; this will be weak until trained.
            from src.model.train_model import build_model
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            model = build_model(num_classes=len(self.labels))
            # Save and reload for consistency
            model.save(MODEL_PATH)
            with open(LABELS_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.labels, f)
        model = tf.keras.models.load_model(MODEL_PATH)
        return model

    def reset(self):
        self.seq.clear()
        self.frame_count = 0
        self.smooth_q.clear()
        self.last_pred = None

    def update(self, keypoints: np.ndarray) -> Tuple[str, float, bool]:
        """Feed one frame's keypoints, return (label, confidence, changed).

        Predicts on every `frame_skip` frames once we have SEQ_LEN samples.
        """
        self.seq.append(keypoints)
        self.frame_count += 1
        if len(self.seq) < SEQ_LEN or (self.frame_count % self.frame_skip) != 0:
            # Not enough data or skipping frames
            label = self.last_pred or "detecting..."
            return label, 0.0, False

        sample = np.expand_dims(np.array(self.seq, dtype=np.float32), axis=0)
        probs = self.model.predict(sample, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        pred_label = self.labels[idx]

        # Temporal smoothing across last few decisions
        self.smooth_q.append((pred_label, conf))
        best_label, best_conf = self._majority_vote()

        changed = False
        if best_conf >= self.threshold:
            changed = best_label != self.last_pred
            self.last_pred = best_label
            return best_label, best_conf, changed
        else:
            return "", best_conf, False

    def _majority_vote(self) -> Tuple[str, float]:
        if not self.smooth_q:
            return "", 0.0
        labels = [l for l, _ in self.smooth_q]
        cnt = Counter(labels)
        best_label, _ = cnt.most_common(1)[0]
        confs = [c for l, c in self.smooth_q if l == best_label]
        return best_label, float(np.mean(confs))

