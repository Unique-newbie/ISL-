import os
import json
from glob import glob
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.utils.constants import (
    ACTIONS,
    DATA_DIR,
    MODEL_DIR,
    MODEL_PATH,
    LABELS_PATH,
    SEQ_LEN,
    FEATURES_LEN,
)


def load_dataset(actions: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx, a in enumerate(actions):
        folder = os.path.join(DATA_DIR, a)
        files = sorted(glob(os.path.join(folder, "*.npy")))
        for f in files:
            arr = np.load(f)
            if arr.shape != (SEQ_LEN, FEATURES_LEN):
                continue
            X.append(arr)
            y.append(idx)
    if not X:
        print("No real data found. Generating synthetic dataset for demo...")
        # Synthetic dataset: distinct magnitude patterns for each class
        rng = np.random.default_rng(42)
        for idx, _ in enumerate(actions):
            for _ in range(60):
                base = np.zeros((SEQ_LEN, FEATURES_LEN), dtype=np.float32)
                # Use different feature bands per class
                start = (idx * 300) % FEATURES_LEN
                end = min(start + 300, FEATURES_LEN)
                pattern = np.linspace(0, 1, SEQ_LEN).reshape(-1, 1)
                noise = rng.normal(0, 0.02, size=(SEQ_LEN, end - start))
                base[:, start:end] = pattern + noise
                X.append(base)
                y.append(idx)
    X = np.array(X, dtype=np.float32)
    y = tf.keras.utils.to_categorical(np.array(y), num_classes=len(actions))
    return X, y


def build_model(num_classes: int) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, FEATURES_LEN)),
        Dropout(0.2),
        LSTM(128),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_dataset(ACTIONS)
    print(f"Dataset: X={X.shape}, y={y.shape}")
    model = build_model(num_classes=len(ACTIONS))
    callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]
    history = model.fit(
        X, y,
        epochs=25,
        batch_size=16,
        validation_split=0.2,
        verbose=2,
        callbacks=callbacks,
    )
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Final training set accuracy: {acc:.3f}")
    model.save(MODEL_PATH)
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(ACTIONS, f, ensure_ascii=False, indent=2)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Labels saved to {LABELS_PATH}")


if __name__ == "__main__":
    main()

