import os
import time
from datetime import datetime
from collections import deque

import cv2
import numpy as np

from src.mediapipe_utils import HolisticWrapper, extract_keypoints, overlay_prediction
from src.utils.constants import ACTIONS, DATA_DIR, SEQ_LEN


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    for a in ACTIONS:
        os.makedirs(os.path.join(DATA_DIR, a), exist_ok=True)


def _open_camera(idx_pref: int | None = None, width: int = 640, height: int = 480):
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    indices = [idx_pref] if (idx_pref is not None and idx_pref >= 0) else list(range(0, 6))
    for i in indices:
        for b in backends:
            cap = cv2.VideoCapture(i, b) if b is not None else cv2.VideoCapture(i)
            if cap is None or not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def main():
    ensure_dirs()
    print("SignLink — Data Recorder")
    print("Press number keys 1-5 to select label:")
    for i, a in enumerate(ACTIONS, start=1):
        print(f"  {i}. {a}")
    print("Press 'r' to record a 30-frame sample, 'q' to quit.")

    cap = _open_camera()
    if cap is None:
        print("Error: Could not open webcam. Close other apps and allow camera permissions in OS settings.")
        return

    holo = HolisticWrapper()
    seq = deque(maxlen=SEQ_LEN)
    cur_label_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = holo.process(frame)
            kps = extract_keypoints(results)
            seq.append(kps)
            vis = holo.draw_landmarks(frame.copy(), results)
            vis = overlay_prediction(vis, f"label: {ACTIONS[cur_label_idx]}", 1.0)
            cv2.imshow("SignLink Recorder", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                cur_label_idx = int(chr(key)) - 1
            if key == ord('r'):
                if len(seq) < SEQ_LEN:
                    print("Need more frames buffered. Hold still for a moment...")
                    continue
                arr = np.array(seq)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out_path = os.path.join(DATA_DIR, ACTIONS[cur_label_idx], f"{ts}.npy")
                np.save(out_path, arr)
                print(f"Saved sample: {out_path}")
                time.sleep(0.3)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
