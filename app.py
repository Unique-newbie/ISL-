import os
import time
import json
import threading
from collections import deque

import cv2
import numpy as np
import streamlit as st

from src.mediapipe_utils import HolisticWrapper, extract_keypoints, overlay_prediction
from src.inference.predictor import SignPredictor
from src.translation.translator import Translator
from src.audio.tts_player import TTSPlayer
from src.utils.constants import ACTIONS, LANGUAGES, DEFAULT_LANG, SEQ_LEN


st.set_page_config(page_title="SignLink • ISL to Text & Speech", layout="wide")

# Sidebar controls
st.sidebar.title("Settings")
lang_label = st.sidebar.selectbox("Target language", list(LANGUAGES.keys()), index=0)
lang_code = LANGUAGES[lang_label]
auto_speak = st.sidebar.checkbox("Auto-speak translation", value=True)
threshold = st.sidebar.slider("Confidence threshold", 0.5, 0.95, 0.70, 0.01)
frame_skip = st.sidebar.slider("Predict every N frames", 1, 10, 5, 1)

# Camera controls
st.sidebar.subheader("Camera")
if 'camera_index' not in st.session_state:
    st.session_state.camera_index = None
if 'camera_backend' not in st.session_state:
    st.session_state.camera_backend = None
if 'camera_resolution' not in st.session_state:
    st.session_state.camera_resolution = (640, 480)
cam_idx_opt = st.sidebar.selectbox("Device index", ["Auto", 0, 1, 2, 3, 4, 5], index=0)
st.session_state.camera_index = None if cam_idx_opt == "Auto" else int(cam_idx_opt)
backend_opt = st.sidebar.selectbox("Backend", ["Auto", "MSMF", "DSHOW"], index=0)
st.session_state.camera_backend = None if backend_opt == "Auto" else backend_opt
res_opt = st.sidebar.selectbox("Resolution", ["640x480", "1280x720"], index=0)
st.session_state.camera_resolution = (640, 480) if res_opt == "640x480" else (1280, 720)


st.title("SignLink — Real-time ISL → Multilingual Text & Speech")
st.caption("Built with AI for Accessibility — Team SignLink")

if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_text' not in st.session_state:
    st.session_state.last_text = ""
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'translator' not in st.session_state:
    st.session_state.translator = Translator()
if 'tts' not in st.session_state:
    st.session_state.tts = TTSPlayer()
if 'thread' not in st.session_state:
    st.session_state.thread = None


start, stop = st.columns(2)
with start:
    if st.button("▶ Start") and not st.session_state.running:
        st.session_state.running = True
with stop:
    if st.button("■ Stop") and st.session_state.running:
        st.session_state.running = False


video_placeholder = st.empty()
pred_col, trans_col = st.columns([1, 1])
with pred_col:
    pred_box = st.empty()
with trans_col:
    trans_box = st.empty()

spk1, spk2 = st.columns([1, 3])
with spk1:
    if st.button("🔊 Speak Translation"):
        if st.session_state.last_text:
            st.session_state.tts.speak(st.session_state.last_text, lang_code)


def _open_camera(idx_pref: int | None = None, backend_pref: str | None = None, width: int = 640, height: int = 480):
    backends = []
    if backend_pref == "MSMF":
        backends = [cv2.CAP_MSMF]
    elif backend_pref == "DSHOW":
        backends = [cv2.CAP_DSHOW]
    else:
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]

    if idx_pref is not None and idx_pref >= 0:
        indices = [idx_pref]
    else:
        indices = list(range(0, 6))

    tried = []
    for i in indices:
        for b in backends:
            cap = cv2.VideoCapture(i, b) if b is not None else cv2.VideoCapture(i)
            if cap is None or not cap.isOpened():
                tried.append((i, b, "open_failed"))
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap, i, b, tried
            tried.append((i, b, "read_failed"))
            cap.release()
    return None, None, None, tried


def run_loop():
    cam_idx = st.session_state.get("camera_index")
    cam_backend = st.session_state.get("camera_backend")
    cam_res = st.session_state.get("camera_resolution", (640, 480))
    cap, use_idx, use_backend, tried = _open_camera(cam_idx, cam_backend, cam_res[0], cam_res[1])
    if cap is None:
        details = "; ".join([f"idx {i} backend {b} => {reason}" for i, b, reason in tried])
        st.error(
            "Could not open webcam. Ensure no other app is using it and that camera access is allowed in Windows Privacy Settings."
        )
        if details:
            st.info(f"Tried: {details}")
        st.session_state.running = False
        return

    holo = HolisticWrapper()
    predictor = SignPredictor(threshold=threshold, frame_skip=frame_skip)
    st.session_state.predictor = predictor

    fps_deque = deque(maxlen=30)
    last_spoken = ""

    try:
        while st.session_state.running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                st.warning("Frame grab failed.")
                break
            frame = cv2.flip(frame, 1)
            results = holo.process(frame)
            kps = extract_keypoints(results)
            label, conf, changed = predictor.update(kps)

            vis = holo.draw_landmarks(frame.copy(), results)
            if label:
                vis = overlay_prediction(vis, label, conf)

            video_placeholder.image(vis, channels="BGR", use_column_width=True)

            if label:
                pred_box.markdown(f"**Detected:** {label}  |  **Confidence:** {conf:.2f}")
                translated = st.session_state.translator.translate(label, lang_code)
                trans_box.markdown(f"**Translated:** {translated}")
                st.session_state.last_text = translated
                if auto_speak and changed and translated and translated != last_spoken:
                    st.session_state.tts.speak(translated, lang_code)
                    last_spoken = translated
            else:
                pred_box.markdown("Detecting…")
                trans_box.markdown("")

            fps = 1.0 / max(1e-6, (time.time() - t0))
            fps_deque.append(fps)
            avg_fps = sum(fps_deque) / len(fps_deque)
            st.sidebar.metric("FPS", f"{avg_fps:.1f}")
    finally:
        cap.release()


if st.session_state.running and (st.session_state.thread is None or not st.session_state.thread.is_alive()):
    st.session_state.thread = threading.Thread(target=run_loop, daemon=True)
    st.session_state.thread.start()

if not st.session_state.running:
    video_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR", use_column_width=True)
    pred_box.markdown("Click Start to begin.")
    trans_box.markdown("")

