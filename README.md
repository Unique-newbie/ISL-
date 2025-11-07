# Real-Time ISL -> Multilingual Translation

## Overview
- Stream webcam frames from the browser, decode ISL gestures with the open-source SignLink LSTM (trained on public ISL gesture clips), and emit multilingual captions plus optional speech with end-to-end latency < 2 s.
- Streaming contract provides `partial_text`, `final_text`, `audio`, `hint`, and `notice` events so the UI can progressively render captions while audio synthesis happens in parallel.
- ML-heavy stages (pose, recognition, MT, TTS, NER, safety) are implemented as hot-swappable stubs with clean interfaces so production models can drop in without touching the WebSocket contract.

## ASCII Architecture
```
+----------------------------+         WebSocket /ws          +-------------------------------+
| React + Vite + Tailwind    |  frame/init messages (JSON)    | FastAPI + uvicorn             |
| - WebRTC camera capture    | <----------------------------> | - Async queue orchestrator    |
| - Streaming captions panel |        partial/final/audio     | - Pose -> ISL -> MT pipeline    |
| - Settings & accessibility |                                | - Privacy + telemetry stubs   |
+----------------------------+                                +-------------------------------+
             |                                                             |
             | docker-compose (frontend @ 5173, backend @ 8000)            |
             +-------------------------------------------------------------+
```

## Quick Start (local dev)
1. **Backend**
   ```bash
   python -m venv .venv && .\.venv\Scripts\activate
   pip install -r backend/requirements.txt
   uvicorn backend.app.main:app --reload
   ```
2. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev -- --host 0.0.0.0 --port 5173
   ```
3. Open `http://localhost:5173`, allow camera, and enable "Audio On" if you want TTS events to play.

### Docker Compose
```bash
docker compose up --build
```
- Frontend: http://localhost:5173
- Backend WebSocket: ws://localhost:8000/ws

### Pretrained model
The FastAPI backend loads the open-source SignLink LSTM weights from `models/signlink_lstm.h5` (labels in `models/labels.json`). These weights come from the accompanying SignLink research repo; if you want to retrain or fine-tune using your own open-source datasets (e.g., WLASL subsets, IIIT-ILSL), drop the updated files in `models/` before starting the backend.

## Streaming Contract
Client messages:
```json
{"type":"init","prefs":{"target_language":"en","domain_mode":"general","tts_enabled":false,"romanize":false,"telemetry_opt_in":false}}
{"type":"frame","window_id":"w0","seq":12,"ts":1700000000000,"image":"<base64 jpeg>"}
```
Server events (JSON):
- `partial_text {type,lang,text,conf,window_id,timestamp,window_revision}`
- `final_text {type,lang,text,conf,start_ms,end_ms,segment_id}`
- `audio {type,lang,audio_url,segment_id}`
- `hint {type,message,hint_kind}`
- `notice {type,message,notice_type}`

Ordering guarantees:
1. Partial text precedes final text for the same segment.
2. Finals are immutable (no downgrades); subsequent revisions arrive as new segments.
3. Audio events reference a published `segment_id`.

## Tests
```bash
pytest
```
- `tests/test_events_contract.py`: validates ordering and language coverage.
- `tests/test_latency_budget.py`: mocks frame bursts to ensure <2 s total with partials <800 ms.

## Swapping Real Models
Each pipeline stage lives in `backend/app/pipeline` with a single public class:
1. **Pose** (`pose.py`): replace `PoseEstimator.estimate()` with MediaPipe or custom gRPC call. Return the same `PoseEstimate` dataclass to keep downstream consumers untouched.
2. **Recognition** (`sign_recognizer.py`): drop in a TensorFlow/ONNX runtime that consumes landmarks and outputs `GlossObservation`.
3. **Segmentation/CTC** (`segmenter.py`): wire real-time decoders that map streaming glosses to stable segments.
4. **Gloss -> Text / MT / TTS** (`gloss_to_text.py`, `mt.py`, `tts.py`): connect to production NMT/TTS services via HTTP or SDKs while keeping the current method signatures.
5. **Safety & NER** (`ner.py`, `safety.py`): plug policy engines or external services.

Because each stage is dependency-injected in `PipelineOrchestrator`, swapping only requires editing the respective module (or subclassing and swapping during orchestrator construction).

## Telemetry & Privacy
- Raw frames never touch disk; only per-frame metrics (latency/confidence) may be logged when `prefs.telemetry_opt_in` is true.
- `safety.apply()` masks digits and runs rudimentary safety checks; extend to integrate with compliance pipelines.
- Enable anonymized telemetry via `prefs.telemetry_opt_in` or `ENABLE_TELEMETRY_DEFAULT=1`.
- CORS allowlist is environment-driven (`CORS_ALLOW_ORIGINS`), defaulting to `http://localhost:5173`.

## Accessibility
- Captions use high-contrast themes, keyboard shortcuts, ARIA roles, and font scaling.
- Romanization toggle (hi/ta/te/bn) keeps captions readable for learners.

---

Happy streaming! Use `frontend/src/lib/ws.ts` to integrate alternative clients or embed the streaming UI elsewhere.

