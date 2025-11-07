from __future__ import annotations

import json
import logging
import os
from typing import Any, Awaitable, Callable, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .pipeline.orchestrator import PipelineOrchestrator
from .schemas import FrameMessage, InitMessage, NoticeEvent
from .utils.timestamps import now_ms

logger = logging.getLogger("signlink.backend")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="SignLink Streaming API", version="0.1.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
if not allowed_origins:
    allowed_origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def healthcheck() -> Dict[str, str]:  # pragma: no cover - trivial
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    orchestrator = PipelineOrchestrator()
    sender: Callable[[Dict[str, Any]], Awaitable[None]] = websocket.send_json
    await orchestrator.attach(sender)
    prefs_set = False

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await sender(
                    NoticeEvent(
                        type="notice",
                        message="Invalid JSON payload",
                        notice_type="error",
                    ).model_dump()
                )
                continue

            msg_type = payload.get("type")
            if msg_type == "init":
                try:
                    init_msg = InitMessage(**payload)
                except Exception as exc:  # noqa: BLE001
                    await sender(
                        NoticeEvent(
                            type="notice",
                            message=f"Init validation failed: {exc}",
                            notice_type="error",
                        ).model_dump()
                    )
                    continue

                orchestrator.update_preferences(init_msg.prefs)
                prefs_set = True
                await sender(
                    NoticeEvent(
                        type="notice",
                        message="Pipeline initialized",
                        notice_type="init_ack",
                    ).model_dump()
                )
                continue

            if msg_type == "frame":
                if not prefs_set:
                    await sender(
                        NoticeEvent(
                            type="notice",
                            message="Send init before frames",
                            notice_type="error",
                        ).model_dump()
                    )
                    continue
                try:
                    frame = FrameMessage(**payload)
                except Exception as exc:  # noqa: BLE001
                    await sender(
                        NoticeEvent(
                            type="notice",
                            message=f"Frame validation failed: {exc}",
                            notice_type="error",
                        ).model_dump()
                    )
                    continue

                await orchestrator.handle_frame(frame)
                continue

            await sender(
                NoticeEvent(
                    type="notice",
                    message=f"Unsupported message type: {msg_type}",
                    notice_type="error",
                ).model_dump()
            )
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    finally:
        await orchestrator.close()
        logger.debug("Connection cleaned up @%s", now_ms())
