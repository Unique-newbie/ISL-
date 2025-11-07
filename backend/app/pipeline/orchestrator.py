from __future__ import annotations

import asyncio
import base64
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..schemas import (
    AudioEvent,
    FinalTextEvent,
    FrameMessage,
    HintEvent,
    NoticeEvent,
    PartialTextEvent,
    Preferences,
)
from ..utils.timestamps import monotonic_ms, now_ms
from .gloss_to_text import GlossTranslator
from .mt import MTGateway
from .ner import NERFormatter
from .pose import PoseEstimator
from .safety import SafetyFilter
from .segmenter import SegmentFinal, SegmentPartial, Segmenter
from .sign_recognizer import SignRecognizer
from .tts import TTSStub

OutputSender = Callable[[Dict[str, Any]], Awaitable[None]]


class PipelineOrchestrator:
    """Coordinates the streaming inference pipeline with backpressure."""

    def __init__(self) -> None:
        self.pose = PoseEstimator()
        self.recognizer = SignRecognizer()
        self.segmenter = Segmenter()
        self.gloss_translator = GlossTranslator()
        self.mt = MTGateway()
        self.tts = TTSStub()
        self.ner = NERFormatter()
        self.safety = SafetyFilter()
        self.prefs = Preferences()
        self.languages = ["en", "hi", "ta", "te", "bn"]
        self.frame_queue: asyncio.Queue[FrameMessage] = asyncio.Queue(maxsize=8)
        self._sender: Optional[OutputSender] = None
        self._worker: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._dropped_frames = 0
        self.telemetry_enabled = bool(int(os.getenv("ENABLE_TELEMETRY_DEFAULT", "0")))

    async def attach(self, sender: OutputSender) -> None:
        self._sender = sender
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    def update_preferences(self, prefs: Preferences) -> None:
        self.prefs = prefs
        self.telemetry_enabled = prefs.telemetry_opt_in or self.telemetry_enabled

    async def handle_frame(self, frame: FrameMessage) -> None:
        if self._closed:
            return
        if self.frame_queue.full():
            try:
                _ = self.frame_queue.get_nowait()
                self._dropped_frames += 1
                self.frame_queue.task_done()
            except asyncio.QueueEmpty:  # pragma: no cover - unlikely
                pass
        await self.frame_queue.put(frame)

    async def close(self) -> None:
        self._closed = True
        if self._worker:
            await self.frame_queue.put(
                FrameMessage(type="frame", window_id="__flush__", seq=-1, ts=now_ms(), image="")
            )
            await self._worker
            self._worker = None

    async def process_frame_for_test(self, frame: FrameMessage) -> List[Any]:
        return await self._process(frame)

    async def _run(self) -> None:
        while not self._closed or not self.frame_queue.empty():
            frame = await self.frame_queue.get()
            if frame.window_id == "__flush__":
                self.frame_queue.task_done()
                break
            events = await self._process(frame)
            if self._sender:
                for event in events:
                    await self._sender(event.model_dump())
            self.frame_queue.task_done()

    async def _process(self, frame: FrameMessage) -> List[Any]:
        start = monotonic_ms()
        events: List[Any] = []
        if not frame.image:
            return events

        try:
            image_bytes = base64.b64decode(frame.image)
        except Exception:  # noqa: BLE001
            return [
                NoticeEvent(type="notice", message="Bad frame encoding", notice_type="error")
            ]

        pose_result = self.pose.estimate(image_bytes, frame.seq)
        obs = self.recognizer.recognize(pose_result, frame.window_id, frame.seq, frame.ts)

        if pose_result.hint:
            events.append(
                HintEvent(type="hint", message=pose_result.hint, hint_kind="environment")
            )
        if pose_result.occluded:
            events.append(
                HintEvent(
                    type="hint",
                    message="Hands or face occluded",
                    hint_kind="occlusion",
                )
            )

        partial, final = self.segmenter.consume(frame.window_id, obs)
        events.extend(self._emit_partial(partial))
        if final:
            events.extend(await self._emit_final(final))

        latency = monotonic_ms() - start
        if self.telemetry_enabled:
            from logging import getLogger

            logger = getLogger("signlink.telemetry")
            logger.info(
                "latency_ms=%s conf=%.2f drop_frames=%s revision=%s window=%s",
                latency,
                obs.confidence,
                self._dropped_frames,
                partial.revision,
                frame.window_id,
            )
        return events

    def _emit_partial(self, partial: SegmentPartial) -> List[Any]:
        events: List[Any] = []
        english_text = self.gloss_translator.partial_text(partial)
        english_text = self.safety.scrub(english_text)
        partial_event = PartialTextEvent(
            lang="en",
            text=self.ner.normalize(english_text),
            conf=partial.confidence,
            window_id=partial.window_id,
            timestamp=partial.timestamp,
            window_revision=partial.revision,
        )
        events.append(partial_event)

        if self.prefs.target_language != "en":
            target_text = self.mt.translate(partial_event.text, self.prefs.target_language, self.prefs.domain_mode)
            events.append(
                PartialTextEvent(
                    lang=self.prefs.target_language,
                    text=target_text,
                    conf=partial.confidence,
                    window_id=partial.window_id,
                    timestamp=partial.timestamp,
                    window_revision=partial.revision,
                )
            )

        if self.safety.needs_hint(partial.confidence):
            events.append(
                HintEvent(
                    type="hint",
                    message="Raise hands higher or slow down",
                    hint_kind="low_confidence",
                )
            )
        return events

    async def _emit_final(self, final: SegmentFinal) -> List[Any]:
        events: List[Any] = []
        english = self.gloss_translator.to_english(final)
        english = self.safety.scrub(self.ner.normalize(english))

        for lang in self.languages:
            text = english if lang == "en" else self.mt.translate(english, lang, self.prefs.domain_mode)
            events.append(
                FinalTextEvent(
                    lang=lang,
                    text=text,
                    conf=final.confidence,
                    start_ms=final.start_ms,
                    end_ms=final.end_ms,
                    segment_id=final.segment_id,
                )
            )
            if self.prefs.tts_enabled and lang == self.prefs.target_language:
                audio_url = await self.tts.synthesize(text, lang, final.segment_id)
                events.append(
                    AudioEvent(
                        lang=lang,
                        audio_url=audio_url,
                        segment_id=final.segment_id,
                    )
                )

        return events
