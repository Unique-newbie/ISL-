import base64
import pytest

from backend.app.pipeline.orchestrator import PipelineOrchestrator
from backend.app.schemas import FrameMessage, Preferences

pytestmark = pytest.mark.asyncio


def make_frame(seq: int) -> FrameMessage:
    payload = base64.b64encode(f"frame-{seq}".encode()).decode()
    return FrameMessage(type="frame", window_id="w0", seq=seq, ts=seq * 100, image=payload)


async def produce_events(orchestrator: PipelineOrchestrator, frames: int = 6):
    events = []
    for seq in range(frames):
        batch = await orchestrator.process_frame_for_test(make_frame(seq))
        events.extend(batch)
    return events


async def test_partial_precedes_final():
    orchestrator = PipelineOrchestrator()
    orchestrator.update_preferences(Preferences(target_language="hi"))
    events = await produce_events(orchestrator, frames=8)
    partial_indices = [idx for idx, event in enumerate(events) if getattr(event, "type", "") == "partial_text"]
    final_indices = [idx for idx, event in enumerate(events) if getattr(event, "type", "") == "final_text"]

    assert partial_indices, "Expected partial events"
    assert final_indices, "Expected final events"
    assert min(final_indices) > min(partial_indices), "Final emitted before partial"

    hindi_finals = [event for event in events if getattr(event, "type", "") == "final_text" and getattr(event, "lang", "") == "hi"]
    assert any(ev.text for ev in hindi_finals), "Hindi final text should not be empty"


async def test_audio_event_references_segment_id():
    orchestrator = PipelineOrchestrator()
    orchestrator.update_preferences(Preferences(target_language="en", tts_enabled=True))
    events = await produce_events(orchestrator, frames=10)
    final_ids = {event.segment_id for event in events if getattr(event, "type", "") == "final_text"}
    audio_segments = {event.segment_id for event in events if getattr(event, "type", "") == "audio"}
    assert audio_segments.issubset(final_ids)
