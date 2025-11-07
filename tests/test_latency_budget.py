import base64
import time
import pytest

from backend.app.pipeline.orchestrator import PipelineOrchestrator
from backend.app.schemas import FrameMessage

pytestmark = pytest.mark.asyncio


def make_frame(seq: int) -> FrameMessage:
    payload = base64.b64encode(b"\xff\xd8\xff demo frame").decode()
    return FrameMessage(type="frame", window_id="w0", seq=seq, ts=seq * 120, image=payload)


async def test_latency_budget_under_two_seconds():
    orchestrator = PipelineOrchestrator()
    start = time.perf_counter()
    first_partial_ms = None
    last_final_time = None

    for seq in range(12):
        events = await orchestrator.process_frame_for_test(make_frame(seq))
        now_ms = (time.perf_counter() - start) * 1000
        if first_partial_ms is None and any(getattr(e, "type", "") == "partial_text" for e in events):
            first_partial_ms = now_ms
        if any(getattr(e, "type", "") == "final_text" for e in events):
            last_final_time = now_ms

    assert first_partial_ms is not None and first_partial_ms < 800
    assert last_final_time is not None and last_final_time < 2000
