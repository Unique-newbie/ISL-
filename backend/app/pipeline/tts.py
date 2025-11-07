from __future__ import annotations

import asyncio
import base64


class TTSStub:
    """Async TTS stub that emits short data URLs to prove wiring."""

    async def synthesize(self, text: str, lang: str, segment_id: str) -> str:
        payload = f"{lang}:{segment_id}:{text}".encode("utf-8")
        audio_b64 = base64.b64encode(payload).decode("ascii")
        await asyncio.sleep(min(0.35, 0.01 * len(text)))
        return f"data:audio/wav;base64,{audio_b64}"
