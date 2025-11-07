from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .sign_recognizer import GlossObservation


@dataclass
class SegmentPartial:
    gloss: str
    confidence: float
    timestamp: int
    revision: int
    window_id: str
    nmm: Dict[str, float]


@dataclass
class SegmentFinal:
    segment_id: str
    tokens: List[str]
    start_ms: int
    end_ms: int
    confidence: float
    nmm_snapshot: Dict[str, float]


@dataclass
class SegmentBuffer:
    tokens: List[str] = field(default_factory=list)
    start_ms: int = 0
    repeat_gloss: str = ""
    repeat_count: int = 0
    revision: int = 0
    nmm_accum: Dict[str, float] = field(default_factory=dict)
    nmm_samples: int = 0


class Segmenter:
    """Greedy streaming stabilizer that emits finals after K repeated glosses."""

    def __init__(self, stable_count: int = 3) -> None:
        self.stable_count = stable_count
        self.buffers: Dict[str, SegmentBuffer] = {}
        self.segment_counter = itertools.count(1)

    def consume(self, window_id: str, observation: GlossObservation) -> Tuple[SegmentPartial, SegmentFinal | None]:
        buffer = self.buffers.setdefault(window_id, SegmentBuffer())
        if observation.gloss == "DETECTING":
            partial = SegmentPartial(
                gloss=observation.gloss,
                confidence=observation.confidence,
                timestamp=observation.timestamp,
                revision=buffer.revision,
                window_id=window_id,
                nmm=observation.nmm,
            )
            return partial, None

        buffer.tokens.append(observation.gloss)
        buffer.revision += 1
        buffer.nmm_samples += 1
        for key, value in observation.nmm.items():
            buffer.nmm_accum[key] = buffer.nmm_accum.get(key, 0.0) + value

        if buffer.repeat_gloss == observation.gloss:
            buffer.repeat_count += 1
        else:
            buffer.repeat_gloss = observation.gloss
            buffer.repeat_count = 1
            buffer.start_ms = observation.timestamp

        partial = SegmentPartial(
            gloss=observation.gloss,
            confidence=observation.confidence,
            timestamp=observation.timestamp,
            revision=buffer.revision,
            window_id=window_id,
            nmm=observation.nmm,
        )

        final: SegmentFinal | None = None
        if buffer.repeat_count >= self.stable_count:
            segment_id = f"{window_id}-{next(self.segment_counter)}"
            final = SegmentFinal(
                segment_id=segment_id,
                tokens=list(buffer.tokens),
                start_ms=buffer.start_ms,
                end_ms=observation.timestamp,
                confidence=observation.confidence,
                nmm_snapshot=self._avg_nmm(buffer),
            )
            self.buffers[window_id] = SegmentBuffer()

        return partial, final

    @staticmethod
    def _avg_nmm(buffer: SegmentBuffer) -> Dict[str, float]:
        if buffer.nmm_samples == 0:
            return {}
        return {key: value / buffer.nmm_samples for key, value in buffer.nmm_accum.items()}
