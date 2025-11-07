from __future__ import annotations

from typing import List

from .segmenter import SegmentFinal, SegmentPartial

GLOSS_MAP = {
    "HELLO": "hello",
    "YES": "yes",
    "NO": "no",
    "THANK-YOU": "thank you",
    "GOOD-MORNING": "good morning",
    "GOOD-NIGHT": "good night",
    "PLEASE": "please",
}


class GlossTranslator:
    """Rule-based gloss -> English mapper with basic NMM awareness."""

    def partial_text(self, segment: SegmentPartial) -> str:
        if segment.gloss == "DETECTING":
            return "detecting gesture..."
        base = self._token_to_text(segment.gloss)
        suffix = "?" if segment.nmm.get("brow_raise", 0.0) > 0.6 else "..."
        return f"{base}{suffix}"

    def to_english(self, segment: SegmentFinal) -> str:
        words: List[str] = [self._token_to_text(token) for token in segment.tokens]
        sentence = " ".join(word for word in words if word).strip()
        if not sentence:
            sentence = "gesture detected"

        if segment.nmm_snapshot.get("headshake", 0.0) > 0.5 and "not" not in sentence:
            sentence = f"not {sentence}"

        if segment.nmm_snapshot.get("brow_raise", 0.0) > 0.6:
            sentence = f"{sentence}?"
        else:
            sentence = f"{sentence}."

        return sentence[:1].upper() + sentence[1:]

    def _token_to_text(self, token: str) -> str:
        if token.startswith("[") and token.endswith("]"):
            return token[1:-1].lower()
        return GLOSS_MAP.get(token, token.replace("-", " ").lower())

