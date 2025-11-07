from __future__ import annotations

import re


class SafetyFilter:
    """Masks digits and enforces privacy-friendly substitutions."""

    def scrub(self, text: str, mask_digits: bool = True) -> str:
        if mask_digits:
            text = re.sub(r"\d", "#", text)
        return text

    def needs_hint(self, confidence: float) -> bool:
        return confidence < 0.65
