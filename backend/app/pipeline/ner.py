from __future__ import annotations

import re


class NERFormatter:
    """Very small formatter to normalize detected entities."""

    NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+)\b")

    def normalize(self, text: str) -> str:
        return self.NAME_PATTERN.sub(lambda match: match.group(1).title(), text)
