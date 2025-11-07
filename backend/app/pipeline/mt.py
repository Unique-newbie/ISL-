from __future__ import annotations

from typing import Dict


class MTGateway:
    """Deterministic multilingual stub with domain-aware fallbacks."""

    _lexicon: Dict[str, Dict[str, str]] = {
        "hi": {
            "hello.": "नमस्ते।",
            "not hello.": "नमस्ते नहीं।",
            "yes.": "हाँ।",
            "no.": "नहीं।",
            "thank you.": "धन्यवाद।",
            "good morning.": "सुप्रभात।",
        },
        "ta": {
            "hello.": "வணக்கம்.",
            "yes.": "ஆம்.",
            "no.": "இல்லை.",
            "thank you.": "நன்றி.",
            "good morning.": "காலை வணக்கம்.",
        },
        "te": {
            "hello.": "నమస్కారం.",
            "yes.": "అవును.",
            "no.": "కాదు.",
            "thank you.": "ధన్యవాదాలు.",
            "good morning.": "శుభోదయం.",
        },
        "bn": {
            "hello.": "নমস্তে।",
            "yes.": "হ্যাঁ।",
            "no.": "না।",
            "thank you.": "ধন্যবাদ।",
            "good morning.": "সুপ্রভাত।",
        },
    }

    def translate(self, text: str, target_lang: str, domain: str) -> str:
        if target_lang == "en":
            return text
        lexicon = self._lexicon.get(target_lang, {})
        normalized = text.lower()
        if normalized in lexicon:
            return lexicon[normalized]
        for key, value in lexicon.items():
            if key.split(".")[0] in normalized:
                return value
        return f"{text} (untranslated:{target_lang})"
