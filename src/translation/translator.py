from functools import lru_cache
from typing import Optional

try:
    from googletrans import Translator as GoogleTranslator
except Exception:
    GoogleTranslator = None

FALLBACK_DICT = {
    # Hindi fallback
    ("hello", "hi"): "नमस्ते",
    ("yes", "hi"): "हाँ",
    ("no", "hi"): "नहीं",
    ("thank you", "hi"): "धन्यवाद",
    ("good morning", "hi"): "सुप्रभात",
}


class Translator:
    def __init__(self):
        self.client = GoogleTranslator() if GoogleTranslator is not None else None

    @lru_cache(maxsize=256)
    def translate(self, text: str, dest_lang: str = "hi") -> str:
        text = (text or "").strip()
        if not text:
            return ""
        # Short-circuit: if dest is English, return as-is
        if dest_lang == "en":
            return text
        # Try googletrans
        if self.client is not None:
            try:
                res = self.client.translate(text, dest=dest_lang)
                if hasattr(res, 'text') and res.text:
                    return res.text
            except Exception:
                pass
        # Fallback dictionary for a few known words
        key = (text.lower(), dest_lang)
        return FALLBACK_DICT.get(key, text)

