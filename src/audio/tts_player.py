import os
import tempfile
from typing import Optional

import pygame

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


class TTSPlayer:
    """Speaks text using gTTS + pygame; falls back to pyttsx3 offline."""

    def __init__(self):
        self._init_pygame()
        self.last_cache_key: Optional[str] = None
        self.last_cached_file: Optional[str] = None
        self.engine = None
        if pyttsx3 is not None:
            try:
                self.engine = pyttsx3.init()
            except Exception:
                self.engine = None

    def _init_pygame(self):
        try:
            if not pygame.get_init():
                pygame.init()
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except Exception:
            # If mixer fails (no audio device), ignore.
            pass

    def speak(self, text: str, lang: str = "hi"):
        text = (text or "").strip()
        if not text:
            return

        # Try gTTS first (network required)
        if gTTS is not None:
            key = f"{lang}:{text}"
            try:
                if self.last_cache_key == key and self.last_cached_file and os.path.exists(self.last_cached_file):
                    self._play_file(self.last_cached_file)
                    return
                tts = gTTS(text=text, lang=lang)
                tmpdir = tempfile.gettempdir()
                out_path = os.path.join(tmpdir, "signlink_tts.mp3")
                tts.save(out_path)
                self.last_cache_key = key
                self.last_cached_file = out_path
                self._play_file(out_path)
                return
            except Exception:
                pass

        # Fallback to pyttsx3 (offline)
        if self.engine is not None:
            try:
                # pyttsx3 might not support non-English voices well
                self.engine.say(text)
                self.engine.runAndWait()
                return
            except Exception:
                pass

    def _play_file(self, path: str):
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception:
            pass

