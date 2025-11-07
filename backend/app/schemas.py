from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class Preferences(BaseModel):
    target_language: Literal["en", "hi", "ta", "te", "bn"] = "en"
    domain_mode: Literal["general", "education", "healthcare", "gov", "workplace"] = "general"
    tts_enabled: bool = False
    romanize: bool = False
    telemetry_opt_in: bool = False
    high_contrast: bool = False

    @field_validator("target_language")
    @classmethod
    def _validate_lang(cls, value: str) -> str:  # noqa: D401
        if value not in {"en", "hi", "ta", "te", "bn"}:
            raise ValueError("Unsupported language")
        return value


class InitMessage(BaseModel):
    type: Literal["init"]
    prefs: Preferences


class FrameMessage(BaseModel):
    type: Literal["frame"]
    window_id: str = Field(..., min_length=1, max_length=16)
    seq: int = Field(..., ge=0)
    ts: int = Field(..., ge=0)
    image: str = Field(..., description="Base64-encoded JPEG payload")


ClientMessage = Union[InitMessage, FrameMessage]


class PartialTextEvent(BaseModel):
    type: Literal["partial_text"] = "partial_text"
    lang: str
    text: str
    conf: float
    window_id: str
    timestamp: int
    window_revision: int


class FinalTextEvent(BaseModel):
    type: Literal["final_text"] = "final_text"
    lang: str
    text: str
    conf: float
    start_ms: int
    end_ms: int
    segment_id: str


class AudioEvent(BaseModel):
    type: Literal["audio"] = "audio"
    lang: str
    audio_url: str
    segment_id: str


class HintEvent(BaseModel):
    type: Literal["hint"] = "hint"
    message: str
    hint_kind: Literal["occlusion", "low_confidence", "environment"]


class NoticeEvent(BaseModel):
    type: Literal["notice"] = "notice"
    message: str
    notice_type: Literal["init_ack", "info", "error"]


ServerEvent = Union[
    PartialTextEvent,
    FinalTextEvent,
    AudioEvent,
    HintEvent,
    NoticeEvent,
]
