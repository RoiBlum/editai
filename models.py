from pydantic import BaseModel
from typing import Optional


class ClipStrategy(BaseModel):
    tone: str
    audience: str
    platform: str
    min_clip_seconds: int
    max_clip_seconds: int
    require_hook: bool
    custom_prompt: Optional[str] = None
    user_id: Optional[str] = "default"


class TranscriptRequest(BaseModel):
    transcript: str
    strategy: ClipStrategy


class FeedbackRequest(BaseModel):
    user_id: str
    clip_text: str
    scores: dict
    verdict: str          # "good" or "bad"
    comment: Optional[str] = None