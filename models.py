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
    verdict: str                          # "good" | "bad" (kept for compat)
    comment: Optional[str] = None
    # ── New detailed feedback fields ──
    overall_score:     Optional[int]  = None   # 1-10
    start_score:       Optional[int]  = None   # 1-10
    middle_score:      Optional[int]  = None   # 1-10
    end_score:         Optional[int]  = None   # 1-10
    message_score:     Optional[int]  = None   # 1-10
    marketing_purpose: Optional[str]  = None   # חשיפה | בניית אמון | מיתוג ומיצוב
    message_captured:  Optional[bool] = None   # was full message captured?
    missing_content:   Optional[str]  = None   # what was missing
    # ── Per-dimension comments ──
    overall_comment: Optional[str] = None
    start_comment:   Optional[str] = None
    middle_comment:  Optional[str] = None
    end_comment:     Optional[str] = None
    message_comment: Optional[str] = None