from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from models import TranscriptRequest, FeedbackRequest, SaveTranscriptRequest
from clip_selector import select_clips
from feedback_store import save_feedback, get_feedback_stats, get_learned_weights
from transcript_store import (
    get_or_create_client, list_clients, get_client,
    save_transcript, get_client_transcripts
)
from component_extractor import extract_and_store
from commentary_processor import process_commentary_recording

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Clip selection ─────────────────────────────────────────────────────────────

@app.post("/select-clips")
def select_clips_endpoint(request: TranscriptRequest):
    clips = select_clips(request.transcript, request.strategy)
    return {"clips": clips}


# ── Feedback ───────────────────────────────────────────────────────────────────

@app.post("/feedback")
def feedback_endpoint(request: FeedbackRequest):
    save_feedback(
        user_id=request.user_id,
        clip_text=request.clip_text,
        scores=request.scores,
        verdict=request.verdict,
        comment=request.comment,
        overall_score=request.overall_score,
        start_score=request.start_score,
        middle_score=request.middle_score,
        end_score=request.end_score,
        message_score=request.message_score,
        marketing_purpose=request.marketing_purpose,
        message_captured=request.message_captured,
        missing_content=request.missing_content,
        overall_comment=request.overall_comment,
        start_comment=request.start_comment,
        middle_comment=request.middle_comment,
        end_comment=request.end_comment,
        message_comment=request.message_comment,
    )
    stats   = get_feedback_stats(request.user_id)
    weights = get_learned_weights(request.user_id)
    return {"saved": True, "stats": stats, "current_weights": weights}


@app.get("/feedback/stats/{user_id}")
def feedback_stats(user_id: str):
    return {
        "stats":   get_feedback_stats(user_id),
        "weights": get_learned_weights(user_id)
    }


# ── Clients ────────────────────────────────────────────────────────────────────

@app.get("/clients")
def get_clients():
    """List all clients."""
    return {"clients": list_clients()}


@app.get("/clients/{client_id}")
def get_client_endpoint(client_id: str):
    """Get a single client with their transcript history."""
    client = get_client(client_id)
    if not client:
        return {"error": "Client not found"}
    transcripts = get_client_transcripts(client_id)
    return {"client": client, "transcripts": transcripts}


@app.post("/clients")
def create_client(body: dict):
    """Create or get a client."""
    client_id = body.get("client_id", "").strip()
    name      = body.get("name", "").strip()
    if not client_id:
        return {"error": "client_id is required"}
    client = get_or_create_client(client_id, name)
    return {"client": client}


# ── Transcripts ────────────────────────────────────────────────────────────────

@app.post("/transcripts/save")
async def save_transcript_endpoint(
    request: SaveTranscriptRequest,
    background_tasks: BackgroundTasks
):
    """
    Called by the frontend after transcription completes.
    Saves the transcript and triggers background component extraction.
    """
    # Ensure client exists
    get_or_create_client(request.client_id)

    # Save transcript
    transcript_id = save_transcript(
        client_id=request.client_id,
        video_id=request.video_id,
        filename=request.filename,
        duration_sec=request.duration_sec,
        speakers=request.speakers,
        full_text=request.full_text,
        raw_segments=request.raw_segments,
    )

    # Extract components in background (takes 30-60 seconds, don't block)
    background_tasks.add_task(
        extract_and_store,
        client_id=request.client_id,
        video_id=request.video_id,
        transcript_id=transcript_id,
        full_text=request.full_text,
    )

    return {
        "saved":          True,
        "transcript_id":  transcript_id,
        "extracting":     True,
        "message":        "Transcript saved. Component extraction running in background."
    }


@app.get("/transcripts/{client_id}")
def get_transcripts(client_id: str):
    return {"transcripts": get_client_transcripts(client_id)}

# ── Commentary processing ──────────────────────────────────────────────────────

@app.post("/commentary/process")
async def process_commentary(body: dict, background_tasks: BackgroundTasks):
    """
    Process a commentary recording transcript.
    The transcript must already have Voice 1 / Voice 2 labels from diarization.

    Body:
      client_id:     str
      video_id:      str
      transcript:    str  — full transcript with Voice N: labels
      extract_rules: bool — whether to extract scoring rules (default False)
    """
    client_id     = body.get("client_id", "default")
    video_id      = body.get("video_id",  "")
    transcript    = body.get("transcript", "")
    extract_rules = body.get("extract_rules", False)

    if not transcript:
        return {"error": "transcript is required"}

    # Run in background — takes 2-5 minutes for a long recording
    background_tasks.add_task(
        process_commentary_recording,
        transcript=transcript,
        client_id=client_id,
        video_id=video_id,
        extract_rules=extract_rules,
    )

    return {
        "started": True,
        "message": "עיבוד פרשנות התחיל ברקע. ייקח 2-5 דקות.",
        "client_id": client_id,
    }


@app.get("/commentary/rules/{client_id}")
def get_scoring_rules(client_id: str):
    """Return the learned scoring rules for a client."""
    from transcript_store import get_client
    client = get_client(client_id)
    if not client:
        return {"error": "client not found"}
    return {
        "client_id": client_id,
        "rules": client.get("scoring_rules") or {},
        "has_rules": bool(client.get("scoring_rules"))
    }