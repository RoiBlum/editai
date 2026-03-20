from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import TranscriptRequest, FeedbackRequest
from clip_selector import select_clips
from feedback_store import save_feedback, get_feedback_stats, get_learned_weights

app = FastAPI()

# Allow the HTML file to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/select-clips")
def select_clips_endpoint(request: TranscriptRequest):
    clips = select_clips(request.transcript, request.strategy)
    return {"clips": clips}


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
    )
    stats = get_feedback_stats(request.user_id)
    weights = get_learned_weights(request.user_id)
    return {
        "saved": True,
        "stats": stats,
        "current_weights": weights
    }


@app.get("/feedback/stats/{user_id}")
def feedback_stats(user_id: str):
    return {
        "stats": get_feedback_stats(user_id),
        "weights": get_learned_weights(user_id)
    }