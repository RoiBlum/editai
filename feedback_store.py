import os
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ── Connect to Supabase ───────────────────────
_url = os.getenv("SUPABASE_URL")
_key = os.getenv("SUPABASE_KEY")

if not _url or not _key:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

supabase = create_client(_url, _key)


# ─────────────────────────────────────────────
# Save feedback
# ─────────────────────────────────────────────

def save_feedback(user_id: str, clip_text: str, scores: dict, verdict: str, comment: str = None):
    supabase.table("feedback").insert({
        "user_id": user_id,
        "clip_text": clip_text[:600],
        "scores": scores,
        "verdict": verdict,
        "comment": comment or "",
        "created_at": datetime.now().isoformat()
    }).execute()


# ─────────────────────────────────────────────
# Load all feedback for a user
# ─────────────────────────────────────────────

def load_feedback(user_id: str) -> list:
    res = supabase.table("feedback") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at") \
        .execute()
    return res.data or []


# ─────────────────────────────────────────────
# Build few-shot examples block for prompt injection
# ─────────────────────────────────────────────

def build_examples_block(user_id: str, n: int = 6) -> str:
    entries = load_feedback(user_id)
    if not entries:
        return ""

    good = [e for e in entries if e["verdict"] == "good"]
    bad  = [e for e in entries if e["verdict"] == "bad"]

    selected = good[-(n//2):] + bad[-(n//2):]
    if not selected:
        return ""

    lines = ["--- PAST FEEDBACK EXAMPLES (learn from these) ---"]
    for e in selected:
        label = "✅ APPROVED" if e["verdict"] == "good" else "❌ REJECTED"
        scores_str = ", ".join([f"{k}: {v}" for k, v in (e.get("scores") or {}).items()])
        comment = f' | User note: "{e["comment"]}"' if e.get("comment") else ""
        lines.append(
            f'{label} | Scores: [{scores_str}]{comment}\n'
            f'Text preview: "{(e.get("clip_text") or "")[:200]}"\n'
        )
    lines.append("--- END EXAMPLES ---\n")
    lines.append("Use the above examples to calibrate your scoring for this user.")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Compute learned weights from feedback history
# ─────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "hook_strength":   0.25,
    "completeness":    0.20,
    "emotional_peak":  0.20,
    "value_density":   0.15,
    "profile_match":   0.15,
    "quotability":     0.05,
}

MIN_SAMPLES_TO_LEARN = 10


def get_learned_weights(user_id: str) -> dict:
    entries = load_feedback(user_id)
    if len(entries) < MIN_SAMPLES_TO_LEARN:
        return DEFAULT_WEIGHTS.copy()

    dim_keys = list(DEFAULT_WEIGHTS.keys())
    approved_avgs = {k: [] for k in dim_keys}
    rejected_avgs = {k: [] for k in dim_keys}

    for e in entries:
        scores = e.get("scores") or {}
        target = approved_avgs if e["verdict"] == "good" else rejected_avgs
        for k in dim_keys:
            if k in scores:
                try:
                    target[k].append(float(scores[k]))
                except (ValueError, TypeError):
                    pass

    separations = {}
    for k in dim_keys:
        a = sum(approved_avgs[k]) / len(approved_avgs[k]) if approved_avgs[k] else 5.0
        r = sum(rejected_avgs[k]) / len(rejected_avgs[k]) if rejected_avgs[k] else 5.0
        separations[k] = max(0.01, a - r)

    total_sep = sum(separations.values())
    learned = {k: separations[k] / total_sep for k in dim_keys}

    blended = {k: (DEFAULT_WEIGHTS[k] * 0.5 + learned[k] * 0.5) for k in dim_keys}
    total = sum(blended.values())
    return {k: round(v / total, 4) for k, v in blended.items()}


def get_feedback_stats(user_id: str) -> dict:
    entries = load_feedback(user_id)
    total = len(entries)
    good  = sum(1 for e in entries if e["verdict"] == "good")
    return {"total": total, "good": good, "bad": total - good}