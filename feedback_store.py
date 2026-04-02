import os
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

_url = os.getenv("SUPABASE_URL")
_key = os.getenv("SUPABASE_KEY")
if not _url or not _key:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

supabase = create_client(_url, _key)


# ── Save feedback ─────────────────────────────────────────────────────────────

def save_feedback(
    user_id: str,
    clip_text: str,
    scores: dict,
    verdict: str,
    comment: str = None,
    overall_score: int = None,
    start_score: int = None,
    middle_score: int = None,
    end_score: int = None,
    message_score: int = None,
    marketing_purpose: str = None,
    message_captured: bool = None,
    missing_content: str = None,
    overall_comment: str = None,
    start_comment:   str = None,
    middle_comment:  str = None,
    end_comment:     str = None,
    message_comment: str = None,
):
    supabase.table("feedback").insert({
        "user_id":           user_id,
        "clip_text":         clip_text[:600],
        "scores":            scores,
        "verdict":           verdict,
        "comment":           comment or "",
        "overall_score":     overall_score,
        "start_score":       start_score,
        "middle_score":      middle_score,
        "end_score":         end_score,
        "message_score":     message_score,
        "marketing_purpose": marketing_purpose,
        "message_captured":  message_captured,
        "missing_content":   missing_content,
        "overall_comment":   overall_comment,
        "start_comment":     start_comment,
        "middle_comment":    middle_comment,
        "end_comment":       end_comment,
        "message_comment":   message_comment,
        "created_at":        datetime.now().isoformat()
    }).execute()


# ── Load feedback ─────────────────────────────────────────────────────────────

def load_feedback(user_id: str) -> list:
    res = supabase.table("feedback") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at") \
        .execute()
    return res.data or []


# ── Build rich few-shot examples for prompt injection ─────────────────────────

def build_examples_block(user_id: str, n: int = 8) -> str:
    entries = load_feedback(user_id)
    if not entries:
        return ""

    # Sort by overall_score descending — best examples first
    def entry_score(e):
        if e.get("overall_score") is not None:
            return int(e["overall_score"])
        return 7 if e["verdict"] == "good" else 3

    entries_scored = sorted(entries, key=entry_score, reverse=True)

    # Take top n//2 approved and bottom n//2 rejected
    good = [e for e in entries_scored if entry_score(e) >= 6]
    bad  = [e for e in entries_scored if entry_score(e) <  6]

    selected = good[:n//2] + bad[:n//2]
    if not selected:
        return ""

    lines = ["--- PAST FEEDBACK FROM THIS USER (calibrate your scoring to match) ---\n"]

    for e in selected:
        overall = e.get("overall_score")
        verdict = "✅ APPROVED" if entry_score(e) >= 6 else "❌ REJECTED"

        # Build a rich description of why this clip was rated this way
        reasons = []

        if e.get("overall_comment"):
            reasons.append(f"Overall: {e['overall_comment']}")
        if e.get("start_comment"):
            reasons.append(f"Start ({e.get('start_score', '?')}/10): {e['start_comment']}")
        if e.get("middle_comment"):
            reasons.append(f"Middle ({e.get('middle_score', '?')}/10): {e['middle_comment']}")
        if e.get("end_comment"):
            reasons.append(f"End ({e.get('end_score', '?')}/10): {e['end_comment']}")
        if e.get("message_comment"):
            reasons.append(f"Message ({e.get('message_score', '?')}/10): {e['message_comment']}")
        if e.get("missing_content"):
            reasons.append(f"Missing from clip: {e['missing_content']}")
        if e.get("marketing_purpose"):
            reasons.append(f"Purpose: {e['marketing_purpose']}")
        if e.get("message_captured") is not None:
            captured = "Yes" if e["message_captured"] else "No"
            reasons.append(f"Full message captured: {captured}")
        if e.get("comment"):
            reasons.append(f"General note: {e['comment']}")

        # Fall back to old-style comment if no rich data
        if not reasons and e.get("comment"):
            reasons.append(e["comment"])

        scores_str = ""
        if e.get("scores"):
            scores_str = " | ".join([f"{k}: {v}" for k, v in e["scores"].items()])

        block = [
            f"{verdict} | Overall score: {overall or '?'}/10",
        ]
        if scores_str:
            block.append(f"AI scores: [{scores_str}]")
        if reasons:
            block.append("User reasoning:")
            for r in reasons:
                block.append(f"  - {r}")
        block.append(f'Clip text: "{(e.get("clip_text") or "")[:250]}"')
        block.append("")

        lines.append("\n".join(block))

    lines.append("--- END EXAMPLES ---")
    lines.append("Use the above to understand exactly what this user considers good vs bad. Match their taste.")
    return "\n".join(lines)


# ── Learned weights ───────────────────────────────────────────────────────────

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
        scores  = e.get("scores") or {}
        overall = e.get("overall_score")
        is_good = (int(overall) >= 6) if overall is not None else (e["verdict"] == "good")
        target  = approved_avgs if is_good else rejected_avgs
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
    learned   = {k: separations[k] / total_sep for k in dim_keys}
    blended   = {k: (DEFAULT_WEIGHTS[k] * 0.5 + learned[k] * 0.5) for k in dim_keys}
    total     = sum(blended.values())
    return {k: round(v / total, 4) for k, v in blended.items()}


def get_feedback_stats(user_id: str) -> dict:
    entries = load_feedback(user_id)
    total = len(entries)
    good  = sum(1 for e in entries if e.get("overall_score", 0) >= 6 or e["verdict"] == "good")
    return {"total": total, "good": good, "bad": total - good}