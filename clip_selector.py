import json
from openai_client import client
from chunker import chunk_transcript
from feedback_store import build_examples_block, get_learned_weights, DEFAULT_WEIGHTS


# ─────────────────────────────────────────────
# The 6-dimension scoring prompt
# ─────────────────────────────────────────────

SCORING_SYSTEM_PROMPT = """You are an expert at selecting viral Hebrew podcast clips for social media marketing.

Your job is to score a transcript segment on 6 dimensions. Be strict and honest — most segments are NOT good clips.

## THE 6 DIMENSIONS

**1. hook_strength (0-10)**
Judge only the FIRST 2-3 sentences. Would someone stop scrolling?
- 9-10: Bold claim, shocking number, direct challenge to a belief, or confession
- 7-8: A question or relatable situation that pulls you in
- 5-6: Starts mid-thought, needs context to understand
- 1-4: Filler, greetings, setup with no visible payoff ("אז כן, אנחנו היום נדבר על...")

**2. completeness (0-10)**
Does this segment stand alone without watching the rest?
- 10: Clear beginning + middle + end. A stranger gets full value
- 5: Good content but missing setup OR ending is cut off
- 1: Only makes sense if you saw what came before

**3. emotional_peak (0-10)**
Is there a moment of genuine strong feeling?
- Humor, anger, vulnerability, excitement, surprise all score high
- Flat monotone informational delivery scores low
- Multiple emotional beats score higher than one

**4. value_density (0-10)**
How much useful/interesting content per minute?
- High: one concrete insight delivered tightly, no filler
- Low: same point repeated, lots of "כאילו", tangents, "נכון?" filler
- Penalize heavily if less than 40% of words carry meaning

**5. profile_match (0-10)**
How directly relevant is this to the stated audience and goals?
- 10: This is exactly what this audience cares about
- 5: Related but indirect
- 1: Off-topic for this specific profile

**6. quotability (0-10)**
Is there a single punchy sentence that could be a caption or thumbnail?
- 10: "אם אתה לא מפחד מהחלטה שלך, כנראה שהיא לא גדולה מספיק"
- 5: Has a good point but no quotable single sentence
- 1: No memorable line at all

## HARD DISQUALIFIERS
If ANY of these are true, set disqualified=true:
- Segment starts or ends mid-sentence with no natural break
- Speaker references the podcast, a previous episode, or that they are being recorded
- More than 30% is crosstalk or interruption
- Contains a specific date or price that makes it time-sensitive
- Shorter than min_seconds or longer than max_seconds

## OUTPUT FORMAT
Return ONLY valid JSON, no extra text:
{
  "scores": {
    "hook_strength": 0-10,
    "completeness": 0-10,
    "emotional_peak": 0-10,
    "value_density": 0-10,
    "profile_match": 0-10,
    "quotability": 0-10
  },
  "disqualified": true/false,
  "disqualify_reason": "reason or empty string",
  "best_quote": "the most quotable sentence from the segment, in the original language",
  "hook_sentence": "the opening sentence rewritten to be stronger (in Hebrew)",
  "reason": "2-sentence explanation of overall assessment in English"
}"""


# ─────────────────────────────────────────────
# Score a single chunk
# ─────────────────────────────────────────────

def score_chunk(chunk_text: str, strategy, weights: dict, examples_block: str) -> dict:

    # Build the user message
    examples_section = f"\n{examples_block}\n" if examples_block else ""

    user_message = f"""{examples_section}
## CLIENT PROFILE
Tone: {strategy.tone}
Audience: {strategy.audience}
Platform: {strategy.platform}
Clip length target: {strategy.min_clip_seconds}-{strategy.max_clip_seconds} seconds
Min seconds: {strategy.min_clip_seconds}
Max seconds: {strategy.max_clip_seconds}

## TRANSCRIPT SEGMENT TO SCORE:
{chunk_text}
"""

    # Allow custom prompt override to replace system prompt
    system = strategy.custom_prompt if getattr(strategy, 'custom_prompt', None) else SCORING_SYSTEM_PROMPT

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"},
        temperature=0.2   # low temp for consistent scoring
    )

    return json.loads(response.choices[0].message.content)


# ─────────────────────────────────────────────
# Compute weighted final score
# ─────────────────────────────────────────────

def compute_final_score(scores: dict, weights: dict) -> float:
    total = 0.0
    for dim, weight in weights.items():
        total += scores.get(dim, 5) * weight
    return round(total, 2)


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def select_clips(transcript: str, strategy) -> list:
    user_id = getattr(strategy, 'user_id', 'default') or 'default'

    # Load personalized weights (learned from past feedback)
    weights = get_learned_weights(user_id)

    # Load few-shot examples from past feedback
    examples_block = build_examples_block(user_id, n=6)

    chunks = chunk_transcript(transcript)
    results = []

    print(f"\n→ Scoring {len(chunks)} chunks for user '{user_id}'")
    print(f"→ Weights: {weights}")
    if examples_block:
        print(f"→ Injecting {examples_block.count('APPROVED') + examples_block.count('REJECTED')} feedback examples")

    for chunk in chunks:
        try:
            analysis = score_chunk(chunk["text"], strategy, weights, examples_block)
        except Exception as e:
            print(f"  Chunk {chunk['index']} error: {e}")
            continue

        scores = analysis.get("scores", {})
        disqualified = analysis.get("disqualified", False)
        final_score = compute_final_score(scores, weights)

        result = {
            "chunk_index": chunk["index"],
            "text": chunk["text"],
            "scores": scores,
            "final_score": final_score,
            "disqualified": disqualified,
            "disqualify_reason": analysis.get("disqualify_reason", ""),
            "best_quote": analysis.get("best_quote", ""),
            "hook_sentence": analysis.get("hook_sentence", ""),
            "reason": analysis.get("reason", ""),
            # Legacy fields for HTML compatibility
            "score": round(final_score),
            "contains_hook": scores.get("hook_strength", 0) >= 6,
        }

        print(f"  Chunk {chunk['index']}: final={final_score:.1f} | hook={scores.get('hook_strength')} | complete={scores.get('completeness')} | {'DISQUALIFIED' if disqualified else 'ok'}")
        results.append(result)

    # Filter: not disqualified, final score >= 5.5
    filtered = [r for r in results if not r["disqualified"] and r["final_score"] >= 5.5]
    filtered.sort(key=lambda x: x["final_score"], reverse=True)

    top = filtered[:5]
    print(f"→ Selected {len(top)} clips from {len(filtered)} passing chunks\n")
    return top