import json
from openai_client import client
from chunker import chunk_transcript
from feedback_store import build_examples_block, get_learned_weights, DEFAULT_WEIGHTS
from hook_finder import find_matching_hooks, find_matching_conclusions


SCORING_SYSTEM_PROMPT = """You are an expert at selecting viral Hebrew podcast clips for social media.

The transcript uses speaker labels like "Voice 1:" and "Voice 2:". Use this information — dialogue clips where speakers challenge each other often perform differently than monologue clips.

## WHAT MAKES A GREAT CLIP (in order of importance)

**1. The hook (first 3-5 seconds) must stop the scroll.**
The very first sentence must make someone stop mid-scroll. The strongest hooks are:
- A bold counter-intuitive claim: "רוב האנשים שקונים דירה עושים טעות אחת קריטית"
- A confession or personal stake: "אני לא סובל את המשפטים האלה"  
- A direct challenge: "בוא תקנה דירה עם אפס הון עצמי — זה רעיון מסוכן"
- A provocative question: "למה כולם מייעצים לך לקנות דירה כשזה לא תמיד נכון?"
If the clip opens mid-conversation or with filler, it fails.

**2. Dialogue dynamics**
When Voice 2 asks a sharp question that Voice 1 then answers — this creates natural tension and is often better than a monologue. Look for:
- Challenge/response patterns
- Moments where one speaker pushes back
- The interviewer question that unlocks a strong answer

**3. The clip must stand completely alone**
Someone who never heard this podcast must get FULL value. No references to "earlier we said", "as I mentioned", "like I explained". The clip must have its own narrative arc: setup → insight → conclusion.

**4. Concrete and specific beats abstract**
Numbers, specific examples, named mistakes, and real consequences always outperform vague advice. "תיקח הלוואה של 200,000 שקל על דירה שלא נמסרה" beats "לפעמים אנשים לוקחים הלוואות".

**5. Emotional authenticity**
Real frustration, genuine passion, or authentic vulnerability. NOT polished radio-style delivery. The moment when the speaker breaks from interview mode and speaks from the gut.

## THE 6 SCORING DIMENSIONS

**hook_strength (0-10)** — First 3-5 seconds only
- 9-10: Would stop someone mid-scroll immediately
- 7-8: Interesting opener, most people would keep watching
- 5-6: Mediocre start, needs warming up
- 1-4: Filler, greeting, or starts mid-thought

**completeness (0-10)** — Does it stand alone?
- 10: Full arc, stranger gets complete value
- 7: Good content but slightly abrupt ending or needs minor context
- 4: Missing key setup or conclusion
- 1: Fragment, only makes sense in context

**emotional_peak (0-10)** — Genuine feeling
- 10: Raw authentic emotion — frustration, passion, shock, humor
- 7: Clear conviction, engaged speaker
- 4: Informational, flat delivery
- 1: Robotic or clearly rehearsed

**value_density (0-10)** — Insight per minute
- 10: Every sentence earns its place, no filler
- 7: Good content with minor filler
- 4: Repeats itself, lots of "אה", "כן, כן", "אז"
- 1: Almost all filler

**profile_match (0-10)** — Match to stated audience and platform
- Consider: does this work as a short clip on TikTok/Instagram?
- Consider: does it speak directly to the stated target audience?

**quotability (0-10)** — One punchy caption-worthy sentence
- 10: A sentence that works standalone as a caption or thumbnail text
- 1: Nothing quotable, all context-dependent

## HARD DISQUALIFIERS — set disqualified=true if ANY apply:
- Clip starts or ends mid-sentence with no natural break
- Speaker says "כפי שאמרתי", "כמו שדיברנו", "כאמור", "קודם אמרתי" (references earlier content)
- Speaker mentions being recorded, the podcast name, or the interviewer by name as a greeting
- More than 40% of the clip is the same point repeated differently
- Contains a specific date, price, or regulation that makes it time-sensitive and will age badly

## OUTPUT FORMAT — return ONLY valid JSON, no extra text:
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
  "clip_type": "monologue" or "dialogue" or "story",
  "speaker_dynamic": "one sentence describing the speaker dynamic if relevant, else empty string",
  "best_quote": "the single most quotable sentence from the segment, in the original Hebrew",
  "hook_sentence": "the opening sentence as-is, or a suggested stronger version in Hebrew",
  "reason": "2-3 sentence honest assessment of why this clip works or doesn't"
}"""


def score_chunk(chunk_text: str, strategy, weights: dict, examples_block: str, rules_block: str = "") -> dict:
    examples_section = f"\n{examples_block}\n" if examples_block else ""
    rules_section    = f"\n{rules_block}\n"  if rules_block    else ""

    user_message = f"""{rules_section}{examples_section}
## CONTENT PROFILE
Tone: {strategy.tone}
Audience: {strategy.audience}
Platform: {strategy.platform}
Target clip length: {strategy.min_clip_seconds}-{strategy.max_clip_seconds} seconds

## TRANSCRIPT SEGMENT TO SCORE:
{chunk_text}
"""
    system = strategy.custom_prompt if getattr(strategy, 'custom_prompt', None) else SCORING_SYSTEM_PROMPT

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message}
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )
    return json.loads(response.choices[0].message.content)


def compute_final_score(scores: dict, weights: dict) -> float:
    total = 0.0
    for dim, weight in weights.items():
        total += scores.get(dim, 5) * weight
    return round(total, 2)


def build_rules_block(client_id: str) -> str:
    """Load learned scoring rules from the client profile and format for prompt injection."""
    try:
        from transcript_store import get_client
        client = get_client(client_id)
        if not client or not client.get("scoring_rules"):
            return ""
        rules = client["scoring_rules"]

        lines = ["--- LEARNED SCORING RULES FROM EXPERT COMMENTARY ---"]
        lines.append(f"General taste: {rules.get('general_taste', '')}")

        for rule in rules.get("hook_rules", []):
            lines.append(f"HOOK: {rule}")
        for rule in rules.get("disqualify_rules", []):
            lines.append(f"DISQUALIFY: {rule}")
        for rule in rules.get("value_density_rules", []):
            lines.append(f"VALUE: {rule}")
        for rule in rules.get("emotional_rules", []):
            lines.append(f"EMOTION: {rule}")
        for rule in rules.get("completeness_rules", []):
            lines.append(f"COMPLETE: {rule}")
        for pattern in rules.get("anti_patterns", []):
            lines.append(f"AVOID: {pattern}")
        lines.append("--- END RULES ---")
        lines.append("Apply these rules when scoring. They override your defaults.")
        return "\n".join(lines)
    except Exception as e:
        print(f"[rules] load failed: {e}")
        return ""


def select_clips(transcript: str, strategy) -> list:
    user_id        = getattr(strategy, 'user_id', 'default') or 'default'
    weights        = get_learned_weights(user_id)
    examples_block = build_examples_block(user_id, n=8)
    rules_block    = build_rules_block(user_id)

    # Chunk by actual time using strategy settings
    min_sec = getattr(strategy, 'min_clip_seconds', 30)
    max_sec = getattr(strategy, 'max_clip_seconds', 60)
    chunks  = chunk_transcript(transcript, min_seconds=min_sec, max_seconds=max_sec)

    results = []

    print(f"\n→ Scoring {len(chunks)} chunks for user '{user_id}'")
    print(f"→ Time range: {min_sec}-{max_sec}s | Weights: { {k: round(v,2) for k,v in weights.items()} }")

    for chunk in chunks:
        try:
            analysis = score_chunk(chunk["text"], strategy, weights, examples_block, rules_block)
        except Exception as e:
            print(f"  Chunk {chunk['index']} error: {e}")
            continue

        scores       = analysis.get("scores", {})
        disqualified = analysis.get("disqualified", False)
        final_score  = compute_final_score(scores, weights)
        start_time   = chunk.get("start_time", 0.0)
        end_time     = chunk.get("end_time",   0.0)

        # ── Hook/conclusion suggestions from vector DB ────────────────────
        suggested_hooks       = []
        suggested_conclusions = []
        video_id              = getattr(strategy, 'video_id', None)

        hook_score = scores.get("hook_strength", 0)
        end_score  = scores.get("completeness", 0)

        # If content is good but hook is weak → search past hooks
        if not disqualified and final_score >= 5.0 and hook_score <= 5:
            try:
                suggested_hooks = find_matching_hooks(
                    client_id=user_id,
                    clip_text=chunk["text"],
                    exclude_video=video_id,
                    top_k=2,
                )
            except Exception as e:
                print(f"  Hook search failed: {e}")

        # If content is good but ending is weak → search past conclusions
        if not disqualified and final_score >= 5.0 and end_score <= 5:
            try:
                suggested_conclusions = find_matching_conclusions(
                    client_id=user_id,
                    clip_text=chunk["text"],
                    exclude_video=video_id,
                    top_k=2,
                )
            except Exception as e:
                print(f"  Conclusion search failed: {e}")

        result = {
            "chunk_index":            chunk["index"],
            "text":                   chunk["text"],
            "scores":                 scores,
            "final_score":            final_score,
            "disqualified":           disqualified,
            "disqualify_reason":      analysis.get("disqualify_reason", ""),
            "clip_type":              analysis.get("clip_type", ""),
            "speaker_dynamic":        analysis.get("speaker_dynamic", ""),
            "best_quote":             analysis.get("best_quote", ""),
            "hook_sentence":          analysis.get("hook_sentence", ""),
            "reason":                 analysis.get("reason", ""),
            "start_time":             start_time,
            "end_time":               end_time,
            "score":                  round(final_score),
            "contains_hook":          scores.get("hook_strength", 0) >= 6,
            "suggested_hooks":        suggested_hooks,
            "suggested_conclusions":  suggested_conclusions,
        }

        status = "DISQ" if disqualified else f"score={final_score:.1f}"
        print(f"  Chunk {chunk['index']}: {start_time:.1f}s-{end_time:.1f}s | {status} | {analysis.get('clip_type','')}")
        results.append(result)

    # Filter: not disqualified, score >= 5.5, return top 5
    filtered = [r for r in results if not r["disqualified"] and r["final_score"] >= 5.5]
    filtered.sort(key=lambda x: x["final_score"], reverse=True)
    top = filtered[:5]
    print(f"→ Selected {len(top)} clips from {len(results)} scored\n")
    return top