"""
component_extractor.py
After transcription, extract hooks / conclusions / messages from the full
transcript and store them as embeddings in clip_components.
Also updates the client profile with patterns found across videos.
"""
import os
import json
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv
from transcript_store import update_client_profile, mark_components_extracted, get_client

load_dotenv()

openai   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

EXTRACTION_PROMPT = """You are analyzing a Hebrew podcast transcript to extract reusable components for social media clip creation.

Extract the following from the transcript:

1. **hooks** — Sentences that would make someone STOP scrolling. Strong opinions, shocking claims, counter-intuitive statements, confessions, strong emotional openings. Max 1-2 sentences each.

2. **conclusions** — Clear endings that complete a message. Actionable takeaways, strong closing statements, memorable summaries. Max 2-3 sentences.

3. **messages** — The core point or insight of a segment. What is the speaker actually trying to say? Distilled to 1-2 sentences.

4. **creator_patterns** — Analyze the whole transcript and describe:
   - main_topics: what subjects does this creator talk about?
   - tone: how do they speak? (direct, emotional, educational, etc.)
   - target_audience: who are they talking to?
   - hook_style: how do they typically open strong moments?
   - recurring_phrases: any phrases they use often?

For hooks and conclusions, include the EXACT timestamp from the transcript line.

Return ONLY valid JSON:
{
  "hooks": [
    {
      "text": "exact text from transcript",
      "start_time": 0.0,
      "end_time": 0.0,
      "speaker": "Voice 1",
      "context": "1-2 surrounding lines for context",
      "why_strong": "one sentence explaining why this is a good hook"
    }
  ],
  "conclusions": [
    {
      "text": "exact text",
      "start_time": 0.0,
      "end_time": 0.0,
      "speaker": "Voice 1",
      "context": "surrounding lines",
      "why_strong": "why this is a good conclusion"
    }
  ],
  "messages": [
    {
      "text": "distilled message",
      "start_time": 0.0,
      "end_time": 0.0,
      "topic": "main topic of this message",
      "speaker": "Voice 1"
    }
  ],
  "creator_patterns": {
    "main_topics": ["topic1", "topic2"],
    "tone": "description",
    "target_audience": "description",
    "hook_style": "description",
    "recurring_phrases": ["phrase1", "phrase2"]
  }
}"""


def embed_text(text: str) -> list:
    """Embed a single text with text-embedding-3-small."""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def extract_and_store(
    client_id:     str,
    video_id:      str,
    transcript_id: int,
    full_text:     str,
) -> dict:
    """
    Extract components from the transcript, embed them, and store in Supabase.
    Returns summary of what was extracted.
    """
    print(f"[extractor] extracting components for client={client_id} transcript={transcript_id}")

    # Truncate if too long — GPT-4o has a context limit
    # For very long podcasts, process in 2 halves
    max_chars = 80000
    if len(full_text) > max_chars:
        print(f"[extractor] transcript too long ({len(full_text)} chars), truncating to {max_chars}")
        full_text = full_text[:max_chars] + "\n[TRANSCRIPT TRUNCATED]"

    # ── Extract with GPT-4o ───────────────────────────────────────────────────
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user",   "content": f"TRANSCRIPT:\n{full_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        extracted = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[extractor] GPT extraction failed: {e}")
        return {"error": str(e)}

    hooks       = extracted.get("hooks", [])
    conclusions = extracted.get("conclusions", [])
    messages    = extracted.get("messages", [])
    patterns    = extracted.get("creator_patterns", {})

    print(f"[extractor] found: {len(hooks)} hooks, {len(conclusions)} conclusions, {len(messages)} messages")

    # ── Store each component with its embedding ───────────────────────────────
    stored_count = 0
    components_to_insert = []

    for component_type, items in [
        ("hook",       hooks),
        ("conclusion", conclusions),
        ("message",    messages),
    ]:
        for item in items:
            text = item.get("text", "").strip()
            if not text or len(text) < 10:
                continue
            try:
                embedding = embed_text(text)
                components_to_insert.append({
                    "client_id":      client_id,
                    "video_id":       video_id,
                    "transcript_id":  transcript_id,
                    "component_type": component_type,
                    "text":           text,
                    "context":        item.get("context", ""),
                    "start_time":     item.get("start_time"),
                    "end_time":       item.get("end_time"),
                    "speaker":        item.get("speaker"),
                    "embedding":      embedding,
                    "metadata": {
                        "why_strong":    item.get("why_strong", ""),
                        "topic":         item.get("topic", ""),
                        "video_id":      video_id,
                    }
                })
                stored_count += 1
            except Exception as e:
                print(f"[extractor] embed failed for {component_type}: {e}")

    if components_to_insert:
        supabase.table("clip_components").insert(components_to_insert).execute()
        print(f"[extractor] stored {stored_count} components")

    # ── Update client profile with patterns ───────────────────────────────────
    if patterns:
        existing = get_client(client_id) or {}
        existing_topics = existing.get("main_topics") or []
        new_topics      = patterns.get("main_topics", [])

        # Merge topics lists
        merged_topics = list(set(existing_topics + new_topics))

        update_client_profile(client_id, {
            "main_topics":      merged_topics,
            "tone":             patterns.get("tone", existing.get("tone", "")),
            "target_audience":  patterns.get("target_audience", existing.get("target_audience", "")),
            "hook_style":       patterns.get("hook_style", existing.get("hook_style", "")),
        })
        print(f"[extractor] updated client profile for {client_id}")

    # Mark transcript as processed
    mark_components_extracted(transcript_id)

    return {
        "hooks":       len(hooks),
        "conclusions": len(conclusions),
        "messages":    len(messages),
        "stored":      stored_count,
        "patterns":    patterns,
    }