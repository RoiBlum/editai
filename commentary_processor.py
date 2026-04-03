"""
commentary_processor.py

Your brother records himself listening to a podcast and giving live commentary.
The recording has two voices interleaved:
  - Voice 1 = podcast content (the guest/host)
  - Voice 2 = your brother's commentary about the content

This processor:
1. Splits the transcript by voice
2. Aligns each commentary segment to the podcast segment it follows
3. Uses GPT-4o to extract structured judgment from the commentary
4. Stores results as labeled training data in Supabase feedback table
   (identical format to manual UI feedback — same system benefits)

The key insight: your brother doesn't rate clips on a scale.
He explains what he sees, what he feels, what's missing or great.
That natural language reasoning is FAR more useful than a number.
"""

import os
import json
import re
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
openai   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# ── Step 1: Split transcript by speaker ───────────────────────────────────────

def split_by_speaker(transcript: str) -> list:
    """
    Parse transcript lines like:
      [12.4s --> 18.1s] Voice 1: some content here
      [18.5s --> 24.0s] Voice 2: my commentary here

    Returns list of:
      { start, end, speaker, text }
    """
    segments = []
    pattern  = re.compile(r'\[(\d+\.?\d*)s\s*-->\s*(\d+\.?\d*)s\]\s*(Voice \d+):\s*(.*)')

    for line in transcript.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            segments.append({
                "start":   float(m.group(1)),
                "end":     float(m.group(2)),
                "speaker": m.group(3),
                "text":    m.group(4).strip()
            })

    return segments


def identify_commentator_voice(segments: list) -> str:
    """
    Figure out which Voice N is your brother (the commentator).
    The commentator voice contains meta-phrases like:
    'אני אוהב', 'אני לא אוהב', 'זה טוב', 'זה לא מעניין',
    'פה יש', 'בואו נמשיך', 'מה שאני מחפש'
    """
    meta_phrases = [
        'אני אוהב', 'אני לא אוהב', 'זה טוב', 'זה לא מעניין',
        'לא מעניין', 'מעניין', 'בואו נמשיך', 'מה שאני מחפש',
        'מה שאני לא אוהב', 'נמשיך להקשיב', 'חזרה', 'פה יש',
        'זה נשמע', 'נחמד', 'מגעיל', 'סייל', 'זה מוצלח'
    ]

    voice_scores = {}
    for seg in segments:
        v = seg["speaker"]
        if v not in voice_scores:
            voice_scores[v] = 0
        for phrase in meta_phrases:
            if phrase in seg["text"]:
                voice_scores[v] += 1

    if not voice_scores:
        return "Voice 2"  # default assumption

    return max(voice_scores, key=voice_scores.get)


# ── Step 2: Align commentary to podcast content ───────────────────────────────

def build_aligned_pairs(segments: list, commentator_voice: str) -> list:
    """
    For each commentary segment, find the podcast content that immediately
    preceded it. Build pairs:
      {
        "podcast_text":     "the actual podcast content",
        "podcast_start":    float,
        "podcast_end":      float,
        "commentary_text":  "brother's reaction",
        "commentary_start": float,
        "commentary_end":   float,
      }
    """
    content_voice = [s for s in segments if s["speaker"] != commentator_voice]
    comment_segs  = [s for s in segments if s["speaker"] == commentator_voice]

    pairs = []
    for comment in comment_segs:
        # Find all podcast segments that ended before this comment started
        preceding = [
            s for s in content_voice
            if s["end"] <= comment["start"] + 2.0   # small tolerance
        ]
        if not preceding:
            continue

        # Take the last 3-5 podcast segments before the comment
        # (this is the content being reacted to)
        window = preceding[-5:]
        podcast_text  = " ".join(s["text"] for s in window)
        podcast_start = window[0]["start"]
        podcast_end   = window[-1]["end"]

        pairs.append({
            "podcast_text":     podcast_text,
            "podcast_start":    podcast_start,
            "podcast_end":      podcast_end,
            "commentary_text":  comment["text"],
            "commentary_start": comment["start"],
            "commentary_end":   comment["end"],
        })

    return pairs


# ── Step 3: Extract structured judgment from commentary ───────────────────────

EXTRACTION_PROMPT = """You are analyzing a Hebrew content expert's live commentary on podcast content.
The expert was listening to a podcast and pausing to give his reaction.

Your job: extract structured judgment from his commentary.

The commentary is in Hebrew and may be casual, fragmentary, or spoken-language style.
Read the CONTENT (what was said in the podcast) and the COMMENTARY (expert's reaction).

Return ONLY valid JSON:
{
  "verdict": "good" or "bad" or "neutral",
  "overall_score": 1-10,
  "what_he_liked": "what specifically he liked, in 1-2 sentences, or empty string",
  "what_he_disliked": "what specifically he disliked, in 1-2 sentences, or empty string",
  "why_not_clip": "if bad/neutral, why this doesn't work as a clip, or empty string",
  "why_good_clip": "if good, why this would work as a clip, or empty string",
  "missing_content": "what was missing to make it a complete clip, or empty string",
  "hook_quality": "strong / weak / none — assessment of the opening",
  "emotional_quality": "authentic / flat / forced",
  "clip_potential": "high / medium / low",
  "content_type": "education / story / opinion / debate / boring",
  "one_line_summary": "summarize the podcast content in one Hebrew sentence"
}

Be strict. If the expert says "זה לא מעניין" (not interesting) that's a bad rating.
If he says "זה נשמע טוב" or gets excited, that's good.
If he's neutral or moves on without comment, that's neutral."""


def extract_judgment(pair: dict) -> dict:
    """Extract structured judgment from one commentary/content pair using GPT-4o."""
    prompt = f"""PODCAST CONTENT:
{pair['podcast_text']}

EXPERT'S COMMENTARY:
{pair['commentary_text']}"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  [extract] GPT failed: {e}")
        return {}


# ── Step 4: Learn scoring rules from ALL judgments ────────────────────────────

RULES_PROMPT = """You are analyzing a collection of labeled examples where an expert Hebrew content curator
judged which podcast segments make good social media clips.

Each example shows:
- What the podcast said
- What the expert said about it (his live reaction)
- The structured judgment extracted from his reaction

Your job: extract CONCRETE SCORING RULES that capture his taste.
These rules will be injected into an AI clip-scoring prompt to make it think like him.

Write rules as specific, testable criteria. NOT vague like "be interesting".
YES: "If a speaker uses the phrase 'בואו נהיה כנים' followed by a personal confession, score hook_strength 8+"
YES: "Content that sounds 'רכילותי' (gossip-like) about a specific person should be disqualified"
YES: "When the speaker gives a concrete number or example, score value_density +2"

Format:
{
  "hook_rules": ["rule1", "rule2", ...],
  "disqualify_rules": ["rule1", "rule2", ...],
  "value_density_rules": ["rule1", "rule2", ...],
  "emotional_rules": ["rule1", "rule2", ...],
  "completeness_rules": ["rule1", "rule2", ...],
  "general_taste": "2-3 sentence summary of what this curator values most",
  "anti_patterns": ["things he consistently dislikes", ...]
}"""


def extract_scoring_rules(all_judgments: list, client_id: str) -> dict:
    """
    After processing many commentary recordings, extract scoring rules
    that encode the expert's taste into the clip selection prompt.
    """
    # Build a compact summary of all labeled examples
    examples = []
    for j in all_judgments[:40]:  # max 40 examples to stay within context
        examples.append({
            "content_preview": j.get("podcast_text", "")[:200],
            "verdict":         j.get("judgment", {}).get("verdict"),
            "score":           j.get("judgment", {}).get("overall_score"),
            "liked":           j.get("judgment", {}).get("what_he_liked"),
            "disliked":        j.get("judgment", {}).get("what_he_disliked"),
            "clip_potential":  j.get("judgment", {}).get("clip_potential"),
        })

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": RULES_PROMPT},
                {"role": "user",   "content": f"LABELED EXAMPLES:\n{json.dumps(examples, ensure_ascii=False, indent=2)}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        rules = json.loads(response.choices[0].message.content)

        # Save rules to Supabase for use in clip selection
        supabase.table("clients").update({
            "scoring_rules": rules
        }).eq("client_id", client_id).execute()

        print(f"[rules] extracted and saved scoring rules for {client_id}")
        return rules
    except Exception as e:
        print(f"[rules] extraction failed: {e}")
        return {}


# ── Step 5: Save everything to Supabase ──────────────────────────────────────

def save_labeled_pairs(pairs_with_judgments: list, client_id: str, video_id: str):
    """
    Save each labeled pair as a feedback row.
    Identical format to manual UI feedback — the clip selector doesn't
    need to know how the feedback was generated.
    """
    saved = 0
    for item in pairs_with_judgments:
        j = item.get("judgment", {})
        if not j or j.get("verdict") == "neutral":
            continue  # skip neutral, not useful for training

        try:
            supabase.table("feedback").insert({
                "user_id":          client_id,
                "clip_text":        item["podcast_text"][:600],
                "scores":           {},   # no AI scores, this is human-labeled
                "verdict":          j.get("verdict", "bad"),
                "comment":          item.get("commentary_text", "")[:400],
                "overall_score":    j.get("overall_score"),
                "marketing_purpose": None,
                "message_captured": j.get("clip_potential") == "high",
                "missing_content":  j.get("missing_content"),
                "overall_comment":  j.get("why_good_clip") or j.get("why_not_clip"),
                "start_comment":    j.get("hook_quality"),
                "message_comment":  j.get("what_he_liked") or j.get("what_he_disliked"),
            }).execute()
            saved += 1
        except Exception as e:
            print(f"  [save] failed: {e}")

    print(f"[commentary] saved {saved} labeled pairs for {client_id}")
    return saved


# ── Main entry point ──────────────────────────────────────────────────────────

def process_commentary_recording(
    transcript:  str,
    client_id:   str,
    video_id:    str,
    extract_rules: bool = False,
) -> dict:
    """
    Full pipeline:
    1. Split transcript by speaker
    2. Identify which voice is the commentator
    3. Align commentary to podcast content
    4. Extract structured judgment from each pair
    5. Save to feedback table
    6. Optionally extract and save scoring rules

    Returns summary dict.
    """
    print(f"\n[commentary] processing recording for client={client_id}")

    # 1. Parse
    segments = split_by_speaker(transcript)
    print(f"[commentary] {len(segments)} segments total")
    if not segments:
        return {"error": "no segments found — check transcript format has Voice 1/Voice 2 labels"}

    # 2. Identify commentator
    commentator = identify_commentator_voice(segments)
    content_voice = "Voice 1" if commentator == "Voice 2" else "Voice 2"
    print(f"[commentary] commentator={commentator}, content={content_voice}")

    # 3. Align
    pairs = build_aligned_pairs(segments, commentator)
    print(f"[commentary] {len(pairs)} commentary/content pairs aligned")

    if not pairs:
        return {"error": "no pairs aligned — ensure diarization ran and produced Voice labels"}

    # 4. Extract judgments
    pairs_with_judgments = []
    for i, pair in enumerate(pairs):
        if len(pair["podcast_text"]) < 30:
            continue   # skip trivial content
        print(f"  [{i+1}/{len(pairs)}] extracting judgment...")
        judgment = extract_judgment(pair)
        pairs_with_judgments.append({**pair, "judgment": judgment})

    # 5. Save
    saved = save_labeled_pairs(pairs_with_judgments, client_id, video_id)

    # 6. Extract rules (optional — do after accumulating many recordings)
    rules = {}
    if extract_rules and len(pairs_with_judgments) >= 10:
        print("[commentary] extracting scoring rules from judgments...")
        rules = extract_scoring_rules(pairs_with_judgments, client_id)

    # Summary
    good    = sum(1 for p in pairs_with_judgments if p.get("judgment", {}).get("verdict") == "good")
    bad     = sum(1 for p in pairs_with_judgments if p.get("judgment", {}).get("verdict") == "bad")
    neutral = sum(1 for p in pairs_with_judgments if p.get("judgment", {}).get("verdict") == "neutral")

    result = {
        "pairs_found":     len(pairs),
        "judgments_made":  len(pairs_with_judgments),
        "saved_to_db":     saved,
        "good":            good,
        "bad":             bad,
        "neutral":         neutral,
        "commentator":     commentator,
        "content_voice":   content_voice,
        "rules_extracted": bool(rules),
    }
    print(f"[commentary] done: {result}")
    return result