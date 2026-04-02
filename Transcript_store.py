"""
transcript_store.py
Saves transcripts and manages client profiles in Supabase.
"""
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# ── Clients ───────────────────────────────────────────────────────────────────

def get_or_create_client(client_id: str, name: str = None) -> dict:
    """Return existing client or create a new one."""
    res = supabase.table("clients").select("*").eq("client_id", client_id).execute()
    if res.data:
        return res.data[0]

    # Create new client
    new_client = {
        "client_id":   client_id,
        "name":        name or client_id,
        "description": "",
        "main_topics": [],
    }
    res = supabase.table("clients").insert(new_client).execute()
    print(f"[clients] created new client: {client_id}")
    return res.data[0]


def update_client_profile(client_id: str, profile: dict):
    """Update client profile fields extracted from transcripts."""
    supabase.table("clients").update(profile).eq("client_id", client_id).execute()


def increment_client_videos(client_id: str):
    """Increment total_videos counter for a client."""
    client = supabase.table("clients").select("total_videos").eq("client_id", client_id).execute()
    if client.data:
        current = client.data[0].get("total_videos", 0) or 0
        supabase.table("clients").update({"total_videos": current + 1}).eq("client_id", client_id).execute()


def list_clients() -> list:
    res = supabase.table("clients").select("*").order("created_at", desc=True).execute()
    return res.data or []


def get_client(client_id: str) -> dict:
    res = supabase.table("clients").select("*").eq("client_id", client_id).execute()
    return res.data[0] if res.data else None


# ── Transcripts ───────────────────────────────────────────────────────────────

def save_transcript(
    client_id:    str,
    video_id:     str,
    filename:     str,
    duration_sec: int,
    speakers:     list,
    full_text:    str,
    raw_segments: list,
) -> int:
    """
    Save a transcript to the database.
    Returns the new transcript's id.
    """
    # Ensure client exists
    get_or_create_client(client_id)

    row = {
        "client_id":    client_id,
        "video_id":     video_id,
        "filename":     filename,
        "duration_sec": duration_sec,
        "speakers":     speakers,
        "speaker_count": len(speakers) if speakers else 1,
        "full_text":    full_text,
        "raw_segments": raw_segments,
        "components_extracted": False,
    }
    res = supabase.table("transcripts").insert(row).execute()
    transcript_id = res.data[0]["id"]

    # Bump client video count
    increment_client_videos(client_id)

    print(f"[transcript_store] saved transcript id={transcript_id} for client={client_id} video={video_id}")
    return transcript_id


def mark_components_extracted(transcript_id: int):
    supabase.table("transcripts").update(
        {"components_extracted": True}
    ).eq("id", transcript_id).execute()


def get_client_transcripts(client_id: str) -> list:
    res = supabase.table("transcripts") \
        .select("id, created_at, video_id, filename, duration_sec, speaker_count, components_extracted") \
        .eq("client_id", client_id) \
        .order("created_at", desc=True) \
        .execute()
    return res.data or []