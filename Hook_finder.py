"""
hook_finder.py
Given a clip that has good content but a weak hook,
search the vector DB for strong hooks from past recordings
that match the clip's topic/message.
"""
import os
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

openai   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def embed_text(text: str) -> list:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def find_matching_hooks(
    client_id:      str,
    clip_text:      str,
    exclude_video:  str = None,   # don't return hooks from the same video
    top_k:          int = 3,
    min_similarity: float = 0.70,
) -> list:
    """
    Embed the clip's message and search for similar hooks from past recordings.
    Returns top_k hooks sorted by similarity.
    """
    try:
        clip_embedding = embed_text(clip_text[:1000])
    except Exception as e:
        print(f"[hook_finder] embed failed: {e}")
        return []

    # Vector similarity search using pgvector cosine distance
    # Supabase RPC call — we need a SQL function for this (see below)
    try:
        result = supabase.rpc("find_similar_hooks", {
            "client_id_input":  client_id,
            "query_embedding":  clip_embedding,
            "match_threshold":  min_similarity,
            "match_count":      top_k + 3,   # fetch extra to filter out same-video
            "exclude_video_id": exclude_video or "",
        }).execute()

        hooks = result.data or []

        # Filter same video if needed (belt and suspenders)
        if exclude_video:
            hooks = [h for h in hooks if h.get("video_id") != exclude_video]

        return hooks[:top_k]

    except Exception as e:
        print(f"[hook_finder] search failed: {e}")
        return []


def find_matching_conclusions(
    client_id:     str,
    clip_text:     str,
    exclude_video: str = None,
    top_k:         int = 2,
) -> list:
    """Same as hook finder but for conclusions."""
    try:
        clip_embedding = embed_text(clip_text[:1000])
    except Exception as e:
        print(f"[hook_finder] embed failed: {e}")
        return []

    try:
        result = supabase.rpc("find_similar_conclusions", {
            "client_id_input":  client_id,
            "query_embedding":  clip_embedding,
            "match_threshold":  0.65,
            "match_count":      top_k + 2,
            "exclude_video_id": exclude_video or "",
        }).execute()

        conclusions = result.data or []
        if exclude_video:
            conclusions = [c for c in conclusions if c.get("video_id") != exclude_video]

        return conclusions[:top_k]

    except Exception as e:
        print(f"[hook_finder] conclusions search failed: {e}")
        return []