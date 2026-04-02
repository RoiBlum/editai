import re


def chunk_transcript(text: str, min_seconds: float = 30, max_seconds: float = 60):
    """
    Split transcript into chunks by TIME duration, not character count.
    Each transcript line looks like:
      [14.2s --> 28.5s] Voice 1: some text here

    A new chunk starts when:
    - The chunk has hit max_seconds duration, OR
    - A natural speaker-change boundary is reached after min_seconds

    Each chunk carries start_time and end_time for the video editor.
    """
    lines   = text.strip().split("\n")
    chunks  = []
    index   = 0

    current_lines = []
    current_start = None
    current_end   = None
    prev_speaker  = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r'\[(\d+\.?\d*)s\s*-->\s*(\d+\.?\d*)s\]\s*(.*)', line)
        if not match:
            continue

        seg_start = float(match.group(1))
        seg_end   = float(match.group(2))
        seg_text  = match.group(3)

        # Detect speaker from "Voice N: text"
        speaker_match = re.match(r'(Voice \d+):\s*(.*)', seg_text)
        curr_speaker  = speaker_match.group(1) if speaker_match else None

        if current_start is None:
            current_start = seg_start

        current_lines.append(line)
        current_end = seg_end
        duration    = current_end - current_start

        # Commit chunk if we hit max duration
        over_max = duration >= max_seconds

        # Commit at a speaker-change boundary after min_seconds
        speaker_changed = (curr_speaker is not None and
                           prev_speaker is not None and
                           curr_speaker != prev_speaker)
        past_min = duration >= min_seconds
        natural_break = past_min and speaker_changed

        if over_max or natural_break:
            chunks.append({
                "index":      index,
                "text":       "\n".join(current_lines),
                "start_time": current_start,
                "end_time":   current_end,
            })
            index         += 1
            current_lines  = []
            current_start  = None
            current_end    = None

        prev_speaker = curr_speaker

    # Last chunk — only keep if long enough to be a real clip
    if current_lines and current_end and current_start:
        duration = current_end - current_start
        if duration >= min_seconds * 0.6:  # allow last chunk to be 60% of min
            chunks.append({
                "index":      index,
                "text":       "\n".join(current_lines),
                "start_time": current_start,
                "end_time":   current_end,
            })

    return chunks