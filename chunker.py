import re


def chunk_transcript(text: str, chunk_size: int = 1200):
    """
    Split transcript into chunks, preserving real timestamps.
    Each transcript line looks like:
      [14.2s --> 28.5s] some text here
    Each chunk gets the start time of its first line and end time of its last line.
    """
    chunks  = []
    lines   = text.strip().split("\n")

    current_text  = ""
    current_start = None
    current_end   = None
    index         = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse timestamp from line like [14.2s --> 28.5s] text
        match = re.match(r'\[(\d+\.?\d*)s\s*-->\s*(\d+\.?\d*)s\]\s*(.*)', line)
        if match:
            seg_start = float(match.group(1))
            seg_end   = float(match.group(2))

            if current_start is None:
                current_start = seg_start
            current_end = seg_end

        current_text += line + "\n"

        if len(current_text) >= chunk_size:
            chunks.append({
                "index":      index,
                "text":       current_text.strip(),
                "start_time": current_start if current_start is not None else 0.0,
                "end_time":   current_end   if current_end   is not None else 0.0,
            })
            index        += 1
            current_text  = ""
            current_start = None
            current_end   = None

    # Last chunk
    if current_text.strip():
        chunks.append({
            "index":      index,
            "text":       current_text.strip(),
            "start_time": current_start if current_start is not None else 0.0,
            "end_time":   current_end   if current_end   is not None else 0.0,
        })

    return chunks