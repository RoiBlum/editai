def chunk_transcript(text: str, chunk_size: int = 1200):
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]

        chunks.append({
            "index": index,
            "text": chunk
        })

        start += chunk_size
        index += 1

    return chunks