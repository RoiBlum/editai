from faster_whisper import WhisperModel
import sys
import os

video_path = sys.argv[1]
output_path = os.path.splitext(video_path)[0] + "_transcript.txt"

print("Loading model... (first run downloads ~3GB, just wait)")

model = WhisperModel(
    "large-v3",
    device="cpu",
compute_type="int8"
)

print(f"Transcribing: {video_path}")

segments, info = model.transcribe(
    video_path,
    language="he",
    beam_size=5,
    word_timestamps=True
)

print(f"Language detected: {info.language} ({info.language_probability:.0%} confidence)")

with open(output_path, "w", encoding="utf-8") as f:
    for segment in segments:
        start = f"{segment.start:.1f}s"
        end = f"{segment.end:.1f}s"
        line = f"[{start} --> {end}] {segment.text.strip()}"
        print(line)
        f.write(line + "\n")

print(f"\nDone. Transcript saved to: {output_path}")