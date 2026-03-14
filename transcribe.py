import os

# Add CUDA DLLs to PATH before anything else loads
os.environ["PATH"] = (
    "C:\\Users\\PC\\Desktop\\EditAI\\venv\\Lib\\site-packages\\nvidia\\cublas\\bin" +
    os.pathsep +
    "C:\\Users\\PC\\Desktop\\EditAI\\venv\\Lib\\site-packages\\nvidia\\cudnn\\bin" +
    os.pathsep +
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2\\bin" +
    os.pathsep +
    os.environ["PATH"]
)

os.add_dll_directory("C:\\Users\\PC\\Desktop\\EditAI\\venv\\Lib\\site-packages\\nvidia\\cublas\\bin")
os.add_dll_directory("C:\\Users\\PC\\Desktop\\EditAI\\venv\\Lib\\site-packages\\nvidia\\cudnn\\bin")

from faster_whisper import WhisperModel
import sys

video_path = sys.argv[1]
output_path = os.path.splitext(video_path)[0] + "_transcript.txt"

print("Loading model... (first run downloads ~3GB, just wait)")

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
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