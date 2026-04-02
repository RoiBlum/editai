import os

# ── Must be before ANY other imports ─────────────────────────────────────────
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

import json
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Video storage folder (shared with video_editor_api) ───────────────────────
VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)

print("Loading Whisper model on CUDA...")

model = WhisperModel(
    "ivrit-ai/whisper-large-v3-turbo-ct2",  # CTranslate2 version
    device="cuda",
    compute_type="float16"
)

print("✓ Model ready on GPU.")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    content      = await file.read()
    file_size_mb = round(len(content) / 1024 / 1024, 1)
    suffix       = os.path.splitext(file.filename)[1] or ".mp4"

    # Save video permanently so video_editor_api can use it later
    video_id   = str(uuid.uuid4())
    saved_path = VIDEOS_DIR / f"{video_id}{suffix}"
    saved_path.write_bytes(content)

    async def stream():
        try:
            yield sse("status", {
                "stage":    "received",
                "message":  f"קובץ התקבל ({file_size_mb} MB)",
                "progress": 2,
                "video_id": video_id
            })
            await asyncio.sleep(0.1)

            yield sse("status", {
                "stage":    "loading",
                "message":  "טוען מודל Whisper על GPU...",
                "progress": 5
            })
            await asyncio.sleep(0.1)

            yield sse("status", {
                "stage":    "transcribing",
                "message":  "מתחיל תמלול...",
                "progress": 8
            })
            await asyncio.sleep(0.1)

            segments_iter, info = model.transcribe(
                str(saved_path),
                language="he",
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            total_duration = info.duration
            lines          = []
            raw_segments   = []
            segment_count  = 0

            yield sse("status", {
                "stage":          "transcribing",
                "message":        f"אורך הקלטה: {int(total_duration // 60)}:{int(total_duration % 60):02d} דקות",
                "progress":       10,
                "total_duration": round(total_duration)
            })

            for segment in segments_iter:
                segment_count += 1
                line = f"[{segment.start:.1f}s --> {segment.end:.1f}s] {segment.text.strip()}"
                lines.append(line)
                raw_segments.append({
                    "start": round(segment.start, 2),
                    "end":   round(segment.end, 2),
                    "text":  segment.text.strip(),
                    "words": [
                        {
                            "word":  w.word,
                            "start": round(w.start, 3),
                            "end":   round(w.end, 3)
                        }
                        for w in (segment.words or [])
                    ]
                })

                progress = min(10 + int((segment.end / total_duration) * 85), 95)
                minutes  = int(segment.start // 60)
                seconds  = int(segment.start % 60)

                yield sse("segment", {
                    "stage":         "transcribing",
                    "message":       f"מתמלל... {minutes}:{seconds:02d} / {int(total_duration // 60)}:{int(total_duration % 60):02d}",
                    "text":          segment.text.strip(),
                    "start":         round(segment.start, 1),
                    "end":           round(segment.end, 1),
                    "segment_index": segment_count,
                    "progress":      progress
                })

                await asyncio.sleep(0)

            yield sse("done", {
                "stage":      "done",
                "message":    f"תמלול הושלם — {segment_count} קטעים",
                "progress":   100,
                "transcript": "\n".join(lines),
                "segments":   raw_segments,        # full structured segments for editor
                "language":   info.language,
                "confidence": round(info.language_probability, 2),
                "video_id":   video_id,            # ← editor uses this to find the file
                "duration":   round(total_duration)
            })

        except Exception as e:
            yield sse("error", {"stage": "error", "message": f"שגיאה בתמלול: {str(e)}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )