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
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)
HF_TOKEN = os.getenv("HF_TOKEN")

# ── Load Whisper ──────────────────────────────────────────────────────────────
print("Loading Whisper model on CUDA...")
model = WhisperModel(
    "ivrit-ai/whisper-large-v3-turbo-ct2",
    device="cuda",
    compute_type="float16"
)
print("✓ Whisper ready on GPU.")

# ── Load pyannote diarization pipeline ───────────────────────────────────────
diarize_pipeline = None
if HF_TOKEN:
    try:
        from pyannote.audio import Pipeline
        import torch
        print("Loading pyannote speaker diarization model...")
        diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
        if torch.cuda.is_available():
            diarize_pipeline = diarize_pipeline.to(torch.device("cuda"))
        print("✓ Pyannote diarization ready.")
    except Exception as e:
        print(f"⚠ Could not load pyannote: {e}")
        print("  Transcription will work without speaker labels.")
else:
    print("⚠ HF_TOKEN not set in .env — running without speaker diarization.")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_speaker_tracks(diarization) -> list:
    tracks = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        tracks.append((segment.start, segment.end, speaker))
    tracks.sort(key=lambda x: x[0])
    return tracks


def find_speaker(tracks: list, timestamp: float) -> str:
    for start, end, speaker in tracks:
        if start <= timestamp <= end:
            return speaker
        if start > timestamp:
            break
    return None


def normalize_speakers(tracks: list) -> dict:
    seen = {}
    counter = 1
    for _, _, speaker in tracks:
        if speaker not in seen:
            seen[speaker] = f"Voice {counter}"
            counter += 1
    return seen


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    content      = await file.read()
    file_size_mb = round(len(content) / 1024 / 1024, 1)
    suffix       = os.path.splitext(file.filename)[1] or ".mp4"

    video_id   = str(uuid.uuid4())
    saved_path = VIDEOS_DIR / f"{video_id}{suffix}"
    saved_path.write_bytes(content)

    async def stream():
        try:
            yield sse("status", {"stage": "received", "message": f"קובץ התקבל ({file_size_mb} MB)", "progress": 2, "video_id": video_id})
            await asyncio.sleep(0.1)

            yield sse("status", {"stage": "transcribing", "message": "מתחיל תמלול...", "progress": 6})
            await asyncio.sleep(0.1)

            # ── Step 1: Whisper ────────────────────────────────────────────────
            segments_iter, info = model.transcribe(
                str(saved_path),
                language="he",
                beam_size=10,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, threshold=0.5),
                initial_prompt="שיחה בעברית ישראלית מדוברת, כולל מונחים בעברית ובאנגלית."
            )

            total_duration = info.duration
            raw_segments   = []
            segment_count  = 0

            yield sse("status", {
                "stage": "transcribing",
                "message": f"אורך הקלטה: {int(total_duration // 60)}:{int(total_duration % 60):02d} דקות",
                "progress": 8,
                "total_duration": round(total_duration)
            })

            for segment in segments_iter:
                segment_count += 1
                raw_segments.append({
                    "start":   round(segment.start, 2),
                    "end":     round(segment.end, 2),
                    "text":    segment.text.strip(),
                    "speaker": None,
                    "words": [
                        {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                        for w in (segment.words or [])
                    ]
                })
                progress = min(8 + int((segment.end / total_duration) * 52), 60)
                minutes  = int(segment.start // 60)
                seconds  = int(segment.start % 60)
                yield sse("segment", {
                    "stage": "transcribing",
                    "message": f"מתמלל... {minutes}:{seconds:02d} / {int(total_duration // 60)}:{int(total_duration % 60):02d}",
                    "text": segment.text.strip(),
                    "start": round(segment.start, 1),
                    "end": round(segment.end, 1),
                    "segment_index": segment_count,
                    "progress": progress
                })
                await asyncio.sleep(0)

            yield sse("status", {"stage": "transcribing", "message": f"תמלול הושלם — {segment_count} קטעים", "progress": 60})

            # ── Step 2: Speaker diarization ────────────────────────────────────
            speaker_labels = {}

            if diarize_pipeline is not None:
                yield sse("status", {"stage": "diarizing", "message": "מזהה דוברים... (1-2 דקות)", "progress": 62})
                await asyncio.sleep(0.1)

                try:
                    import concurrent.futures
                    loop = asyncio.get_event_loop()

                    def run_diarization():
                        # soundfile cannot read MP4/MOV — extract audio to a temp WAV
                        # first via FFmpeg, then load with soundfile into memory so
                        # pyannote never needs torchcodec (which is broken on this machine).
                        import soundfile as sf
                        import torch
                        import tempfile, subprocess, os as _os

                        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
                        _os.close(fd)
                        try:
                            subprocess.run(
                                ["ffmpeg", "-y", "-i", str(saved_path),
                                 "-ac", "1", "-ar", "16000", "-vn", tmp_wav],
                                check=True, capture_output=True
                            )
                            waveform, sample_rate = sf.read(tmp_wav, dtype="float32", always_2d=True)
                        finally:
                            try: _os.remove(tmp_wav)
                            except: pass

                        # soundfile returns (time, channels) — pyannote needs (channels, time)
                        waveform_tensor = torch.from_numpy(waveform.T)
                        audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}
                        return diarize_pipeline(audio_input)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        diarization = await loop.run_in_executor(executor, run_diarization)

                    tracks = build_speaker_tracks(diarization)
                    speaker_labels = normalize_speakers(tracks)

                    for seg in raw_segments:
                        mid = (seg["start"] + seg["end"]) / 2
                        pyannote_id = find_speaker(tracks, mid)
                        seg["speaker"] = speaker_labels.get(pyannote_id, "Voice 1")

                    unique_speakers = sorted(set(speaker_labels.values()))
                    yield sse("status", {
                        "stage": "diarizing",
                        "message": f"זוהו {len(unique_speakers)} דוברים: {', '.join(unique_speakers)}",
                        "progress": 88
                    })
                    print(f"[diarize] found {len(unique_speakers)} speakers: {unique_speakers}")

                except Exception as e:
                    print(f"[diarize] failed: {e} — continuing without speaker labels")
                    yield sse("status", {"stage": "diarizing", "message": "זיהוי דוברים נכשל — ממשיך ללא תיוג", "progress": 88})
                    for seg in raw_segments:
                        seg["speaker"] = "Voice 1"
            else:
                for seg in raw_segments:
                    seg["speaker"] = None

            # ── Step 3: Build transcript lines ────────────────────────────────
            lines = []
            for seg in raw_segments:
                prefix = f"{seg['speaker']}: " if seg.get("speaker") else ""
                lines.append(f"[{seg['start']:.1f}s --> {seg['end']:.1f}s] {prefix}{seg['text']}")

            yield sse("done", {
                "stage":      "done",
                "message":    f"הושלם — {segment_count} קטעים",
                "progress":   100,
                "transcript": "\n".join(lines),
                "segments":   raw_segments,
                "language":   info.language,
                "confidence": round(info.language_probability, 2),
                "video_id":   video_id,
                "duration":   round(total_duration),
                "speakers":   sorted(set(speaker_labels.values())) if speaker_labels else []
            })

        except Exception as e:
            yield sse("error", {"stage": "error", "message": f"שגיאה בתמלול: {str(e)}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )