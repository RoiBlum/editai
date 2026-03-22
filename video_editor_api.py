import os
import json
import uuid
import asyncio
import tempfile
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Folders ────────────────────────────────────────────────────────────────────
VIDEOS_DIR  = Path("videos")   # saved uploaded videos (from transcribe_api)
OUTPUTS_DIR = Path("outputs")  # edited clip outputs
VIDEOS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── In-memory job store ────────────────────────────────────────────────────────
# job_id -> { "status": "processing|done|error", "progress": 0-100, "steps": [], "error": "" }
jobs: dict = {}


# ── Models ────────────────────────────────────────────────────────────────────

class Word(BaseModel):
    word:  str
    start: float
    end:   float

class Segment(BaseModel):
    start: float
    end:   float
    text:  str
    words: Optional[List[Word]] = None

class EditRequest(BaseModel):
    video_id: str               # UUID returned by transcribe_api after saving the video
    start: float                # clip start in seconds
    end: float                  # clip end in seconds
    segments: List[Segment]     # whisper segments that fall in this clip (for subtitles)
    portrait: bool = True       # crop to 9:16
    subtitles: bool = True      # burn Hebrew subtitles


# ── SSE helper ─────────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Subtitle generation ────────────────────────────────────────────────────────

MAX_WORDS_PER_LINE  = 5   # max words on a single subtitle line
MAX_LINES           = 2   # never more than 2 lines on screen at once
MAX_WORDS_PER_CHUNK = MAX_WORDS_PER_LINE * MAX_LINES   # 10 words total max
MAX_CHUNK_SECONDS   = 3.5  # also split if a chunk exceeds this duration
SYNC_OFFSET_MS      = -0.08  # show subtitle 80ms early for better perceived sync


def make_srt(segments: List[Segment], clip_start: float, output_path: Path):
    """
    Generate an SRT subtitle file from Whisper word-level timestamps.

    Chunks words into subtitle cards of at most MAX_WORDS_PER_CHUNK words
    OR MAX_CHUNK_SECONDS seconds — whichever comes first.
    Each card is split into at most MAX_LINES lines of MAX_WORDS_PER_LINE words.
    Times are relative to the clip (subtract clip_start).
    """

    def fmt_time(seconds: float) -> str:
        seconds = max(0.0, seconds)
        h  = int(seconds // 3600)
        m  = int((seconds % 3600) // 60)
        s  = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # ── Collect all words from all segments ───────────────────────────────────
    all_words = []
    for seg in segments:
        if seg.words:
            # Use precise word-level timestamps
            for w in seg.words:
                rel_start = w.start - clip_start
                rel_end   = w.end   - clip_start
                # Skip words that fall outside the clip
                if rel_end < 0 or rel_start > (seg.end - clip_start + 1):
                    continue
                all_words.append({
                    "word":  w.word.strip(),
                    "start": rel_start,
                    "end":   rel_end
                })
        else:
            # Fallback: no word timestamps — distribute evenly across segment
            text  = seg.text.strip()
            words = text.split()
            if not words:
                continue
            rel_start = seg.start - clip_start
            rel_end   = seg.end   - clip_start
            if rel_end <= 0:
                continue
            dur_per_word = (rel_end - rel_start) / len(words)
            for i, word in enumerate(words):
                all_words.append({
                    "word":  word,
                    "start": rel_start + i * dur_per_word,
                    "end":   rel_start + (i + 1) * dur_per_word
                })

    if not all_words:
        output_path.write_text("", encoding="utf-8")
        return

    # ── Chunk words into subtitle cards ──────────────────────────────────────
    chunks = []
    current_chunk = []

    for word in all_words:
        if not word["word"]:
            continue

        current_chunk.append(word)

        chunk_duration = current_chunk[-1]["end"] - current_chunk[0]["start"]
        over_word_limit = len(current_chunk) >= MAX_WORDS_PER_CHUNK
        over_time_limit = chunk_duration >= MAX_CHUNK_SECONDS

        if over_word_limit or over_time_limit:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    # ── Build SRT entries ─────────────────────────────────────────────────────
    srt_lines = []
    idx = 1

    for chunk in chunks:
        if not chunk:
            continue

        start_time = chunk[0]["start"]  + SYNC_OFFSET_MS
        end_time   = chunk[-1]["end"]

        if end_time <= 0:
            continue

        # Split words into max 2 lines
        words_text = [w["word"] for w in chunk]
        if len(words_text) <= MAX_WORDS_PER_LINE:
            # Single line
            text_display = " ".join(words_text)
        else:
            # Two lines — split roughly in half
            split_at = (len(words_text) + 1) // 2
            # Don't let either line exceed MAX_WORDS_PER_LINE
            split_at = min(split_at, MAX_WORDS_PER_LINE)
            line1 = " ".join(words_text[:split_at])
            line2 = " ".join(words_text[split_at:])
            text_display = line1 + "\n" + line2

        # RTL unicode mark for Hebrew
        text_display = "\u202B" + text_display

        srt_lines.append(str(idx))
        srt_lines.append(f"{fmt_time(start_time)} --> {fmt_time(end_time)}")
        srt_lines.append(text_display)
        srt_lines.append("")
        idx += 1

    output_path.write_text("\n".join(srt_lines), encoding="utf-8")
    print(f"[subtitles] {idx-1} subtitle cards from {len(all_words)} words")


# ── FFmpeg helpers ─────────────────────────────────────────────────────────────

async def run_ffmpeg(args: list) -> tuple[int, str]:
    """Run ffmpeg asynchronously so it does not block the event loop."""
    cmd = ["ffmpeg", "-y"] + args
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stderr.decode(errors="replace")


async def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Return (width, height) using ffprobe - async, non-blocking."""
    import json as _json, re as _re

    # Method 1: JSON format
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    try:
        data = _json.loads(out.decode(errors="replace"))
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                w, h = int(stream["width"]), int(stream["height"])
                print(f"[dimensions] {video_path.name}: {w}x{h}")
                return w, h
    except Exception as e:
        print(f"[dimensions] json failed: {e}")

    # Method 2: key=value format
    proc2 = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "default=noprint_wrappers=1", str(video_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out2, _ = await proc2.communicate()
    try:
        w, h = None, None
        for line in out2.decode(errors="replace").splitlines():
            if line.startswith("width="):  w = int(line.split("=")[1].strip())
            elif line.startswith("height="): h = int(line.split("=")[1].strip())
        if w and h:
            print(f"[dimensions] method2: {w}x{h}")
            return w, h
    except Exception as e:
        print(f"[dimensions] method2 failed: {e}")

    # Method 3: parse from ffmpeg stderr
    proc3 = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", str(video_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, err3 = await proc3.communicate()
    match = _re.search(r"(\d{2,5})x(\d{2,5})", err3.decode(errors="replace"))
    if match:
        w, h = int(match.group(1)), int(match.group(2))
        print(f"[dimensions] method3: {w}x{h}")
        return w, h

    print("[dimensions] all methods failed, using 1920x1080")
    return 1920, 1080


# ── Main edit pipeline ─────────────────────────────────────────────────────────

async def run_edit_pipeline(job_id: str, request: EditRequest):
    """
    Full editing pipeline. Updates jobs[job_id] at each step.
    Steps:
      1. Locate source video
      2. Cut segment
      3. Generate SRT subtitles
      4. Portrait crop (9:16)
      5. Burn subtitles
      6. Done
    """

    def update(step_name: str, progress: int, status: str = "processing", error: str = ""):
        jobs[job_id]["steps"].append({"name": step_name, "progress": progress})
        jobs[job_id]["progress"] = progress
        jobs[job_id]["status"]   = status
        jobs[job_id]["current"]  = step_name
        if error:
            jobs[job_id]["error"] = error

    try:
        # ── Step 1: Find source video ────────────────────────────────────────
        update("מאתר קובץ וידאו...", 5)
        await asyncio.sleep(0.1)

        source_video = VIDEOS_DIR / f"{request.video_id}.mp4"
        if not source_video.exists():
            # Try other extensions
            for ext in [".mov", ".m4a", ".mp3", ".wav", ".webm", ".mkv"]:
                candidate = VIDEOS_DIR / f"{request.video_id}{ext}"
                if candidate.exists():
                    source_video = candidate
                    break

        if not source_video.exists():
            raise FileNotFoundError(
                f"קובץ הוידאו לא נמצא. וודא שהתמלול נשמר עם video_id: {request.video_id}"
            )

        update("קובץ וידאו נמצא ✓", 10)
        await asyncio.sleep(0.1)

        # ── Step 2: Cut segment ──────────────────────────────────────────────
        update("חותך קטע מהוידאו...", 20)

        duration  = request.end - request.start
        cut_path  = OUTPUTS_DIR / f"{job_id}_cut.mp4"

        rc, err = await run_ffmpeg([
            "-i", str(source_video),
            "-ss", str(request.start),   # accurate seek AFTER -i
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-movflags", "+faststart",
            str(cut_path)
        ])

        if rc != 0:
            raise RuntimeError(f"שגיאה בחיתוך: {err[-300:]}")
        
        # Verify cut file is valid before continuing
        cut_w, cut_h = await get_video_dimensions(cut_path)
        print(f"[cut] output dimensions: {cut_w}x{cut_h}")
        if cut_w == 0 or cut_h == 0:
            raise RuntimeError("קובץ החיתוך פגום — מידות 0")

        update("חיתוך הושלם ✓", 35)
        await asyncio.sleep(0.1)

        # ── Step 3: Portrait crop (9:16) ─────────────────────────────────────
        cropped_path = OUTPUTS_DIR / f"{job_id}_cropped.mp4"

        if request.portrait:
            update("חותך לפורמט אנכי 9:16...", 50)

            w, h = await get_video_dimensions(cut_path)
            print(f"[crop] source dimensions: {w}x{h}")

            # Must ensure dimensions are even numbers (required by libx264)
            # Target 9:16 crop — take center slice of the width
            target_w = int(h * 9 / 16)
            target_w = min(target_w, w)            # cannot exceed source width
            target_w = target_w - (target_w % 2)  # make even
            target_h = h - (h % 2)                # make even
            crop_x   = (w - target_w) // 2
            crop_x   = crop_x - (crop_x % 2)      # make even
            crop_x   = max(0, crop_x)              # never negative

            # Final bounds check — crop box must fit inside source
            if crop_x + target_w > w:
                crop_x = 0
                target_w = w - (w % 2)

            print(f"[crop] source={w}x{h} crop={target_w}x{target_h} x={crop_x}")

            if target_w <= 0 or target_h <= 0:
                # Dimensions invalid — skip crop, copy as-is
                print("[crop] invalid dimensions, skipping crop")
                import shutil
                shutil.copy(cut_path, cropped_path)
                update("חיתוך פורמט דולג (מידות לא תקינות)", 60)
            else:
                rc, err = await run_ffmpeg([
                    "-i", str(cut_path),
                    "-vf", f"crop={target_w}:{target_h}:{crop_x}:0",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-preset", "fast",
                    str(cropped_path)
                ])

                if rc != 0:
                    raise RuntimeError(f"שגיאה בחיתוך פורמט: {err[-400:]}")

                update("פורמט אנכי ✓", 60)

        else:
            # No crop — just use the cut file
            import shutil
            shutil.copy(cut_path, cropped_path)
            update("ללא חיתוך פורמט", 60)

        await asyncio.sleep(0.1)

        # ── Step 4: Generate SRT subtitles ───────────────────────────────────
        final_path = OUTPUTS_DIR / f"{job_id}_final.mp4"
        srt_path   = OUTPUTS_DIR / f"{job_id}.srt"

        if request.subtitles and request.segments:
            update("מייצר כתוביות עבריות...", 70)

            make_srt(request.segments, request.start, srt_path)
            update("כתוביות נוצרו ✓", 75)
            await asyncio.sleep(0.1)

            # ── Step 5: Burn subtitles ───────────────────────────────────────
            update("צורב כתוביות על הוידאו...", 80)

            # Use libass via the subtitles filter
            # force_style controls font, size, color, border
            srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")

            subtitle_style = (
                "FontName=Arial,"
                "FontSize=12,"
                "PrimaryColour=&H00FFFFFF,"  # white text
                "OutlineColour=&H00000000,"  # black outline
                "BackColour=&H80000000,"     # semi-transparent background
                "BorderStyle=3,"             # box background style
                "Outline=2,"
                "Shadow=0,"
                "Alignment=2,"              # bottom center
                "MarginV=40"
            )

            rc, err = await run_ffmpeg([
                "-i", str(cropped_path),
                "-vf", f"subtitles='{srt_escaped}':force_style='{subtitle_style}'",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                str(final_path)
            ])

            if rc != 0:
                # Subtitles failed — fall back to no subtitles
                print(f"Subtitle burn failed, falling back: {err[-200:]}")
                import shutil
                shutil.copy(cropped_path, final_path)
                update("כתוביות נכשלו — שמור ללא כתוביות", 90)
            else:
                update("כתוביות נצרבו ✓", 90)

        else:
            import shutil
            shutil.copy(cropped_path, final_path)
            update("ללא כתוביות", 90)

        await asyncio.sleep(0.1)

        # ── Cleanup temp files ───────────────────────────────────────────────
        for f in [cut_path, cropped_path]:
            try:
                f.unlink()
            except Exception:
                pass

        # ── Done ─────────────────────────────────────────────────────────────
        update("הקליפ מוכן! ✓", 100, status="done")

    except Exception as e:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["error"]   = str(e)
        jobs[job_id]["current"] = f"שגיאה: {str(e)}"
        print(f"Edit pipeline error for {job_id}: {e}")


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.post("/edit")
async def start_edit(request: EditRequest, background_tasks: BackgroundTasks):
    """Start an edit job. Returns job_id immediately."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status":   "processing",
        "progress": 0,
        "steps":    [],
        "current":  "מתחיל...",
        "error":    ""
    }
    background_tasks.add_task(run_edit_pipeline, job_id, request)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def stream_status(job_id: str):
    """
    SSE stream of job progress.
    Client receives events until status is 'done' or 'error'.
    """
    async def stream():
        while True:
            job = jobs.get(job_id)
            if not job:
                yield sse("error", {"message": "Job not found"})
                return

            yield sse("progress", {
                "status":   job["status"],
                "progress": job["progress"],
                "current":  job["current"],
                "error":    job.get("error", "")
            })

            if job["status"] in ("done", "error"):
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/output/{job_id}")
async def get_output(job_id: str):
    """Stream the final edited video file."""
    final_path = OUTPUTS_DIR / f"{job_id}_final.mp4"
    if not final_path.exists():
        return {"error": "Output not ready yet"}
    return FileResponse(
        path=str(final_path),
        media_type="video/mp4",
        filename=f"clip_{job_id[:8]}.mp4",
        headers={"Accept-Ranges": "bytes"}
    )


@app.delete("/job/{job_id}")
async def cleanup_job(job_id: str):
    """Delete output files for a job to free disk space."""
    for pattern in [f"{job_id}_final.mp4", f"{job_id}.srt"]:
        f = OUTPUTS_DIR / pattern
        if f.exists():
            f.unlink()
    jobs.pop(job_id, None)
    return {"deleted": job_id}