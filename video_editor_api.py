import os
import json
import uuid
import asyncio
import subprocess
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

class Segment(BaseModel):
    start: float
    end: float
    text: str

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

def make_srt(segments: List[Segment], clip_start: float, output_path: Path):
    """
    Generate an SRT subtitle file from Whisper segments.
    Times are relative to the clip (subtract clip_start from each timestamp).
    """
    def fmt_time(seconds: float) -> str:
        h  = int(seconds // 3600)
        m  = int((seconds % 3600) // 60)
        s  = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    idx = 1
    for seg in segments:
        rel_start = max(0.0, seg.start - clip_start)
        rel_end   = max(0.0, seg.end   - clip_start)
        if rel_end <= 0:
            continue
        # Add RTL unicode mark so libass renders right-to-left
        text = "\u202B" + seg.text.strip()
        lines.append(f"{idx}")
        lines.append(f"{fmt_time(rel_start)} --> {fmt_time(rel_end)}")
        lines.append(text)
        lines.append("")
        idx += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ── FFmpeg helpers ─────────────────────────────────────────────────────────────

def run_ffmpeg(args: list) -> tuple[int, str]:
    """Run ffmpeg and return (returncode, stderr)."""
    cmd = ["ffmpeg", "-y"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stderr


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Return (width, height) using ffprobe - multiple fallback methods."""
    # Method 1: try JSON format
    import json as _json
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True
    )
    try:
        data = _json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                w, h = int(stream["width"]), int(stream["height"])
                print(f"[dimensions] method1: {w}x{h}")
                return w, h
    except Exception as e:
        print(f"[dimensions] method1 failed: {e}")

    # Method 2: use ffprobe with explicit stream entry format
    result2 = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "default=noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True
    )
    try:
        w, h = None, None
        for line in result2.stdout.splitlines():
            if line.startswith("width="):
                w = int(line.split("=")[1].strip())
            elif line.startswith("height="):
                h = int(line.split("=")[1].strip())
        if w and h:
            print(f"[dimensions] method2: {w}x{h}")
            return w, h
    except Exception as e:
        print(f"[dimensions] method2 failed: {e}")

    # Method 3: parse from ffmpeg stderr
    result3 = subprocess.run(
        ["ffmpeg", "-i", str(video_path)],
        capture_output=True, text=True
    )
    import re
    match = re.search(r"(\d{2,5})x(\d{2,5})", result3.stderr)
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
        print(f"[edit] start={request.start} end={request.end} duration={request.end - request.start} video_id={request.video_id}")
        duration = request.end - request.start
        if duration <= 0:
            raise ValueError(f"משך הקליפ לא תקין: start={request.start} end={request.end}. הגדר timestamps בכרטיס הקליפ.")
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

        rc, err = run_ffmpeg([
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
        cut_w, cut_h = get_video_dimensions(cut_path)
        print(f"[cut] output dimensions: {cut_w}x{cut_h}")
        if cut_w == 0 or cut_h == 0:
            raise RuntimeError("קובץ החיתוך פגום — מידות 0")

        update("חיתוך הושלם ✓", 35)
        await asyncio.sleep(0.1)

        # ── Step 3: Portrait crop (9:16) ─────────────────────────────────────
        cropped_path = OUTPUTS_DIR / f"{job_id}_cropped.mp4"

        if request.portrait:
            update("חותך לפורמט אנכי 9:16...", 50)

            w, h = get_video_dimensions(cut_path)
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
                rc, err = run_ffmpeg([
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
                "FontSize=18,"
                "PrimaryColour=&H00FFFFFF,"  # white text
                "OutlineColour=&H00000000,"  # black outline
                "BackColour=&H80000000,"     # semi-transparent background
                "BorderStyle=3,"             # box background style
                "Outline=2,"
                "Shadow=0,"
                "Alignment=2,"              # bottom center
                "MarginV=40"
            )

            rc, err = run_ffmpeg([
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