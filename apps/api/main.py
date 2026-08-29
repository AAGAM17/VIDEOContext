"""VideoContext REST API — FastAPI service for video processing and querying.

Endpoints:
- POST   /v1/videos              Upload and create a video job
- GET    /v1/videos/{video_id}   Get video metadata
- POST   /v1/videos/{video_id}/process  Start processing a video
- GET    /v1/videos/{video_id}/status   Get processing status
- POST   /v1/videos/{video_id}/search   Search video content
- POST   /v1/videos/{video_id}/ask      Ask a question about the video
- GET    /v1/videos/{video_id}/timeline Get timeline view
- GET    /v1/videos/{video_id}/segments Get segments
- GET    /v1/videos/{video_id}/frames   Get sampled frames
- GET    /health                     Health check
- GET    /ready                      Readiness check
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from videocontent.config import ProcessingConfig, load_config
from videocontent.sdk import Video, load, process
from videocontent.schema.v1 import VideoContextDocument

# In-memory job store (replace with Redis in production)
jobs: dict[str, dict[str, Any]] = {}
video_files: dict[str, Path] = {}
video_docs: dict[str, VideoContextDocument] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    yield
    # Shutdown - cleanup temp files
    for path in video_files.values():
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


app = FastAPI(
    title="VideoContext API",
    description="Turn video into timestamped, searchable context for AI agents and applications.",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Pydantic models ---


class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    status: str = "uploaded"


class ProcessRequest(BaseModel):
    config: dict[str, Any] | None = None


class ProcessResponse(BaseModel):
    video_id: str
    status: str = "processing"


class JobStatus(BaseModel):
    video_id: str
    status: str  # uploaded, processing, completed, failed
    progress: float = 0.0
    error: str | None = None
    result_path: str | None = None


class SearchRequest(BaseModel):
    query: str
    modalities: list[str] | None = None
    top_k: int = 10
    start: float | None = None
    end: float | None = None
    min_score: float | None = None


class SearchHit(BaseModel):
    timecode: str
    start: float
    end: float
    modality: str
    text: str
    score: float
    reason: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    total: int
    took_ms: float


class AskRequest(BaseModel):
    question: str
    modalities: list[str] | None = None
    top_k: int = 5
    min_score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    evidence: list[SearchHit]


# --- Context models ---


class ContextRequest(BaseModel):
    task: str
    max_tokens: int = 4000
    modalities: list[str] | None = None


class ContextResponse(BaseModel):
    task: str
    task_type: str
    confidence: float
    context: str
    profiles: dict[str, Any] = {}
    evidence: list[SearchHit] = []
    frames: list[dict[str, Any]] = []
    global_context: dict[str, Any] | None = None
    summaries: dict[str, str] = {}
    token_estimate: int = 0


class ProfileRequest(BaseModel):
    profile_name: str
    force: bool = False


class ProfileResponse(BaseModel):
    profile_name: str
    profile: dict[str, Any] | None = None
    available: bool = True


class TimelineEntry(BaseModel):
    start: float
    end: float
    modality: str
    text: str


class TimelineResponse(BaseModel):
    video_id: str
    timeline: list[TimelineEntry]


# --- Helpers ---


def _get_job(video_id: str) -> dict[str, Any]:
    if video_id not in jobs:
        raise HTTPException(status_code=404, detail="Video not found")
    return jobs[video_id]


def _get_doc(video_id: str) -> VideoContextDocument:
    if video_id not in video_docs:
        raise HTTPException(status_code=404, detail="Video not processed yet")
    return video_docs[video_id]


# --- Routes ---


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness check endpoint."""
    # Check if we can process videos (FFmpeg available)
    from videocontent.media import ffmpeg

    if not ffmpeg.available():
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "ffmpeg not available"},
        )
    return {"status": "ready"}


@app.post("/v1/videos", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
) -> VideoUploadResponse:
    """Upload a video file and create a video record."""
    video_id = str(uuid.uuid4())

    # Save to temp location
    upload_dir = Path("/tmp/videocontent_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{video_id}_{file.filename}"

    content = await file.read()
    file_path.write_bytes(content)

    video_files[video_id] = file_path
    jobs[video_id] = {
        "video_id": video_id,
        "filename": file.filename,
        "status": "uploaded",
        "progress": 0.0,
        "error": None,
        "result_path": None,
    }

    return VideoUploadResponse(
        video_id=video_id,
        filename=file.filename,
        status="uploaded",
    )


@app.get("/v1/videos/{video_id}")
async def get_video(video_id: str) -> dict[str, Any]:
    """Get video metadata."""
    job = _get_job(video_id)
    return {
        "video_id": video_id,
        "filename": job["filename"],
        "status": job["status"],
    }


@app.post("/v1/videos/{video_id}/process", response_model=ProcessResponse)
async def process_video(video_id: str, request: ProcessRequest) -> ProcessResponse:
    """Start processing a video."""
    job = _get_job(video_id)
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="Already processing")

    job["status"] = "processing"
    job["progress"] = 0.0

    # Get video file
    file_path = video_files.get(video_id)
    if not file_path or not file_path.exists():
        job["status"] = "failed"
        job["error"] = "Video file not found"
        raise HTTPException(status_code=404, detail="Video file not found")

    # Build config
    config = ProcessingConfig()
    if request.config:
        # Apply overrides
        try:
            config = load_config(overrides=request.config)
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = f"Invalid config: {exc}"
            raise HTTPException(status_code=400, detail=str(exc))

    # Process in background (in production, use a job queue)
    try:
        video = Video(file_path, config=config)
        video.process()
        doc = video.document

        # Save .vctx
        output_dir = Path("/tmp/videocontent_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_id}.vctx"
        video.save(output_path)

        job["status"] = "completed"
        job["progress"] = 1.0
        job["result_path"] = str(output_path)
        video_docs[video_id] = doc

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    return ProcessResponse(video_id=video_id, status="processing")


@app.get("/v1/videos/{video_id}/status", response_model=JobStatus)
async def get_status(video_id: str) -> JobStatus:
    """Get processing status."""
    job = _get_job(video_id)
    return JobStatus(**job)


@app.get("/v1/videos/{video_id}/download")
async def download_vctx(video_id: str) -> FileResponse:
    """Download the .vctx file."""
    job = _get_job(video_id)
    if job["status"] != "completed" or not job["result_path"]:
        raise HTTPException(status_code=404, detail="No result available")
    return FileResponse(
        job["result_path"],
        media_type="application/vnd.videocontext+json",
        filename=f"{video_id}.vctx",
    )


@app.post("/v1/videos/{video_id}/search", response_model=SearchResponse)
async def search_video(video_id: str, request: SearchRequest) -> SearchResponse:
    """Search video content."""
    doc = _get_doc(video_id)
    video = load(doc=doc)

    result = video.search(
        request.query,
        modalities=request.modalities,
        top_k=request.top_k,
        start=request.start,
        end=request.end,
        min_score=request.min_score,
    )

    hits = [
        SearchHit(
            timecode=span.timecode,
            start=span.start,
            end=span.end,
            modality=span.modality,
            text=span.text,
            score=span.score,
            reason=span.reason,
        )
        for span in result.spans
    ]

    return SearchResponse(
        query=request.query,
        hits=hits,
        total=result.total,
        took_ms=result.took_ms,
    )


@app.post("/v1/videos/{video_id}/ask", response_model=AskResponse)
async def ask_video(video_id: str, request: AskRequest) -> AskResponse:
    """Ask a question about the video."""
    doc = _get_doc(video_id)
    video = load(doc=doc)

    answer = video.ask(
        request.question,
        modalities=request.modalities,
        top_k=request.top_k,
        min_score=request.min_score,
    )

    evidence = [
        SearchHit(
            timecode=span.timecode,
            start=span.start,
            end=span.end,
            modality=span.modality,
            text=span.text,
            score=span.score,
            reason=span.reason,
        )
        for span in answer.evidence
    ]

    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        confidence=answer.confidence,
        evidence=evidence,
    )


@app.get("/v1/videos/{video_id}/timeline", response_model=TimelineResponse)
async def get_timeline(
    video_id: str,
    start: float | None = None,
    end: float | None = None,
) -> TimelineResponse:
    """Get timeline view of the video."""
    doc = _get_doc(video_id)

    timeline: list[TimelineEntry] = []

    # Add transcript
    for utt in doc.transcript:
        if start is not None and utt.end < start:
            continue
        if end is not None and utt.start > end:
            continue
        timeline.append(
            TimelineEntry(
                start=utt.start,
                end=utt.end,
                modality="transcript",
                text=utt.text,
            )
        )

    # Add OCR
    for ocr in doc.ocr:
        if start is not None and ocr.end < start:
            continue
        if end is not None and ocr.start > end:
            continue
        timeline.append(
            TimelineEntry(
                start=ocr.start,
                end=ocr.end,
                modality="ocr",
                text=ocr.text,
            )
        )

    # Add events
    for evt in doc.events:
        if start is not None and evt.end < start:
            continue
        if end is not None and evt.start > end:
            continue
        desc = evt.description or evt.type.replace("_", " ")
        timeline.append(
            TimelineEntry(
                start=evt.start,
                end=evt.end,
                modality="events",
                text=f"[{evt.type}] {desc}",
            )
        )

    # Sort by start time
    timeline.sort(key=lambda x: (x.start, x.modality))

    return TimelineResponse(video_id=video_id, timeline=timeline)


@app.get("/v1/videos/{video_id}/segments")
async def get_segments(video_id: str) -> list[dict[str, Any]]:
    """Get segments."""
    doc = _get_doc(video_id)
    return [
        {
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "transcript_ids": seg.transcript_ids,
            "ocr_ids": seg.ocr_ids,
            "vision_ids": seg.vision_ids,
            "event_ids": seg.event_ids,
        }
        for seg in doc.segments
    ]


@app.get("/v1/videos/{video_id}/frames")
async def get_frames(video_id: str) -> list[dict[str, Any]]:
    """Get sampled frames."""
    doc = _get_doc(video_id)
    return [
        {
            "id": frame.id,
            "ts": frame.ts,
            "path": frame.path,
            "reason": frame.reason,
            "sharpness": frame.sharpness,
        }
        for frame in doc.frames
    ]


@app.post("/v1/videos/{video_id}/context", response_model=ContextResponse)
async def get_context(video_id: str, request: ContextRequest) -> ContextResponse:
    """Get optimized AI context for a task."""
    doc = _get_doc(video_id)
    video = load(doc=doc)

    from videocontent.routing import ContextBudget

    context = video.context_for(
        request.task,
        max_tokens=request.max_tokens,
        modalities=request.modalities,
    )

    # Convert evidence to SearchHit format
    evidence_hits = []
    for span in context.evidence:
        evidence_hits.append(SearchHit(
            timecode=span.timecode,
            start=span.start,
            end=span.end,
            modality=span.modality,
            text=span.text,
            score=span.score,
            reason=span.reason,
        ))

    # Convert frames
    frame_data = []
    for frame in context.frames:
        frame_data.append({
            "id": frame.get("id", ""),
            "ts": frame.get("ts", 0.0),
            "path": frame.get("path", ""),
            "reason": frame.get("reason", ""),
        })

    # Convert profiles to dict
    profiles_dict = {}
    for name, profile in context.profiles.items():
        if hasattr(profile, "model_dump"):
            profiles_dict[name] = profile.model_dump()
        else:
            profiles_dict[name] = str(profile)

    # Convert global context
    global_ctx = None
    if context.global_context:
        if hasattr(context.global_context, "model_dump"):
            global_ctx = context.global_context.model_dump()
        else:
            global_ctx = context.global_context

    # Convert summaries
    summaries = context.summaries if context.summaries else {}

    return ContextResponse(
        task=context.task,
        task_type=context.task_type.value if hasattr(context.task_type, "value") else str(context.task_type),
        confidence=context.confidence,
        context=context.context,
        profiles=profiles_dict,
        evidence=evidence_hits,
        frames=frame_data,
        global_context=global_ctx,
        summaries=summaries,
        token_estimate=context.token_estimate,
    )


@app.post("/v1/videos/{video_id}/profile", response_model=ProfileResponse)
async def get_profile(video_id: str, request: ProfileRequest) -> ProfileResponse:
    """Get a specific semantic profile."""
    doc = _get_doc(video_id)
    video = load(doc=doc)

    try:
        profile = video.profile(request.profile_name)
        if hasattr(profile, "model_dump"):
            profile_data = profile.model_dump()
        else:
            profile_data = profile
        return ProfileResponse(
            profile_name=request.profile_name,
            profile=profile_data,
            available=True,
        )
    except ValueError as e:
        return ProfileResponse(
            profile_name=request.profile_name,
            profile=None,
            available=False,
        )


@app.get("/v1/videos/{video_id}/profiles")
async def list_profiles(video_id: str) -> dict[str, Any]:
    """Get all available semantic profiles for this video."""
    doc = _get_doc(video_id)
    video = load(doc=doc)

    profiles = video.profiles()
    result = {}
    for name, profile in profiles.items():
        if hasattr(profile, "model_dump"):
            result[name] = profile.model_dump()
        else:
            result[name] = str(profile)

    return {"profiles": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)