"""Job queue implementation using Redis + arq."""

from __future__ import annotations

import importlib.util
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ...config import ProcessingConfig, load_config
from ...logging import get_logger
from ...sdk import Video

log = get_logger("jobs.queue")


def _arq_installed() -> bool:
    return importlib.util.find_spec("arq") is not None


def _redis_installed() -> bool:
    return importlib.util.find_spec("redis") is not None


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    video_path: str
    config: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result_path: str | None = None
    error: str | None = None
    progress: float = 0.0


class JobQueue:
    """Redis-backed job queue for video processing."""

    def __init__(self, redis_url: str = "redis://localhost:6379", queue_name: str = "videocontent") -> None:
        if not _redis_installed():
            raise RuntimeError("redis not installed. pip install redis")

        import redis

        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.queue_name = queue_name
        self.jobs_key = f"{queue_name}:jobs"
        self.queue_key = f"{queue_name}:queue"

    def enqueue(self, video_path: str, config: ProcessingConfig | None = None) -> str:
        """Add a video processing job to the queue."""
        job_id = str(uuid.uuid4())
        cfg = config or ProcessingConfig()

        job = Job(
            id=job_id,
            video_path=video_path,
            config=cfg.model_dump(mode="json", exclude={"workdir", "log_level", "log_format"}),
        )

        # Store job
        self.redis.hset(self.jobs_key, job_id, json.dumps({
            "id": job.id,
            "video_path": job.video_path,
            "config": job.config,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "result_path": job.result_path,
            "error": job.error,
            "progress": job.progress,
        }))

        # Add to queue
        self.redis.lpush(self.queue_key, job_id)

        log.info("job.enqueued", extra={"job_id": job_id, "video": video_path})
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        """Get job status."""
        data = self.redis.hget(self.jobs_key, job_id)
        if not data:
            return None
        j = json.loads(data)
        return Job(
            id=j["id"],
            video_path=j["video_path"],
            config=j["config"],
            status=JobStatus(j["status"]),
            created_at=datetime.fromisoformat(j["created_at"]),
            started_at=datetime.fromisoformat(j["started_at"]) if j["started_at"] else None,
            completed_at=datetime.fromisoformat(j["completed_at"]) if j["completed_at"] else None,
            result_path=j["result_path"],
            error=j["error"],
            progress=j["progress"],
        )

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        progress: float | None = None,
        result_path: str | None = None,
        error: str | None = None,
    ) -> bool:
        """Update job fields."""
        job = self.get_job(job_id)
        if not job:
            return False

        if status:
            job.status = status
            if status == JobStatus.PROCESSING and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.utcnow()
        if progress is not None:
            job.progress = progress
        if result_path:
            job.result_path = result_path
        if error:
            job.error = error

        self.redis.hset(self.jobs_key, job_id, json.dumps({
            "id": job.id,
            "video_path": job.video_path,
            "config": job.config,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "result_path": job.result_path,
            "error": job.error,
            "progress": job.progress,
        }))
        return True

    def get_next_job(self) -> Job | None:
        """Get next job from queue (for worker)."""
        job_id = self.redis.rpop(self.queue_key)
        if not job_id:
            return None
        return self.get_job(job_id)

    def list_jobs(self, status: JobStatus | None = None) -> list[Job]:
        """List all jobs, optionally filtered by status."""
        jobs = []
        for job_id in self.redis.hkeys(self.jobs_key):
            job = self.get_job(job_id)
            if job and (status is None or job.status == status):
                jobs.append(job)
        return jobs


async def process_job(job: Job, redis_url: str) -> None:
    """Process a video job (worker function)."""
    import arq

    queue = JobQueue(redis_url)
    queue.update_job(job.id, status=JobStatus.PROCESSING, progress=0.1)

    try:
        # Load config
        config = ProcessingConfig(**job.config)

        # Process video
        video = Video(job.video_path, config=config)
        queue.update_job(job.id, progress=0.3)
        video.process()
        queue.update_job(job.id, progress=0.8)

        # Save result
        import tempfile
        output_dir = Path(tempfile.gettempdir()) / "videocontent_outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{job.id}.vctx"
        video.save(output_path)

        queue.update_job(
            job.id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            result_path=str(output_path),
        )
        log.info("job.completed", extra={"job_id": job.id, "output": str(output_path)})

    except Exception as exc:
        log.exception("job.failed", extra={"job_id": job.id})
        queue.update_job(
            job.id,
            status=JobStatus.FAILED,
            error=f"{type(exc).__name__}: {exc}",
        )


class ArqWorkerSettings:
    """arq worker settings for video processing."""

    functions = [process_job]
    redis_settings = None  # Configured at runtime
    queue_name = "videocontent"

    @classmethod
    def create(cls, redis_url: str):
        if not _arq_installed():
            raise RuntimeError("arq not installed. pip install arq")

        class Settings(cls):
            redis_settings = arq.RedisSettings.from_url(redis_url)
            queue_name = "videocontent"

        return Settings


__all__ = ["Job", "JobQueue", "JobStatus", "ArqWorkerSettings", "process_job"]