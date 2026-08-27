"""Configuration.

Resolution order (later wins): built-in defaults → ``config.yaml`` → environment
(``VIDEO_CONTEXT_*``) → Python/CLI keyword arguments. Defaults are chosen so that
``Video("x.mp4").process()`` works offline with no configuration at all — the first-five-
minutes rule from the brief (§23).

Every config object is hashable via :meth:`ProcessingConfig.stage_hash`, which is what makes
per-stage caching correct: changing the sampling rate must invalidate frames/OCR/vision but
not ASR.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import ConfigurationError

ENV_PREFIX = "VIDEO_CONTEXT_"
DEFAULT_WORKDIR_NAME = ".videocontent"


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SamplingConfig(_Cfg):
    """Frame sampling — the single biggest determinant of cost and latency."""

    mode: Literal["fixed", "scene", "adaptive"] = "adaptive"
    base_fps: float = Field(default=1.0, gt=0, le=60, description="Baseline frames per second.")
    max_fps: float = Field(default=4.0, gt=0, le=60, description="Ceiling for adaptive bursts.")
    min_fps: float = Field(default=0.2, gt=0, le=60, description="Floor for static content.")
    scene_detection: bool = True
    max_frames: int = Field(default=2000, gt=0, description="Hard cap; protects long videos.")
    scale_width: int | None = Field(
        default=1280, description="Downscale extracted frames; None keeps source size."
    )
    jpeg_quality: int = Field(default=4, ge=1, le=31, description="FFmpeg -q:v (lower=better).")
    dedupe_identical: bool = True
    phash_threshold: int = Field(
        default=4, ge=0, le=64, description="Max Hamming distance treated as a duplicate frame."
    )
    burst_s: float = Field(
        default=1.2, gt=0,
        description="Seconds around each scene boundary sampled at max_fps.",
    )
    static_scene_s: float = Field(
        default=8.0, gt=0,
        description="Scenes longer than this are treated as static and sampled at min_fps.",
    )


class SceneConfig(_Cfg):
    detector: str = "ffmpeg"
    threshold: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Score at or above which a cut is always accepted, regardless of density.",
    )
    min_scene_duration: float = Field(default=1.5, gt=0)
    use_ocr_signal: bool = True
    # analysis pass (cheap: decode small and slow, not full resolution at native rate)
    analysis_fps: float = Field(default=8.0, gt=0, le=60)
    analysis_width: int = Field(default=320, gt=0)
    score_floor: float = Field(
        default=0.015, ge=0.0, le=1.0,
        description="Candidates below this are never considered cuts.",
    )
    max_boundaries_per_minute: float = Field(
        default=30.0, gt=0,
        description="Density budget. Slide decks cut rarely and score low; action footage "
                    "scores high constantly. Budgeting boundaries per minute keeps both "
                    "usable without a per-video threshold.",
    )
    black_detect: bool = True


class OCRConfig(_Cfg):
    provider: str = "tesseract"
    enabled: bool = True
    languages: list[str] = Field(default_factory=lambda: ["eng"])
    min_confidence: float = Field(
        default=0.20, ge=0.0, le=1.0,
        description="Drop words Tesseract scores below this. Deliberately low: Tesseract's "
                    "word confidence is dictionary-biased, so it scores exactly the text this "
                    "project exists to read — URLs, shell commands, file paths, error strings "
                    "— as though it were garbage. A correctly-read 'localhost:3000/login' came "
                    "back at 0.05. On the fixture, raising this to 0.40 removed no junk at all "
                    "and cost real text, so it is a floor for gross garbage only; temporal "
                    "persistence (frame_count/stable) is the precision signal that works. "
                    "See scripts/bench_ocr.py.",
    )
    psm: int = Field(
        default=6, ge=0, le=13,
        description="Tesseract page segmentation mode. 6 (uniform block) measured best on "
                    "this project's fixture: 41 of 47 rendered strings recovered with 0 "
                    "ungrounded events, against 40 and 4 for --psm 11 at the same upscale. "
                    "The sparse modes 11/12 read isolated UI text slightly better but "
                    "fragment tables into disconnected cells and invent more text. "
                    "See scripts/bench_ocr.py.",
    )
    upscale: float = Field(
        default=1.75, ge=1.0, le=4.0,
        description="Enlarge frames before recognition. Small on-screen text — browser URL "
                    "bars, status lines — is where layout analysis fails: at native size "
                    "Tesseract misreads 'localhost' as '{ocalhost', and 1.75x reads it. "
                    "Not a monotonic quality knob, which is why it is not set higher: psm 6 "
                    "assumes one uniform block, and as enlargement widens the gaps between "
                    "elements that assumption breaks. At 2x the fixture's footer line and all "
                    "four chart labels are dropped entirely — 38 of 47 strings against 41 with "
                    "no upscaling at all, for several times the cost. 1.0 disables. "
                    "See scripts/bench_ocr.py.",
    )
    # frame-run deduplication (§27): recognise one frame per run of identical frames
    frame_dedupe: bool = True
    frame_dedupe_threshold: float = Field(
        default=1.0, ge=0.0, le=255.0,
        description="Grey levels (0-255) of most-changed-tile difference below which two "
                    "frames count as the same image. Measured on the fixture: JPEG re-encode "
                    "noise reaches 0.31, a caption covering 0.46% of the frame reaches 4.44.",
    )
    # temporal deduplication (spec §8)
    dedupe: bool = True
    similarity_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    max_gap_s: float = Field(default=3.0, gt=0, description="Bridge gaps up to this long.")
    min_text_length: int = Field(default=2, ge=1)
    drop_numeric_noise: bool = Field(
        default=True,
        description="Drop lines with no alphanumeric character at all — a panel border or "
                    "underline that read as '|' or '--'. It is not a numeric filter despite "
                    "the name: short numbers such as a '$29' price cell are kept, because an "
                    "earlier rule that dropped them caught no real noise on the fixture and "
                    "deleted three prices. See videocontent.processing.ocr.tesseract._is_noise.",
    )


class ASRConfig(_Cfg):
    provider: str = "faster-whisper"
    enabled: bool = True
    model: str = Field(default="base", description="Model size/id, provider-specific.")
    language: str | None = Field(default=None, description="None = auto-detect.")
    task: Literal["transcribe", "translate"] = "transcribe"
    word_timestamps: bool = True
    vad_filter: bool = True
    beam_size: int = Field(default=5, ge=1)
    compute_type: str = "auto"
    device: str = "auto"
    fallback_to_subtitles: bool = True


class VisionConfig(_Cfg):
    provider: str = "null"
    enabled: bool = False
    model: str | None = None
    max_frames: int = Field(default=60, gt=0)
    batch_size: int = Field(default=4, gt=0)
    max_concurrency: int = Field(default=2, gt=0)
    prompt: str | None = None


class EmbeddingConfig(_Cfg):
    provider: str = "null"
    enabled: bool = False
    model: str | None = None
    batch_size: int = Field(default=32, gt=0)
    store: str = "memory"


class SegmentConfig(_Cfg):
    align_to_scenes: bool = True
    max_duration: float = Field(default=45.0, gt=0)
    min_duration: float = Field(default=4.0, gt=0)
    split_on_utterance_boundary: bool = True


class RetrievalConfig(_Cfg):
    modalities: list[str] = Field(
        default_factory=lambda: ["transcript", "ocr", "vision", "events"]
    )
    top_k: int = Field(default=10, gt=0)
    lexical_weight: float = Field(default=1.0, ge=0.0)
    vector_weight: float = Field(default=1.0, ge=0.0)
    cooccurrence_boost: float = Field(
        default=0.25, ge=0.0, description="Bonus when a query hits >1 modality in one span."
    )
    merge_adjacent_s: float = Field(default=2.0, ge=0.0)
    min_score: float = Field(default=0.0, ge=0.0)


class LLMConfig(_Cfg):
    """LLM provider for answer synthesis (used by ask())."""

    provider: str = "null"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, gt=0)


class LimitsConfig(_Cfg):
    """Untrusted-input guardrails (ARCHITECTURE §8)."""

    max_file_size_mb: int = Field(default=8192, gt=0)
    max_duration_s: float = Field(default=6 * 3600, gt=0)
    ffmpeg_timeout_s: float = Field(default=1800, gt=0)
    probe_timeout_s: float = Field(default=60, gt=0)
    allowed_containers: list[str] = Field(
        default_factory=lambda: [
            "mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm", "avi", "mpegts", "flv", "asf", "ogg",
        ]
    )


class ProcessingConfig(_Cfg):
    """The complete knob surface. All fields have working defaults."""

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    scenes: SceneConfig = Field(default_factory=SceneConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    segments: SegmentConfig = Field(default_factory=SegmentConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    workdir: Path | None = Field(
        default=None, description="Artifact dir; default: alongside the video."
    )
    cache_enabled: bool = True
    keep_artifacts: bool = True
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    # -- hashing -----------------------------------------------------------

    def stage_hash(self, *sections: str) -> str:
        """Stable hash of the given config sections; the cache key's config component."""
        payload = {s: getattr(self, s).model_dump(mode="json") for s in sorted(sections)}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def full_hash(self) -> str:
        blob = self.model_dump(mode="json", exclude={"workdir", "log_level", "log_format"})
        return hashlib.sha256(
            json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

#: env var suffix -> dotted config path
ENV_MAP: dict[str, str] = {
    "VISION_PROVIDER": "vision.provider",
    "ASR_PROVIDER": "asr.provider",
    "ASR_MODEL": "asr.model",
    "ASR_LANGUAGE": "asr.language",
    "OCR_PROVIDER": "ocr.provider",
    "OCR_LANGUAGES": "ocr.languages",
    "EMBEDDING_PROVIDER": "embeddings.provider",
    "VECTOR_DB": "embeddings.store",
    "SAMPLING_MODE": "sampling.mode",
    "SAMPLING_BASE_FPS": "sampling.base_fps",
    "SAMPLING_MAX_FPS": "sampling.max_fps",
    "SCENE_DETECTOR": "scenes.detector",
    "SCENE_THRESHOLD": "scenes.threshold",
    "LLM_PROVIDER": "llm.provider",
    "LLM_MODEL": "llm.model",
    "LLM_API_KEY": "llm.api_key",
    "LLM_BASE_URL": "llm.base_url",
    "WORKDIR": "workdir",
    "LOG_LEVEL": "log_level",
    "LOG_FORMAT": "log_format",
    "CACHE_ENABLED": "cache_enabled",
    "MAX_FILE_SIZE_MB": "limits.max_file_size_mb",
    "MAX_DURATION_S": "limits.max_duration_s",
}

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _coerce(value: str) -> Any:
    low = value.strip().lower()
    if low in _TRUTHY:
        return True
    if low in _FALSY:
        return False
    if "," in value:
        return [v.strip() for v in value.split(",") if v.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _assign(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def from_env(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Extract a config overlay from ``VIDEO_CONTEXT_*`` variables."""
    env = environ if environ is not None else dict(os.environ)
    overlay: dict[str, Any] = {}
    for key, raw in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        suffix = key[len(ENV_PREFIX) :]
        dotted = ENV_MAP.get(suffix)
        if dotted is None:
            continue
        _assign(overlay, dotted, _coerce(raw))
    # Turning a provider on implies enabling its stage — the obvious intent.
    for section, provider_key in (("vision", "provider"), ("embeddings", "provider")):
        node = overlay.get(section)
        if isinstance(node, dict) and node.get(provider_key) not in (None, "null"):
            node.setdefault("enabled", True)
    return overlay


def from_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.is_file():
        raise ConfigurationError(f"config file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"{p}: top level must be a mapping")
    return data.get("videocontent", data)


def find_config_file(start: Path | None = None) -> Path | None:
    """Look for ``videocontent.yaml`` / ``config.yaml`` from cwd upward."""
    cur = (start or Path.cwd()).resolve()
    for directory in (cur, *cur.parents):
        for name in ("videocontent.yaml", "videocontent.yml", "config.yaml"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
        if (directory / ".git").exists():
            break
    return None


def load_config(
    overrides: dict[str, Any] | None = None,
    *,
    config_file: str | os.PathLike[str] | None = None,
    use_env: bool = True,
    search: bool = True,
) -> ProcessingConfig:
    """Build a :class:`ProcessingConfig` from all sources in precedence order."""
    layers: dict[str, Any] = {}
    path = Path(config_file) if config_file else (find_config_file() if search else None)
    if path:
        layers = _deep_merge(layers, from_yaml(path))
    if use_env:
        layers = _deep_merge(layers, from_env())
    if overrides:
        layers = _deep_merge(layers, overrides)
    try:
        return ProcessingConfig.model_validate(layers)
    except Exception as exc:
        raise ConfigurationError(f"invalid configuration: {exc}") from exc


__all__ = [
    "DEFAULT_WORKDIR_NAME",
    "ENV_MAP",
    "ENV_PREFIX",
    "ASRConfig",
    "EmbeddingConfig",
    "LimitsConfig",
    "LLMConfig",
    "OCRConfig",
    "ProcessingConfig",
    "RetrievalConfig",
    "SamplingConfig",
    "SceneConfig",
    "SegmentConfig",
    "VisionConfig",
    "find_config_file",
    "from_env",
    "from_yaml",
    "load_config",
]
