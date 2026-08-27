"""Exception hierarchy.

One root (``VideoContextError``) so callers can catch everything from this library, with
specific subclasses that carry actionable remediation text — the ``hint`` is surfaced by the
CLI, because "OCR failed" without "brew install tesseract" wastes the user's afternoon.
"""

from __future__ import annotations


class VideoContextError(Exception):
    """Base class for every error raised by videocontent."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\nhint: {self.hint}" if self.hint else self.message


class ConfigurationError(VideoContextError):
    """Invalid or contradictory configuration."""


class DependencyMissingError(VideoContextError):
    """A required external binary or optional Python package is unavailable."""


class MediaError(VideoContextError):
    """Base for media-layer failures."""


class UnsupportedMediaError(MediaError):
    """The file is not a media file we can process, or the container is not allowed."""


class CorruptMediaError(MediaError):
    """The container or streams could not be decoded."""


class MediaTooLargeError(MediaError):
    """The input exceeds a configured size or duration limit."""


class FFmpegError(MediaError):
    """An FFmpeg/FFprobe invocation failed."""

    def __init__(self, message: str, *, command: list[str] | None = None,
                 stderr: str | None = None, returncode: int | None = None,
                 hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.command = command or []
        self.stderr = stderr or ""
        self.returncode = returncode


class StageError(VideoContextError):
    """A pipeline stage failed. Non-fatal by design: recorded, then degraded."""

    def __init__(self, stage: str, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.stage = stage


class SchemaError(VideoContextError):
    """A ``.vctx`` document is malformed or violates the specification."""


class UnsupportedVersionError(SchemaError):
    """The document's major version is newer than this reader supports."""


class ProviderError(VideoContextError):
    """A provider adapter (ASR/OCR/vision/embedding) failed."""


class SecurityError(VideoContextError):
    """A request would escape a sandboxed path or violate an input limit."""


__all__ = [
    "ConfigurationError",
    "CorruptMediaError",
    "DependencyMissingError",
    "FFmpegError",
    "MediaError",
    "MediaTooLargeError",
    "ProviderError",
    "SchemaError",
    "SecurityError",
    "StageError",
    "UnsupportedMediaError",
    "UnsupportedVersionError",
    "VideoContextError",
]
