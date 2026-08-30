"""Structured logging.

Two formats: human-readable text for the CLI, JSON for production. Both emit *fields*, not
interpolated prose, so timings and counts are queryable.

**Privacy rule (ARCHITECTURE §8):** log identifiers, durations and counts — never transcript
text, OCR text, frame bytes or prompts. :func:`redact` exists for the rare case where a
snippet genuinely helps debugging; it truncates and marks the value.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any

LOGGER_NAME = "videocontent"
_CONFIGURED = False

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
    "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(
            {
                k: v for k, v in record.__dict__.items()
                if k not in _RESERVED and not k.startswith("_")
            }
        )
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[2;37m", "INFO": "\033[36m", "WARNING": "\033[33m",
        "ERROR": "\033[31m", "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def __init__(self, *, color: bool = True) -> None:
        super().__init__()
        self.color = color and sys.stderr.isatty() and not os.environ.get("NO_COLOR")

    def format(self, record: logging.LogRecord) -> str:
        fields = {
            k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
        }
        extra = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
        level = record.levelname.lower()
        if self.color:
            tint = self.COLORS.get(record.levelname, "")
            level = f"{tint}{level:<7}{self.RESET}"
        else:
            level = f"{level:<7}"
        line = f"{level} {record.getMessage()}"
        if extra:
            line = f"{line}  {extra}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    text = str(value)
    return f'"{text}"' if " " in text else text


def configure(level: str = "INFO", fmt: str = "text", *, force: bool = False) -> None:
    """Install a handler on the ``videocontent`` logger. Idempotent."""
    global _CONFIGURED
    logger = logging.getLogger(LOGGER_NAME)
    if _CONFIGURED and not force:
        logger.setLevel(level.upper())
        return
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


def redact(text: str, limit: int = 24) -> str:
    """Truncate potentially sensitive content for debug logs."""
    if len(text) <= limit:
        return f"<{len(text)}c>"
    return f"<{len(text)}c:{text[:limit]!r}…>"


@contextmanager
def timed(logger: logging.Logger, event: str, **fields: Any):
    """Log ``event`` with its wall-clock duration, on success or failure."""
    start = time.perf_counter()
    logger.debug(f"{event}.start", extra=fields)
    try:
        yield fields
    except Exception:
        fields["duration_s"] = round(time.perf_counter() - start, 3)
        logger.warning(f"{event}.failed", extra=fields, exc_info=True)
        raise
    else:
        fields["duration_s"] = round(time.perf_counter() - start, 3)
        logger.info(f"{event}.done", extra=fields)


__all__ = [
    "LOGGER_NAME",
    "JsonFormatter",
    "TextFormatter",
    "configure",
    "get_logger",
    "redact",
    "timed",
]
