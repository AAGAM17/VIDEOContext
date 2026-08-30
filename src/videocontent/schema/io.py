"""Reading, writing and validating ``.vctx`` documents.

Serialization is intentionally boring: UTF-8 JSON, optional gzip when the path ends in
``.gz``, atomic writes via a temp file + rename so an interrupted write never truncates an
existing document.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from ..errors import SchemaError, UnsupportedVersionError
from .v1 import VCTX_VERSION, VideoContextDocument

READER_MAJOR = int(VCTX_VERSION.split(".")[0])


def _check_version(raw: dict[str, Any]) -> None:
    version = raw.get("vctx_version")
    if not version:
        raise SchemaError(
            "not a .vctx document: missing 'vctx_version'",
            hint="The file may be plain JSON or from another tool.",
        )
    try:
        major = int(str(version).split(".")[0])
    except ValueError as exc:  # pragma: no cover - defensive
        raise SchemaError(f"malformed vctx_version: {version!r}") from exc
    if major > READER_MAJOR:
        raise UnsupportedVersionError(
            f"document version {version} is newer than this reader (supports "
            f"{READER_MAJOR}.x)",
            hint="Upgrade videocontent: pip install -U videocontent",
        )


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Bring an older document up to the current schema.

    No migrations exist yet (v1.0 is the first release); the seam is here so that a future
    breaking change has an obvious, testable home rather than growing conditionals in the
    models themselves.
    """
    _check_version(raw)
    return raw


def loads(data: str | bytes) -> VideoContextDocument:
    raw = json.loads(data)
    if not isinstance(raw, dict):
        raise SchemaError("a .vctx document must be a JSON object")
    return VideoContextDocument.model_validate(migrate(raw))


def dumps(doc: VideoContextDocument, *, indent: int | None = 2) -> str:
    return doc.model_dump_json(indent=indent, exclude_none=False)


def load(path: str | os.PathLike[str]) -> VideoContextDocument:
    """Load a ``.vctx`` (or ``.vctx.gz``) document."""
    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"no such .vctx file: {p}")
    data = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    try:
        return loads(data)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{p} is not valid JSON: {exc}") from exc


def save(
    doc: VideoContextDocument,
    path: str | os.PathLike[str],
    *,
    indent: int | None = 2,
    compress: bool | None = None,
) -> Path:
    """Write a document atomically. Compresses when the path ends in ``.gz``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dumps(doc, indent=indent).encode("utf-8")
    if compress is None:
        compress = p.suffix == ".gz"
    if compress:
        payload = gzip.compress(payload)

    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, p)
    return p


def json_schema() -> dict[str, Any]:
    """JSON Schema for the document — used by docs and by non-Python consumers."""
    return VideoContextDocument.model_json_schema()


__all__ = ["dumps", "json_schema", "load", "loads", "migrate", "save"]
