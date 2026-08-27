"""Normative validation of a ``.vctx`` document (spec §16).

Pydantic guarantees *shape*; these are the *semantic* rules that make a document
trustworthy: sorted timelines, resolvable references, non-overlapping scenes and segments,
timestamps inside the media duration.

``validate()`` returns findings instead of raising, so the CLI can print all problems at
once and tests can assert on specific violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from .v1 import VideoContextDocument

DURATION_TOLERANCE = 0.5  # container durations round; don't fail a document over 40 ms

_TIMED = ("scenes", "transcript", "ocr", "vision", "objects", "events", "segments")

_REF_FIELDS: dict[str, str] = {
    "scene_ids": "scenes",
    "transcript_ids": "transcript",
    "ocr_ids": "ocr",
    "vision_ids": "vision",
    "event_ids": "events",
    "object_ids": "objects",
    "frame_ids": "frames",
}


@dataclass(frozen=True)
class Finding:
    rule: str
    message: str
    severity: str = "error"  # error | warning

    def __str__(self) -> str:
        return f"[{self.severity}] {self.rule}: {self.message}"


def validate(doc: VideoContextDocument, *, strict: bool = False) -> list[Finding]:
    """Check the normative rules. ``strict`` promotes warnings to errors."""
    findings: list[Finding] = []
    duration = doc.video.duration
    limit = duration + DURATION_TOLERANCE

    # Rule 1 — version
    if not doc.vctx_version:
        findings.append(Finding("version", "vctx_version is missing"))

    # Rules 2 & 3 & 6 — bounds, ordering, confidence
    for coll_name in _TIMED:
        items: list[Any] = getattr(doc, coll_name)
        previous = None
        for item in items:
            where = f"{coll_name}[{getattr(item, 'id', '?')}]"
            if item.start < 0:
                findings.append(Finding("bounds", f"{where} start {item.start} < 0"))
            if duration and item.end > limit:
                findings.append(
                    Finding("bounds", f"{where} end {item.end} exceeds duration {duration}")
                )
            if item.end < item.start:
                findings.append(Finding("bounds", f"{where} end < start"))
            if previous is not None and (item.start, item.end) < (previous.start, previous.end):
                findings.append(
                    Finding("ordering", f"{coll_name} is not sorted at {where}")
                )
            conf = getattr(item, "confidence", None)
            if conf is not None and not (0.0 <= conf <= 1.0):
                findings.append(Finding("confidence", f"{where} confidence {conf} outside [0,1]"))
            previous = item

    for frame in doc.frames:
        if frame.ts < 0 or (duration and frame.ts > limit):
            findings.append(
                Finding("bounds", f"frames[{frame.id}] ts {frame.ts} outside media duration")
            )

    # Rule 4 — unique IDs
    seen: dict[str, str] = {}
    for coll_name in (*_TIMED, "frames"):
        for item in getattr(doc, coll_name):
            oid = item.id
            if oid in seen:
                findings.append(
                    Finding("unique_id", f"duplicate id {oid!r} in {seen[oid]} and {coll_name}")
                )
            else:
                seen[oid] = coll_name

    # Rule 5 — references resolve
    known = set(seen)
    for segment in doc.segments:
        for field in _REF_FIELDS:
            for ref in getattr(segment, field, []) or []:
                if ref not in known:
                    findings.append(
                        Finding(
                            "dangling_ref",
                            f"segments[{segment.id}].{field} -> {ref!r} does not exist",
                        )
                    )
    for event in doc.events:
        for target, refs in (event.refs or {}).items():
            for ref in refs:
                if ref not in known:
                    findings.append(
                        Finding(
                            "dangling_ref",
                            f"events[{event.id}].refs[{target}] -> {ref!r} does not exist",
                        )
                    )

    # Rule 7 — scenes and segments tile without overlap
    for coll_name in ("scenes", "segments"):
        items = getattr(doc, coll_name)
        for a, b in pairwise(items):
            if b.start < a.end - 1e-6:
                findings.append(
                    Finding(
                        "overlap",
                        f"{coll_name} overlap: {a.id} ends {a.end}, {b.id} starts {b.start}",
                    )
                )

    # Rule 8 — one record per stage name
    stage_names: set[str] = set()
    for record in doc.stages:
        if record.name in stage_names:
            findings.append(Finding("stages", f"duplicate stage record {record.name!r}"))
        stage_names.add(record.name)

    # Warnings: things that are legal but usually mistakes.
    if doc.segments and not doc.stages:
        findings.append(
            Finding("provenance", "document has segments but no stage records", "warning")
        )
    if doc.ocr and not any(o.frame_count > 1 for o in doc.ocr) and len(doc.ocr) > 20:
        findings.append(
            Finding(
                "ocr_dedup",
                "no OCR entry spans more than one frame — temporal dedup may not have run",
                "warning",
            )
        )

    if strict:
        findings = [Finding(f.rule, f.message, "error") for f in findings]
    return findings


def is_valid(doc: VideoContextDocument) -> bool:
    return not [f for f in validate(doc) if f.severity == "error"]


__all__ = ["DURATION_TOLERANCE", "Finding", "is_valid", "validate"]
