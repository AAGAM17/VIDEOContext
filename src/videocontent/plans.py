"""Task-aware processing plans — inspectable, never auto-executed.

A plan answers "what would this task need?" as data: which pipeline stages matter
and why. It composes the existing stage catalog; it never duplicates processors
and never triggers processing on its own (an unneeded vision call costs real money
and may move data off-machine, so execution stays an explicit operator decision).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .logging import get_logger

log = get_logger("plans")


@dataclass(frozen=True)
class PlannedStage:
    name: str
    why: str
    required: bool = True


@dataclass(frozen=True)
class ProcessingPlan:
    task: str
    stages: tuple[PlannedStage, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def required_stages(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.stages if s.required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "stages": [{"name": s.name, "why": s.why, "required": s.required}
                       for s in self.stages],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CoverageGap:
    stage: str
    hint: str


@dataclass(frozen=True)
class CoverageReport:
    satisfied: tuple[str, ...] = ()
    missing: tuple[CoverageGap, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "satisfied": list(self.satisfied),
            "missing": [{"stage": g.stage, "hint": g.hint} for g in self.missing],
            "complete": self.complete,
        }


#: signal keyword → (stages, explanation). Order matters: first match wins per stage.
_SIGNAL_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("speech|said|saying|transcript|spoken|audio|quote", ("audio", "asr"),
     "the task references what was said"),
    ("text|ocr|screen|slide|written|shown|display|error|command|visible", ("frames", "ocr"),
     "the task needs visible text"),
    ("visual|look|design|color|layout|appearance|ui|interface|animation", ("frames", "vision"),
     "the task needs visual understanding"),
    ("scene|shot|cut|transition", ("scenes",),
     "the task needs temporal transitions"),
    ("event|happen|changed|click|navigate|action", ("events",),
     "the task needs typed temporal events"),
)

_RERUN_HINTS = {
    "audio": "reprocess with asr.enabled=true",
    "asr": "reprocess with asr.enabled=true (and a working ASR provider)",
    "frames": "reprocess with a higher sampling rate",
    "ocr": "reprocess with ocr.enabled=true (requires tesseract)",
    "vision": "reprocess with vision.enabled=true and a configured vision provider",
    "scenes": "reprocess with sampling.scene_detection=true",
    "events": "reprocess normally — events derive from extracted modalities",
    "segments": "reprocess normally — segments derive from extracted modalities",
}


def plan_for_task(task: str) -> ProcessingPlan:
    """Map a task description to the stages it needs, with reasons.

    Always includes ``segments`` (the retrieval unit). ``vision`` stays optional
    unless the task needs visual understanding — it is the only stage that can
    send data off-machine, so it must never be selected silently by default.
    """
    lowered = task.lower()
    wanted: dict[str, str] = {}
    for pattern, stages, why in _SIGNAL_RULES:
        if re.search(pattern, lowered):
            for stage in stages:
                wanted.setdefault(stage, why)
    if not wanted:
        wanted = {
            "scenes": "general understanding needs temporal structure",
            "frames": "general understanding needs visual anchors",
            "ocr": "general understanding needs visible text",
            "audio": "general understanding needs the audio track",
            "asr": "general understanding needs speech",
            "events": "general understanding needs typed events",
        }
    ordered = ["scenes", "frames", "ocr", "audio", "asr", "vision", "events", "segments"]
    stages = [PlannedStage(name, wanted[name], True)
              for name in ordered if name in wanted]
    stages.append(PlannedStage(
        "segments", "segments fuse modalities into the retrieval unit", True))
    notes = []
    if not any(s.name == "vision" for s in stages):
        notes.append("vision not selected: enable explicitly if the task needs pixels, "
                     "not just text")
    return ProcessingPlan(task, tuple(stages), tuple(notes))


def check_coverage(doc: Any, plan: ProcessingPlan) -> CoverageReport:
    """Which planned stages already have results in this document?

    Reuses the document's own stage records — the same ``ran`` guard that keeps
    "nothing found" distinct from "never ran".
    """
    satisfied: list[str] = []
    missing: list[CoverageGap] = []
    for stage in plan.required_stages:
        if doc.ran(stage):
            satisfied.append(stage)
        else:
            missing.append(CoverageGap(stage, _RERUN_HINTS.get(stage, "reprocess")))
    return CoverageReport(tuple(satisfied), tuple(missing))


__all__ = ["CoverageGap", "CoverageReport", "PlannedStage", "ProcessingPlan",
           "check_coverage", "plan_for_task"]
