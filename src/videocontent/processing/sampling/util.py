"""Budget enforcement shared by all samplers.

``max_frames`` is the guardrail that stops a three-hour video from silently becoming a
100 000-frame OCR job. Enforcing it *in the plan* rather than by truncating the extracted
frames matters: truncation would cover only the beginning of the video and quietly leave the
rest unanalysed, while scaling the rates keeps coverage uniform across the whole timeline.
"""

from __future__ import annotations

from ...interfaces import SamplePlan, SampleWindow
from ...logging import get_logger

log = get_logger("sampling.budget")


def fit_budget(plan: SamplePlan, *, max_frames: int, min_fps: float = 0.05) -> SamplePlan:
    """Scale window rates down until the plan fits ``max_frames``.

    Explicit timestamps (scene keyframes) are never scaled away — they are the cheapest,
    highest-value frames in the plan. Only the continuous windows are thinned.
    """
    estimate = plan.estimated_frames()
    if estimate <= max_frames or not plan.windows:
        return plan

    explicit = len(plan.explicit)
    room = max(1, max_frames - explicit)
    window_estimate = max(1, estimate - explicit)
    factor = room / window_estimate

    scaled = [
        SampleWindow(w.start, w.end, max(min_fps, w.fps * factor), w.reason)
        for w in plan.windows
    ]
    fitted = SamplePlan(windows=scaled, explicit=list(plan.explicit), duration=plan.duration)
    log.info(
        "sampling.budget_applied",
        extra={
            "estimated": estimate,
            "max_frames": max_frames,
            "factor": round(factor, 4),
            "now": fitted.estimated_frames(),
        },
    )
    return fitted


__all__ = ["fit_budget"]
