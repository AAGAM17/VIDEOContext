"""Uniform sampling — the baseline, and the honest comparison point.

Every frame-sampling claim in this project is measured against this sampler: it is what
"sample at 1 fps" means everywhere else, so benchmarks can show what adaptive sampling
actually buys instead of asserting it.
"""

from __future__ import annotations

from ...config import SamplingConfig
from ...interfaces import FrameContext, SamplePlan, SampleWindow, SceneSpan
from .util import fit_budget


class FixedSampler:
    """Sample the whole timeline at one constant rate."""

    name = "fixed"
    version = "1.0.0"

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self.config = config or SamplingConfig()

    def plan(self, ctx: FrameContext, scenes: list[SceneSpan] | None = None) -> SamplePlan:
        if ctx.duration <= 0:
            return SamplePlan(duration=0.0)
        plan = SamplePlan(
            windows=[SampleWindow(0.0, ctx.duration, self.config.base_fps, "fixed")],
            duration=ctx.duration,
        )
        return fit_budget(plan, max_frames=self.config.max_frames)


__all__ = ["FixedSampler"]
