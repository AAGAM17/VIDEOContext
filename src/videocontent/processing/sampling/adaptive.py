"""Adaptive sampling — the default mode.

The rule the brief insists on: *never process every frame blindly.* Information in video is
distributed extremely unevenly. A slide held for forty seconds contains one screen's worth of
content; the half-second where it is replaced contains the transition, the new title, and the
moment a viewer would point at. Uniform sampling spends the same budget on both.

So the rate follows the content:

===============================  ==========  ===================================================
region                           rate        why
===============================  ==========  ===================================================
around a scene boundary          ``max_fps``  where change happens; worth oversampling
a scene shorter than a burst     ``max_fps``  too brief to risk missing entirely
a scene longer than static_s     ``min_fps``  held content; one read plus drift checks is enough
everything else                  ``base_fps`` ordinary pacing
each scene's keyframe            explicit     guarantees every scene is represented at all
===============================  ==========  ===================================================

Without scene information this degrades to uniform ``base_fps`` sampling — stated plainly
rather than pretending to be adaptive.
"""

from __future__ import annotations

from ...config import SamplingConfig
from ...interfaces import FrameContext, SamplePlan, SampleWindow, SceneSpan
from ...logging import get_logger
from .util import fit_budget

log = get_logger("sampling.adaptive")


class AdaptiveSampler:
    name = "adaptive"
    version = "1.0.0"

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self.config = config or SamplingConfig()

    def plan(self, ctx: FrameContext, scenes: list[SceneSpan] | None = None) -> SamplePlan:
        cfg = self.config
        duration = ctx.duration
        if duration <= 0:
            return SamplePlan(duration=0.0)

        if not scenes:
            log.debug("sampling.uniform_fallback", extra={"reason": "no_scenes"})
            plan = SamplePlan(
                windows=[SampleWindow(0.0, duration, cfg.base_fps, "fixed")],
                duration=duration,
            )
            return fit_budget(plan, max_frames=cfg.max_frames)

        windows: list[SampleWindow] = []
        explicit: list[tuple[float, str]] = []

        for scene in scenes:
            length = max(0.0, scene.end - scene.start)
            if length <= cfg.burst_s * 2:
                rate, reason = cfg.max_fps, "short_scene"
            elif length > cfg.static_scene_s:
                rate, reason = cfg.min_fps, "static_scene"
            else:
                rate, reason = cfg.base_fps, "scene"
            windows.append(SampleWindow(scene.start, scene.end, rate, reason))

            keyframe = scene.keyframe_ts if scene.keyframe_ts is not None else scene.start
            explicit.append(
                (round(min(keyframe, max(scene.start, scene.end - 0.05)), 3), "scene_keyframe")
            )

            # Oversample across the incoming boundary — the transition itself plus the first
            # moment of new content, which is where titles and slide headers appear.
            if scene.start > 0.0:
                windows.append(
                    SampleWindow(
                        max(0.0, scene.start - cfg.burst_s / 2),
                        min(duration, scene.start + cfg.burst_s),
                        cfg.max_fps,
                        "boundary_burst",
                    )
                )

        plan = SamplePlan(windows=windows, explicit=explicit, duration=duration)
        fitted = fit_budget(plan, max_frames=cfg.max_frames)
        log.info(
            "sampling.planned",
            extra={
                "mode": self.name,
                "scenes": len(scenes),
                "windows": len(fitted.windows),
                "estimated_frames": fitted.estimated_frames(),
            },
        )
        return fitted


__all__ = ["AdaptiveSampler"]
