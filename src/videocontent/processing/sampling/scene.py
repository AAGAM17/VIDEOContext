"""Scene-driven sampling — the cheapest useful mode.

One representative frame per scene, plus a slow sweep through scenes long enough that their
content is likely to change without the camera cutting. For slide decks and screen recordings
this reads every distinct screen for a tiny fraction of the frames a uniform sweep would
decode; for continuous footage it degrades to the ``min_fps`` sweep, which is the honest
answer when there are no cuts to exploit.
"""

from __future__ import annotations

from ...config import SamplingConfig
from ...interfaces import FrameContext, SamplePlan, SampleWindow, SceneSpan
from .util import fit_budget


class SceneSampler:
    name = "scene"
    version = "1.0.0"

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self.config = config or SamplingConfig()

    def plan(self, ctx: FrameContext, scenes: list[SceneSpan] | None = None) -> SamplePlan:
        cfg = self.config
        if ctx.duration <= 0:
            return SamplePlan(duration=0.0)

        if not scenes:
            # No scene information: fall back to a slow uniform sweep rather than guessing.
            plan = SamplePlan(
                windows=[SampleWindow(0.0, ctx.duration, cfg.min_fps, "no_scenes")],
                duration=ctx.duration,
            )
            return fit_budget(plan, max_frames=cfg.max_frames)

        windows: list[SampleWindow] = []
        explicit: list[tuple[float, str]] = []
        for scene in scenes:
            keyframe = scene.keyframe_ts if scene.keyframe_ts is not None else scene.start
            explicit.append((round(min(keyframe, max(scene.start, scene.end - 0.05)), 3),
                             "scene_keyframe"))
            if scene.end - scene.start > cfg.static_scene_s:
                windows.append(
                    SampleWindow(scene.start, scene.end, cfg.min_fps, "long_scene")
                )

        plan = SamplePlan(windows=windows, explicit=explicit, duration=ctx.duration)
        return fit_budget(plan, max_frames=cfg.max_frames)


__all__ = ["SceneSampler"]
