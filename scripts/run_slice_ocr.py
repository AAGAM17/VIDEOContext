"""Ad-hoc slice runner: probe -> scenes -> adaptive sampling -> frames -> OCR -> dedupe.

Not a test. This is the executable proof that the OCR stage works on real video before it is
wired into the pipeline, per the brief's rule that nothing is assumed to work unexecuted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videocontext.config import ProcessingConfig
from videocontext.interfaces import FrameContext
from videocontext.media.frames import extract_plan
from videocontext.media.probe import probe
from videocontext.media.workspace import Workspace
from videocontext.processing.ocr import deduplicate
from videocontext.processing.ocr.tesseract import TesseractOCR
from videocontext.processing.sampling.adaptive import AdaptiveSampler
from videocontext.processing.scenes.ffmpeg_scene import FFmpegSceneDetector
from videocontext.processing.scenes.refine import refine_with_text

VIDEO = Path("tests/fixtures/demo.mp4")


def main() -> int:
    cfg = ProcessingConfig()
    ws = Workspace.for_video(VIDEO, workdir=".vctx-work")

    t0 = time.perf_counter()
    media = probe(VIDEO)
    ctx = FrameContext(
        duration=media.duration or 0.0,
        fps=media.fps,
        width=media.width,
        height=media.height,
    )
    print(f"probe        {media.duration:.2f}s {media.width}x{media.height} "
          f"{media.fps:.2f}fps  [{time.perf_counter() - t0:.2f}s]")

    t0 = time.perf_counter()
    scenes = FFmpegSceneDetector(cfg.scenes).detect(VIDEO, ctx)
    print(f"scenes       {len(scenes)} spans  [{time.perf_counter() - t0:.2f}s]")

    plan = AdaptiveSampler(cfg.sampling).plan(ctx, scenes)
    print(f"plan         {len(plan.windows)} windows, {len(plan.explicit)} explicit, "
          f"~{plan.estimated_frames()} frames")

    t0 = time.perf_counter()
    frames = extract_plan(VIDEO, plan, ws.frames_dir, scale_width=1280, quality=3)
    print(f"frames       {len(frames)} extracted  [{time.perf_counter() - t0:.2f}s]")

    engine = TesseractOCR(cfg.ocr)
    print(f"engine       tesseract {engine.version} available={engine.available()}")

    t0 = time.perf_counter()
    observations = engine.extract(frames, ctx)
    ocr_s = time.perf_counter() - t0
    print(f"ocr          {len(observations)} observations  [{ocr_s:.2f}s, "
          f"{ocr_s / max(1, len(frames)) * 1000:.0f}ms/frame]")

    t0 = time.perf_counter()
    events = deduplicate(
        observations,
        config=cfg.ocr,
        duration=ctx.duration,
        frame_ts=[f.ts for f in frames],
        engine=engine.name,
    )
    print(f"dedupe       {len(observations)} -> {len(events)} events "
          f"({100 * (1 - len(events) / max(1, len(observations))):.0f}% reduction)  "
          f"[{time.perf_counter() - t0:.2f}s]")

    refined = refine_with_text(scenes, events, duration=ctx.duration,
                               min_scene_duration=cfg.scenes.min_scene_duration)
    print(f"refine       {len(scenes)} -> {len(refined)} scenes after text turnover")

    print("\n--- temporal OCR events ---")
    for event in events:
        flag = "S" if event.stable else " "
        print(f"{flag} [{event.start:6.2f} - {event.end:6.2f}] n={event.frame_count:<3d} "
              f"c={event.confidence if event.confidence is not None else 0:.2f}  "
              f"{event.text[:64]!r}")

    expected = json.loads(Path("tests/fixtures/demo.manifest.json").read_text())
    print("\n--- expected on-screen strings ---")
    haystack = " || ".join(e.text.lower() for e in events)
    for phrase in expected.get("expected", {}).get("on_screen_text", []):
        print(f"{'HIT ' if phrase.lower() in haystack else 'MISS'} {phrase!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
