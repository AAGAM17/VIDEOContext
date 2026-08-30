"""Measure the OCR stage over a whole fixture, end to end, across its cost/quality knobs.

Choosing Tesseract's ``--psm``, an upscale factor and a confidence floor by intuition is how
OCR quality quietly regresses: the mode that reads a dense slide best is not the mode that
reads browser chrome best, and the floor that removes junk also removes correctly-read URLs.
This runs the real stage — engine plus temporal deduplication — once per combination and
scores it, so the defaults in :class:`OCRConfig` are measured choices (brief §33).

Two scores, because recall alone is not quality:

* **recall** — how many of the strings actually drawn into the fixture come back.
* **junk** — events matching nothing that was ever on screen. A setting that reads everything
  by also inventing everything is not an improvement, and this is the axis a
  five-string acceptance list cannot see.

``GROUND_TRUTH`` is the text ``scripts/make_test_video.py`` draws, transcribed from its drawer
functions. The manifest's ``expected.on_screen_text`` is the smaller acceptance list the tests
assert on; this is the fuller list needed to compare settings.

Usage::

    python scripts/run_slice_ocr.py      # extract frames first
    python scripts/bench_ocr.py          # then score them
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videocontent.config import OCRConfig
from videocontent.interfaces import FrameContext, FrameImage
from videocontent.processing.ocr import deduplicate, normalize, similarity
from videocontent.processing.ocr.tesseract import TesseractOCR

DEFAULT_FRAMES = Path(".vctx-work/demo/frames/pass")
MANIFEST = Path("tests/fixtures/demo.manifest.json")

#: Every string drawn by scripts/make_test_video.py, per slide. Keep in step with its drawers.
GROUND_TRUTH: list[str] = [
    # title / outro
    "Quarterly Business Review", "Product, Pricing and Competition",
    "VideoContext demo fixture",
    # agenda
    "Agenda", "1. Revenue update", "2. Pricing changes", "3. Competitor analysis",
    "4. Live demo",
    # revenue — the bar labels are small, muted and isolated, which is exactly where layout
    # analysis struggles; omitting them once made a correct read score as invented text.
    "Revenue", "Revenue Rs 42L", "up 18% quarter over quarter",
    "Q1", "Q2", "Q3", "Q4",
    # pricing table — the figures are the point: a mode that reads the labels but drops
    # the prices has lost the slide's content while still scoring well on labels.
    "Pricing", "Plan", "Seats", "Price / month",
    "Starter", "$29", "Growth", "$149", "Scale", "$499",
    "Enterprise", "custom", "contact sales",
    # competitor
    "Competitor pricing", "Competitor A charges $2 per hour of video",
    "Competitor B bundles OCR at $1.40 per hour", "We win on local processing",
    # transition / objects
    "TRANSITION", "Objects on screen", "red car", "green ball", "yellow triangle",
    # browser
    "localhost:3000/login", "Sign in", "email@example.com", "Password", "Log in",
    # terminal
    "bash", "$ pytest -q tests/", "collected 12 items",
    "ConnectionError: refused on port 5432", "1 failed, 11 passed in 3.42s",
]

#: An event counts as grounded if it looks like something that was on screen. Generous on
#: purpose: the junk score should flag invented text, not punish a slightly ragged read.
JUNK_SIMILARITY = 0.60


def grounded(text: str, truth: list[str]) -> bool:
    norm = normalize(text)
    if not norm:
        return False
    for line in truth:
        low = normalize(line)
        if norm in low or low in norm or similarity(norm, low) >= JUNK_SIMILARITY:
            return True
    return False


def score(events, truth: list[str]) -> tuple[list[str], list]:
    """Return (missed ground-truth strings, ungrounded events)."""
    haystack = " || ".join(normalize(e.text) for e in events)
    missed = [line for line in truth if normalize(line) not in haystack]
    junk = [e for e in events if not grounded(e.text, truth)]
    return missed, junk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--psm", type=int, nargs="+", default=[6, 11])
    ap.add_argument("--upscale", type=float, nargs="+", default=[1.0, 1.75, 2.0])
    ap.add_argument("--min-conf", type=float, nargs="+", default=[0.20, 0.40])
    ap.add_argument(
        "--drop-noise", type=int, nargs="+", default=[1, 0], choices=(0, 1),
        help="Punctuation-only-line filter (OCRConfig.drop_numeric_noise): 1 on, 0 off. "
             "Swept by default because an earlier, wider version of that filter deleted the "
             "pricing slide's figures at --psm 11 while catching nothing at --psm 6.",
    )
    ap.add_argument("--duration", type=float, default=0.0, help="0 = read from manifest")
    ap.add_argument("--detail", action="store_true", help="Print per-string and junk detail.")
    args = ap.parse_args()

    paths = sorted(args.frames.glob("f_*.jpg"))
    if not paths:
        print(f"no frames in {args.frames}; run scripts/run_slice_ocr.py first")
        return 1

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    duration = args.duration or float(manifest.get("duration") or 0.0) or len(paths)
    acceptance = manifest.get("expected", {}).get("on_screen_text", [])

    # Timestamps are not recoverable from filenames, so use a uniform grid: this comparison is
    # about text, and deduplication only needs a consistent, ordered clock.
    step = duration / len(paths)
    frames = [
        FrameImage(ts=round(i * step, 3), path=p, index=i, width=1280, height=720)
        for i, p in enumerate(paths)
    ]
    ctx = FrameContext(duration=duration, fps=30.0, width=1280, height=720)

    print(f"{len(frames)} frames | {len(GROUND_TRUTH)} on-screen strings | "
          f"{len(acceptance)} acceptance strings\n")
    hdr = (f"{'psm':>3} {'up':>4} {'conf':>5} {'nf':>3} | {'obs':>4} {'evt':>4} {'secs':>5} | "
           f"{'recall':>7} {'accept':>6} {'junk':>5}")
    print(hdr)
    print("-" * len(hdr))

    results: dict[tuple[int, float, float, int], tuple] = {}
    for psm in args.psm:
        for upscale in args.upscale:
            for min_conf in args.min_conf:
                for drop_noise in args.drop_noise:
                    cfg = OCRConfig(
                        psm=psm, upscale=upscale, min_confidence=min_conf,
                        drop_numeric_noise=bool(drop_noise),
                    )
                    engine = TesseractOCR(cfg)
                    t0 = time.perf_counter()
                    observations = engine.extract(frames, ctx)
                    elapsed = time.perf_counter() - t0
                    events = deduplicate(
                        observations, config=cfg, duration=duration,
                        frame_ts=[f.ts for f in frames], engine=engine.name,
                    )
                    missed, junk = score(events, GROUND_TRUTH)
                    hay = " || ".join(normalize(e.text) for e in events)
                    accepted = sum(1 for p in acceptance if normalize(p) in hay)
                    results[(psm, upscale, min_conf, drop_noise)] = (events, missed, junk)
                    print(f"{psm:>3} {upscale:>4} {min_conf:>5} {drop_noise:>3} | "
                          f"{len(observations):>4} {len(events):>4} {elapsed:>5.1f} | "
                          f"{len(GROUND_TRUTH) - len(missed):>3}/{len(GROUND_TRUTH):<3} "
                          f"{accepted:>3}/{len(acceptance):<2} {len(junk):>5}")

    if args.detail:
        for key, (_events, missed, junk) in results.items():
            print(f"\n=== psm={key[0]} upscale={key[1]} min_conf={key[2]} drop_noise={key[3]} ===")
            print(f"  missed ({len(missed)}): {'; '.join(missed) or '-'}")
            for event in junk:
                conf = event.confidence if event.confidence is not None else 0.0
                print(f"  junk c={conf:.2f} {event.text[:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
