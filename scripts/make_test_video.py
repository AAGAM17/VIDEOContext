#!/usr/bin/env python3
"""Generate the canonical test video for VideoContext (brief §51).

The fixture deliberately contains every property the pipeline must handle:

* spoken narration (synthesized) and a silent stretch
* a presentation with changing slides, including a **fast transition** (0.4 s)
* dense on-screen text: a revenue figure, a pricing table, a competitor slide
* a simulated browser navigating to ``localhost:3000/login``
* a terminal where ``pytest -q`` is typed and an error appears
* multiple coloured objects and several distinct scenes

Usage::

    python scripts/make_test_video.py                       # -> tests/fixtures/demo.mp4
    python scripts/make_test_video.py --out /tmp/demo.mp4 --no-speech

Speech uses macOS ``say`` or Linux ``espeak``/``espeak-ng`` when available; otherwise the
clip is silent and the ASR stage simply reports zero utterances (which is itself a case
worth testing).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 30

BG = (17, 19, 24)
FG = (238, 240, 245)
MUTED = (150, 158, 172)
ACCENT = (99, 179, 237)
GOOD = (72, 187, 120)
BAD = (245, 101, 101)
PANEL = (28, 31, 38)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
MONO_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def _font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    for path in MONO_CANDIDATES if mono else FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


@dataclass
class Shot:
    """One visual segment plus the narration spoken over it."""

    name: str
    duration: float
    draw: str
    narration: str = ""
    payload: dict = field(default_factory=dict)


def _canvas(bg=BG) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), bg)
    return img, ImageDraw.Draw(img)


def slide_title(shot: Shot) -> Image.Image:
    img, d = _canvas()
    d.rectangle([0, 0, W, 8], fill=ACCENT)
    d.text((90, 240), "Quarterly Business Review", font=_font(64), fill=FG)
    d.text((90, 330), "Product, Pricing and Competition", font=_font(36), fill=MUTED)
    d.text((90, 600), "VideoContext demo fixture", font=_font(24), fill=MUTED)
    return img


def slide_agenda(shot: Shot) -> Image.Image:
    img, d = _canvas()
    d.text((90, 90), "Agenda", font=_font(52), fill=FG)
    for i, line in enumerate(
        ["1. Revenue update", "2. Pricing changes", "3. Competitor analysis", "4. Live demo"]
    ):
        d.text((110, 220 + i * 78), line, font=_font(38), fill=FG if i != 3 else ACCENT)
    return img


def slide_revenue(shot: Shot) -> Image.Image:
    img, d = _canvas()
    d.text((90, 80), "Revenue", font=_font(52), fill=FG)
    d.text((90, 180), "Revenue Rs 42L", font=_font(72), fill=GOOD)
    d.text((90, 290), "up 18% quarter over quarter", font=_font(32), fill=MUTED)
    bars = [(0.42, "Q1"), (0.55, "Q2"), (0.68, "Q3"), (0.92, "Q4")]
    base_y, bar_w = 640, 90
    for i, (value, label) in enumerate(bars):
        x = 200 + i * 180
        height = int(240 * value)
        d.rectangle([x, base_y - height, x + bar_w, base_y], fill=ACCENT if i < 3 else GOOD)
        d.text((x + 20, base_y + 12), label, font=_font(26), fill=MUTED)
    return img


def slide_pricing(shot: Shot) -> Image.Image:
    img, d = _canvas()
    d.text((90, 70), "Pricing", font=_font(52), fill=FG)
    rows = [
        ("Plan", "Seats", "Price / month"),
        ("Starter", "3", "$29"),
        ("Growth", "25", "$149"),
        ("Scale", "100", "$499"),
        ("Enterprise", "custom", "contact sales"),
    ]
    top, row_h = 190, 84
    for r, row in enumerate(rows):
        y = top + r * row_h
        if r == 0:
            d.rectangle([90, y, W - 90, y + row_h], fill=PANEL)
        for c, cell in enumerate(row):
            d.text(
                (120 + c * 380, y + 24), cell,
                font=_font(34 if r else 30), fill=FG if r else ACCENT,
            )
        d.line([90, y + row_h, W - 90, y + row_h], fill=(60, 65, 75), width=2)
    return img


def slide_competitor(shot: Shot) -> Image.Image:
    img, d = _canvas()
    d.text((90, 80), "Competitor pricing", font=_font(52), fill=FG)
    d.text((90, 200), "Competitor A charges $2 per hour of video", font=_font(34), fill=FG)
    d.text((90, 270), "Competitor B bundles OCR at $1.40 per hour", font=_font(34), fill=FG)
    d.text((90, 380), "We win on local processing", font=_font(40), fill=GOOD)
    return img


def slide_flash(shot: Shot) -> Image.Image:
    """The fast transition: a high-contrast frame that must not fool scene detection."""
    img, d = _canvas(bg=(250, 250, 252))
    d.text((90, 300), "TRANSITION", font=_font(96), fill=(20, 20, 24))
    return img


def slide_objects(shot: Shot) -> Image.Image:
    img, d = _canvas(bg=(24, 33, 46))
    d.text((90, 70), "Objects on screen", font=_font(44), fill=FG)
    d.rectangle([160, 320, 420, 460], fill=(220, 60, 60))       # red car body
    d.rectangle([210, 260, 370, 330], fill=(200, 90, 90))
    d.ellipse([190, 440, 250, 500], fill=(30, 30, 34))
    d.ellipse([330, 440, 390, 500], fill=(30, 30, 34))
    d.ellipse([620, 300, 800, 480], fill=(72, 187, 120))        # green ball
    d.polygon([(950, 480), (1080, 260), (1210, 480)], fill=(240, 200, 80))  # yellow triangle
    d.text((160, 540), "red car   green ball   yellow triangle", font=_font(28), fill=MUTED)
    return img


def screen_browser(shot: Shot) -> Image.Image:
    img, d = _canvas(bg=(240, 242, 246))
    d.rectangle([0, 0, W, 96], fill=(222, 226, 232))
    for i, colour in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse([30 + i * 30, 34, 50 + i * 30, 54], fill=colour)
    d.rounded_rectangle([150, 26, W - 60, 70], radius=10, fill=(255, 255, 255))
    d.text((172, 36), shot.payload.get("url", "localhost:3000"), font=_font(26), fill=(40, 44, 52))
    d.text((100, 200), shot.payload.get("heading", "Sign in"), font=_font(54), fill=(20, 22, 28))
    d.rounded_rectangle([100, 300, 620, 360], radius=8, fill=(255, 255, 255),
                        outline=(180, 186, 196), width=2)
    d.text((118, 316), "email@example.com", font=_font(28), fill=(120, 126, 138))
    d.rounded_rectangle([100, 390, 620, 450], radius=8, fill=(255, 255, 255),
                        outline=(180, 186, 196), width=2)
    d.text((118, 406), "Password", font=_font(28), fill=(120, 126, 138))
    d.rounded_rectangle([100, 490, 320, 550], radius=8, fill=(56, 132, 255))
    d.text((160, 506), "Log in", font=_font(30), fill=(255, 255, 255))
    return img


def screen_terminal(shot: Shot) -> Image.Image:
    img, d = _canvas(bg=(12, 14, 18))
    d.rectangle([0, 0, W, 60], fill=(30, 34, 42))
    d.text((24, 16), "bash — videocontent", font=_font(24), fill=MUTED)
    mono = _font(26, mono=True)
    y = 110
    for line, colour in shot.payload.get("lines", []):
        d.text((40, y), line, font=mono, fill=colour)
        y += 40
    return img


DRAWERS = {
    "title": slide_title,
    "agenda": slide_agenda,
    "revenue": slide_revenue,
    "pricing": slide_pricing,
    "competitor": slide_competitor,
    "flash": slide_flash,
    "objects": slide_objects,
    "browser": screen_browser,
    "terminal": screen_terminal,
}


def storyboard() -> list[Shot]:
    return [
        Shot("title", 6.0, "title",
             "Welcome to the quarterly business review."),
        Shot("agenda", 6.0, "agenda",
             "Here is the agenda. Revenue, pricing, competitors, and a live demo."),
        Shot("revenue", 8.0, "revenue",
             "Revenue reached forty two lakh rupees this quarter, up eighteen percent."),
        Shot("pricing", 9.0, "pricing",
             "Our pricing starts at twenty nine dollars for the starter plan, "
             "and four hundred ninety nine dollars at scale."),
        Shot("flash", 0.4, "flash", ""),
        Shot("competitor", 8.0, "competitor",
             "Competitor pricing is roughly two dollars per hour of video."),
        Shot("silence", 4.0, "objects", ""),  # deliberately silent stretch
        Shot("browser", 7.0, "browser",
             "Now the demo. The browser opens the login page on localhost port three thousand.",
             payload={"url": "localhost:3000/login", "heading": "Sign in"}),
        Shot("terminal", 9.0, "terminal",
             "In the terminal we run pytest, and one test fails with a connection error.",
             payload={"lines": [
                 ("$ pytest -q tests/", FG),
                 ("collected 12 items", MUTED),
                 ("...........F", FG),
                 ("E   ConnectionError: refused on port 5432", BAD),
                 ("1 failed, 11 passed in 3.42s", BAD),
             ]}),
        Shot("outro", 5.0, "title",
             "That concludes the review. Thank you."),
    ]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def run(cmd: list[str], *, fatal: bool = True) -> bool:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if fatal:
            sys.exit(f"command failed: {' '.join(cmd[:6])}…\n{proc.stderr[-1500:]}")
        return False
    return True


def speech_tool() -> str | None:
    for tool in ("say", "espeak-ng", "espeak"):
        if shutil.which(tool):
            return tool
    return None


PREFERRED_VOICES = ("Samantha", "Alex", "Daniel")
MIN_AUDIO_BYTES = 4096


def say_voice() -> list[str]:
    """``-v VOICE`` for a voice that is actually installed, else the system default."""
    if not shutil.which("say"):
        return []
    listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    installed = listing.stdout
    for voice in PREFERRED_VOICES:
        if voice in installed:
            return ["-v", voice]
    return []


def synth(text: str, out: Path, tool: str) -> bool:
    """Render narration to a 16 kHz mono WAV.

    Returns False on failure. ``say`` exits 0 even when it could not open the output file
    (e.g. an unsupported --data-format), so the result is validated by size rather than by
    exit code — otherwise the fixture silently ends up with no audio at all.
    """
    raw = out.with_suffix(".raw.aiff" if tool == "say" else ".raw.wav")
    raw.unlink(missing_ok=True)
    try:
        if tool == "say":
            # Default AIFF output only: passing --data-format here fails with "fmt?".
            ok = run(["say", *say_voice(), "-o", str(raw), text], fatal=False)
        else:
            ok = run([tool, "-w", str(raw), text], fatal=False)
        if not ok or not raw.exists() or raw.stat().st_size < MIN_AUDIO_BYTES:
            print(f"! speech synthesis produced no audio for: {text[:40]!r}")
            return False
        ok = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                  "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)], fatal=False)
    finally:
        raw.unlink(missing_ok=True)
    if not ok or not out.exists() or out.stat().st_size < MIN_AUDIO_BYTES:
        print(f"! could not convert narration audio for: {text[:40]!r}")
        return False
    return True


def build(out_path: Path, *, speech: bool = True, keep: bool = False) -> dict:
    shots = storyboard()
    work = out_path.parent / f".build_{out_path.stem}"
    work.mkdir(parents=True, exist_ok=True)
    tool = speech_tool() if speech else None
    if speech and not tool:
        print("! no speech synthesizer found (say/espeak); building a silent video")

    timeline: list[dict] = []
    clips: list[Path] = []
    cursor = 0.0
    narrated = 0
    failed_narration: list[str] = []
    wanted_narration = sum(1 for s in shots if s.narration)

    for i, shot in enumerate(shots):
        drawer = DRAWERS[shot.draw]
        png = work / f"{i:02d}_{shot.name}.png"
        drawer(shot).save(png)

        audio: Path | None = None
        if tool and shot.narration:
            candidate = work / f"{i:02d}_{shot.name}.wav"
            if synth(shot.narration, candidate, tool):
                audio = candidate
                narrated += 1
            else:
                failed_narration.append(shot.name)

        clip = work / f"{i:02d}_{shot.name}.mp4"
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-loop", "1", "-framerate", str(FPS), "-t", f"{shot.duration}", "-i", str(png)]
        if audio:
            # Narration first, then silence to fill the shot: apad + shortest keeps the
            # visual duration authoritative so timestamps in the storyboard stay true.
            args += ["-i", str(audio), "-af", "apad", "-shortest"]
        else:
            args += ["-f", "lavfi", "-t", f"{shot.duration}",
                     "-i", "anullsrc=channel_layout=mono:sample_rate=16000"]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-r", str(FPS),
                 "-c:a", "aac", "-b:a", "96k", "-ar", "44100", "-ac", "1",
                 "-t", f"{shot.duration}", "-movflags", "+faststart", str(clip)]
        run(args)
        clips.append(clip)

        timeline.append({
            "shot": shot.name, "screen": shot.draw,
            "start": round(cursor, 2), "end": round(cursor + shot.duration, 2),
            "narration": shot.narration,
            "has_audio": audio is not None,
        })
        cursor += shot.duration

    if tool and narrated == 0 and wanted_narration:
        sys.exit(
            f"speech synthesis failed for every shot ({tool}); refusing to write a fixture "
            "that claims to contain narration. Re-run with --no-speech to build silent video."
        )
    if failed_narration:
        print(f"! narration missing for {len(failed_narration)} shot(s): "
              f"{', '.join(failed_narration)}")

    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out_path)])

    manifest = {
        "video": str(out_path), "duration": round(cursor, 2), "fps": FPS,
        "width": W, "height": H,
        "speech": narrated > 0,
        "narrated_shots": narrated,
        "shots": timeline,
        "expected": {
            "on_screen_text": ["Revenue Rs 42L", "localhost:3000/login", "$ pytest -q tests/",
                               "ConnectionError", "Competitor A charges $2 per hour of video"],
            "spoken_phrases": ["revenue", "pricing", "competitor", "pytest"],
            "silent_shot": "silence",
            "min_scenes": 8,
        },
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/fixtures/demo.mp4")
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--keep-build", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = build(out, speech=not args.no_speech, keep=args.keep_build)

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({manifest['duration']:.1f}s, {size_mb:.1f} MB, "
          f"{len(manifest['shots'])} shots, speech={manifest['speech']})")
    print(f"wrote {out.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
