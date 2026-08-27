"""Collapsing runs of unchanged frames before they reach an OCR engine.

Adaptive sampling decides *when* to look. This decides whether looking again is worth
anything. On this project's fixture, 85% of consecutive sampled frames are bit-identical —
a slide is held while the narrator talks over it — and re-recognising them produced 342
observations that deduplicated to 36 events. Nine tenths of the recognition work was spent
re-reading text already known.

Frames are therefore grouped into *runs* of visually identical frames. The engine recognises
one representative per run, and the result is replayed onto every frame in the run. This is
not an approximation: the frames are the same image, so the text was on screen at each of
those timestamps, and temporal deduplication still sees a continuous observation per frame
and derives the same lifespans it would have derived from full recognition.

**Why a tile maximum and not a mean.** A caption changing in the corner of an otherwise static
1280x720 slide moves perhaps 0.3% of the pixels. Averaged over the frame that is a difference
of well under one grey level — indistinguishable from noise — so a mean-difference test would
declare the frames identical and silently lose the new text. The signature is therefore split
into tiles and the *most changed* tile decides. A localised change saturates its own tile even
when the frame as a whole barely moves.

**Degradation.** Camera grain, film grain and aggressive encoding lift every tile above the
threshold, so nothing is grouped and every frame is recognised. That is the safe direction:
the optimisation stops applying rather than starting to miss text.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, UnidentifiedImageError

from ...interfaces import FrameImage
from ...logging import get_logger

log = get_logger("ocr.runs")

#: Side length of the greyscale signature. 64 keeps a 1280-wide frame's text blocks
#: distinguishable while costing ~10 ms per frame to compute.
SIGNATURE_SIZE = 64
#: Signature is split into TILES x TILES tiles; the most-changed one decides.
TILES = 16


def signature(path, *, size: int = SIGNATURE_SIZE) -> np.ndarray | None:
    """Small greyscale fingerprint of a frame, or None if the image cannot be read."""
    try:
        with Image.open(path) as im:
            return np.asarray(
                im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32
            )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        log.debug("ocr.signature_failed", extra={"path": str(path), "error": str(exc)})
        return None


def tile_difference(a: np.ndarray, b: np.ndarray, *, tiles: int = TILES) -> float:
    """Largest mean absolute difference of any tile, in grey levels (0-255)."""
    if a.shape != b.shape:
        return 255.0
    diff = np.abs(a - b)
    side = max(1, diff.shape[0] // tiles)
    trimmed = diff[: side * tiles, : side * tiles]
    if trimmed.size == 0:  # pragma: no cover - signature smaller than the tile grid
        return float(diff.mean())
    return float(trimmed.reshape(tiles, side, tiles, side).mean(axis=(1, 3)).max())


def group_runs(
    frames: list[FrameImage], *, threshold: float, tiles: int = TILES
) -> list[list[FrameImage]]:
    """Split ``frames`` (ascending by ts) into runs of visually identical frames.

    A run always starts a new group when the signature cannot be computed, so an unreadable
    frame is never treated as identical to its neighbour.
    """
    if not frames:
        return []
    runs: list[list[FrameImage]] = [[frames[0]]]
    previous = signature(frames[0].path)
    for frame in frames[1:]:
        current = signature(frame.path)
        if previous is None or current is None:
            runs.append([frame])
        elif tile_difference(previous, current, tiles=tiles) <= threshold:
            runs[-1].append(frame)
        else:
            runs.append([frame])
        previous = current

    grouped = sum(len(r) - 1 for r in runs)
    log.info(
        "ocr.runs_grouped",
        extra={"frames": len(frames), "runs": len(runs), "skipped": grouped,
               "threshold": threshold},
    )
    return runs


__all__ = ["SIGNATURE_SIZE", "TILES", "group_runs", "signature", "tile_difference"]
