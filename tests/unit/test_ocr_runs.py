"""Frame-run collapsing, and the statistic that makes it safe.

The optimisation is only sound if "these frames look the same" never means "these frames look
the same apart from the text that changed". The central test here is the one that proves the
chosen statistic can tell those apart when the obvious statistic — a whole-frame mean — cannot.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from videocontent.interfaces import FrameImage
from videocontent.processing.ocr.runs import (
    SIGNATURE_SIZE,
    group_runs,
    signature,
    tile_difference,
)


def write(path, array) -> None:
    Image.fromarray(array.astype("uint8"), mode="L").save(path)


def frame(path, ts: float, index: int = 0) -> FrameImage:
    return FrameImage(ts=ts, path=path, index=index, width=64, height=64)


class TestSignature:
    def test_shape_is_square_and_greyscale(self, tmp_path):
        path = tmp_path / "a.png"
        write(path, np.full((720, 1280), 128))
        sig = signature(path)
        assert sig is not None
        assert sig.shape == (SIGNATURE_SIZE, SIGNATURE_SIZE)

    def test_unreadable_file_is_none_not_an_error(self, tmp_path):
        broken = tmp_path / "broken.jpg"
        broken.write_bytes(b"not an image")
        assert signature(broken) is None

    def test_missing_file_is_none(self, tmp_path):
        assert signature(tmp_path / "absent.jpg") is None


class TestTileDifference:
    def test_identical_is_zero(self):
        a = np.full((64, 64), 100.0, dtype=np.float32)
        assert tile_difference(a, a.copy()) == 0.0

    def test_shape_mismatch_is_maximal(self):
        # Never claim two differently-shaped signatures are the same image.
        assert tile_difference(np.zeros((64, 64), np.float32),
                               np.zeros((32, 32), np.float32)) == 255.0

    def test_localised_change_separates_from_global_noise(self):
        """The reason the statistic is a tile maximum and not a frame mean.

        Two changes, each compared against the same base frame:

        * codec noise — every pixel drifts by a fraction of a grey level
        * a caption appearing — a small patch changes completely

        A whole-frame mean cannot separate them: spread over 4096 pixels, the caption moves the
        average *less* than the noise floor does in some encodings, and no single threshold can
        keep one and drop the other. The tile maximum separates them by more than 600x, because
        a local change saturates its own tile.
        """
        base = np.zeros((64, 64), dtype=np.float32)

        noisy = base + 0.4  # uniform low-level drift, the whole frame
        caption = base.copy()
        caption[0:4, 0:4] = 255.0  # one tile's worth of new text, 0.4% of the frame

        mean_noise = float(np.abs(base - noisy).mean())
        mean_caption = float(np.abs(base - caption).mean())
        tile_noise = tile_difference(base, noisy)
        tile_caption = tile_difference(base, caption)

        # A mean cannot be thresholded to keep the caption and drop the noise: they land on
        # the same side of any cut you could pick. (approx, not ==: the signatures are float32.)
        assert mean_noise == pytest.approx(0.4)
        assert mean_caption < 1.0
        assert abs(mean_caption - mean_noise) < 1.0

        # The tile maximum puts them 600x apart, so a threshold of 1.0 is unambiguous.
        assert tile_noise == pytest.approx(0.4)
        assert tile_caption == 255.0
        assert tile_caption / tile_noise > 100


class TestGroupRuns:
    def test_empty(self):
        assert group_runs([], threshold=1.0) == []

    def test_identical_frames_form_one_run(self, tmp_path):
        frames = []
        for i in range(5):
            path = tmp_path / f"f{i}.png"
            write(path, np.full((64, 64), 120))
            frames.append(frame(path, float(i), i))
        runs = group_runs(frames, threshold=1.0)
        assert len(runs) == 1
        assert len(runs[0]) == 5
        assert runs[0][0].ts == 0.0, "the representative is the earliest frame of the run"

    def test_changed_frame_starts_a_new_run(self, tmp_path):
        frames = []
        for i, value in enumerate((120, 120, 20, 20)):
            path = tmp_path / f"f{i}.png"
            write(path, np.full((64, 64), value))
            frames.append(frame(path, float(i), i))
        runs = group_runs(frames, threshold=1.0)
        assert [len(r) for r in runs] == [2, 2]

    def test_small_caption_change_is_not_grouped_away(self, tmp_path):
        """The failure this module exists to prevent, end to end."""
        first = tmp_path / "a.png"
        second = tmp_path / "b.png"
        base = np.full((256, 256), 30, dtype=np.uint8)
        write(first, base)
        with_caption = base.copy()
        with_caption[240:252, 8:120] = 240  # a caption strip along the bottom
        write(second, with_caption)
        runs = group_runs([frame(first, 0.0, 0), frame(second, 1.0, 1)], threshold=1.0)
        assert len(runs) == 2, "text that appeared must be recognised, not skipped"

    def test_unreadable_frame_never_joins_its_neighbour(self, tmp_path):
        good = tmp_path / "good.png"
        write(good, np.full((64, 64), 120))
        broken = tmp_path / "broken.jpg"
        broken.write_bytes(b"not an image")
        runs = group_runs([frame(good, 0.0, 0), frame(broken, 1.0, 1)], threshold=1.0)
        assert len(runs) == 2

    def test_threshold_zero_groups_only_exact_matches(self, tmp_path):
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        write(a, np.full((64, 64), 120))
        write(b, np.full((64, 64), 121))
        assert len(group_runs([frame(a, 0.0, 0), frame(b, 1.0, 1)], threshold=0.0)) == 2
