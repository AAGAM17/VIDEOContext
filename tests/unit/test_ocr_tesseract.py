"""The Tesseract boundary: argv contract, TSV parsing, and batch failure handling.

None of this needs Tesseract installed — the point is to pin the parts that were wrong once and
would be silently wrong again. The argument-order test in particular guards a fault that cost
real debugging time: Tesseract accepted a misordered command line, exited 0, and returned an
empty result, reporting the problem only as a line on stderr.
"""

from __future__ import annotations

import pytest

from videocontext.config import OCRConfig
from videocontext.errors import DependencyMissingError
from videocontext.interfaces import FrameContext, FrameImage
from videocontext.processing.ocr import tesseract as tess

CTX = FrameContext(duration=10.0, fps=30.0, width=1280, height=720)

HEADER = "\t".join((
    "level", "page_num", "block_num", "par_num", "line_num", "word_num",
    "left", "top", "width", "height", "conf", "text",
))


def row(level, page, block, par, line, word, left, top, w, h, conf, text) -> str:
    return "\t".join(str(v) for v in
                     (level, page, block, par, line, word, left, top, w, h, conf, text))


def tsv(*rows: str) -> str:
    return "\n".join((HEADER, *rows)) + "\n"


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(tess, "_binary", lambda: "/usr/bin/tesseract")
    return tess.TesseractOCR(OCRConfig())


class TestArgs:
    def test_tsv_config_file_comes_last(self, engine):
        """``tesseract IMAGE OUTPUT [options] [configfile]``.

        ``tsv`` is a config file name, so anything after it is parsed as further config files.
        With ``... tsv -l eng`` Tesseract looks for config files called ``-l`` and ``eng``,
        prints ``read_params_file: Can't open -l``, exits 0, and emits only a TSV header.
        """
        args = engine._args("/frames/frame.jpg")
        assert args[-1] == "tsv"

    def test_image_and_output_precede_options(self, engine):
        args = engine._args("/frames/f_00001.jpg")
        assert args[1] == "/frames/f_00001.jpg"
        assert args[2] == "stdout"

    def test_language_and_psm_are_passed(self, monkeypatch):
        monkeypatch.setattr(tess, "_binary", lambda: "/usr/bin/tesseract")
        engine = tess.TesseractOCR(OCRConfig(languages=["eng", "deu"], psm=11))
        args = engine._args("/frames/f.jpg")
        assert args[args.index("-l") + 1] == "eng+deu"
        assert args[args.index("--psm") + 1] == "11"

    def test_missing_binary_raises_with_an_install_hint(self, monkeypatch):
        monkeypatch.setattr(tess, "_binary", lambda: None)
        engine = tess.TesseractOCR(OCRConfig())
        assert engine.available() is False
        with pytest.raises(DependencyMissingError) as excinfo:
            engine.extract([FrameImage(ts=0.0, path=__file__, index=0)], CTX)
        assert "tesseract" in str(excinfo.value).lower()


class TestParseTsv:
    def test_words_group_into_lines(self):
        payload = tsv(
            row(1, 1, 0, 0, 0, 0, 0, 0, 1280, 720, -1, ""),
            row(5, 1, 1, 1, 1, 1, 100, 200, 80, 30, 95, "Hello"),
            row(5, 1, 1, 1, 1, 2, 190, 200, 90, 30, 90, "World"),
            row(5, 1, 1, 1, 2, 1, 100, 240, 60, 30, 88, "Second"),
        )
        pages = tess._parse_tsv(payload)
        assert set(pages) == {1}
        assert [len(line) for line in pages[1].lines] == [2, 1]
        assert pages[1].width == 1280
        assert pages[1].height == 720

    def test_page_number_maps_batch_entries(self):
        # page_num is a 1-based counter over the file-list order; it is how a batch result is
        # attributed back to the frame that produced it.
        payload = tsv(
            row(5, 1, 1, 1, 1, 1, 0, 0, 10, 10, 90, "first"),
            row(5, 3, 1, 1, 1, 1, 0, 0, 10, 10, 90, "third"),
        )
        pages = tess._parse_tsv(payload)
        assert set(pages) == {1, 3}

    def test_header_only_output_is_empty(self):
        assert tess._parse_tsv(tsv()) == {}

    def test_blank_and_malformed_rows_are_skipped(self):
        payload = tsv(
            "",
            "garbage",
            "\t".join(["1", "2", "3"]),
            row("x", 1, 1, 1, 1, 1, 0, 0, 10, 10, 90, "bad-level"),
            row(5, 1, 1, 1, 1, 1, 0, 0, 10, 10, 90, "good"),
        )
        pages = tess._parse_tsv(payload)
        assert [w.text for w in pages[1].lines[0]] == ["good"]

    def test_empty_text_rows_are_dropped(self):
        payload = tsv(
            row(5, 1, 1, 1, 1, 1, 0, 0, 10, 10, 90, "   "),
            row(5, 1, 1, 1, 1, 2, 0, 0, 10, 10, 90, "kept"),
        )
        assert [w.text for w in tess._parse_tsv(payload)[1].lines[0]] == ["kept"]

    def test_non_word_levels_are_ignored(self):
        payload = tsv(
            row(2, 1, 1, 0, 0, 0, 0, 0, 100, 50, -1, "block"),
            row(4, 1, 1, 1, 1, 0, 0, 0, 100, 20, -1, "line"),
            row(5, 1, 1, 1, 1, 1, 0, 0, 40, 20, 90, "word"),
        )
        assert [w.text for w in tess._parse_tsv(payload)[1].lines[0]] == ["word"]


class TestObservations:
    def _page(self, *rows):
        return tess._parse_tsv(tsv(*rows))[1]

    def test_line_text_and_geometry(self, engine):
        page = self._page(
            row(1, 1, 0, 0, 0, 0, 0, 0, 1280, 720, -1, ""),
            row(5, 1, 1, 1, 1, 1, 100, 200, 80, 30, 95, "Sign"),
            row(5, 1, 1, 1, 1, 2, 190, 200, 40, 30, 85, "in"),
        )
        frame = FrameImage(ts=1.5, path="x.jpg", index=1, width=1280, height=720)
        [obs] = engine._observations_for(frame, page, CTX)
        assert obs.text == "Sign in"
        assert obs.ts == 1.5
        assert obs.confidence == 0.9
        assert obs.bbox == (100.0, 200.0, 230.0, 230.0)
        assert obs.frame_width == 1280

    def test_upscaled_geometry_is_divided_back_to_source_pixels(self, engine):
        """Stored boxes are always source-frame pixels, whatever was recognised.

        Without this, enabling ``upscale`` would silently double every bbox and every
        normalised coordinate would fall outside the frame.
        """
        page = self._page(
            row(1, 1, 0, 0, 0, 0, 0, 0, 2560, 1440, -1, ""),
            row(5, 1, 1, 1, 1, 1, 200, 400, 160, 60, 95, "Password"),
        )
        frame = FrameImage(ts=0.0, path="x.jpg", index=0, width=1280, height=720)
        [obs] = engine._observations_for(frame, page, CTX, scale=2.0)
        assert obs.bbox == (100.0, 200.0, 180.0, 230.0)
        assert obs.frame_width == 1280
        assert obs.frame_height == 720

    def test_low_confidence_words_are_dropped(self, monkeypatch):
        monkeypatch.setattr(tess, "_binary", lambda: "/usr/bin/tesseract")
        engine = tess.TesseractOCR(OCRConfig(min_confidence=0.5))
        page = self._page(
            row(5, 1, 1, 1, 1, 1, 0, 0, 40, 20, 95, "keep"),
            row(5, 1, 1, 1, 1, 2, 50, 0, 40, 20, 10, "drop"),
        )
        frame = FrameImage(ts=0.0, path="x.jpg", index=0, width=1280, height=720)
        [obs] = engine._observations_for(frame, page, CTX)
        assert obs.text == "keep"

    def test_short_text_is_dropped(self, monkeypatch):
        monkeypatch.setattr(tess, "_binary", lambda: "/usr/bin/tesseract")
        engine = tess.TesseractOCR(OCRConfig(min_text_length=4))
        page = self._page(row(5, 1, 1, 1, 1, 1, 0, 0, 10, 20, 95, "ab"))
        frame = FrameImage(ts=0.0, path="x.jpg", index=0)
        assert engine._observations_for(frame, page, CTX) == []


class TestCleanLine:
    def test_border_glyphs_at_the_ends_are_removed(self):
        # An input field's rounded border reads as a lone pipe either side of its contents.
        assert tess._clean_line("| email@example.com |") == "email@example.com"

    def test_interior_pipe_is_content(self):
        # A pipe inside the text is real — most obviously in a shell command.
        assert tess._clean_line("$ cat log | grep ERROR") == "$ cat log | grep ERROR"

    def test_unchanged_when_clean(self):
        assert tess._clean_line("Quarterly Business Review") == "Quarterly Business Review"


class TestIsNoise:
    @pytest.mark.parametrize("text", ["-", "()", "|", "--", "~", ""])
    def test_punctuation_only_lines_are_noise(self, text):
        assert tess._is_noise(text) is True

    @pytest.mark.parametrize(
        "text", ["Q1", "$29", "$149", "$499", "12", "0:00", "3.4", "ab", "1 failed in 3.42s"]
    )
    def test_anything_alphanumeric_is_kept(self, text):
        assert tess._is_noise(text) is False

    def test_short_price_cells_survive(self):
        """A regression test for real evidence this filter used to delete.

        The rule was once "a short run with no letter is probably a clock tick", which makes
        ``$29`` noise. Measured over the fixture that rule caught nothing at ``--psm 6`` and,
        at ``--psm 11`` where each table cell becomes its own line, deleted all three of the
        pricing slide's figures — the most searchable text on that slide.
        """
        assert [t for t in ("$29", "$149", "$499") if tess._is_noise(t)] == []


class TestBatchTruncation:
    """A bad image aborts a batch: the remaining images are never attempted.

    It truncates the output rather than shifting page numbers, so the recovery is to notice a
    short result and retry the tail one file at a time. Without it, one unreadable frame would
    silently drop every later frame's text — the quiet data loss the evidence-first contract
    exists to prevent.
    """

    def _frames(self, n: int) -> list[tuple[FrameImage, str, float]]:
        return [
            (FrameImage(ts=float(i), path=f"f{i}.jpg", index=i, width=1280, height=720),
             f"f{i}.jpg", 1.0)
            for i in range(n)
        ]

    def test_truncated_batch_retries_the_untouched_tail(self, engine, monkeypatch, tmp_path):
        calls: list[str] = []

        def fake_run(argv, *, timeout):
            target = argv[1]
            calls.append(target)
            if target.endswith(".txt"):
                # Reached only the first entry, then aborted.
                return 0, tsv(row(5, 1, 1, 1, 1, 1, 0, 0, 40, 20, 95, "first")), "Error!"
            return 0, tsv(row(5, 1, 1, 1, 1, 1, 0, 0, 40, 20, 95, f"retried-{target}")), ""

        monkeypatch.setattr(tess, "_run", fake_run)
        monkeypatch.chdir(tmp_path)
        prepared = self._frames(3)
        results = engine._extract_batch([(f, tmp_path / p, s) for f, p, s in prepared], CTX)

        assert len(results) == 3, "every frame must get a result, reached or retried"
        assert results[0][0].text == "first"
        assert results[1][0].text.startswith("retried-")
        assert results[2][0].text.startswith("retried-")
        assert sum(1 for c in calls if c.endswith(".txt")) == 1, "one batch attempt"
        assert len(calls) == 3, "batch, then one retry per unreached frame"

    def test_failed_batch_falls_back_entirely(self, engine, monkeypatch, tmp_path):
        def fake_run(argv, *, timeout):
            if argv[1].endswith(".txt"):
                return 1, "", "boom"
            return 0, tsv(row(5, 1, 1, 1, 1, 1, 0, 0, 40, 20, 95, "single")), ""

        monkeypatch.setattr(tess, "_run", fake_run)
        prepared = [(f, tmp_path / p, s) for f, p, s in self._frames(2)]
        results = engine._extract_batch(prepared, CTX)
        assert [r[0].text for r in results] == ["single", "single"]

    def test_list_file_is_always_removed(self, engine, monkeypatch, tmp_path):
        seen: list[str] = []

        def fake_run(argv, *, timeout):
            seen.append(argv[1])
            return 0, tsv(), ""

        monkeypatch.setattr(tess, "_run", fake_run)
        prepared = [(f, tmp_path / p, s) for f, p, s in self._frames(2)]
        engine._extract_batch(prepared, CTX)
        assert not list(tmp_path.glob("_ocr_batch_*.txt")), "temp list file must not survive"

    def test_frame_failure_returns_no_observations(self, engine, monkeypatch, tmp_path):
        monkeypatch.setattr(tess, "_run", lambda argv, *, timeout: (1, "", "cannot read"))
        path = tmp_path / "x.jpg"
        frame = FrameImage(ts=0.0, path=path, index=0)
        assert engine._extract_one(frame, path, 1.0, CTX) == []
