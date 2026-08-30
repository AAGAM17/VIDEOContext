"""Tesseract OCR engine.

Tesseract is called as a binary rather than through ``pytesseract`` so that the argv, the
timeout and the process group stay under this project's control (ARCHITECTURE §8) — the same
reasoning as :mod:`videocontent.media.ffmpeg`.

**Batching.** One ``tesseract`` process per frame costs ~0.15 s of start-up on top of ~0.15 s
of recognition, so a 91-frame video spends half its OCR budget spawning processes. Tesseract
accepts a *file list* in place of an image path and recognises every entry in one process,
which halves that. Two behaviours of that mode are load-bearing here:

* ``page_num`` in the TSV output is a 1-based counter over the list order, so it maps back to
  the frame that produced each row.
* An unreadable image **aborts the whole batch** — the remaining images are not attempted and
  no output is produced for them. It does not shift the page numbering, it truncates it. So a
  short batch is detected and its remainder retried one file at a time; otherwise a single bad
  frame would silently drop every later frame's text, which is precisely the sort of quiet
  data loss the evidence-first contract forbids.

**Cost.** Two measured decisions keep this affordable. Frames that show the same screen are
collapsed into runs by :mod:`videocontent.processing.ocr.runs` and only one member of each run
is recognised — on this project's fixture that is 10 recognitions instead of 91, cutting OCR
from 18.3 s to 2.8 s. Part of that saving is then spent enlarging frames before recognition,
because small text (a browser's URL bar) is unreadable at native size. Enlargement is not a
free win: it widens the gaps between elements, which degrades the ``--psm 6`` single-block
assumption, so past ~1.75x it starts dropping isolated lines that were read fine at native
size. Both knobs are scored over the whole fixture by ``scripts/bench_ocr.py``.

**Argument order.** ``tesseract IMAGE OUTPUT [options] [configfile]``. The trailing ``tsv`` is
a *config file name* and must come last: anything after it is parsed as further config files,
so ``… tsv -l eng`` makes Tesseract look for a config file called ``-l``. It reports this only
as ``read_params_file: Can't open -l`` on stderr and still exits 0 with an empty result.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from PIL import Image

from ...config import OCRConfig
from ...errors import DependencyMissingError
from ...interfaces import FrameContext, FrameImage, OCRObservation
from ...logging import get_logger
from .runs import group_runs

log = get_logger("ocr.tesseract")

#: Frames per ``tesseract`` invocation. Large enough to amortise start-up, small enough that
#: one unreadable frame costs a short retry rather than the whole video.
BATCH_SIZE = 32

_INSTALL_HINT = (
    "Install Tesseract — macOS: brew install tesseract · "
    "Debian/Ubuntu: apt install tesseract-ocr · Windows: winget install UB-Mannheim.TesseractOCR"
)

#: TSV columns emitted by Tesseract's ``tsv`` config, in order.
_LEVEL, _PAGE, _BLOCK, _PAR, _LINE, _WORD = 0, 1, 2, 3, 4, 5
_LEFT, _TOP, _WIDTH, _HEIGHT, _CONF, _TEXT = 6, 7, 8, 9, 10, 11
_LEVEL_PAGE, _LEVEL_WORD = 1, 5

#: Tesseract ``traineddata`` name → BCP-47, which is what the spec requires on every textual
#: object (spec §3). Tesseract names its models with ISO 639-2/T codes and its own script
#: suffixes; both have to be translated, because ``"eng"`` in a document field that a consumer
#: filters on as ``"en"`` is a silent miss. Only codes that differ are listed — a two-letter
#: code, and a three-letter one with no shorter equivalent (``fil``, ``tgl``), are already
#: valid tags and pass through.
_BCP47: dict[str, str] = {
    "afr": "af", "amh": "am", "ara": "ar", "aze": "az", "bel": "be", "ben": "bn",
    "bul": "bg", "cat": "ca", "ces": "cs", "chi_sim": "zh-Hans", "chi_tra": "zh-Hant",
    "cym": "cy", "dan": "da", "deu": "de", "ell": "el", "eng": "en", "est": "et",
    "eus": "eu", "fas": "fa", "fin": "fi", "fra": "fr", "gle": "ga", "glg": "gl",
    "guj": "gu", "heb": "he", "hin": "hi", "hrv": "hr", "hun": "hu", "hye": "hy",
    "ind": "id", "isl": "is", "ita": "it", "jpn": "ja", "kan": "kn", "kat": "ka",
    "kaz": "kk", "khm": "km", "kor": "ko", "lao": "lo", "lav": "lv", "lit": "lt",
    "mal": "ml", "mar": "mr", "mkd": "mk", "mon": "mn", "msa": "ms", "mya": "my",
    "nep": "ne", "nld": "nl", "nor": "no", "ori": "or", "pan": "pa", "pol": "pl",
    "por": "pt", "ron": "ro", "rus": "ru", "sin": "si", "slk": "sk", "slv": "sl",
    "spa": "es", "sqi": "sq", "srp": "sr", "srp_latn": "sr-Latn", "swa": "sw",
    "swe": "sv", "tam": "ta", "tel": "te", "tha": "th", "tur": "tr", "ukr": "uk",
    "urd": "ur", "uzb": "uz", "vie": "vi", "yid": "yi",
}

#: Tesseract models that are not languages: orientation/script detection and the equation
#: model. Loading one says nothing about what language a line is in.
_NOT_A_LANGUAGE = frozenset({"osd", "equ"})

#: Suffixes Tesseract appends for a variant of the same language. Stripped before lookup so
#: ``chi_sim_vert`` resolves as ``chi_sim``; ``_frak`` marks Fraktur typesetting, not a
#: different language.
_MODEL_SUFFIXES = ("_vert", "_frak")


def language_tag(codes: list[str]) -> str | None:
    """The BCP-47 tag for a Tesseract language configuration, or ``None`` if unknowable.

    Returns ``None`` for a multi-language configuration on purpose. Tesseract loads several
    models at once and reports *no* per-line attribution, so with ``eng+hin`` enabled there is
    no way to say which language a given line was read as. The spec defines ``null`` as
    "unknown", which is exactly the state we are in — where the previous behaviour emitted
    ``"eng+hin"``, a string that is not a language tag in any registry and that no consumer
    could match against.
    """
    usable = [c for c in (code.strip().lower() for code in codes) if c and c not in _NOT_A_LANGUAGE]
    if len(usable) != 1:
        return None
    code = usable[0]
    for suffix in _MODEL_SUFFIXES:
        code = code.removesuffix(suffix)
    return _BCP47.get(code, code)



@lru_cache(maxsize=1)
def _binary() -> str | None:
    override = os.environ.get("VIDEO_CONTEXT_TESSERACT")
    if override and Path(override).exists():
        return override
    return shutil.which("tesseract")


@lru_cache(maxsize=1)
def _version() -> str | None:
    exe = _binary()
    if not exe:
        return None
    code, out, err = _run([exe, "--version"], timeout=15.0)
    if code != 0:
        return None
    first = (out or err).strip().splitlines()
    if not first:
        return None
    parts = first[0].split()
    return parts[1] if len(parts) > 1 else first[0]


def _run(argv: list[str], *, timeout: float) -> tuple[int, str, str]:
    """Spawn ``argv`` with the same hardening as the FFmpeg boundary.

    Deliberately not :func:`videocontent.media.ffmpeg.run`: that function owns FFmpeg's flag
    contract and error taxonomy, and widening it to arbitrary binaries would make it leaky in
    exchange for saving these few lines.
    """
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        raise DependencyMissingError(
            f"could not start tesseract: {exc}", hint=_INSTALL_HINT
        ) from exc
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
            proc.kill()
        out, err = proc.communicate()
        log.warning("ocr.timeout", extra={"timeout_s": timeout})
        return 124, out, err
    return proc.returncode, out, err


class TesseractOCR:
    """Local OCR. Nothing leaves the machine."""

    name = "tesseract"
    remote = False

    def __init__(self, config: OCRConfig | None = None, *, timeout: float = 600.0) -> None:
        self.config = config or OCRConfig()
        self.timeout = timeout

    @property
    def version(self) -> str:
        return _version() or "unknown"

    def available(self) -> bool:
        return _binary() is not None

    # -- invocation ---------------------------------------------------------

    def _args(self, target: str) -> list[str]:
        exe = _binary()
        if not exe:  # pragma: no cover - guarded by available()
            raise DependencyMissingError("tesseract was not found on PATH", hint=_INSTALL_HINT)
        langs = "+".join(self.config.languages) or "eng"
        # Options first, `tsv` config file LAST — see module docstring.
        return [exe, target, "stdout", "-l", langs, "--psm", str(self.config.psm), "tsv"]

    def extract(
        self, frames: list[FrameImage], ctx: FrameContext
    ) -> list[OCRObservation]:
        if not frames:
            return []
        if not self.available():
            raise DependencyMissingError(
                "tesseract was not found on PATH", hint=_INSTALL_HINT
            )

        readable = [f for f in frames if f.path.is_file()]
        if len(readable) != len(frames):
            log.warning(
                "ocr.frames_missing", extra={"missing": len(frames) - len(readable)}
            )
        if not readable:
            return []

        readable.sort(key=lambda f: f.ts)
        if self.config.frame_dedupe:
            runs = group_runs(readable, threshold=self.config.frame_dedupe_threshold)
        else:
            runs = [[frame] for frame in readable]

        per_run = self._recognise([run[0] for run in runs], ctx)

        # Replay the representative's readings onto the rest of its run. The frames are the
        # same image, so the text was on screen at those timestamps too; carrying them through
        # is what lets temporal deduplication derive the same lifespans it would have derived
        # from recognising every frame.
        observations: list[OCRObservation] = []
        for run, found in zip(runs, per_run, strict=True):
            observations.extend(found)
            for member in run[1:]:
                observations.extend(replace(obs, ts=member.ts) for obs in found)

        observations.sort(key=lambda o: (o.ts, o.block_index or 0))
        log.info(
            "ocr.extracted",
            extra={
                "engine": self.name,
                "frames": len(readable),
                "recognised": len(runs),
                "observations": len(observations),
            },
        )
        return observations

    def _recognise(
        self, frames: list[FrameImage], ctx: FrameContext
    ) -> list[list[OCRObservation]]:
        """Recognise each frame, returning one observation list per input frame."""
        results: list[list[OCRObservation]] = []
        for start in range(0, len(frames), BATCH_SIZE):
            chunk = frames[start : start + BATCH_SIZE]
            with self._prepared(chunk) as prepared:
                results += self._extract_batch(prepared, ctx)
        return results

    @contextmanager
    def _prepared(self, frames: list[FrameImage]):
        """Yield ``(frame, path_to_recognise, scale)`` triples, upscaling when configured.

        Enlarged copies are written beside the frames — a directory already known writable and
        on the same filesystem — and removed on the way out, including on error (§32: no
        derived image outlives the run that made it).
        """
        scale = float(self.config.upscale)
        if scale <= 1.0:
            yield [(frame, frame.path, 1.0) for frame in frames]
            return

        made: list[Path] = []
        try:
            triples: list[tuple[FrameImage, Path, float]] = []
            for frame in frames:
                target = frame.path.parent / f"_ocr_up_{int(frame.ts * 1000):09d}.jpg"
                try:
                    with Image.open(frame.path) as image:
                        image.convert("RGB").resize(
                            (int(image.width * scale), int(image.height * scale)),
                            Image.LANCZOS,
                        ).save(target, quality=95)
                except (OSError, ValueError) as exc:
                    log.debug(
                        "ocr.upscale_failed", extra={"ts": frame.ts, "error": str(exc)}
                    )
                    triples.append((frame, frame.path, 1.0))
                    continue
                made.append(target)
                triples.append((frame, target, scale))
            yield triples
        finally:
            for path in made:
                path.unlink(missing_ok=True)

    def _extract_batch(
        self, prepared: list[tuple[FrameImage, Path, float]], ctx: FrameContext
    ) -> list[list[OCRObservation]]:
        """One process for the batch, falling back per-file for whatever it did not reach."""
        if not prepared:
            return []
        if len(prepared) == 1:
            return [self._extract_one(*prepared[0], ctx)]

        # The list file goes beside the frames: that directory is known-writable and on the
        # same filesystem, and it is cleaned up with the rest of the workspace.
        first = prepared[0][1]
        list_path = first.parent / f"_ocr_batch_{int(prepared[0][0].ts * 1000):09d}.txt"
        try:
            list_path.write_text(
                "".join(f"{path.resolve()}\n" for _f, path, _s in prepared), encoding="utf-8"
            )
            code, out, err = _run(self._args(str(list_path)), timeout=self.timeout)
        finally:
            list_path.unlink(missing_ok=True)

        if code != 0:
            log.warning("ocr.batch_failed", extra={"returncode": code, "n": len(prepared)})
            return [self._extract_one(f, p, s, ctx) for f, p, s in prepared]

        pages = _parse_tsv(out)
        reached = max(pages, default=0)
        results: list[list[OCRObservation]] = []
        for page, (frame, _path, scale) in enumerate(prepared, start=1):
            if page > reached:
                break
            results.append(
                self._observations_for(frame, pages.get(page, _Page()), ctx, scale)
            )

        if reached < len(prepared):
            # Batch aborted part-way (see module docstring). Retry the untouched tail.
            tail = prepared[reached:]
            log.warning(
                "ocr.batch_truncated",
                extra={"reached": reached, "expected": len(prepared), "retrying": len(tail)},
            )
            if err.strip():
                log.debug("ocr.batch_stderr", extra={"stderr": err.strip()[-400:]})
            results += [self._extract_one(f, p, s, ctx) for f, p, s in tail]
        return results

    def _extract_one(
        self, frame: FrameImage, path: Path, scale: float, ctx: FrameContext
    ) -> list[OCRObservation]:
        code, out, err = _run(self._args(str(path.resolve())), timeout=self.timeout)
        if code != 0:
            log.warning(
                "ocr.frame_failed",
                extra={"ts": frame.ts, "returncode": code, "stderr": err.strip()[-200:]},
            )
            return []
        pages = _parse_tsv(out)
        return self._observations_for(frame, pages.get(1, _Page()), ctx, scale)

    # -- interpretation -----------------------------------------------------

    def _observations_for(
        self, frame: FrameImage, page: _Page, ctx: FrameContext, scale: float = 1.0
    ) -> list[OCRObservation]:
        cfg = self.config
        min_conf = cfg.min_confidence * 100.0
        language = language_tag(cfg.languages)
        # Geometry comes back in the coordinates of whatever was recognised, so an upscaled
        # pass must be divided back down: stored boxes are always source-frame pixels.
        factor = scale if scale > 0 else 1.0
        width = int(page.width / factor) if page.width else (frame.width or ctx.width)
        height = int(page.height / factor) if page.height else (frame.height or ctx.height)

        observations: list[OCRObservation] = []
        for index, line in enumerate(page.lines):
            words = [w for w in line if w.conf >= min_conf and w.text.strip()]
            if not words:
                continue
            text = _clean_line(" ".join(w.text for w in words))
            if len(text) < cfg.min_text_length:
                continue
            if cfg.drop_numeric_noise and _is_noise(text):
                continue
            observations.append(
                OCRObservation(
                    text=text,
                    ts=frame.ts,
                    confidence=round(sum(w.conf for w in words) / len(words) / 100.0, 4),
                    bbox=(
                        round(min(w.left for w in words) / factor, 1),
                        round(min(w.top for w in words) / factor, 1),
                        round(max(w.left + w.width for w in words) / factor, 1),
                        round(max(w.top + w.height for w in words) / factor, 1),
                    ),
                    language=language,
                    block_index=index,
                    frame_width=width,
                    frame_height=height,
                )
            )
        return observations


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------


class _Word:
    __slots__ = ("conf", "height", "left", "text", "top", "width")

    def __init__(self, left: int, top: int, width: int, height: int, conf: float, text: str):
        self.left, self.top, self.width, self.height = left, top, width, height
        self.conf, self.text = conf, text


class _Page:
    """Words of one recognised image, grouped into lines in reading order."""

    __slots__ = ("_lines", "height", "width")

    def __init__(self) -> None:
        self.width: int | None = None
        self.height: int | None = None
        self._lines: dict[tuple[int, int, int], list[_Word]] = {}

    def add(self, key: tuple[int, int, int], word: _Word) -> None:
        self._lines.setdefault(key, []).append(word)

    @property
    def lines(self) -> list[list[_Word]]:
        return [self._lines[k] for k in sorted(self._lines)]


def _parse_tsv(payload: str) -> dict[int, _Page]:
    """Group Tesseract's TSV rows by page number (1-based, in list order)."""
    pages: dict[int, _Page] = {}
    for raw in payload.splitlines():
        if not raw or raw.startswith("level\t"):
            continue
        cols = raw.split("\t")
        if len(cols) <= _TEXT:
            continue
        try:
            level = int(cols[_LEVEL])
            page_num = int(cols[_PAGE])
        except ValueError:
            continue
        page = pages.setdefault(page_num, _Page())
        if level == _LEVEL_PAGE:
            try:
                page.width = int(cols[_WIDTH])
                page.height = int(cols[_HEIGHT])
            except ValueError:
                pass
            continue
        if level != _LEVEL_WORD:
            continue
        text = "\t".join(cols[_TEXT:]).strip()
        if not text:
            continue
        try:
            page.add(
                (int(cols[_BLOCK]), int(cols[_PAR]), int(cols[_LINE])),
                _Word(
                    int(cols[_LEFT]), int(cols[_TOP]), int(cols[_WIDTH]),
                    int(cols[_HEIGHT]), float(cols[_CONF]), text,
                ),
            )
        except ValueError:
            continue
    return pages


def _clean_line(text: str) -> str:
    """Drop border glyphs Tesseract reads off UI chrome.

    An input field's rounded border becomes a standalone ``|`` token on either side of its
    contents (``| email@example.com |``), which would otherwise end up in the searchable text
    and in the box geometry. Only *lone* pipes at the ends are removed — a pipe inside the
    text is real content, most obviously in a shell command.
    """
    tokens = text.split()
    while tokens and tokens[0] in {"|", "!", "]", "["}:
        tokens.pop(0)
    while tokens and tokens[-1] in {"|", "!", "]", "["}:
        tokens.pop()
    return " ".join(tokens).strip()


def _is_noise(text: str) -> bool:
    """Lines with nothing searchable in them: no letter, no digit, only punctuation.

    Deliberately narrow, because the obvious wider rule was measured and was wrong. This
    previously dropped any short run without a letter — "digits and punctuation, so probably a
    clock tick". Over the whole fixture that rule fired **zero** times at the default
    ``--psm 6`` and, at ``--psm 11`` where table cells become their own lines, its every firing
    was real on-screen text: ``$29``, ``$149``, ``$499`` and two chart-axis numbers. A price is
    exactly the kind of short numeric string a user searches a video for, so the filter was
    pure downside — no noise caught, real evidence deleted, and it made ``--psm 11`` look 3
    strings worse than it is.

    Punctuation-only lines are worth dropping: a panel border or an underline reads as ``|``,
    ``--`` or ``~``, which is unsearchable by construction. Across the 24-configuration sweep
    the narrowed rule removes exactly one ungrounded event and no real text, which is the whole
    of its earned keep — pass ``--drop-noise 1 0`` to ``scripts/bench_ocr.py`` to see both
    columns. (:func:`_clean_line` removes border glyphs flanking real text; this removes a line
    that was *only* ever chrome.) Anything with an alphanumeric character survives here and is
    judged on temporal persistence instead — see ``OCRConfig.min_confidence``.
    """
    return not any(ch.isalnum() for ch in text)


__all__ = ["BATCH_SIZE", "TesseractOCR", "language_tag"]
