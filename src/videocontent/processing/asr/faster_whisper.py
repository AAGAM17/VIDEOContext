"""faster-whisper ASR adapter.

`faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_ is a CTranslate2 reimplementation
of OpenAI's Whisper: same weights, no PyTorch, and it runs usefully on a laptop CPU — which is
what makes the local-first promise (brief §21) hold for speech, the most expensive stage.

Three details here are not incidental:

* **The import is lazy.** ``faster-whisper`` pulls in CTranslate2 and tokenizers, ~100 MB of
  wheels. ``import videocontent`` must not pay for that, so nothing is imported at module
  scope and :meth:`available` answers by looking for the module rather than loading it.
* **Models are cached across videos.** Loading ``base`` costs seconds; a batch run would pay
  that per file. The cache is keyed on everything that changes the weights.
* **``device`` and ``compute_type`` are resolved, not guessed.** ``"auto"`` is asked of
  CTranslate2 — which quantisations this machine actually supports — instead of a hardcoded
  string that raises on the platforms where it happens to be wrong.

The model download on first use is the one moment this stage touches the network. It goes to
the Hugging Face cache, carries no video data, and is the only externally-reaching step in
the default configuration (brief §32); ``videocontent doctor`` reports whether it has already
happened.
"""

from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from ...config import ASRConfig
from ...errors import DependencyMissingError, ProviderError
from ...interfaces import ASROutput, FrameContext
from ...logging import get_logger
from ...schema.v1 import Word
from .normalize import finalize, utterance

log = get_logger("asr.faster_whisper")

_INSTALL_HINT = (
    "Install the speech extra — pip install 'videocontent[asr]' — or configure another "
    "provider: VIDEO_CONTEXT_ASR_PROVIDER=subtitles (embedded track) or =null (skip speech)."
)

#: Preferred quantisations, best first, filtered against what the device reports supporting.
_CPU_PREFERENCE = ("int8", "int8_float32", "float32")
_GPU_PREFERENCE = ("float16", "int8_float16", "float32")


def installed() -> bool:
    """Whether the package is importable — checked without importing it."""
    return importlib.util.find_spec("faster_whisper") is not None


def resolve_device(requested: str) -> str:
    """Turn ``"auto"`` into a concrete device, so ``compute_type`` can be chosen for it."""
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    # Broad by design: this is a capability probe, and a probe that raises means "no".
    except Exception as exc:
        log.debug("asr.device_probe_failed", extra={"error": str(exc)})
        return "cpu"


def resolve_compute_type(requested: str, device: str) -> str:
    """Pick a quantisation this device supports, preferring speed at equal quality.

    ``"auto"`` is CTranslate2's own token for this, but it resolves per *build* rather than
    per device and returns ``float32`` on CPU builds that support ``int8`` — several times
    slower for no accuracy that matters at ``base``. Asking which types are supported and
    choosing from a preference list is both faster and, unlike a hardcoded ``"int8"``,
    correct on machines where it is unavailable.
    """
    if requested != "auto":
        return requested
    preference = _GPU_PREFERENCE if device.startswith("cuda") else _CPU_PREFERENCE
    try:
        import ctranslate2

        supported = ctranslate2.get_supported_compute_types(device)
    # Broad by design: an unusable probe must degrade to CTranslate2's own default.
    except Exception as exc:
        log.debug("asr.compute_probe_failed", extra={"error": str(exc)})
        return "default"
    for candidate in preference:
        if candidate in supported:
            return candidate
    return "default"


@lru_cache(maxsize=2)
def _load_model(model: str, device: str, compute_type: str) -> Any:
    """Load and cache a WhisperModel. Keyed on everything that changes the weights.

    ``maxsize=2`` is deliberate: a cached model holds hundreds of megabytes resident, and the
    access pattern that matters — many videos, one configuration — needs exactly one slot.
    """
    from faster_whisper import WhisperModel

    log.info(
        "asr.model_loading",
        extra={"model": model, "device": device, "compute_type": compute_type},
    )
    try:
        return WhisperModel(model, device=device, compute_type=compute_type)
    # Broad by design: model loading fails in many library-specific ways; all of them
    # mean the same thing to a caller, so they arrive as one error type.
    except Exception as exc:
        raise ProviderError(
            f"could not load Whisper model {model!r}: {type(exc).__name__}: {exc}",
            hint="Check the model name, or the network/HF cache if this is a first run.",
        ) from exc


def _confidence(avg_logprob: float | None) -> float | None:
    """Whisper reports mean per-token log probability; exponentiating gives a usable score.

    The result is the geometric mean of the token probabilities — a real quantity, not a
    rescaling: 0.9 means the model was, on average, 90% confident per token. Values are
    clamped into ``[0, 1]`` because a positive ``avg_logprob`` is possible in principle.

    **It is a per-window score, not a per-utterance one.** Whisper decodes 30-second windows,
    and faster-whisper copies the window's ``avg_logprob`` onto every segment that came out of
    it — so consecutive utterances routinely carry the *identical* confidence (measured on the
    test fixture: five utterances at 0.7975, then three at 0.7463). That is the model's own
    number and it is reported unchanged rather than replaced by something derived, but a caller
    that wants a score specific to one utterance should use its word probabilities, which are
    carried through per word in :func:`_words`.
    """
    if avg_logprob is None:
        return None
    try:
        return min(1.0, max(0.0, math.exp(float(avg_logprob))))
    except (OverflowError, ValueError):  # pragma: no cover - defensive
        return None


class FasterWhisperASR:
    """Local speech-to-text. Audio never leaves the machine; only weights are fetched."""

    name = "faster-whisper"
    remote = False

    def __init__(self, config: ASRConfig | None = None) -> None:
        self.config = config or ASRConfig()

    @property
    def version(self) -> str:
        if not installed():
            return "unavailable"
        try:
            from importlib.metadata import version as pkg_version

            return f"{pkg_version('faster-whisper')}/{self.config.model}"
        # Broad by design: provenance is nice to have; it must not fail a run.
        except Exception:
            return self.config.model

    def available(self) -> bool:
        return installed()

    def transcribe(self, audio_path: Path, ctx: FrameContext) -> ASROutput:
        if not installed():
            raise DependencyMissingError(
                "faster-whisper is not installed", hint=_INSTALL_HINT
            )
        audio = Path(audio_path)
        if not audio.is_file():
            raise ProviderError(f"audio file not found: {audio}")

        cfg = self.config
        device = resolve_device(cfg.device)
        compute_type = resolve_compute_type(cfg.compute_type, device)
        model = _load_model(cfg.model, device, compute_type)

        try:
            segments, info = model.transcribe(
                str(audio),
                language=cfg.language or ctx.language,
                task=cfg.task,
                beam_size=cfg.beam_size,
                word_timestamps=cfg.word_timestamps,
                vad_filter=cfg.vad_filter,
            )
            # `segments` is a generator: the work happens here, not in the call above.
            raw = list(segments)
        # Broad by design: CTranslate2 raises its own types; re-raised in our taxonomy.
        except Exception as exc:
            raise ProviderError(
                f"transcription failed: {type(exc).__name__}: {exc}",
                hint="Try asr.vad_filter=false, or a different model size.",
            ) from exc

        language = getattr(info, "language", None) or cfg.language or ctx.language
        duration = float(getattr(info, "duration", 0.0) or 0.0) or ctx.duration

        utterances = finalize(
            [
                utterance(
                    segment.text,
                    segment.start,
                    segment.end,
                    confidence=_confidence(getattr(segment, "avg_logprob", None)),
                    language=language,
                    no_speech_prob=getattr(segment, "no_speech_prob", None),
                    words=_words(segment),
                )
                for segment in raw
            ],
            duration=duration,
            language=language,
        )

        warnings: list[str] = []
        if not utterances:
            # An empty transcript is a *result*, not a failure: the audio may be music or
            # silence. Saying so keeps "ran and found no speech" distinguishable from
            # "never ran" in the document (ARCHITECTURE §7).
            warnings.append("no speech detected in the audio")

        log.info(
            "asr.transcribed",
            extra={
                "model": cfg.model,
                "device": device,
                "compute_type": compute_type,
                "language": language,
                "utterances": len(utterances),
                "words": sum(len(u.words) for u in utterances),
            },
        )
        return ASROutput(
            utterances=utterances,
            language=language,
            model=f"{self.name}:{cfg.model}",
            duration_s=duration,
            warnings=warnings,
        )


def _words(segment: Any) -> list[Word]:
    """Convert faster-whisper word timings, dropping the leading space it emits."""
    out: list[Word] = []
    for word in getattr(segment, "words", None) or []:
        text = str(getattr(word, "word", "")).strip()
        if not text:
            continue
        out.append(
            Word(
                text=text,
                start=max(0.0, float(word.start)),
                end=max(float(word.start), float(word.end)),
                confidence=getattr(word, "probability", None),
            )
        )
    return out


__all__ = [
    "FasterWhisperASR",
    "installed",
    "resolve_compute_type",
    "resolve_device",
]
