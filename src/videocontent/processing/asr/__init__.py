"""Speech transcription.

Same two-part split as :mod:`videocontent.processing.ocr`: an
:class:`~videocontent.interfaces.ASREngine` turns audio into utterances, and
:func:`~videocontent.processing.asr.normalize.finalize` turns whatever it produced into
well-formed, id-assigned, timeline-ordered spans. A contributor adding WhisperX, a cloud
speech API or a diarizing engine writes the first part only.

**The fallback chain.** Speech is the one stage whose provider can be absent on a working
install — the model weights are a separate download, and the base package deliberately does
not depend on them (ARCHITECTURE §7). :func:`resolve_engine` implements the documented
degradation: the configured provider, then the video's own embedded subtitle track, then the
explicit no-op. Each step down is returned as a *warning*, not swallowed, so the ``.vctx``
document records that a substitution happened and against what — a transcript from subtitles
and a transcript from Whisper are both valid evidence, but they are not the same claim.
"""

from __future__ import annotations

from typing import Any

from ...config import ASRConfig
from ...errors import ConfigurationError
from ...logging import get_logger
from .normalize import finalize, utterance
from .null import NullASR

log = get_logger("asr")

#: Providers tried after the configured one, in order. ``null`` is last and always succeeds.
FALLBACK_ORDER = ("subtitles", "null")

#: Built-ins that can read a transcript out of the container rather than the audio, and so
#: need the video path. Any other engine is constructed with its config alone.
_NEEDS_SOURCE = frozenset({"subtitles"})


def resolve_engine(
    config: ASRConfig | None = None, *, source: Any = None
) -> tuple[Any, list[str]]:
    """Return ``(engine, warnings)``: the first available provider in the chain.

    ``source`` is the video path, needed only by the subtitle engine (see its module
    docstring for why the protocol hands engines the audio instead). Warnings name every
    provider that was skipped, so a document that says "transcribed from subtitles" also says
    what it tried first.
    """
    cfg = config or ASRConfig()
    if not cfg.enabled:
        # `enabled=False` is the off-switch, and it must not fall through to the chain: a
        # disabled stage that quietly resolved to Whisper would be the opposite of what the
        # caller asked for. NullASR still records that the stage ran and why it is empty.
        return NullASR(cfg), []

    candidates = [cfg.provider]
    for name in FALLBACK_ORDER:
        if name == "subtitles" and not cfg.fallback_to_subtitles:
            continue
        if name not in candidates:
            candidates.append(name)

    warnings: list[str] = []
    skipped: list[str] = []
    for name in candidates:
        engine = _build(name, cfg, source, skipped)
        if engine is None:
            continue
        if engine.available():
            if skipped:
                warnings.append(
                    f"ASR provider {cfg.provider!r} unavailable "
                    f"({'; '.join(skipped)}); substituted {name!r}"
                )
                log.warning(
                    "asr.provider_substituted",
                    extra={"requested": cfg.provider, "using": name},
                )
            return engine, warnings
        skipped.append(f"{name}: not installed or not usable here")

    # Only reachable if the registry has been emptied — NullASR is always available.
    raise ConfigurationError(
        f"no usable ASR provider: tried {', '.join(candidates)}",
        hint="Set asr.provider to 'null' to skip speech entirely.",
    )


def _build(name: str, cfg: ASRConfig, source: Any, skipped: list[str]) -> Any | None:
    from ...registry import create

    kwargs: dict[str, Any] = {"config": cfg}
    if name in _NEEDS_SOURCE and source is not None:
        kwargs["source"] = source
    try:
        return create("asr", name, **kwargs)
    except ConfigurationError as exc:
        # An unregistered name is a configuration mistake worth reporting, not a crash: the
        # chain exists precisely so a typo in one provider still yields a usable run.
        # `.message` rather than `str(exc)`: the hint is multi-line, and this text ends up in
        # a JSON document's warnings list, not on a terminal.
        skipped.append(f"{name}: {exc.message}")
        return None


__all__ = ["FALLBACK_ORDER", "finalize", "resolve_engine", "utterance"]
