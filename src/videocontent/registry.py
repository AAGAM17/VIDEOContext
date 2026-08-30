"""Plugin registry.

Capabilities are resolved by *name* so configuration can be plain strings from an env var,
a YAML file or a CLI flag. Registration supports two forms:

* eager — ``@register_ocr("tesseract")`` on a class in this codebase
* lazy — a ``"module:attr"`` spec, imported only when the plugin is actually instantiated

The lazy form is what keeps ``import videocontent`` fast and the base install small: the
faster-whisper adapter's dependencies are not touched unless ASR actually runs.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from .errors import ConfigurationError
from .logging import get_logger

T = TypeVar("T")

log = get_logger("registry")

Capability = str
Factory = Callable[..., Any]

#: capability -> name -> factory or "module:attr" spec
_REGISTRY: dict[Capability, dict[str, Factory | str]] = {}
_ALIASES: dict[Capability, dict[str, str]] = {}
_ENTRYPOINTS_LOADED = False

ENTRYPOINT_GROUP = "videocontent.plugins"

CAPABILITIES = (
    "sampler",
    "scene_detector",
    "ocr",
    "asr",
    "vision",
    "event_detector",
    "embedding",
    "vector_store",
    "llm",
    "storage",
)

#: Built-ins, declared lazily so heavy adapters stay unimported until used.
_BUILTINS: dict[Capability, dict[str, str]] = {
    "sampler": {
        "fixed": "videocontent.processing.sampling.fixed:FixedSampler",
        "scene": "videocontent.processing.sampling.scene:SceneSampler",
        "adaptive": "videocontent.processing.sampling.adaptive:AdaptiveSampler",
    },
    "scene_detector": {
        "ffmpeg": "videocontent.processing.scenes.ffmpeg_scene:FFmpegSceneDetector",
        "null": "videocontent.processing.scenes.null:NullSceneDetector",
    },
    "ocr": {
        "tesseract": "videocontent.processing.ocr.tesseract:TesseractOCR",
        "null": "videocontent.processing.ocr.null:NullOCR",
    },
    "asr": {
        "faster-whisper": "videocontent.processing.asr.faster_whisper:FasterWhisperASR",
        "subtitles": "videocontent.processing.asr.subtitles:SubtitleASR",
        "null": "videocontent.processing.asr.null:NullASR",
    },
    "vision": {
        "null": "videocontent.processing.vision.null:NullVision",
        "openai": "videocontent.processing.vision.remote:OpenAIVision",
        "gemini": "videocontent.processing.vision.remote:GeminiVision",
        "local-vlm": "videocontent.processing.vision.remote:LocalVLMVision",
    },
    "event_detector": {
        "rules": "videocontent.processing.events.rules:RuleEventDetector",
    },
    "embedding": {
        "local": "videocontent.embeddings.local:LocalEmbeddings",
    },
    "vector_store": {
        "faiss": "videocontent.storage.faiss_store:FAISSStore",
        "qdrant": "videocontent.storage.qdrant_store:QdrantStore",
    },
    "llm": {
        "openai": "videocontent.llm.openai_llm:OpenAILLM",
        "local": "videocontent.llm.local:LocalLLM",
        "null": "videocontent.llm.null:NullLLM",
    },
    "storage": {},
}

_BUILTIN_ALIASES: dict[Capability, dict[str, str]] = {
    "asr": {"whisper": "faster-whisper", "auto": "faster-whisper"},
    "ocr": {"auto": "tesseract"},
}


def _ensure_capability(capability: str) -> None:
    if capability not in CAPABILITIES:
        raise ConfigurationError(
            f"unknown capability {capability!r}",
            hint=f"known capabilities: {', '.join(CAPABILITIES)}",
        )


def register(capability: str, name: str, factory: Factory | str, *, override: bool = False) -> None:
    """Register ``factory`` (a callable or ``"module:attr"`` spec) under ``name``."""
    _ensure_capability(capability)
    table = _REGISTRY.setdefault(capability, {})
    if name in table and not override:
        raise ConfigurationError(
            f"{capability} provider {name!r} is already registered",
            hint="Pass override=True to replace it.",
        )
    table[name] = factory


def alias(capability: str, alias_name: str, target: str) -> None:
    _ensure_capability(capability)
    _ALIASES.setdefault(capability, {})[alias_name] = target


def _decorator(capability: str) -> Callable[[str], Callable[[type[T]], type[T]]]:
    def outer(name: str) -> Callable[[type[T]], type[T]]:
        def inner(cls: type[T]) -> type[T]:
            register(capability, name, cls, override=True)
            return cls

        return inner

    return outer


register_sampler = _decorator("sampler")
register_scene_detector = _decorator("scene_detector")
register_ocr = _decorator("ocr")
register_asr = _decorator("asr")
register_vision = _decorator("vision")
register_event_detector = _decorator("event_detector")
register_embedding = _decorator("embedding")
register_vector_store = _decorator("vector_store")
register_llm = _decorator("llm")
register_storage = _decorator("storage")


def _load_entrypoints() -> None:
    """Discover third-party plugins.

    A broken plugin must not prevent the core from working, so every failure here is
    contained. It is *logged* rather than discarded: "the provider you configured does not
    exist" and "the plugin that provides it raised on import" are different problems, and
    without the log there is nothing to tell them apart. ``doctor`` reports them separately.
    """
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group=ENTRYPOINT_GROUP):
            try:
                hook = ep.load()
                hook()  # plugin calls register_* itself
            # Broad by design: this is third-party code and it must not escape.
            except Exception as exc:
                log.warning(
                    "registry.plugin_failed",
                    extra={"entrypoint": ep.name, "error": f"{type(exc).__name__}: {exc}"},
                )
                continue
    # Broad by design: importlib.metadata itself can misbehave on odd installs.
    except Exception as exc:
        log.debug("registry.entrypoints_unavailable", extra={"error": str(exc)})
        return


def _bootstrap() -> None:
    for capability, table in _BUILTINS.items():
        for name, spec in table.items():
            _REGISTRY.setdefault(capability, {}).setdefault(name, spec)
    for capability, table in _BUILTIN_ALIASES.items():
        for name, target in table.items():
            _ALIASES.setdefault(capability, {}).setdefault(name, target)
    _load_entrypoints()


def _resolve_spec(spec: Factory | str) -> Factory:
    if callable(spec):
        return spec
    module_name, _, attr = str(spec).partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def factory(capability: str, name: str | None = None) -> Factory:
    """Resolve a provider's factory *without* calling it.

    The pipeline needs this to see what a constructor accepts before handing it anything: a
    third-party provider must be usable without core knowing its signature (§25), and one
    built-in — ``NullSceneDetector`` — takes no arguments at all. Deciding that by inspecting
    the callable is exact, where catching ``TypeError`` from a failed ``create`` would also
    swallow a genuine ``TypeError`` raised inside the constructor's body.
    """
    _bootstrap()
    _ensure_capability(capability)
    table = _REGISTRY.get(capability, {})
    if not table:
        raise ConfigurationError(f"no {capability} providers are registered")

    resolved = name or next(iter(_BUILTINS.get(capability, table)))
    resolved = _ALIASES.get(capability, {}).get(resolved, resolved)

    if resolved not in table:
        raise ConfigurationError(
            f"unknown {capability} provider {resolved!r}",
            hint=f"available: {', '.join(sorted(table))}",
        )
    return _resolve_spec(table[resolved])


def create(capability: str, name: str | None = None, /, **kwargs: Any) -> Any:
    """Instantiate a provider. ``None`` resolves to the capability's first built-in."""
    return factory(capability, name)(**kwargs)


def names(capability: str) -> list[str]:
    _bootstrap()
    _ensure_capability(capability)
    return sorted(_REGISTRY.get(capability, {}))


def all_names() -> dict[str, list[str]]:
    return {cap: names(cap) for cap in CAPABILITIES}


def registered(capability: str, name: str) -> bool:
    _bootstrap()
    table = _REGISTRY.get(capability, {})
    resolved = _ALIASES.get(capability, {}).get(name, name)
    return resolved in table


def clear(capabilities: Iterable[str] | None = None) -> None:
    """Reset the registry — for tests only."""
    global _ENTRYPOINTS_LOADED
    for cap in capabilities or list(_REGISTRY):
        _REGISTRY.pop(cap, None)
        _ALIASES.pop(cap, None)
    _ENTRYPOINTS_LOADED = False


__all__ = [
    "CAPABILITIES",
    "ENTRYPOINT_GROUP",
    "alias",
    "all_names",
    "clear",
    "create",
    "factory",
    "names",
    "register",
    "register_asr",
    "register_embedding",
    "register_event_detector",
    "register_llm",
    "register_ocr",
    "register_sampler",
    "register_scene_detector",
    "register_storage",
    "register_vector_store",
    "register_vision",
    "registered",
]
