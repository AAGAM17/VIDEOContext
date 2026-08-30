"""VideoContext — the open-source semantic layer for video.

Turn video into timestamped, searchable context for AI agents and applications::

    from videocontext import Video

    video = Video("demo.mp4")
    video.process()
    for hit in video.search("pricing"):
        print(hit.timecode, hit.text)

Heavy submodules are imported lazily, so ``import videocontext`` stays fast and the base
install stays small (ARCHITECTURE §4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

from .config import ProcessingConfig, load_config
from .errors import VideoContextError
from .logging import configure as configure_logging
from .logging import get_logger
from .schema import VCTX_VERSION, VideoContextDocument

if TYPE_CHECKING:  # pragma: no cover
    from .retrieval.query import EvidenceSpan, SearchResult
    from .sdk import Video

_LAZY: dict[str, tuple[str, str]] = {
    "Video": (".sdk", "Video"),
    "process": (".sdk", "process"),
    "load": (".sdk", "load"),
    "EvidenceSpan": (".retrieval.query", "EvidenceSpan"),
    "SearchResult": (".retrieval.query", "SearchResult"),
    "Pipeline": (".processing.pipeline", "Pipeline"),
    "registry": (".registry", None),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attributes: pay for a subsystem only when it is used."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    import importlib

    module = importlib.import_module(module_name, __package__)
    return module if attr is None else getattr(module, attr)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "VCTX_VERSION",
    "EvidenceSpan",
    "ProcessingConfig",
    "SearchResult",
    "Video",
    "VideoContextDocument",
    "VideoContextError",
    "__version__",
    "configure_logging",
    "get_logger",
    "load_config",
]
