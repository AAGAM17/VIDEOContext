"""Local embeddings via sentence-transformers.

This provider runs entirely on the machine — no API key, no network calls after the
initial model download. It uses `sentence-transformers` which wraps Hugging Face
transformers and supports CPU, CUDA, and MPS (Apple Silicon).

The default model is ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim, fast,
strong baseline). For multilingual support, ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``
is a drop-in replacement.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

from ...config import EmbeddingConfig
from ...errors import DependencyMissingError, ProviderError
from ...interfaces import EmbeddingProvider
from ...logging import get_logger

log = get_logger("embeddings.local")

_INSTALL_HINT = (
    "Install the embeddings extra — pip install 'videocontent[embeddings]' — "
    "or configure a remote provider."
)


def _installed() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str | None = None) -> Any:
    """Load and cache a sentence-transformers model.

    The cache key includes the model name and device so that a change in either
    correctly invalidates the cache. ``maxsize=2`` covers the common case of
    one model + one fallback device.
    """
    from sentence_transformers import SentenceTransformer

    log.info("embeddings.model_loading", extra={"model": model_name, "device": device})
    try:
        model = SentenceTransformer(model_name, device=device)
        return model
    except Exception as exc:
        raise ProviderError(
            f"could not load embedding model {model_name!r}: {type(exc).__name__}: {exc}",
            hint="Check the model name and Hugging Face cache permissions.",
        ) from exc


def _resolve_device(requested: str | None) -> str | None:
    """Resolve device string, with auto-detection for 'auto'."""
    if requested is None or requested == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return requested


class LocalEmbeddings:
    """Local sentence-transformers embedding provider."""

    name = "local"
    remote = False

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model: Any | None = None

    @property
    def version(self) -> str:
        if not _installed():
            return "unavailable"
        try:
            from importlib.metadata import version as pkg_version

            return f"sentence-transformers-{pkg_version('sentence-transformers')}/{self.config.model}"
        except Exception:
            return self.config.model or "unknown"

    @property
    def dim(self) -> int:
        """Return the embedding dimension for the configured model."""
        # Known dimensions for common models
        known_dims = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
            "e5-small-v2": 384,
            "e5-base-v2": 768,
        }
        model_name = self.config.model or "all-MiniLM-L6-v2"
        short_name = model_name.split("/")[-1]
        return known_dims.get(short_name, 384)  # default to 384

    def available(self) -> bool:
        return _installed()

    def _ensure_model(self) -> Any:
        if self._model is None:
            if not self.available():
                raise DependencyMissingError(
                    "sentence-transformers is not installed", hint=_INSTALL_HINT
                )
            device = _resolve_device(self.config.device)
            model_name = self.config.model or "all-MiniLM-L6-v2"
            self._model = _load_model(model_name, device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        # sentence-transformers returns numpy arrays; convert to list[list[float]]
        embeddings = model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


__all__ = ["LocalEmbeddings", "_installed", "_resolve_device"]