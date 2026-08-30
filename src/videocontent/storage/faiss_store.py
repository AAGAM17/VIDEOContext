"""FAISS vector store — local, in-process, no server required.

FAISS (Facebook AI Similarity Search) is the default vector backend for local
development. It runs in-process, requires no external service, and is fast enough
for documents with tens of thousands of vectors.

The store is intentionally simple: it holds vectors in an ``IndexFlatIP`` (inner
product, which equals cosine similarity when vectors are normalised) and persists
them to disk alongside the ``.vctx`` document. A production deployment should
migrate to Qdrant, pgvector, or another server-backed store.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ...errors import DependencyMissingError
from ...interfaces import VectorStore
from ...logging import get_logger

log = get_logger("storage.faiss")

_INSTALL_HINT = "pip install 'videocontent[vectors]'"


def _installed() -> bool:
    return importlib.util.find_spec("faiss") is not None


class FAISSStore:
    """In-process FAISS vector store with disk persistence."""

    name = "faiss"

    def __init__(self, path: Path | str | None = None, dim: int | None = None) -> None:
        if not _installed():
            raise DependencyMissingError("faiss is not installed", hint=_INSTALL_HINT)

        import faiss

        self._faiss = faiss
        self.path = Path(path) if path else None
        self.dim = dim
        self._index: faiss.Index | None = None
        self._id_map: list[str] = []  # index position -> external id

        if self.path and self.path.exists():
            self._load()
        elif self.dim is not None:
            self._index = faiss.IndexFlatIP(self.dim)

    def _load(self) -> None:
        """Load index and id map from disk."""
        index_path = self.path.with_suffix(".faiss")
        map_path = self.path.with_suffix(".faiss.map")
        if index_path.exists() and map_path.exists():
            self._index = self._faiss.read_index(str(index_path))
            self._id_map = json.loads(map_path.read_text())
            log.info("faiss.loaded", extra={"path": str(self.path), "count": self._index.ntotal})
        else:
            # No existing index; will create on first add
            pass

    def _save(self) -> None:
        """Persist index and id map to disk."""
        if self._index is None or self.path is None:
            return
        index_path = self.path.with_suffix(".faiss")
        map_path = self.path.with_suffix(".faiss.map")
        self._faiss.write_index(self._index, str(index_path))
        map_path.write_text(json.dumps(self._id_map))
        log.debug("faiss.saved", extra={"path": str(self.path), "count": self._index.ntotal})

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(dim)
            self.dim = dim

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        """Add vectors to the index. ``ids`` must be unique and in the same order."""
        if not ids:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if arr.shape[0] != len(ids):
            raise ValueError("ids and vectors length mismatch")
        self._ensure_index(arr.shape[1])
        # FAISS expects normalised vectors for IP == cosine
        # The embedding provider already normalises, but we defensively re-normalise
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-12)
        self._index.add(arr)
        self._id_map.extend(ids)
        self._save()

    def query(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return top-k (id, score) pairs. Score is cosine similarity in [-1, 1]."""
        if self._index is None or self._index.ntotal == 0:
            return []
        arr = np.asarray([vector], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-12)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(arr, k)
        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx >= 0 and idx < len(self._id_map):
                results.append((self._id_map[idx], float(score)))
        return results

    def __len__(self) -> int:
        return self._index.ntotal if self._index else 0

    def clear(self) -> None:
        """Remove all vectors."""
        if self._index is not None:
            self._index.reset()
            self._id_map.clear()
            self._save()


__all__ = ["FAISSStore", "_installed"]