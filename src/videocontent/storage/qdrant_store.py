"""Qdrant vector store — production-grade vector database.

Qdrant is a vector similarity search engine with filtering support.
This adapter implements the VectorStore protocol.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from ...errors import DependencyMissingError
from ...interfaces import VectorStore
from ...logging import get_logger

log = get_logger("storage.qdrant")


def _installed() -> bool:
    return importlib.util.find_spec("qdrant_client") is not None


_INSTALL_HINT = "pip install qdrant-client"


class QdrantStore:
    """Qdrant vector store adapter."""

    name = "qdrant"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "videocontent",
        dim: int | None = None,
    ) -> None:
        if not _installed():
            raise DependencyMissingError("qdrant-client is not installed", hint=_INSTALL_HINT)

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.dim = dim

        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            if dim is None:
                raise ValueError("dim is required when creating a new collection")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            log.info("qdrant.collection_created", extra={"collection": collection_name, "dim": dim})

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        """Add vectors to the collection."""
        if not ids:
            return

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=id_, vector=vector, payload={})
            for id_, vector in zip(ids, vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        log.debug("qdrant.added", extra={"count": len(points)})

    def query(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """Query for nearest neighbors."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
        )
        return [(str(r.id), r.score) for r in results]

    def __len__(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def clear(self) -> None:
        """Delete all points in the collection."""
        self.client.delete_collection(self.collection_name)
        # Recreate
        from qdrant_client.models import Distance, VectorParams
        if self.dim is None:
            raise ValueError("Cannot recreate collection without dim")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )


__all__ = ["QdrantStore", "_installed"]