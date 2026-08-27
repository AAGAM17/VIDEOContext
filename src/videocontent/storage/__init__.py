"""Vector stores — FAISS for local, Qdrant for production.

This module implements the :class:`~videocontent.interfaces.VectorStore` protocol.
Each backend is an adapter; the retrieval layer uses the protocol and never imports
a vendor SDK directly.
"""

from __future__ import annotations

from .faiss_store import FAISSStore
from .qdrant_store import QdrantStore

__all__ = ["FAISSStore", "QdrantStore"]