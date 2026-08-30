"""Embedding providers — vector representations for semantic retrieval.

The embedding layer is the bridge between the lexical index (always available) and
semantic search (when configured). Providers are resolved the same way as other
capabilities: ``VIDEO_CONTEXT_EMBEDDING_PROVIDER=local`` + ``pip install
'videocontent[embeddings]'`` enables the default local path; remote providers (OpenAI,
Cohere, Voyage, etc.) are adapters with the same interface.
"""

from __future__ import annotations

from .local import LocalEmbeddings

__all__ = ["LocalEmbeddings"]