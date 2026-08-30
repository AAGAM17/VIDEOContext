"""Retrieval over a ``.vctx`` document — lexical now, hybrid by construction.

    from videocontext.retrieval import search
    for span in search(doc, "pricing"):
        print(span.timecode, span.modality, span.text)

The public surface is :func:`search`, :func:`at` and :class:`Retriever`; the modules below them
(``index``, ``lexical``, ``fusion``) are the implementation and may change shape. See
``docs/ARCHITECTURE.md`` §4 Layer 4 for why the default is lexical rather than vector.
"""

from __future__ import annotations

from .index import MODALITIES, Record, build_records
from .query import EvidenceSpan, Retriever, SearchResult, at, search

__all__ = [
    "MODALITIES",
    "EvidenceSpan",
    "Record",
    "Retriever",
    "SearchResult",
    "at",
    "build_records",
    "search",
]
