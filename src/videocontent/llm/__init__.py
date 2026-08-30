"""LLM providers — the model that generates answers from retrieved evidence.

The ``ask()`` pipeline is: question → retrieval → context assembly → LLM → answer.
The LLM provider is the last step and is *never* required for processing or search —
it is an optional convenience for the common case of "just give me the answer".
"""

from __future__ import annotations

from .local import LocalLLM
from .null import NullLLM
from .openai_llm import OpenAILLM

__all__ = ["OpenAILLM", "LocalLLM", "NullLLM"]