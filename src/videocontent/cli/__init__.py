"""The ``videocontent`` and ``vctx`` commands.

The typer application lives in :mod:`videocontent.cli.main` and the rendering in
:mod:`videocontent.cli.render`; this module exists so that ``videocontent.cli:main`` is a stable
entry point regardless of how the commands are organised behind it.
"""

from __future__ import annotations

from .main import app, main

__all__ = ["app", "main"]
