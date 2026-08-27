"""The no-ASR engine.

Not a placeholder. It is how a caller turns transcription off *explicitly*, and it is the
last link in the fallback chain: when no speech model is installed and the video carries no
subtitle track, the run still completes and the document records an ASR stage that ran with
a substitution rather than a stage that silently produced nothing (ARCHITECTURE §7, edge
cases). It is also what lets the whole pipeline be exercised on a machine with no model
downloaded.
"""

from __future__ import annotations

from pathlib import Path

from ...config import ASRConfig
from ...interfaces import ASROutput, FrameContext


class NullASR:
    name = "null"
    version = "1.0.0"
    remote = False

    def __init__(self, config: ASRConfig | None = None) -> None:
        self.config = config or ASRConfig()

    def available(self) -> bool:
        return True

    def transcribe(self, audio_path: Path, ctx: FrameContext) -> ASROutput:
        return ASROutput(
            utterances=[],
            language=ctx.language,
            model=None,
            duration_s=ctx.duration,
            warnings=["transcription disabled (null ASR engine)"],
        )


__all__ = ["NullASR"]
