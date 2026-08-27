"""Artifact workspace.

Derived artifacts (frames, audio, stage cache) live in a workspace directory next to the
video by default. Two properties matter:

* **confinement** — every path handed to FFmpeg resolves inside the workspace, so a crafted
  filename cannot write outside it (:meth:`Workspace.path`)
* **reconstructibility** — everything here is derivable from ``media + config``, which is
  what makes eviction and retention policies safe (ARCHITECTURE §5)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ..config import DEFAULT_WORKDIR_NAME
from ..errors import SecurityError

SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def sanitize(name: str, *, fallback: str = "video") -> str:
    """Reduce an arbitrary filename to a safe directory-name component."""
    cleaned = "".join(c if c in SAFE_CHARS else "_" for c in name).strip("._-")
    return (cleaned or fallback)[:80]


class Workspace:
    """A confined directory tree for one video's derived artifacts."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_video(
        cls,
        video_path: str | os.PathLike[str],
        *,
        workdir: str | os.PathLike[str] | None = None,
        key: str | None = None,
    ) -> Workspace:
        video = Path(str(video_path))
        base = Path(workdir).resolve() if workdir else video.resolve().parent / DEFAULT_WORKDIR_NAME
        slug = sanitize(video.stem)
        if key:
            slug = f"{slug}-{key[:12]}"
        return cls(base / slug)

    # -- paths -------------------------------------------------------------

    def path(self, *parts: str) -> Path:
        """Resolve a path inside the workspace, refusing anything that escapes it."""
        candidate = self.root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise SecurityError(
                f"path {candidate} escapes the workspace {self.root}",
                hint="Artifact names must be relative and must not contain '..'.",
            ) from None
        return candidate

    def subdir(self, name: str) -> Path:
        directory = self.path(name)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def frames_dir(self) -> Path:
        return self.subdir("frames")

    @property
    def audio_dir(self) -> Path:
        return self.subdir("audio")

    @property
    def cache_dir(self) -> Path:
        return self.subdir("cache")

    def relative(self, path: Path) -> str:
        """Path relative to the workspace root, for storing in a ``.vctx`` document."""
        try:
            return str(Path(path).resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    # -- lifecycle ---------------------------------------------------------

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def clear(self, *subdirs: str) -> None:
        targets = [self.path(s) for s in subdirs] if subdirs else [self.root]
        for target in targets:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Workspace({self.root})"


@contextmanager
def scratch(prefix: str = "videocontent-"):
    """A temp directory removed even when the body raises (privacy: §32)."""
    directory = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


__all__ = ["Workspace", "sanitize", "scratch"]
