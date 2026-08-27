"""The FFmpeg boundary — the only module in the project that spawns a media process.

Hardening applied to every invocation (ARCHITECTURE §8):

* argv arrays, never a shell string — user filenames cannot inject arguments
* ``-nostdin`` so a subprocess can never consume our stdin or block
* explicit timeouts, with the process group killed on expiry
* ``-protocol_whitelist`` so a crafted container cannot make FFmpeg fetch remote URLs
  (the classic SSRF-via-playlist trick); the whitelist widens only for inputs the caller
  explicitly passed as ``http(s)``
* stderr captured and translated into :class:`~videocontent.errors.FFmpegError`
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

from ..errors import DependencyMissingError, FFmpegError
from ..logging import get_logger

log = get_logger("media.ffmpeg")

LOCAL_PROTOCOLS = "file,crypto,data"
REMOTE_PROTOCOLS = "file,crypto,data,http,https,tcp,tls"

_INSTALL_HINT = (
    "Install FFmpeg — macOS: brew install ffmpeg · Debian/Ubuntu: apt install ffmpeg · "
    "Windows: winget install ffmpeg"
)


@dataclass(frozen=True, slots=True)
class CompletedRun:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


@cache
def _which(binary: str) -> str | None:
    env_override = os.environ.get(f"VIDEO_CONTEXT_{binary.upper()}")
    if env_override and Path(env_override).exists():
        return env_override
    return shutil.which(binary)


def ffmpeg_path() -> str:
    path = _which("ffmpeg")
    if not path:
        raise DependencyMissingError("ffmpeg was not found on PATH", hint=_INSTALL_HINT)
    return path


def ffprobe_path() -> str:
    path = _which("ffprobe")
    if not path:
        raise DependencyMissingError("ffprobe was not found on PATH", hint=_INSTALL_HINT)
    return path


def available() -> bool:
    return bool(_which("ffmpeg") and _which("ffprobe"))


@lru_cache(maxsize=1)
def version() -> str | None:
    """FFmpeg version string, or None when unavailable."""
    if not _which("ffmpeg"):
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [ffmpeg_path(), "-hide_banner", "-version"],
            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    match = re.search(r"ffmpeg version (\S+)", out)
    return match.group(1) if match else None


def is_remote(source: str | os.PathLike[str]) -> bool:
    return str(source).startswith(("http://", "https://"))


def input_args(source: str | os.PathLike[str]) -> list[str]:
    """Input options that constrain what FFmpeg is allowed to reach."""
    protocols = REMOTE_PROTOCOLS if is_remote(source) else LOCAL_PROTOCOLS
    return ["-protocol_whitelist", protocols, "-i", str(source)]


def run(
    args: list[str],
    *,
    binary: str = "ffmpeg",
    timeout: float = 1800.0,
    check: bool = True,
    cwd: Path | None = None,
    loglevel: str = "error",
) -> CompletedRun:
    """Invoke ffmpeg/ffprobe with ``args`` appended to the hardened base flags.

    ``loglevel`` must be raised to ``"info"`` for filters that report through the log
    (``silencedetect``, ``blackdetect``); filters that write to stdout are unaffected.
    """
    exe = ffmpeg_path() if binary == "ffmpeg" else ffprobe_path()
    # -nostdin is an ffmpeg-only option; ffprobe rejects it. stdin=DEVNULL below covers both.
    base = [exe, "-hide_banner", "-loglevel", loglevel]
    if binary == "ffmpeg":
        base += ["-nostdin", "-y"]
    argv = [*base, *args]

    log.debug("ffmpeg.exec", extra={"binary": binary, "argc": len(argv)})
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            cwd=str(cwd) if cwd else None,
            start_new_session=True,  # isolate the process group so we can kill children
        )
    except OSError as exc:
        raise FFmpegError(
            f"could not start {binary}: {exc}", command=argv, hint=_INSTALL_HINT
        ) from exc

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(proc)
        stdout, stderr = proc.communicate()
        raise FFmpegError(
            f"{binary} timed out after {timeout:.0f}s",
            command=argv, stderr=stderr, returncode=None,
            hint="Raise limits.ffmpeg_timeout_s, or trim the input first.",
        ) from None
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _kill(proc)
        raise

    result = CompletedRun(argv, proc.returncode, stdout, stderr)
    if check and proc.returncode != 0:
        raise FFmpegError(
            f"{binary} failed with exit code {proc.returncode}",
            command=argv, stderr=stderr, returncode=proc.returncode,
            hint=_diagnose(stderr),
        )
    return result


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
        proc.kill()


_DIAGNOSTICS: tuple[tuple[str, str], ...] = (
    ("No such file or directory", "Check the path; the file does not exist or is unreadable."),
    ("Invalid data found",
     "The file is corrupt or not a media file. Try: ffmpeg -i FILE -f null -"),
    ("moov atom not found", "Truncated MP4 — the file is incomplete."),
    ("Permission denied", "The process cannot read the input or write the output directory."),
    ("Protocol not on whitelist",
     "The container references a remote resource; blocked on purpose."),
    ("does not contain any stream", "No decodable audio or video streams."),
    ("Output file is empty", "Nothing was decoded — the requested time range may be past the end."),
)


def _diagnose(stderr: str) -> str | None:
    for needle, hint in _DIAGNOSTICS:
        if needle.lower() in stderr.lower():
            return hint
    tail = [line for line in stderr.strip().splitlines() if line.strip()]
    return tail[-1] if tail else None


__all__ = [
    "CompletedRun",
    "available",
    "ffmpeg_path",
    "ffprobe_path",
    "input_args",
    "is_remote",
    "run",
    "version",
]
