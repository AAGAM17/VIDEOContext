"""Source adapters: local files and direct HTTP(S) URLs.

Each adapter implements the :class:`~videocontent.interfaces.SourceAdapter` protocol:
``can_handle`` (cheap string check), ``inspect`` (metadata without full fetch),
``materialize`` (return a :class:`VideoAsset` with a local file), ``cleanup``.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..errors import (
    DownloadError,
    SourceError,
    SourceNotFoundError,
    UnsupportedSourceError,
)
from ..logging import get_logger
from . import security
from .types import (
    MediaCapabilities,
    SourceInspection,
    SourceType,
    VideoAsset,
    VideoSource,
    stable_id,
)

log = get_logger("sources")

VIDEO_CONTENT_PREFIXES = ("video/", "audio/")
OCTET_TYPES = ("application/octet-stream", "binary/octet-stream", "application/mp4")
# Explicitly non-media responses: almost certainly a login/block page, not a video.
REJECTED_TYPES = ("text/html", "text/plain", "application/json", "application/xml", "text/xml")

_FILENAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _safe_filename(name: str, fallback: str = "video") -> str:
    cleaned = "".join(c if c in _FILENAME_CHARS else "_" for c in (name or "")).strip("._-")
    return (cleaned or fallback)[:100]


class LocalFileAdapter:
    """Adapter for files already on this machine — zero-copy, zero network."""

    name = "local"
    version = "1"

    def can_handle(self, locator: str) -> bool:
        s = locator.strip()
        if s.startswith(("http://", "https://")):
            return False
        if s.startswith("file://"):
            return True
        # Any other URI scheme (ftp://, s3://, …) is not a local path.
        if "://" in s:
            return False
        # A path that exists, or looks like a path (has a media suffix / separator),
        # belongs here. Anything else falls through to the invalid-source error.
        p = Path(s)
        return p.exists() or p.suffix.lower() in (
            ".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".mpg", ".mpeg",
            ".ts", ".flv", ".ogv", ".m4a", ".mp3", ".wav",
        ) or "/" in s or s.startswith((".", "~"))

    def _path(self, locator: str) -> Path:
        s = locator.strip()
        if s.startswith("file://"):
            from urllib.parse import urlparse
            from urllib.request import url2pathname

            parsed = urlparse(s)
            if parsed.netloc not in ("", "localhost"):
                raise SourceError(f"remote file:// hosts are not supported: {s!r}",
                                  code="UNSUPPORTED_SOURCE")
            s = url2pathname(parsed.path)
        return Path(os.path.expanduser(s))

    def to_source(self, locator: str) -> VideoSource:
        path = self._path(locator)
        resolved = str(path.resolve()) if path.exists() else str(path)
        canonical = f"local:{resolved}"
        return VideoSource(
            source_id=stable_id("src", canonical),
            source_type=SourceType.LOCAL_FILE,
            provider=self.name,
            locator=locator,
            canonical_id=canonical,
            metadata={"path": resolved},
            capabilities=MediaCapabilities(),
        )

    def inspect(self, source: VideoSource, ctx: Any = None) -> SourceInspection:
        path = self._path(source.locator)
        if not path.exists():
            return SourceInspection(source=source, accessible=False,
                                    reason=f"no such file: {path}",
                                    requires_authentication=False)
        if path.is_dir():
            return SourceInspection(source=source, accessible=False,
                                    reason=f"{path} is a directory, not a video file")
        size = path.stat().st_size
        media: dict[str, Any] = {"filename": path.name, "size_bytes": size}
        try:
            from ..media.probe import raw_probe

            payload = raw_probe(path, timeout=30.0)
            fmt = payload.get("format") or {}
            streams = payload.get("streams") or []
            media.update({
                "duration": float(fmt.get("duration") or 0.0),
                "container": fmt.get("format_name"),
                "streams": len(streams),
                "has_audio": any(s.get("codec_type") == "audio" for s in streams),
                "has_video": any(s.get("codec_type") == "video" for s in streams),
            })
        except Exception as exc:  # inspection must not raise on unprobed media
            media["probe_error"] = f"{type(exc).__name__}: {exc}"
        return SourceInspection(
            source=source, accessible=True, media=media,
            capabilities=MediaCapabilities(), estimated_size_bytes=size,
        )

    def materialize(self, source: VideoSource, dest_dir: Path, ctx: Any = None) -> VideoAsset:
        # Existence is validated by the media probe (which owns container errors), not here:
        # the pipeline's fake-probe tests and the "fail fast with a decodable error" contract
        # both depend on probe being the single validator for local paths.
        path = self._path(source.locator)
        resolved = str(path.resolve()) if path.exists() else str(path)
        size = path.stat().st_size if path.is_file() else None
        return VideoAsset(
            asset_id=stable_id("asset", source.canonical_id),
            source=source,
            local_path=resolved,
            temporary=False,
            size_bytes=size,
            provenance={"adapter": self.name, "materialized_at": time.time()},
        )

    def cleanup(self, asset: VideoAsset) -> None:
        return None


class DirectURLAdapter:
    """Adapter for direct http(s) media URLs — validated download to a temp file."""

    name = "http"
    version = "1"

    def can_handle(self, locator: str) -> bool:
        return locator.strip().lower().startswith(("http://", "https://"))

    def to_source(self, locator: str) -> VideoSource:
        canonical = f"url:{security.canonicalize_url(locator)}"
        return VideoSource(
            source_id=stable_id("src", canonical),
            source_type=SourceType.DIRECT_URL,
            provider=self.name,
            locator=locator,
            canonical_id=canonical,
            metadata={"url_redacted": security.redact(locator)},
            capabilities=MediaCapabilities(
                supports_streaming=True, supports_range_requests=True,
                supports_partial_download=False,
            ),
        )

    # -- HTTP primitives (stdlib only: base install stays small) -----------------

    def _opener(self, timeout: float, user_agent: str) -> Any:
        opener = urllib.request.build_opener(_NoRedirectHandler())
        opener.addheaders = [("User-Agent", user_agent)]
        return opener

    def _fetch_headers(self, url: str, *, timeout: float, user_agent: str,
                       max_redirects: int, allow_private: bool) -> tuple[str, Any]:
        """Follow redirects manually (revalidating DNS each hop); return final URL + headers."""
        current = url
        opener = self._opener(timeout, user_agent)
        for index in range(max_redirects + 1):
            security.validate_url(current, allow_private_ips=allow_private)
            # Validated above: http(s) only, no credentials, DNS/IP safety checked.
            req = urllib.request.Request(  # noqa: S310 - URL validated by security.validate_url
                current, method="HEAD", headers={"User-Agent": user_agent})
            try:
                resp = opener.open(req, timeout=timeout)
            except urllib.error.HTTPError as exc:
                # Some origins reject HEAD (405/501). Fall back to a ranged GET below.
                if exc.code in (400, 403, 404, 405, 501):
                    return current, {"_head_status": exc.code}
                raise DownloadError(
                    f"HEAD {security.redact(current)} failed: HTTP {exc.code}",
                    hint="The server rejected the request.") from exc
            except urllib.error.URLError as exc:
                raise DownloadError(
                    f"could not reach {security.redact(current)}: {exc.reason}",
                    hint="Check the URL and network access.") from exc
            status = getattr(resp, "status", 200)
            if status in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    raise DownloadError(f"redirect ({status}) without a Location header",
                                        hint="The server returned a malformed redirect.")
                current = security.resolve_redirect(current, location, index=index)
                continue
            return current, resp.headers
        raise DownloadError(f"too many redirects (>{max_redirects}) for {security.redact(url)}",
                            hint="The URL redirects in a loop.")

    def inspect(self, source: VideoSource, ctx: Any = None) -> SourceInspection:
        cfg = _source_cfg(ctx)
        try:
            final_url, headers = self._fetch_headers(
                source.locator, timeout=min(cfg.download_timeout_s, 30.0),
                user_agent=cfg.user_agent, max_redirects=cfg.max_redirects,
                allow_private=cfg.allow_private_ips,
            )
        except Exception as exc:
            return SourceInspection(source=source, accessible=False,
                                    reason=f"{type(exc).__name__}: {exc}")
        warnings: list[str] = []
        media: dict[str, Any] = {"final_url_redacted": security.redact(final_url)}
        estimated: int | None = None
        if hasattr(headers, "get"):
            ctype = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
            clen = headers.get("Content-Length")
            accept = (headers.get("Accept-Ranges") or "").lower()
            media.update({"content_type": ctype or None,
                          "accept_ranges": accept or None,
                          "etag": headers.get("ETag")})
            if clen:
                try:
                    estimated = int(clen)
                    media["content_length"] = estimated
                except ValueError:
                    warnings.append(f"ignoring malformed Content-Length {clen!r}")
            if ctype and any(ctype.startswith(t) for t in REJECTED_TYPES):
                return SourceInspection(
                    source=source, accessible=False,
                    reason=f"URL serves {ctype}, not media — refusing to download a web page",
                    media=media, warnings=warnings,
                    hint="Point VIDEOContext at the direct media file, not a watch page.",
                )
            if ctype and not (ctype.startswith(VIDEO_CONTENT_PREFIXES) or ctype in OCTET_TYPES):
                warnings.append(f"unexpected Content-Type {ctype!r}; will verify by probing bytes")
            caps = MediaCapabilities(
                supports_streaming=True,
                supports_range_requests=(accept == "bytes"),
                supports_subtitles=False,
            )
        else:
            caps = MediaCapabilities(supports_streaming=True)
        max_bytes = cfg.max_download_mb * 1024 * 1024
        if estimated and estimated > max_bytes:
            limit_mb = cfg.max_download_mb
            return SourceInspection(
                source=source, accessible=False,
                reason=f"remote object is {estimated / 1e6:.0f} MB (limit {limit_mb} MB)",
                media=media, warnings=warnings,
                hint="Raise sources.max_download_mb, or process a smaller clip.",
            )
        return SourceInspection(source=source, accessible=True, media=media,
                                capabilities=caps, estimated_size_bytes=estimated,
                                warnings=warnings)

    def materialize(self, source: VideoSource, dest_dir: Path, ctx: Any = None) -> VideoAsset:
        cfg = _source_cfg(ctx)
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_url, headers = self._fetch_headers(
            source.locator, timeout=cfg.download_timeout_s, user_agent=cfg.user_agent,
            max_redirects=cfg.max_redirects, allow_private=cfg.allow_private_ips,
        )
        etag = headers.get("ETag") if hasattr(headers, "get") else None
        last_mod = headers.get("Last-Modified") if hasattr(headers, "get") else None
        max_bytes = cfg.max_download_mb * 1024 * 1024
        suffix = _suffix_for(final_url, headers)
        stem = _safe_filename(Path(final_url.split("?")[0]).name, "remote")
        out_path = dest_dir / f"{stem}{suffix}"
        # Avoid collisions when the same name downloads twice.
        if out_path.exists():
            out_path = dest_dir / f"{out_path.stem}-{source.source_id}{suffix}"

        digest = hashlib.sha256()
        total = 0
        opener = self._opener(cfg.download_timeout_s, cfg.user_agent)
        # Validated above via _fetch_headers (scheme, credentials, DNS/IP per hop).
        req = urllib.request.Request(  # noqa: S310 - URL validated by security.validate_url
            final_url, headers={"User-Agent": cfg.user_agent})
        try:
            resp = opener.open(req, timeout=cfg.download_timeout_s)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                from ..errors import SourceAccessError

                raise SourceAccessError(
                    f"server denied access (HTTP {exc.code}) for {security.redact(final_url)}",
                    hint="The object is private; use a signed URL or download it first.",
                ) from exc
            if exc.code == 404:
                raise SourceNotFoundError(
                    f"remote object not found (HTTP 404): {security.redact(final_url)}"
                ) from exc
            raise DownloadError(
                f"GET failed: HTTP {exc.code} for {security.redact(final_url)}",
                hint="Retry; the failure may be transient.") from exc
        except urllib.error.URLError as exc:
            raise DownloadError(
                f"could not download {security.redact(final_url)}: {exc.reason}"
            ) from exc

        # Re-check DNS for the final host right before streaming bytes (TOCTOU guard:
        # headers were validated, but the body connection resolves again inside urlopen).
        host = urllib.request.urlparse(final_url).hostname or ""
        security.check_hostname(host.strip("[]"), allow_private=cfg.allow_private_ips)

        ctype = ""
        if hasattr(resp, "headers") and resp.headers:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if any(ctype.startswith(t) for t in REJECTED_TYPES):
                raise DownloadError(
                    f"URL serves {ctype}, not media — refusing to download a web page",
                    retryable=False,
                    hint="Point VIDEOContext at the direct media file, not a watch page.",
                )
        try:
            with open(out_path, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        fh.close()
                        out_path.unlink(missing_ok=True)
                        raise DownloadError(
                            f"download exceeded the {cfg.max_download_mb} MB limit",
                            retryable=False,
                            hint="Raise sources.max_download_mb, or process a smaller clip.",
                        )
                    digest.update(chunk)
                    fh.write(chunk)
        finally:
            with contextlib.suppress(Exception):
                resp.close()
        if total == 0:
            out_path.unlink(missing_ok=True)
            raise DownloadError("remote object was empty (0 bytes)", retryable=False)
        content_hash = f"sha256:{digest.hexdigest()}"
        log.info("sources.downloaded", extra={"bytes": total, "sha": content_hash[:19]})
        return VideoAsset(
            asset_id=stable_id("asset", f"{source.canonical_id}|{etag or content_hash}"),
            source=source,
            local_path=str(out_path.resolve()),
            temporary=True,
            size_bytes=total,
            content_hash=content_hash,
            etag=etag,
            last_modified=last_mod,
            provenance={
                "adapter": self.name,
                "final_url_redacted": security.redact(final_url),
                "content_type": ctype or None,
                "materialized_at": time.time(),
            },
        )

    def cleanup(self, asset: VideoAsset) -> None:
        if not asset.temporary:
            return
        with contextlib.suppress(OSError):
            Path(asset.local_path).unlink(missing_ok=True)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Disable automatic redirects so each hop passes the SSRF revalidation."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _suffix_for(url: str, headers: Any) -> str:
    from urllib.request import urlparse as _up

    path_suffix = Path(_up(url).path).suffix.lower()
    if path_suffix in (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".mpg",
                       ".mpeg", ".ts", ".flv", ".ogv", ".m4a", ".mp3", ".wav"):
        return path_suffix
    ctype = ""
    if hasattr(headers, "get"):
        ctype = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype.startswith("video/mp4") or ctype == "application/mp4":
        return ".mp4"
    if ctype.startswith("video/webm") or ctype == "audio/webm":
        return ".webm"
    if ctype.startswith("video/"):
        return ".mp4"
    if ctype.startswith("audio/"):
        return ".m4a"
    return ".mp4"


def _source_cfg(ctx: Any) -> Any:
    from ..config import ProcessingConfig, SourceConfig

    if isinstance(ctx, ProcessingConfig):
        return ctx.sources
    if isinstance(ctx, SourceConfig):
        return ctx
    if isinstance(ctx, dict) and "sources" in ctx:
        return ctx["sources"]
    return SourceConfig()


def adapter_for(locator: str, *, enabled: tuple[str, ...] = ("local", "http")) -> Any:
    """Return the adapter instance owning ``locator`` (registry-backed, lazy)."""
    from .. import registry as _registry
    from ..logging import get_logger as _get_logger

    _log = _get_logger("sources.resolve")
    for name in enabled:
        try:
            adapter = _registry.create("source", name)
        except Exception as exc:  # misconfigured installs must not break resolution
            _log.debug("sources.adapter_unavailable", extra={"adapter": name, "error": str(exc)})
            continue
        try:
            if adapter.can_handle(locator):
                return adapter
        except Exception as exc:  # a buggy third-party can_handle must not escape
            _log.debug("sources.can_handle_failed", extra={"adapter": name, "error": str(exc)})
            continue
    # Nothing claimed it: distinguish "unsupported scheme" from "typo'd path".
    lowered = locator.strip().lower()
    if "://" in lowered and not lowered.startswith(("http://", "https://", "file://")):
        scheme = lowered.split("://", 1)[0]
        raise UnsupportedSourceError(
            f"unsupported source scheme {scheme!r}: local files and http(s) URLs only",
            hint="Object storage / video platforms need a dedicated adapter (see docs).",
        )
    raise SourceNotFoundError(
        f"could not resolve source {locator!r}",
        hint="Pass an existing file path or an http(s) URL to a media file.",
    )


__all__ = ["DirectURLAdapter", "LocalFileAdapter", "adapter_for"]
