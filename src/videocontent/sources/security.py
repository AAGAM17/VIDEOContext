"""Remote-resource safety boundary.

Remote media is untrusted input. Every URL is validated as::

    syntax → scheme → DNS → IP safety → (per redirect, revalidate) → response limits

DNS is resolved with :func:`socket.getaddrinfo` and *every* returned address is
checked — a hostname that resolves to both a public and a private address is
rejected, which closes the DNS-rebinding hole that string matching leaves open.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ..errors import SecurityError, SourceError

ALLOWED_SCHEMES = ("http", "https")
MAX_URL_LENGTH = 8192

#: Query parameters that identify a *request*, not the *object*. Stripped from the
#: canonical identity so the same object shared via two signed URLs deduplicates.
EPHEMERAL_PARAMS = frozenset(
    {
        "token", "auth", "authorization", "signature", "sig", "sign",
        "expires", "expiry", "exp", "session", "sessionid", "session_id",
        "key", "api_key", "apikey", "access_token", "auth_token",
        "x-amz-signature", "x-amz-expires", "x-amz-credential", "x-amz-date",
        "x-goog-signature", "x-goog-expires",
        "st", "e", "dl", "download",
    }
)


def _fail(message: str, *, hint: str | None = None) -> SecurityError:
    return SecurityError(message, hint=hint)


def parse_url(url: str) -> Any:
    """Parse and enforce syntax + scheme rules. Returns :func:`urlparse` result."""
    from urllib.parse import urlparse as _parse

    if not url or not url.strip():
        raise SourceError("empty source URL", code="INVALID_SOURCE",
                          hint="Pass a local path or an http(s) URL.")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise SourceError(f"URL is {len(url)} chars (limit {MAX_URL_LENGTH})",
                          code="INVALID_SOURCE")
    parsed = _parse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise _fail(
            f"unsupported URL scheme {parsed.scheme!r}: only http and https are allowed",
            hint="Local files, S3/GCS/Azure and video platforms need their own adapter.",
        )
    if not parsed.hostname:
        raise SourceError(f"URL has no host: {redact(url)!r}", code="INVALID_SOURCE")
    if parsed.username or parsed.password:
        # Credentials in URLs leak into logs/shell history; refuse rather than redact-and-fetch.
        raise _fail(
            "URLs with embedded credentials (user:pass@host) are not accepted",
            hint="Use a signed URL without basic-auth credentials, or download the file first.",
        )
    # Block non-http(s) confusion: 'http://host#@evil' etc. is handled by urlparse, but
    # control characters and spaces never belong in a fetch target.
    if any(c.isspace() or ord(c) < 32 for c in url):
        raise SourceError("URL contains whitespace or control characters",
                          code="INVALID_SOURCE")
    return parsed


def is_ip_blocked(ip: ipaddress._BaseAddress, *, allow_private: bool = False) -> str | None:
    """Return a reason string when ``ip`` must not be fetched, else None."""
    if allow_private:
        return None
    if ip.is_unspecified:
        return "unspecified address (0.0.0.0)"
    if ip.is_loopback:
        return "loopback address"
    if ip.is_link_local:
        return "link-local address"
    if ip.is_multicast:
        return "multicast address"
    if ip.is_reserved:
        return "reserved address"
    if ip.is_private:
        return "private-network address"
    return None


def check_hostname(hostname: str, *, allow_private: bool = False) -> list[str]:
    """Resolve ``hostname`` and reject non-global targets. Returns string IPs.

    Raises :class:`SecurityError` when resolution fails or any address is blocked.
    Every address is checked (not just the first) so a hostname round-robining
    between a public CDN and an internal host cannot slip through.
    """
    # Fast path: literal IP (covers hex/octal forms via getaddrinfo below too, but this
    # gives a clearer error and avoids a DNS lookup for the common attack probe).
    try:
        literal = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None:
        reason = is_ip_blocked(literal, allow_private=allow_private)
        if reason is not None:
            raise _fail(
                f"URL resolves to a blocked {reason}: {hostname}",
                hint="Pass --allow-private-ips only for trusted test targets.",
            )
        return [str(literal)]

    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SourceError(f"could not resolve host {hostname!r}: {exc}",
                          code="SOURCE_NOT_FOUND",
                          hint="Check the hostname for typos.") from exc
    ips = sorted({info[4][0] for info in infos})
    if not ips:
        raise SourceError(f"host {hostname!r} resolved to no addresses",
                          code="SOURCE_NOT_FOUND")
    for raw in ips:
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            raise _fail(f"host {hostname!r} resolved to an invalid address") from None
        reason = is_ip_blocked(ip, allow_private=allow_private)
        if reason is not None:
            raise _fail(
                f"URL target {hostname!r} resolves to a blocked {reason} ({raw})",
                hint="Remote ingestion blocks localhost/private/link-local targets (SSRF).",
            )
    return ips


def validate_url(url: str, *, allow_private_ips: bool = False) -> Any:
    """Full pre-flight: syntax, scheme, DNS, and IP safety. Returns parsed URL."""
    parsed = parse_url(url)
    hostname = parsed.hostname or ""
    # Strip brackets for IPv6 literals before the check.
    check_hostname(hostname.strip("[]"), allow_private=allow_private_ips)
    return parsed


def resolve_redirect(base: str, location: str, *, index: int) -> str:
    """Join a Location header against ``base`` and enforce scheme rules."""
    from urllib.parse import urljoin

    target = urljoin(base, location)
    parsed = parse_url(target)  # reuses scheme/credential checks
    _ = parsed
    return target


def canonicalize_url(url: str) -> str:
    """Stable identity for a direct URL: normalized, credentials and ephemeral params removed."""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    default = {"http": 80, "https": 443}.get(scheme)
    netloc = host if port in (None, default) else f"{host}:{port}"
    # Drop trailing '/' on empty paths so 'https://h/v.mp4/' and '...v.mp4' differ only
    # when they really are different objects — actually keep path verbatim except case of
    # host: paths are case-sensitive, do not lowercase.
    path = parsed.path or "/"
    kept = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in EPHEMERAL_PARAMS]
    query = urlencode(kept, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def redact(url: str) -> str:
    """Safe-to-log form: drops userinfo and ephemeral query params."""
    try:
        parsed = urlparse(url)
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc += f":{parsed.port}"
        kept = [(k, "***" if k.lower() in EPHEMERAL_PARAMS else v)
                for k, v in parse_qsl(parsed.query, keep_blank_values=True)]
        return urlunparse((parsed.scheme, netloc, parsed.path, "",
                           urlencode(kept, doseq=True), ""))
    except Exception:
        return "<unparseable-url>"


__all__ = [
    "ALLOWED_SCHEMES",
    "EPHEMERAL_PARAMS",
    "canonicalize_url",
    "check_hostname",
    "is_ip_blocked",
    "parse_url",
    "redact",
    "resolve_redirect",
    "validate_url",
]
