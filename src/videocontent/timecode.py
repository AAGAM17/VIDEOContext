"""Timecodes — the one format every surface prints and parses.

Seconds are the format's only representation of time (spec: floats, seconds from media start).
Humans do not read 201.4 as "three and a half minutes in", so every surface has to render, and
a CLI has to accept, ``MM:SS`` and ``HH:MM:SS``. Doing that in two places would eventually
produce two answers for the same instant, so it is done here.

Parsing is deliberately liberal about *shape* and strict about *value*: ``3:21``, ``03:21.5``,
``00:03:21.450`` and a bare ``201.4`` all parse, while ``3:75`` does not, because a minute has
sixty seconds and accepting it would silently answer a question about a different moment than
the one asked about.
"""

from __future__ import annotations

import math
import re

#: ``[HH:]MM:SS[.mmm]`` — hours optional, fractional seconds optional. Anchored: a timecode is
#: the whole argument, so ``1:23 and something`` is an error rather than a silent truncation.
_CLOCK = re.compile(r"^(?:(\d+):)?([0-5]?\d):([0-5]?\d)(\.\d+)?$")
_SECONDS = re.compile(r"^\d+(\.\d+)?$")


def format_timecode(seconds: float, *, millis: bool = True) -> str:
    """``HH:MM:SS.mmm`` (or ``HH:MM:SS``). Hours are always shown, so timecodes sort as text."""
    if not math.isfinite(seconds):
        raise ValueError(f"not a finite number of seconds: {seconds!r}")
    negative = seconds < 0
    total = abs(float(seconds))
    hours, rest = divmod(total, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    if millis:
        # Round to milliseconds first: formatting 59.9996 as "%06.3f" gives "60.000", which
        # would print 00:01:60.000 for what is really the next minute.
        secs = round(secs, 3)
        if secs >= 60.0:
            secs -= 60.0
            minutes += 1
            if minutes >= 60.0:
                minutes -= 60.0
                hours += 1
        body = f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"
    else:
        body = f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    return f"-{body}" if negative else body


def parse_timecode(text: str) -> float:
    """Seconds from ``MM:SS``, ``HH:MM:SS``, either with fractional seconds, or plain seconds."""
    value = text.strip()
    if _SECONDS.match(value):
        return float(value)
    match = _CLOCK.match(value)
    if match is None:
        raise ValueError(
            f"not a timecode: {text!r} (expected MM:SS, HH:MM:SS, or seconds)"
        )
    hours, minutes, seconds, frac = match.groups()
    return (
        int(hours or 0) * 3600
        + int(minutes) * 60
        + int(seconds)
        + (float(frac) if frac else 0.0)
    )


def format_span(start: float, end: float, *, millis: bool = True) -> str:
    """``HH:MM:SS.mmm → HH:MM:SS.mmm``, or a single timecode when the span is an instant."""
    left = format_timecode(start, millis=millis)
    if end <= start:
        return left
    return f"{left} → {format_timecode(end, millis=millis)}"


__all__ = ["format_span", "format_timecode", "parse_timecode"]
