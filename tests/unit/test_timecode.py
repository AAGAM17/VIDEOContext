"""Timecodes: the rendering every surface shares, and the parsing a CLI has to survive.

Small module, but two of its failure modes are the kind that get shipped: a timecode that reads
``00:01:60.000`` because seconds were rounded after being split off, and a parser that accepts
``3:75`` and silently answers a question about a different moment than the one asked.
"""

from __future__ import annotations

import pytest

from videocontent.timecode import format_span, format_timecode, parse_timecode


class TestFormat:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00.000"),
            (1.5, "00:00:01.500"),
            (61.0, "00:01:01.000"),
            (201.45, "00:03:21.450"),
            (3600.0, "01:00:00.000"),
            (3661.001, "01:01:01.001"),
        ],
    )
    def test_it_renders_hours_minutes_seconds_millis(self, seconds, expected):
        assert format_timecode(seconds) == expected

    def test_hours_are_always_shown(self):
        # So that a column of timecodes sorts correctly as text, in a terminal or a CSV.
        assert format_timecode(5.0).startswith("00:")

    def test_a_value_just_under_a_minute_does_not_render_as_sixty_seconds(self):
        # The regression: splitting first and rounding afterwards gives "00:00:60.000", which is
        # not a time. 59.9996 s is one millisecond short of a minute and must round up to it.
        assert format_timecode(59.9996) == "00:01:00.000"

    def test_the_carry_propagates_to_hours(self):
        assert format_timecode(3599.9999) == "01:00:00.000"

    def test_millis_can_be_dropped(self):
        assert format_timecode(201.45, millis=False) == "00:03:21"

    def test_a_negative_offset_keeps_its_sign(self):
        # Not produced by the pipeline, but a caller computing `ts - window` can reach it, and
        # "00:00:01.500" for -1.5 would be a lie.
        assert format_timecode(-1.5) == "-00:00:01.500"

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_values_are_refused(self, value):
        with pytest.raises(ValueError, match="finite"):
            format_timecode(value)


class TestParse:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("0", 0.0),
            ("201", 201.0),
            ("201.45", 201.45),
            ("3:21", 201.0),
            ("03:21", 201.0),
            ("03:21.45", 201.45),
            ("00:03:21.450", 201.45),
            ("1:00:00", 3600.0),
            ("  3:21  ", 201.0),
        ],
    )
    def test_it_accepts_the_forms_a_person_types(self, text, expected):
        assert parse_timecode(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["3:75", "1:2:3:4", "", "abc", "3:21 pm", "-5", "1:60"])
    def test_it_refuses_what_it_cannot_mean(self, text):
        with pytest.raises(ValueError):
            parse_timecode(text)

    def test_it_round_trips_with_the_formatter(self):
        for seconds in (0.0, 1.5, 201.45, 3661.001):
            assert parse_timecode(format_timecode(seconds)) == pytest.approx(seconds)


class TestSpan:
    def test_a_span_shows_both_ends(self):
        assert format_span(1.0, 2.0) == "00:00:01.000 → 00:00:02.000"

    def test_an_instant_shows_one_timecode(self):
        # Zero-length events are how the detector represents a boundary; rendering them as
        # "x → x" reads as a bug in the formatter.
        assert format_span(1.0, 1.0) == "00:00:01.000"
