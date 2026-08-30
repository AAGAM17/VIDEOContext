"""SRT/WebVTT parsing, and the engine that turns a track into a transcript.

A subtitle track is *better* evidence than a speech model's output — human-checked and
precisely timed — which is exactly why the parser has to be pedantic. Every case here is one
where a sloppier reading would put words into the transcript that nobody said, or lose words
that were.
"""

from __future__ import annotations

import pytest

from conftest import needs_ffmpeg
from videocontent.config import ASRConfig
from videocontent.interfaces import FrameContext
from videocontent.processing.asr.subtitles import (
    MAX_SUBTITLE_BYTES,
    SubtitleASR,
    parse_cues,
    read_subtitle_file,
)

SRT = """\
1
00:00:01,000 --> 00:00:04,500
Welcome to the <i>VideoContext</i> demo.

2
00:01:02,250 --> 00:01:05,000
Revenue grew to
forty two lakh.
"""

VTT = """\
WEBVTT - Demo

NOTE This comment spans
two lines and is not dialogue.

STYLE
::cue { color: peachpuff }

intro
01:02.500 --> 01:07.000 line:0 position:20%
<v.loud Priya>We should talk about {\\an8}pricing.
"""


def ctx(duration: float = 62.4, language: str | None = "en") -> FrameContext:
    return FrameContext(
        duration=duration, fps=30.0, width=1280, height=720, language=language
    )


class TestParseCuesFormats:
    def test_srt(self):
        assert parse_cues(SRT) == [
            (1.0, 4.5, "Welcome to the VideoContext demo.", None),
            (62.25, 65.0, "Revenue grew to forty two lakh.", None),
        ]

    def test_webvtt(self):
        assert parse_cues(VTT) == [(62.5, 67.0, "We should talk about pricing.", "Priya")]

    def test_webvtt_short_timestamp_omits_hours(self):
        assert parse_cues("WEBVTT\n\n01:02.500 --> 01:07.000\nhi\n")[0][:2] == (62.5, 67.0)

    def test_hours_are_read(self):
        cue = parse_cues("1\n01:01:02,500 --> 01:01:07,000\nhi\n")[0]
        assert cue[:2] == (3662.5, 3667.0)

    def test_short_fraction_pads_right(self):
        # ",5" is five hundred milliseconds, not five.
        assert parse_cues("1\n00:00:01,5 --> 00:00:02,05\nhi\n")[0][:2] == (1.5, 2.05)

    def test_crlf(self):
        payload = "1\r\n00:00:01,000 --> 00:00:02,000\r\nhi\r\n\r\n2\r\n" \
                  "00:00:03,000 --> 00:00:04,000\r\nthere\r\n"
        assert [c[2] for c in parse_cues(payload)] == ["hi", "there"]

    def test_bom_does_not_eat_the_first_cue(self):
        assert len(parse_cues("﻿" + SRT)) == 2


class TestParseCuesRejectsNonDialogue:
    def test_header_note_and_style_blocks_are_not_cues(self):
        assert len(parse_cues(VTT)) == 1

    def test_cue_index_is_not_dialogue(self):
        assert parse_cues(SRT)[1][2].startswith("Revenue")

    def test_webvtt_cue_identifier_is_not_dialogue(self):
        assert parse_cues("WEBVTT\n\nintro\n00:00:01.000 --> 00:00:02.000\nhi\n")[0][2] == "hi"

    def test_empty_cue_yields_nothing(self):
        assert parse_cues("1\n00:00:01,000 --> 00:00:02,000\n\n") == []

    def test_prose_yields_nothing(self):
        assert parse_cues("not a subtitle file at all\njust prose\n") == []

    def test_timing_line_without_timestamps_is_not_a_cue(self):
        assert parse_cues("1\nsomewhere --> elsewhere\nhi\n") == []


class TestParseCuesNeverFabricates:
    """The failure mode that matters: text in the transcript that was never spoken."""

    def test_missing_blank_line_does_not_leak_a_timestamp_into_dialogue(self):
        payload = (
            "1\n00:00:01,000 --> 00:00:02,000\nhello\n"
            "2\n00:00:03,000 --> 00:00:04,000\nworld\n"
        )
        assert parse_cues(payload) == [
            (1.0, 2.0, "hello", None),
            (3.0, 4.0, "world", None),
        ]

    def test_tags_are_removed_without_inserting_a_space(self):
        # `demo</i>.` must not become `demo .`, which no phrase query would match.
        assert parse_cues("1\n00:00:01,000 --> 00:00:02,000\nthe <i>demo</i>.\n")[0][2] == (
            "the demo."
        )

    def test_adjacent_tagged_words_are_not_glued_or_split_wrongly(self):
        payload = "1\n00:00:01,000 --> 00:00:02,000\n<b>Video</b><b>Context</b>\n"
        assert parse_cues(payload)[0][2] == "VideoContext"

    def test_display_line_wrapping_becomes_a_space_not_a_join(self):
        payload = "1\n00:00:01,000 --> 00:00:02,000\nRevenue grew\nforty two lakh.\n"
        assert parse_cues(payload)[0][2] == "Revenue grew forty two lakh."

    def test_dialogue_containing_an_arrow_survives(self):
        payload = "1\n00:00:01,000 --> 00:00:02,000\nrequest --> response\n"
        assert parse_cues(payload)[0][2] == "request --> response"

    def test_a_cue_ending_in_a_number_keeps_it(self):
        payload = (
            "1\n00:00:01,000 --> 00:00:02,000\nchapter 42\n"
            "2\n00:00:03,000 --> 00:00:04,000\nnext\n"
        )
        assert [c[2] for c in parse_cues(payload)] == ["chapter 42", "next"]


class TestReadSubtitleFile:
    def test_reads_and_strips_a_bom(self, tmp_path):
        path = tmp_path / "a.srt"
        path.write_bytes("﻿1\n".encode())
        assert read_subtitle_file(path) == "1\n"

    def test_undecodable_bytes_do_not_raise(self):
        # A subtitle file is untrusted input; a mislabelled encoding must not end the run.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.srt"
            path.write_bytes(b"1\n00:00:01,000 --> 00:00:02,000\ncaf\xe9\n")
            assert "caf" in read_subtitle_file(path)

    def test_oversized_file_is_refused(self, tmp_path, monkeypatch):
        path = tmp_path / "big.srt"
        path.write_text("x" * 64)
        monkeypatch.setattr(
            "videocontent.processing.asr.subtitles.MAX_SUBTITLE_BYTES", 16
        )
        with pytest.raises(ValueError, match="over the"):
            read_subtitle_file(path)

    def test_the_limit_is_generous_enough_for_a_real_track(self):
        # A feature film's subtitles are well under a megabyte; the cap exists for hostile
        # input, and must never reject a legitimate file.
        assert MAX_SUBTITLE_BYTES >= 1024 * 1024


class TestSubtitleEngine:
    def test_sidecar_file_becomes_utterances(self, tmp_path):
        path = tmp_path / "demo.srt"
        path.write_text(SRT)
        out = SubtitleASR(ASRConfig(), source=path).transcribe(path, ctx())

        assert [u.id for u in out.utterances] == ["utt_0000", "utt_0001"]
        assert out.utterances[0].text == "Welcome to the VideoContext demo."
        assert (out.utterances[0].start, out.utterances[0].end) == (1.0, 4.5)
        assert out.warnings == []
        assert out.model == "subtitles:srt"

    def test_speaker_label_comes_from_the_file_not_invention(self, tmp_path):
        path = tmp_path / "demo.vtt"
        path.write_text(VTT)
        out = SubtitleASR(ASRConfig(), source=path).transcribe(path, ctx())
        assert out.utterances[0].speaker == "Priya"

    def test_confidence_is_absent_because_none_was_reported(self, tmp_path):
        path = tmp_path / "demo.srt"
        path.write_text(SRT)
        out = SubtitleASR(ASRConfig(), source=path).transcribe(path, ctx())
        assert all(u.confidence is None for u in out.utterances)

    def test_cues_past_the_end_are_clamped_to_the_duration(self, tmp_path):
        path = tmp_path / "demo.srt"
        path.write_text("1\n00:00:58,000 --> 00:01:11,000\ntail\n")
        out = SubtitleASR(ASRConfig(), source=path).transcribe(path, ctx(duration=62.4))
        assert out.utterances[0].end == 62.4

    def test_a_sidecar_file_is_available_without_ffmpeg(self, tmp_path):
        path = tmp_path / "demo.srt"
        path.write_text(SRT)
        assert SubtitleASR(ASRConfig(), source=path).available() is True

    def test_a_missing_sidecar_file_is_not_available(self, tmp_path):
        assert SubtitleASR(ASRConfig(), source=tmp_path / "nope.srt").available() is False

    def test_an_empty_track_is_a_warning_not_a_crash(self, tmp_path):
        path = tmp_path / "empty.srt"
        path.write_text("WEBVTT\n\nNOTE nothing here\n")
        out = SubtitleASR(ASRConfig(), source=path).transcribe(path, ctx())
        assert out.utterances == []
        assert out.warnings and "no subtitle cues" in out.warnings[0]

    def test_language_comes_from_the_context(self, tmp_path):
        path = tmp_path / "demo.srt"
        path.write_text(SRT)
        out = SubtitleASR(ASRConfig(), source=path).transcribe(path, ctx(language="hi"))
        assert out.language == "hi"
        assert all(u.language == "hi" for u in out.utterances)

    @needs_ffmpeg
    def test_a_video_without_a_track_reports_the_absence(self, tmp_path):
        # Handed a media file rather than a sidecar, the engine probes the container and says
        # plainly that there was nothing to read — an empty result, not a failure.
        media = tmp_path / "silent.wav"
        media.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        out = SubtitleASR(ASRConfig(), source=media).transcribe(media, ctx())
        assert out.utterances == []
        assert out.warnings != []
