"""Rule-based event detection: does every event point back at the fact that produced it?

The events layer is the one place where the document stops reporting measurements and starts
making claims — "a command was entered", "the screen changed". Spec §11 makes those claims
auditable by requiring ``refs``, so the assertions that matter here are not "an event was
produced" but *which* facts it cites and *whether its timestamps came from evidence rather
than from an estimate*. A text_appeared event at an interpolated span boundary rather than at a
frame that was actually sampled would be exactly the fabricated timestamp the format forbids.

The regex tables get their own class because both of them guard against fabrication of a
different kind: a markdown heading recorded as a typed shell command is an event that never
happened.
"""

from __future__ import annotations

import pytest

from videocontext.interfaces import EventDetector, FrameContext
from videocontext.processing.events.rules import (
    RuleEventDetector,
    _command_in,
    _error_in,
)
from videocontext.registry import create
from videocontext.schema.v1 import (
    OCRText,
    Scene,
    Utterance,
    VideoContextDocument,
    VideoInfo,
)

DURATION = 60.0


def scene(index: int, start: float, end: float, **kw) -> Scene:
    return Scene(
        id=f"scene_{index:04d}",
        start=start,
        end=end,
        confidence=kw.pop("confidence", 0.8),
        detector=kw.pop("detector", "ffmpeg"),
        change_score=kw.pop("change_score", 0.42),
        signals=kw.pop("signals", ["histogram"]),
        **kw,
    )


def ocr(index: int, text: str, start: float, end: float, **kw) -> OCRText:
    """An OCR event as the dedup stage emits it: a span plus the frames that witnessed it."""
    return OCRText(
        id=f"ocr_{index:04d}",
        text=text,
        start=start,
        end=end,
        confidence=kw.pop("confidence", 0.9),
        frame_count=kw.pop("frame_count", 3),
        # Deliberately inset from start/end: these are the raw evidence, and the tests below
        # check that the events use them in preference to the derived span.
        first_frame_ts=kw.pop("first_frame_ts", start + 0.25),
        last_frame_ts=kw.pop("last_frame_ts", end - 0.25),
        stable=kw.pop("stable", True),
        engine=kw.pop("engine", "tesseract"),
        **kw,
    )


def utt(index: int, text: str, start: float, end: float, **kw) -> Utterance:
    return Utterance(
        id=f"utt_{index:04d}", text=text, start=start, end=end,
        confidence=kw.pop("confidence", 0.8), **kw,
    )


def doc(*, duration: float = DURATION, scenes=(), ocr=(), transcript=()):
    """A real document, not a stand-in: the detector has to work on what the pipeline passes."""
    return VideoContextDocument(
        id="vid_test",
        video=VideoInfo(id="vid_test", filename="demo.mp4", duration=duration, has_audio=True),
        scenes=list(scenes),
        ocr=list(ocr),
        transcript=list(transcript),
    )


def ctx(duration: float = DURATION) -> FrameContext:
    return FrameContext(duration=duration, fps=30.0, width=1280, height=720)


def run(document, **kw) -> list:
    return RuleEventDetector(**kw).detect(document, ctx(document.video.duration))


def of_type(events, kind: str) -> list:
    return [e for e in events if e.type == kind]


class TestSceneChanges:
    def test_one_event_per_boundary_not_per_scene(self):
        # Three scenes have two boundaries between them. The first scene starting is the video
        # starting, which is not a change.
        events = run(doc(scenes=[scene(0, 0, 10), scene(1, 10, 20), scene(2, 20, 30)]))
        assert len(of_type(events, "scene_changed")) == 2

    def test_a_single_scene_produces_no_change(self):
        assert of_type(run(doc(scenes=[scene(0, 0, DURATION)])), "scene_changed") == []

    def test_no_scenes_produces_no_change(self):
        assert of_type(run(doc()), "scene_changed") == []

    def test_the_event_is_an_instant_at_the_boundary(self):
        events = of_type(run(doc(scenes=[scene(0, 0, 10), scene(1, 10, 20)])), "scene_changed")
        assert (events[0].start, events[0].end) == (10.0, 10.0)

    def test_it_refs_both_scenes(self):
        events = of_type(run(doc(scenes=[scene(0, 0, 10), scene(1, 10, 20)])), "scene_changed")
        assert events[0].refs == {"scenes": ["scene_0000", "scene_0001"]}

    def test_it_carries_the_detector_that_found_the_boundary(self):
        events = of_type(
            run(doc(scenes=[scene(0, 0, 10), scene(1, 10, 20, detector="content")])),
            "scene_changed",
        )
        assert events[0].detector == "rule:scene-change/content"

    def test_it_forwards_the_scene_signals_and_score(self):
        second = scene(1, 10, 20, signals=["histogram", "text"], change_score=0.7)
        events = of_type(run(doc(scenes=[scene(0, 0, 10), second])), "scene_changed")
        assert events[0].attributes == {"signals": ["histogram", "text"], "change_score": 0.7}

    def test_confidence_comes_from_the_scene(self):
        scenes = [scene(0, 0, 10), scene(1, 10, 20, confidence=0.55)]
        assert of_type(run(doc(scenes=scenes)), "scene_changed")[0].confidence == 0.55

    def test_source_is_visual(self):
        events = of_type(run(doc(scenes=[scene(0, 0, 10), scene(1, 10, 20)])), "scene_changed")
        assert events[0].source == ["visual"]


class TestTextLifespans:
    def test_stable_text_gets_an_appeared_and_a_disappeared(self):
        events = run(doc(ocr=[ocr(0, "Pricing", 10.0, 20.0)]))
        assert len(of_type(events, "text_appeared")) == 1
        assert len(of_type(events, "text_disappeared")) == 1

    def test_the_timestamps_are_the_witnessed_frames_not_the_span(self):
        """The evidence-first assertion: an event may only claim a moment a frame was sampled."""
        events = run(doc(ocr=[ocr(0, "Pricing", 10.0, 20.0)]))  # frames at 10.25 and 19.75
        assert of_type(events, "text_appeared")[0].start == 10.25
        assert of_type(events, "text_disappeared")[0].start == 19.75

    def test_it_falls_back_to_the_span_when_no_frame_is_recorded(self):
        item = ocr(0, "Pricing", 10.0, 20.0, first_frame_ts=None, last_frame_ts=None)
        events = run(doc(ocr=[item]))
        assert of_type(events, "text_appeared")[0].start == 10.0
        assert of_type(events, "text_disappeared")[0].start == 20.0

    def test_unstable_text_gets_no_events(self):
        # Text seen in a single frame is as likely to be a compression artefact as a caption,
        # and "text appeared" is a stronger claim than one frame supports.
        events = run(doc(ocr=[ocr(0, "flicker", 10.0, 10.4, stable=False, frame_count=1)]))
        assert of_type(events, "text_appeared") == []
        assert of_type(events, "text_disappeared") == []

    def test_the_description_is_the_text_that_appeared(self):
        events = run(doc(ocr=[ocr(0, "Q3 Revenue", 10.0, 20.0)]))
        assert of_type(events, "text_appeared")[0].description == "Q3 Revenue"

    def test_both_ends_ref_the_same_ocr_event(self):
        events = run(doc(ocr=[ocr(0, "Pricing", 10.0, 20.0)]))
        for kind in ("text_appeared", "text_disappeared"):
            assert of_type(events, kind)[0].refs == {"ocr": ["ocr_0000"]}

    def test_the_pair_does_not_share_one_refs_object(self):
        # Both events are built from the same literal; if the dict were aliased, editing one
        # event's refs downstream would silently edit the other's.
        events = run(doc(ocr=[ocr(0, "Pricing", 10.0, 20.0)]))
        appeared = of_type(events, "text_appeared")[0]
        disappeared = of_type(events, "text_disappeared")[0]
        assert appeared.refs is not disappeared.refs

    def test_frame_count_is_carried_as_an_attribute(self):
        events = run(doc(ocr=[ocr(0, "Pricing", 10.0, 20.0, frame_count=7)]))
        assert of_type(events, "text_appeared")[0].attributes["frame_count"] == 7


class TestTerminalEvents:
    def test_a_prompt_line_becomes_a_command_event(self):
        events = run(doc(ocr=[ocr(0, "$ pytest -q tests/", 48.0, 57.0)]))
        assert len(of_type(events, "command_entered")) == 1

    def test_the_span_is_the_ocr_events_own_span(self):
        # The command was on screen for exactly as long as its text was, and that is the only
        # duration there is evidence for.
        event = of_type(run(doc(ocr=[ocr(0, "$ pytest -q", 48.0, 57.0)])), "command_entered")[0]
        assert (event.start, event.end) == (48.0, 57.0)

    def test_the_command_is_isolated_from_the_prompt(self):
        event = of_type(run(doc(ocr=[ocr(0, "$ pytest -q", 48.0, 57.0)])), "command_entered")[0]
        assert event.attributes["command"] == "pytest -q"
        assert event.attributes["line"] == "$ pytest -q"

    def test_an_error_signature_becomes_an_error_event(self):
        item = ocr(0, "E ConnectionError: refused on port 5432", 50.0, 57.0)
        events = of_type(run(doc(ocr=[item])), "error_shown")
        assert len(events) == 1
        assert events[0].detector == "rule:error-signature/exception"

    def test_one_block_can_produce_both(self):
        item = ocr(0, "$ pytest -q\nE AssertionError: nope", 48.0, 57.0)
        events = run(doc(ocr=[item]))
        assert len(of_type(events, "command_entered")) == 1
        assert len(of_type(events, "error_shown")) == 1

    def test_prose_produces_neither(self):
        item = ocr(0, "Our pricing starts at $29 for the starter plan", 20.0, 29.0)
        events = run(doc(ocr=[item]))
        assert of_type(events, "command_entered") == []
        assert of_type(events, "error_shown") == []

    def test_command_detection_can_be_switched_off(self):
        item = ocr(0, "$ pytest -q", 48.0, 57.0)
        assert of_type(run(doc(ocr=[item]), detect_commands=False), "command_entered") == []

    def test_error_detection_can_be_switched_off(self):
        item = ocr(0, "Traceback (most recent call last):", 50.0, 57.0)
        assert of_type(run(doc(ocr=[item]), detect_errors=False), "error_shown") == []

    def test_unstable_text_still_counts_here(self):
        # A command flashes: one frame is enough evidence that it was typed, and unlike a
        # caption's lifespan the claim does not depend on how long it stayed up.
        item = ocr(0, "$ rm -rf build", 48.0, 48.4, stable=False, frame_count=1)
        assert len(of_type(run(doc(ocr=[item])), "command_entered")) == 1

    def test_blank_text_is_skipped(self):
        assert run(doc(ocr=[ocr(0, "   ", 10.0, 20.0, stable=False)])) == []


class TestSilences:
    """Each test pins ``duration`` to the last utterance's end unless it is exercising the
    trailing gap — otherwise every fixture also produces a legitimate silence running from the
    last word to the end of the video, and the count under test stops meaning anything."""

    def test_a_long_gap_becomes_a_pair(self):
        pair = [utt(0, "one", 0.0, 2.0), utt(1, "two", 6.0, 8.0)]
        events = run(doc(duration=8.0, transcript=pair))
        assert len(of_type(events, "silence_started")) == 1
        assert len(of_type(events, "silence_ended")) == 1
        assert of_type(events, "silence_started")[0].start == 2.0
        assert of_type(events, "silence_ended")[0].start == 6.0

    def test_a_short_gap_is_a_breath_not_a_silence(self):
        pair = [utt(0, "one", 0.0, 2.0), utt(1, "two", 2.5, 4.0)]
        assert of_type(run(doc(duration=4.0, transcript=pair)), "silence_started") == []

    def test_the_threshold_is_configurable(self):
        pair = [utt(0, "one", 0.0, 2.0), utt(1, "two", 3.0, 4.0)]
        assert of_type(run(doc(duration=4.0, transcript=pair)), "silence_started") == []
        assert of_type(
            run(doc(duration=4.0, transcript=pair), min_silence_s=0.5), "silence_started"
        ) != []

    def test_zero_disables_the_rule(self):
        # Rather than emitting a silence between every pair of utterances.
        pair = [utt(0, "one", 0.0, 2.0), utt(1, "two", 6.0, 8.0)]
        assert of_type(run(doc(transcript=pair), min_silence_s=0), "silence_started") == []

    def test_leading_silence_is_detected(self):
        events = run(doc(transcript=[utt(0, "late", 5.0, 8.0)]))
        assert of_type(events, "silence_started")[0].start == 0.0

    def test_trailing_silence_runs_to_the_media_duration(self):
        events = run(doc(transcript=[utt(0, "early", 0.0, 4.0)]))
        ended = of_type(events, "silence_ended")
        assert ended[-1].start == DURATION

    def test_trailing_silence_is_skipped_when_the_duration_is_unknown(self):
        # Without a duration the trailing gap has no end, and inventing one would be a claim
        # about a stretch of video nothing measured.
        events = run(doc(duration=0.0, transcript=[utt(0, "early", 0.0, 4.0)]))
        assert of_type(events, "silence_ended") == []

    def test_it_refs_the_utterances_on_both_sides(self):
        pair = [utt(0, "one", 0.0, 2.0), utt(1, "two", 6.0, 8.0)]
        events = run(doc(duration=8.0, transcript=pair))
        assert of_type(events, "silence_started")[0].refs == {
            "transcript": ["utt_0000", "utt_0001"]
        }

    def test_it_carries_no_confidence(self):
        # Derived from an absence. A number here would read as a scored detection.
        pair = [utt(0, "one", 0.0, 2.0), utt(1, "two", 6.0, 8.0)]
        for event in of_type(run(doc(transcript=pair)), "silence_started"):
            assert event.confidence is None

    def test_the_gap_length_is_recorded(self):
        pair = [utt(0, "one", 0.0, 2.0), utt(1, "two", 6.0, 8.0)]
        assert of_type(run(doc(duration=8.0, transcript=pair)), "silence_started")[
            0
        ].attributes == {"duration_s": 4.0}

    def test_a_trailing_gap_is_reported_alongside_an_interior_one(self):
        # Both are real: the fixture stops talking at 8s in a 60s video.
        pair = [utt(0, "one", 0.0, 2.0), utt(1, "two", 6.0, 8.0)]
        starts = [e.start for e in of_type(run(doc(transcript=pair)), "silence_started")]
        assert starts == [2.0, 8.0]

    def test_an_empty_transcript_produces_nothing(self):
        assert of_type(run(doc()), "silence_started") == []

    def test_a_span_never_escapes_the_media_duration(self):
        # An ASR stage can report an end slightly past the container duration; clipping keeps
        # the event inside the video it describes.
        events = run(doc(duration=10.0, transcript=[utt(0, "x", 0.0, 1.0)]))
        for event in events:
            assert 0.0 <= event.start <= 10.0


class TestOutputInvariants:
    @pytest.fixture
    def full(self):
        return doc(
            scenes=[scene(0, 0, 20), scene(1, 20, 48), scene(2, 48, DURATION)],
            ocr=[
                ocr(0, "Quarterly Business Review", 0.5, 6.0),
                ocr(1, "Pricing", 20.0, 29.0),
                ocr(2, "$ pytest -q tests/", 48.0, 57.0),
                ocr(3, "E ConnectionError: refused", 50.0, 57.0),
                ocr(4, "flicker", 30.0, 30.4, stable=False, frame_count=1),
            ],
            transcript=[
                utt(0, "welcome", 0.5, 2.0),
                utt(1, "pricing starts at", 20.0, 24.0),
                utt(2, "the terminal", 48.0, 52.0),
            ],
        )

    def test_ids_are_sequential_and_spec_shaped(self, full):
        events = run(full)
        assert [e.id for e in events] == [f"evt_{i:04d}" for i in range(len(events))]

    def test_events_are_in_timeline_order(self, full):
        starts = [e.start for e in run(full)]
        assert starts == sorted(starts)

    def test_every_event_cites_at_least_one_fact(self, full):
        # The invariant the whole layer exists to uphold (spec §11).
        assert all(e.refs for e in run(full))

    def test_every_ref_points_at_something_in_the_document(self, full):
        known = {
            "scenes": {s.id for s in full.scenes},
            "ocr": {o.id for o in full.ocr},
            "transcript": {u.id for u in full.transcript},
        }
        dangling = [
            (e.id, kind, ref)
            for e in run(full)
            for kind, refs in e.refs.items()
            for ref in refs
            if ref not in known.get(kind, set())
        ]
        assert dangling == []

    def test_no_event_refs_the_unstable_ocr_reading(self, full):
        # ocr_0004 is a single-frame flicker with no command or error in it; nothing should
        # cite it, and a text_appeared event citing it would be the bug this pins.
        cited = {ref for e in run(full) for refs in e.refs.values() for ref in refs}
        assert "ocr_0004" not in cited

    def test_every_event_names_its_rule(self, full):
        assert all(e.detector and e.detector.startswith("rule:") for e in run(full))

    def test_every_event_names_its_modality(self, full):
        assert all(e.source for e in run(full))

    def test_spans_are_well_formed(self, full):
        for event in run(full):
            assert event.end >= event.start
            assert 0.0 <= event.start <= full.video.duration

    def test_detection_is_deterministic(self, full):
        first = run(full)
        second = run(full)
        assert [e.model_dump() for e in first] == [e.model_dump() for e in second]

    def test_it_does_not_mutate_the_document(self, full):
        before = full.model_dump()
        run(full)
        assert full.model_dump() == before

    def test_an_empty_document_yields_no_events(self):
        assert run(doc()) == []

    def test_a_document_with_only_scenes_still_works(self):
        assert len(run(doc(scenes=[scene(0, 0, 30), scene(1, 30, DURATION)]))) == 1


class TestCommandRecognition:
    """Guards the line between a shell prompt and the punctuation of a slide."""

    @pytest.mark.parametrize(
        ("line", "command"),
        [
            ("$ pytest -q tests/", "pytest -q tests/"),
            ("> npm run build", "npm run build"),
            ("% ls -la", "ls -la"),
            ("user@host:~/proj$ git status", "git status"),
            ("root@box:/etc# systemctl restart nginx", "systemctl restart nginx"),
            ("PS C:\\Users\\dev> dotnet build", "dotnet build"),
            ("C:\\src> make", "make"),
            ("VideoContext demo\n$ pytest -q tests/\n1 failed", "pytest -q tests/"),
        ],
    )
    def test_recognised(self, line, command):
        assert _command_in(line) == command

    @pytest.mark.parametrize(
        "line",
        [
            "# make install",          # markdown heading, not a root prompt
            "# Introduction",
            "> Note: this matters",    # blockquote with a label
            "$29 per month",           # price: no separator after the sigil
            "Revenue > Pricing > Demo",
            "$",                       # a prompt with nothing typed at it
            "$ ",
            "100% complete",
            "Q1 > Q2 growth",
            "",
        ],
    )
    def test_rejected(self, line):
        assert _command_in(line) is None


class TestErrorRecognition:
    @pytest.mark.parametrize(
        ("text", "signal"),
        [
            ("Traceback (most recent call last):", "traceback"),
            ("E ConnectionError: refused on port 5432", "exception"),
            ("FAILED tests/test_api.py::test_login", "failure"),
            ("500 Internal Server Error", "http_status"),
            ("404 Not Found", "http_status"),
            ("429 Too Many Requests", "http_status"),
        ],
    )
    def test_recognised(self, text, signal):
        assert _error_in(text) == signal

    def test_the_most_specific_signature_wins(self):
        # Ordering is load-bearing: "500 Internal Server Error" contains "Error", so a broader
        # pattern placed first would record an HTTP response as a thrown exception.
        assert _error_in("Traceback ...\n500 Internal Server Error") == "traceback"
        assert _error_in("500 Internal Server Error") == "http_status"

    @pytest.mark.parametrize(
        "text",
        [
            "1 failed, 2 passed",             # lowercase: a summary line, not an error
            "an error occurred somewhere",    # prose about errors is not an error being shown
            "200 OK",
            "errors: 0",
            "Pricing starts at $29",
            "Listening on port 4000 Ready",
            "",
        ],
    )
    def test_rejected(self, text):
        assert _error_in(text) is None


class TestPluginContract:
    def test_it_satisfies_the_detector_protocol(self):
        # §25: a third-party detector is accepted on the strength of its shape.
        assert isinstance(RuleEventDetector(), EventDetector)

    def test_it_resolves_through_the_registry(self):
        # Regression: the registry declared this module before it existed, so `create` raised
        # ModuleNotFoundError for the only built-in detector.
        assert create("event_detector", "rules").name == "rules"

    def test_a_negative_threshold_is_clamped_not_honoured(self):
        assert RuleEventDetector(min_silence_s=-5).min_silence_s == 0.0
