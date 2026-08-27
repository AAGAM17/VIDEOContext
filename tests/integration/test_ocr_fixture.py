"""OCR against the real fixture: does the text come back *at the right time*?

The unit tests pin the mechanics with synthetic input. This asserts the thing the format
actually promises (brief §17, §52): a caller asking "when was this on screen?" gets an answer
that is true. A test that only checks the string appears *somewhere* in the document would pass
on a stage that returned one event spanning the whole video — which is exactly the unsupported
timestamp the evidence-first contract forbids.

So every acceptance string is tied to the shot the fixture generator drew it in, and the event
carrying it must be located inside that shot. The shot windows come from
``tests/fixtures/demo.manifest.json``, written by ``scripts/make_test_video.py`` — the same
source of truth as the video itself, so this cannot drift from the fixture.

Sampling is deliberately :class:`FixedSampler`, not the adaptive one: a uniform 1 fps grid over
the whole timeline keeps this a test of OCR rather than a test of scene detection, which has its
own tests and its own failure modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# tests/conftest.py, which pytest has already imported as the top-level module `conftest`
# (its directory goes on sys.path under the default `prepend` import mode). Importing the gates
# rather than re-deriving them keeps one definition of "is tesseract installed".
from conftest import needs_demo, needs_ffmpeg, needs_tesseract
from videocontent.config import ProcessingConfig
from videocontent.interfaces import FrameContext
from videocontent.media.frames import extract_plan
from videocontent.media.probe import probe
from videocontent.processing.ocr import deduplicate, normalize
from videocontent.processing.ocr.tesseract import TesseractOCR
from videocontent.processing.sampling.fixed import FixedSampler

pytestmark = [pytest.mark.integration, needs_ffmpeg, needs_tesseract, needs_demo]

#: Acceptance string -> the manifest shot whose screen draws it. Every string here is also in
#: the manifest's ``expected.on_screen_text``; the mapping adds *where* it must be found.
TEXT_IN_SHOT: dict[str, str] = {
    "Revenue Rs 42L": "revenue",
    "Competitor A charges $2 per hour of video": "competitor",
    "localhost:3000/login": "browser",
    "$ pytest -q tests/": "terminal",
    "ConnectionError": "terminal",
}

#: A sampled frame lands within 1/base_fps of any instant, and a span end is estimated as the
#: midpoint to the next frame, so an event boundary can legitimately sit a frame outside its
#: shot. Anything beyond that is a real localisation error, not sampling slack.
BOUNDARY_SLACK_S = 1.5


@pytest.fixture(scope="module")
def ocr_result(demo_video: Path, tmp_path_factory) -> tuple[list, list[float], float]:
    """Run the real stage once: probe -> fixed plan -> frames -> tesseract -> dedupe.

    Frames go to a pytest temp directory rather than the repo workspace so the test neither
    reads a stale cache nor leaves derived images behind (§32).
    """
    cfg = ProcessingConfig()
    media = probe(demo_video)
    ctx = FrameContext(
        duration=media.duration or 0.0,
        fps=media.fps,
        width=media.width,
        height=media.height,
    )
    plan = FixedSampler(cfg.sampling).plan(ctx)
    frames = extract_plan(
        demo_video, plan, tmp_path_factory.mktemp("frames"), scale_width=1280, quality=3
    )
    assert frames, "no frames extracted; ffmpeg produced nothing"

    engine = TesseractOCR(cfg.ocr)
    observations = engine.extract(frames, ctx)
    frame_ts = [f.ts for f in frames]
    events = deduplicate(
        observations, config=cfg.ocr, duration=ctx.duration,
        frame_ts=frame_ts, engine=engine.name,
    )
    return events, frame_ts, ctx.duration


@pytest.fixture(scope="module")
def shots(manifest: dict) -> dict[str, tuple[float, float]]:
    return {s["shot"]: (float(s["start"]), float(s["end"])) for s in manifest["shots"]}


def _matches(events, phrase: str) -> list:
    """Events whose text contains ``phrase``, compared the way a search would compare it."""
    needle = normalize(phrase)
    return [e for e in events if needle in normalize(e.text)]


class TestAcceptanceStringsAreLocated:
    """Each expected string is found, and found inside the shot that drew it."""

    @pytest.mark.parametrize(("phrase", "shot"), sorted(TEXT_IN_SHOT.items()))
    def test_phrase_is_present(self, ocr_result, phrase, shot):
        events, _ts, _dur = ocr_result
        assert _matches(events, phrase), f"{phrase!r} was not recovered from the fixture at all"

    @pytest.mark.parametrize(("phrase", "shot"), sorted(TEXT_IN_SHOT.items()))
    def test_phrase_is_timestamped_inside_its_shot(self, ocr_result, shots, phrase, shot):
        events, _ts, _dur = ocr_result
        start, end = shots[shot]
        found = _matches(events, phrase)
        assert found, f"{phrase!r} missing"

        # The midpoint test is what rejects a span that merely *overlaps* the shot because it
        # covers most of the video. At least one event must genuinely sit in the window.
        located = [e for e in found if start <= (e.start + e.end) / 2 <= end]
        assert located, (
            f"{phrase!r} was read but not localised to shot {shot!r} ({start}-{end}); "
            f"got spans {[(round(e.start, 2), round(e.end, 2)) for e in found]}"
        )

    @pytest.mark.parametrize(("phrase", "shot"), sorted(TEXT_IN_SHOT.items()))
    def test_phrase_span_does_not_leak_past_its_shot(self, ocr_result, shots, phrase, shot):
        """A span may be a frame wide at the edges; it may not bleed into other shots.

        This is the assertion that would fail if run-collapsing ever replayed a reading onto
        frames that do not show it — the specific way this optimisation could go wrong.
        """
        events, _ts, _dur = ocr_result
        start, end = shots[shot]
        located = [
            e for e in _matches(events, phrase) if start <= (e.start + e.end) / 2 <= end
        ]
        assert located, f"{phrase!r} missing"
        for event in located:
            assert event.start >= start - BOUNDARY_SLACK_S, (
                f"{phrase!r} claims to start {start - event.start:.2f}s before shot {shot!r}"
            )
            assert event.end <= end + BOUNDARY_SLACK_S, (
                f"{phrase!r} claims to persist {event.end - end:.2f}s past shot {shot!r}"
            )


class TestEvidenceInvariants:
    """Properties every event must satisfy for a timestamp to count as evidence."""

    def test_events_were_produced(self, ocr_result):
        events, _ts, _dur = ocr_result
        assert len(events) > 10, "a 10-shot fixture full of text should yield many events"

    def test_spans_are_ordered_and_well_formed(self, ocr_result):
        events, _ts, duration = ocr_result
        for event in events:
            assert event.start <= event.end, f"inverted span on {event.text!r}"
            assert 0.0 <= event.start <= duration
            assert 0.0 <= event.end <= duration, (
                f"{event.text!r} ends at {event.end} past the {duration}s video"
            )
        starts = [e.start for e in events]
        assert starts == sorted(starts), "events must be emitted in timeline order"

    def test_frame_timestamps_are_real_sampled_frames(self, ocr_result):
        """The core evidence-first invariant.

        ``start``/``end`` are *estimates* derived from sampling, but ``first_frame_ts`` and
        ``last_frame_ts`` are raw evidence: each must be a frame that was actually decoded and
        recognised. If one is not, the document is asserting it saw text in a frame that never
        existed.
        """
        events, frame_ts, _dur = ocr_result
        sampled = set(frame_ts)
        for event in events:
            assert event.first_frame_ts in sampled, (
                f"{event.text!r} cites frame {event.first_frame_ts} which was never sampled"
            )
            assert event.last_frame_ts in sampled, (
                f"{event.text!r} cites frame {event.last_frame_ts} which was never sampled"
            )
            assert event.first_frame_ts <= event.last_frame_ts

    def test_span_contains_its_own_evidence(self, ocr_result):
        events, _ts, _dur = ocr_result
        for event in events:
            assert event.start <= event.first_frame_ts + 1e-6
            assert event.last_frame_ts <= event.end + 1e-6

    def test_no_empty_text_and_confidence_is_a_fraction(self, ocr_result):
        events, _ts, _dur = ocr_result
        for event in events:
            assert event.text.strip(), "an event with no text is not evidence of anything"
            if event.confidence is not None:
                assert 0.0 <= event.confidence <= 1.0

    def test_normalised_boxes_stay_inside_the_frame(self, ocr_result):
        """Guards the upscale path: a missing scale correction puts boxes outside the frame."""
        events, _ts, _dur = ocr_result
        for event in events:
            if not event.bbox_normalized:
                continue
            for value in event.bbox_normalized:
                assert -0.01 <= value <= 1.01, (
                    f"{event.text!r} has a normalised box outside the frame: "
                    f"{event.bbox_normalized}"
                )

    def test_deduplication_actually_collapses(self, ocr_result):
        """A slide held for seconds must become one event, not one per frame."""
        events, frame_ts, _dur = ocr_result
        held = [e for e in events if e.frame_count > 1]
        assert held, "nothing was collapsed; temporal deduplication is not running"
        assert len(events) < len(frame_ts), (
            "more events than frames means text is being re-emitted per frame"
        )


class TestManifestAgreement:
    def test_every_manifest_acceptance_string_is_covered_here(self, manifest):
        """Keeps this file honest as the fixture grows.

        If a string is added to the manifest's acceptance list, it must also be given a shot
        here — otherwise the new expectation would be silently untested.
        """
        expected = set(manifest.get("expected", {}).get("on_screen_text", []))
        assert expected <= set(TEXT_IN_SHOT), (
            f"manifest expects text with no shot mapping in this test: "
            f"{sorted(expected - set(TEXT_IN_SHOT))}"
        )

    def test_mapped_shots_exist_in_the_manifest(self, shots):
        unknown = sorted(set(TEXT_IN_SHOT.values()) - set(shots))
        assert not unknown, f"TEXT_IN_SHOT references shots the manifest does not define: {unknown}"


def test_manifest_min_scenes_is_plausible_for_this_fixture(manifest):
    """A cheap consistency check on the fixture's own expectations."""
    assert len(manifest["shots"]) >= manifest["expected"]["min_scenes"]


@needs_demo
def test_fixture_json_is_loadable_without_the_video(manifest):
    """The manifest must stand alone — the docs and benchmarks read it directly."""
    assert manifest["duration"] > 0
    assert Path(manifest["video"]).name.endswith(".mp4")
    json.dumps(manifest)  # must round-trip
