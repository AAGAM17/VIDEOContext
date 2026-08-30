"""Real Whisper against the fixture: does a transcript's timestamps mean anything?

This is the speech counterpart to ``test_ocr_fixture.py``, and it exists for the same reason.
A test that only checked "the word 'revenue' appears in the transcript" would pass on a stage
that returned one utterance spanning the whole video, which is precisely the unsupported claim
the format forbids (spec §7). So every phrase is checked against the shot whose *narration*
produced it, using the manifest as ground truth — and the two silent shots are checked to be
empty, which is the assertion that fails if a model's hallucinated filler ever gets through.

Marked ``slow``: the first run downloads the ``base`` weights (~140 MB) into the Hugging Face
cache. That download is the only externally-reaching step (brief §32); no audio leaves the
machine.
"""

from __future__ import annotations

import json

import pytest

from conftest import DEMO_MANIFEST, DEMO_VIDEO, needs_demo, needs_ffmpeg
from videocontext.config import ASRConfig
from videocontext.interfaces import FrameContext
from videocontext.media.audio import extract_audio
from videocontext.media.probe import probe
from videocontext.processing.asr import resolve_engine
from videocontext.processing.asr.faster_whisper import installed as whisper_installed

needs_whisper = pytest.mark.skipif(
    not whisper_installed(), reason="faster-whisper is not installed"
)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    needs_ffmpeg,
    needs_demo,
    needs_whisper,
]

#: Manifest ``spoken_phrases`` mapped to the shot whose narration contains them. The `base`
#: model mishears some of the fixture's narration ("forty two lakh" comes out as "42-lock"),
#: so the phrases asserted here are the ones the manifest already commits to.
PHRASE_SHOTS: dict[str, str] = {
    "revenue": "revenue",
    "pricing": "pricing",
    "competitor": "competitor",
    "pytest": "terminal",
}

#: Speech runs a little inside its shot (the narration is shorter than the slide), and a
#: model's segment boundaries drift by a beat either way. Wide enough to absorb that, narrow
#: enough that a whole-video utterance still fails.
BOUNDARY_SLACK_S = 2.0


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(DEMO_MANIFEST.read_text())


@pytest.fixture(scope="module")
def transcript(tmp_path_factory):
    """Run the real ASR stage once: probe -> extract audio -> transcribe."""
    info = probe(DEMO_VIDEO, compute_hash=False)
    ctx = FrameContext(
        duration=info.duration, fps=info.fps, width=info.width, height=info.height
    )
    # The WAV goes to a pytest temp dir so the test neither reads a stale cache nor leaves
    # extracted audio behind (brief §32).
    wav = extract_audio(DEMO_VIDEO, tmp_path_factory.mktemp("audio") / "demo.wav")
    assert wav is not None, "fixture is supposed to have an audio stream"

    engine, warnings = resolve_engine(ASRConfig(), source=DEMO_VIDEO)
    assert engine.name == "faster-whisper", f"expected Whisper, got {engine.name}: {warnings}"
    return engine.transcribe(wav, ctx), ctx


def shot(manifest: dict, name: str) -> dict:
    for entry in manifest["shots"]:
        if entry["shot"] == name:
            return entry
    raise AssertionError(f"no shot named {name!r} in the manifest")


def contains(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


class TestAcceptancePhrasesAreLocated:
    def test_every_spoken_phrase_is_transcribed(self, transcript, manifest):
        out, _ = transcript
        joined = " ".join(u.text for u in out.utterances).lower()
        missing = [p for p in manifest["expected"]["spoken_phrases"] if p not in joined]
        assert not missing, f"not transcribed: {missing}"

    @pytest.mark.parametrize(("phrase", "shot_name"), sorted(PHRASE_SHOTS.items()))
    def test_phrase_is_timestamped_inside_its_shot(
        self, transcript, manifest, phrase, shot_name
    ):
        """The assertion a whole-video utterance cannot pass: the midpoint must land right."""
        out, _ = transcript
        window = shot(manifest, shot_name)
        hits = [u for u in out.utterances if contains(u.text, phrase)]
        assert hits, f"{phrase!r} was not transcribed at all"

        midpoints = [(u.start + u.end) / 2 for u in hits]
        assert any(window["start"] <= mid <= window["end"] for mid in midpoints), (
            f"{phrase!r} was transcribed but no utterance is centred in shot "
            f"{shot_name!r} ({window['start']}-{window['end']}); midpoints={midpoints}"
        )

    @pytest.mark.parametrize(("phrase", "shot_name"), sorted(PHRASE_SHOTS.items()))
    def test_phrase_does_not_leak_far_past_its_shot(
        self, transcript, manifest, phrase, shot_name
    ):
        out, _ = transcript
        window = shot(manifest, shot_name)
        best = min(
            (u for u in out.utterances if contains(u.text, phrase)),
            key=lambda u: abs((u.start + u.end) / 2 - (window["start"] + window["end"]) / 2),
        )
        assert best.start >= window["start"] - BOUNDARY_SLACK_S
        assert best.end <= window["end"] + BOUNDARY_SLACK_S


class TestSilenceStaysSilent:
    """Hallucinated filler over silence is the classic Whisper failure; it must not appear."""

    def test_the_silent_shot_has_no_utterances(self, transcript, manifest):
        out, _ = transcript
        window = shot(manifest, manifest["expected"]["silent_shot"])
        inside = [
            u
            for u in out.utterances
            if u.start >= window["start"] and u.end <= window["end"]
        ]
        assert inside == [], f"speech invented over silence: {[u.text for u in inside]}"

    def test_no_utterance_covers_a_shot_that_has_no_audio(self, transcript, manifest):
        out, _ = transcript
        silent = [s for s in manifest["shots"] if not s["has_audio"]]
        assert silent, "manifest is supposed to contain silent shots"
        for window in silent:
            covering = [
                u.text
                for u in out.utterances
                if u.start <= window["start"] and u.end >= window["end"]
            ]
            assert covering == [], f"{window['shot']!r} is silent but covered by {covering}"


class TestEvidenceInvariants:
    def test_one_utterance_per_narrated_shot(self, transcript, manifest):
        # The fixture narrates 8 of its 10 shots with one sentence group each. This is the
        # single strongest check that segmentation tracked the audio rather than guessing.
        out, _ = transcript
        assert len(out.utterances) == manifest["narrated_shots"]

    def test_spans_are_ordered_and_well_formed(self, transcript):
        out, _ = transcript
        previous = -1.0
        for utt in out.utterances:
            assert utt.end >= utt.start
            assert utt.start >= previous, "utterances are not in timeline order"
            previous = utt.start

    def test_no_span_exceeds_the_media_duration(self, transcript):
        out, ctx = transcript
        for utt in out.utterances:
            assert 0.0 <= utt.start <= ctx.duration
            assert utt.end <= ctx.duration

    def test_ids_are_sequential_and_spec_shaped(self, transcript):
        out, _ = transcript
        assert [u.id for u in out.utterances] == [
            f"utt_{i:04d}" for i in range(len(out.utterances))
        ]

    def test_no_utterance_is_blank(self, transcript):
        out, _ = transcript
        assert all(u.text.strip() for u in out.utterances)

    def test_no_utterance_spans_most_of_the_video(self, transcript, manifest):
        # Explicitly rejects the degenerate "one span for everything" result that would let a
        # broken stage pass a naive keyword test.
        out, _ = transcript
        widest = max(u.end - u.start for u in out.utterances)
        assert widest < manifest["duration"] / 2

    def test_language_was_detected(self, transcript):
        out, _ = transcript
        assert out.language == "en"

    def test_the_model_is_named_in_the_output(self, transcript):
        # Provenance: a document has to say which engine produced its transcript.
        out, _ = transcript
        assert out.model == "faster-whisper:base"

    def test_no_warnings_on_a_video_that_does_contain_speech(self, transcript):
        out, _ = transcript
        assert out.warnings == []


class TestWordTimings:
    def test_words_were_produced(self, transcript):
        out, _ = transcript
        assert sum(len(u.words) for u in out.utterances) > 40

    def test_every_word_lies_inside_its_utterance(self, transcript):
        out, _ = transcript
        for utt in out.utterances:
            for w in utt.words:
                assert utt.start <= w.start <= w.end <= utt.end, f"{w.text!r} escapes {utt.id}"

    def test_words_are_in_order_within_an_utterance(self, transcript):
        out, _ = transcript
        for utt in out.utterances:
            starts = [w.start for w in utt.words]
            assert starts == sorted(starts)

    def test_word_text_has_no_leading_space(self, transcript):
        # faster-whisper prefixes each word with a space; carrying that into the document
        # would break exact-match lookup of a single word.
        out, _ = transcript
        assert all(w.text == w.text.strip() for u in out.utterances for w in u.words)

    def test_word_confidence_is_a_fraction(self, transcript):
        out, _ = transcript
        reported = [w.confidence for u in out.utterances for w in u.words if w.confidence]
        assert reported, "word probabilities were requested but none came through"
        assert all(0.0 <= c <= 1.0 for c in reported)


class TestConfidenceIsWindowLevel:
    """Pins a faster-whisper behaviour that otherwise looks like a bug in this code."""

    def test_confidence_is_reported(self, transcript):
        out, _ = transcript
        assert all(u.confidence is not None for u in out.utterances)
        assert all(0.0 <= u.confidence <= 1.0 for u in out.utterances)

    def test_neighbours_may_share_a_confidence(self, transcript):
        # Whisper decodes 30-second windows and copies the window's avg_logprob onto every
        # segment from it, so there are far fewer distinct values than utterances. Asserting
        # that keeps a future reader from "fixing" the duplication.
        out, _ = transcript
        distinct = {u.confidence for u in out.utterances}
        assert len(distinct) < len(out.utterances)


class TestManifestAgreement:
    def test_every_manifest_phrase_has_a_shot_mapping(self, manifest):
        """Fails if a phrase is added to the manifest without being located here."""
        unmapped = set(manifest["expected"]["spoken_phrases"]) - set(PHRASE_SHOTS)
        assert not unmapped, f"add these to PHRASE_SHOTS: {sorted(unmapped)}"

    def test_mapped_shots_exist_in_the_manifest(self, manifest):
        names = {s["shot"] for s in manifest["shots"]}
        assert set(PHRASE_SHOTS.values()) <= names

    def test_each_mapped_shot_narration_really_contains_its_phrase(self, manifest):
        # Ground truth check on the mapping itself: if this fails, the test is asserting the
        # wrong window and would pass or fail for the wrong reason.
        for phrase, shot_name in PHRASE_SHOTS.items():
            narration = shot(manifest, shot_name)["narration"]
            assert contains(narration, phrase), f"{shot_name!r} does not narrate {phrase!r}"
