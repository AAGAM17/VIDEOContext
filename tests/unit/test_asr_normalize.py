"""Utterance normalisation: the clamps, drops and orderings every engine shares.

The contract under test is narrow on purpose. :func:`finalize` may reject, clamp and reorder
what an engine produced; it may not invent. Each test below pins one half of that: either a
malformation that must be repaired, or a piece of engine output that must survive untouched.
"""

from __future__ import annotations

from videocontext.processing.asr import finalize, utterance
from videocontext.schema.v1 import Word


def word(text: str, start: float, end: float, conf: float | None = None) -> Word:
    return Word(text=text, start=start, end=end, confidence=conf)


class TestUtteranceBuilder:
    def test_id_is_left_for_finalize(self):
        assert utterance("hello", 1.0, 2.0).id == ""

    def test_negative_start_is_clamped(self):
        # Whisper's alignment pass can report a slightly negative first timestamp.
        assert utterance("hello", -0.4, 2.0).start == 0.0

    def test_degenerate_instant_is_kept_not_dropped(self):
        # The text is still evidence that something was said at that moment.
        utt = utterance("hi", 5.0, 5.0)
        assert (utt.start, utt.end) == (5.0, 5.0)

    def test_inverted_span_is_not_silently_reversed_here(self):
        # The builder only satisfies the schema's ordering rule; finalize records the swap.
        utt = utterance("hi", 9.0, 4.0)
        assert utt.start == 9.0 and utt.end == 9.0

    def test_unreported_fields_stay_unreported(self):
        utt = utterance("hi", 1.0, 2.0)
        assert utt.confidence is None
        assert utt.speaker is None
        assert utt.no_speech_prob is None
        assert utt.words == []


class TestFinalizeOrderingAndIds:
    def test_ids_follow_the_timeline_not_the_input_order(self):
        out = finalize([utterance("second", 5.0, 6.0), utterance("first", 1.0, 2.0)])
        assert [u.id for u in out] == ["utt_0000", "utt_0001"]
        assert [u.text for u in out] == ["first", "second"]

    def test_id_format_matches_the_spec(self):
        out = finalize([utterance("x", float(i), float(i) + 0.5) for i in range(11)])
        assert out[10].id == "utt_0010"

    def test_ties_are_broken_by_end(self):
        out = finalize([utterance("long", 1.0, 9.0), utterance("short", 1.0, 2.0)])
        assert [u.text for u in out] == ["short", "long"]


class TestFinalizeRepairs:
    def test_blank_text_is_dropped(self):
        out = finalize([utterance("   ", 1.0, 2.0), utterance("real", 3.0, 4.0)])
        assert [u.text for u in out] == ["real"]

    def test_internal_whitespace_is_collapsed(self):
        out = finalize([utterance("  we   grew\n revenue ", 1.0, 2.0)])
        assert out[0].text == "we grew revenue"

    def test_inverted_span_is_swapped(self):
        # Built directly rather than via utterance(), which would have flattened it.
        utt = utterance("hi", 4.0, 9.0).model_copy(update={"start": 9.0, "end": 4.0})
        out = finalize([utt])
        assert (out[0].start, out[0].end) == (4.0, 9.0)

    def test_span_past_the_end_is_clamped_to_duration(self):
        out = finalize([utterance("tail", 58.0, 71.0)], duration=62.4)
        assert out[0].end == 62.4

    def test_unknown_duration_does_not_clamp(self):
        # Clamping to a duration we do not know would be the invention this module forbids.
        out = finalize([utterance("tail", 58.0, 71.0)], duration=0.0)
        assert out[0].end == 71.0

    def test_confidence_is_kept_on_scale(self):
        out = finalize([utterance("a", 0.0, 1.0, confidence=1.4)])
        assert out[0].confidence == 1.0

    def test_missing_confidence_stays_none(self):
        assert finalize([utterance("a", 0.0, 1.0)])[0].confidence is None

    def test_language_fills_only_where_absent(self):
        out = finalize(
            [utterance("a", 0.0, 1.0), utterance("b", 2.0, 3.0, language="hi")],
            language="en",
        )
        assert [u.language for u in out] == ["en", "hi"]


class TestFinalizeWords:
    def test_words_are_clamped_into_their_utterance(self):
        # Whisper aligns words in a separate pass, so a word can land outside its segment.
        out = finalize([utterance("hi", 10.0, 12.0, words=[word("hi", 9.7, 12.3)])])
        w = out[0].words[0]
        assert (w.start, w.end) == (10.0, 12.0)

    def test_word_order_follows_time(self):
        out = finalize(
            [utterance("a b", 0.0, 4.0, words=[word("b", 2.0, 3.0), word("a", 0.5, 1.0)])]
        )
        assert [w.text for w in out[0].words] == ["a", "b"]

    def test_blank_words_are_dropped_and_text_is_stripped(self):
        out = finalize(
            [utterance("hi", 0.0, 4.0, words=[word(" ", 1.0, 1.1), word(" hi ", 2.0, 2.5)])]
        )
        assert [w.text for w in out[0].words] == ["hi"]

    def test_word_probability_survives(self):
        out = finalize([utterance("hi", 0.0, 4.0, words=[word("hi", 1.0, 2.0, 0.87)])])
        assert out[0].words[0].confidence == 0.87


class TestFinalizeIsIdempotent:
    """The pipeline calls this defensively on third-party output, so twice must equal once."""

    def _messy(self):
        return [
            utterance("  second  ", 5.0, 71.0, confidence=1.9),
            utterance("", 2.0, 3.0),
            utterance("first", 1.0, 2.0, words=[word("first", 0.2, 9.0, 0.5)]),
        ]

    def test_second_pass_changes_nothing(self):
        once = finalize(self._messy(), duration=62.4, language="en")
        twice = finalize(once, duration=62.4, language="en")
        assert [u.model_dump() for u in once] == [u.model_dump() for u in twice]

    def test_it_did_do_something_the_first_time(self):
        # Guards against the idempotence test above passing because finalize is a no-op.
        once = finalize(self._messy(), duration=62.4, language="en")
        assert len(once) == 2
        assert once[0].id == "utt_0000"
        assert once[1].end == 62.4
        assert once[1].confidence == 1.0

    def test_empty_input(self):
        assert finalize([]) == []
