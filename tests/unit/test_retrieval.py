"""Search over a document: is every answer traceable, and does the temporal signal work?

The assertion this file exists for is the first one in ``TestEvidence``: **a returned span's
timestamps are always the timestamps of facts named in its own ``ref_ids``.** If that ever
fails, the product's central claim fails with it — a timestamp nobody observed is exactly the
hallucination the evidence-first design is built to make impossible, and it would be
indistinguishable from a correct answer to anyone reading the output.

The rest covers the parts of retrieval that are specific to *video* rather than to text:
temporal co-occurrence across modalities, merging evidence that is one moment split into
several facts, and the point-in-time lookup that "what was on screen at 03:21" actually needs.

No media and no models: a hand-built document, so a ranking change is a ranking failure rather
than an OCR one.
"""

from __future__ import annotations

import json

import pytest

from videocontent.config import RetrievalConfig
from videocontent.retrieval import Retriever, at, build_records, search
from videocontent.retrieval.fusion import Candidate, boost_cooccurrence, merge_adjacent, rrf
from videocontent.retrieval.index import Record
from videocontent.schema.v1 import (
    Event,
    OCRText,
    Segment,
    Utterance,
    VideoContextDocument,
    VideoInfo,
    VisionNote,
)
from videocontent.timecode import format_timecode

DURATION = 60.0


def utt(index: int, text: str, start: float, end: float, **kw) -> Utterance:
    return Utterance(id=f"utt_{index:04d}", text=text, start=start, end=end, **kw)


def ocr(index: int, text: str, start: float, end: float, **kw) -> OCRText:
    return OCRText(id=f"ocr_{index:04d}", text=text, start=start, end=end, frame_count=3, **kw)


def event(index: int, type_: str, start: float, end: float, **kw) -> Event:
    return Event(id=f"evt_{index:04d}", type=type_, start=start, end=end, **kw)


def doc(**kw) -> VideoContextDocument:
    """A document with one segment per 30 s, so every fact has a containing segment."""
    facts = {
        "transcript": list(kw.get("transcript", ())),
        "ocr": list(kw.get("ocr", ())),
        "vision": list(kw.get("vision", ())),
        "events": list(kw.get("events", ())),
    }
    segments = [
        Segment(
            id=f"segment_{i:04d}",
            start=float(i * 30),
            end=float((i + 1) * 30),
            transcript_ids=[f.id for f in facts["transcript"] if i * 30 <= f.start < (i + 1) * 30],
            ocr_ids=[f.id for f in facts["ocr"] if i * 30 <= f.start < (i + 1) * 30],
            vision_ids=[f.id for f in facts["vision"] if i * 30 <= f.start < (i + 1) * 30],
            event_ids=[f.id for f in facts["events"] if i * 30 <= f.start < (i + 1) * 30],
        )
        for i in range(2)
    ]
    return VideoContextDocument(
        id="vid_test",
        video=VideoInfo(id="vid_test", filename="demo.mp4", duration=DURATION, has_audio=True),
        segments=kw.get("segments", segments),
        **facts,
    )


def sample() -> VideoContextDocument:
    """Speech and on-screen text that agree at 10 s, and a URL with no speech around it."""
    return doc(
        transcript=[
            utt(0, "let me show you the pricing page for our plans", 10.0, 14.0, language="en"),
            utt(1, "revenue grew forty percent last quarter", 30.0, 34.0, language="en"),
        ],
        ocr=[
            ocr(0, "Pricing", 9.5, 20.0, language="en", stable=True),
            ocr(1, "localhost:3000/login", 40.0, 45.0),
        ],
        events=[
            event(
                1,
                "url_shown",
                40.0,
                40.0,
                description="localhost:3000/login",
                refs={"ocr": ["ocr_0001"]},
            )
        ],
    )


def fact_index(document: VideoContextDocument) -> dict[str, object]:
    facts: dict[str, object] = {}
    for group in (document.transcript, document.ocr, document.vision, document.events):
        for item in group:
            facts[item.id] = item
    return facts


class TestEvidence:
    """The central invariant: nothing is returned that the document does not contain."""

    @pytest.mark.parametrize("query", ["pricing", "revenue", "localhost", "login page", "plans"])
    def test_a_span_timestamps_are_its_own_references(self, query):
        document = sample()
        facts = fact_index(document)
        for span in search(document, query):
            sources = [facts[ref] for ref in span.ref_ids]
            assert sources, f"{span.text!r} was returned citing nothing"
            assert span.start == pytest.approx(min(f.start for f in sources))
            assert span.end == pytest.approx(max(f.end for f in sources))

    def test_span_text_comes_from_the_referenced_facts(self):
        document = sample()
        facts = fact_index(document)
        for span in search(document, "pricing"):
            for ref in span.ref_ids:
                source = facts[ref]
                shown = getattr(source, "text", None) or getattr(source, "description", "")
                assert shown.strip() in span.text

    def test_every_span_names_the_segment_it_came_from(self):
        # The bridge from a precise hit to enough context to answer with: a caller that wants
        # the surrounding 30 seconds must not have to search for it by timestamp.
        for span in search(sample(), "pricing"):
            assert span.segment_ids

    def test_a_query_matching_nothing_returns_nothing(self):
        # Not "the closest thing available". An empty result is a correct answer, and inventing
        # a weak one is how an evidence-first system starts producing wrong timestamps.
        result = search(sample(), "kubernetes")
        assert not result
        assert result.spans == () and result.total == 0

    def test_matched_terms_explain_the_hit(self):
        span = search(sample(), "pricing")[0]
        assert "pricing" in span.matched_terms
        assert span.reason


class TestFilters:
    def test_modalities_narrow_the_search(self):
        result = search(sample(), "pricing", modalities=["ocr"])
        assert {span.modality for span in result} == {"ocr"}

    def test_a_modality_absent_from_the_document_is_not_reported_as_searched(self):
        # `vision` is in the default config but this document has none. Claiming to have
        # searched it would tell a user their query found no vision matches, when the truth is
        # that no vision provider ever ran.
        assert "vision" not in search(sample(), "pricing").modalities

    def test_a_time_window_excludes_facts_outside_it(self):
        result = search(sample(), "pricing", start=25.0)
        assert result.spans == ()

    def test_a_time_window_keeps_a_fact_that_straddles_the_boundary(self):
        # The OCR event runs 9.5-20.0; a window starting at 15 must still see it, or on-screen
        # text that was present throughout would vanish from a windowed search.
        result = search(sample(), "pricing", start=15.0, end=25.0)
        assert "ocr_0000" in {ref for span in result for ref in span.ref_ids}

    def test_top_k_truncates_but_total_reports_the_whole(self):
        result = search(sample(), "pricing", top_k=1)
        assert len(result.spans) == 1
        assert result.total == 2

    def test_top_k_zero_returns_everything(self):
        # "Find every occurrence" needs a way to say so.
        assert len(search(sample(), "pricing", top_k=0).spans) == 2

    def test_min_score_filters(self):
        assert search(sample(), "pricing", min_score=99.0).spans == ()

    def test_config_defaults_are_used_when_no_argument_is_given(self):
        config = RetrievalConfig(top_k=1, modalities=["transcript"])
        result = Retriever(sample(), config).search("pricing")
        assert len(result.spans) == 1
        assert result.spans[0].modality == "transcript"


class TestCooccurrence:
    def test_agreement_across_modalities_raises_the_score(self):
        document = sample()
        plain = Retriever(document, RetrievalConfig(cooccurrence_boost=0.0)).search("pricing")
        boosted = Retriever(document, RetrievalConfig(cooccurrence_boost=0.5)).search("pricing")
        assert boosted[0].score > plain[0].score

    def test_the_boost_is_stated_in_the_reason(self):
        # A score that moved for an unexplained reason is not debuggable by the person whose
        # query went wrong.
        span = search(sample(), "pricing")[0]
        assert "also matched in" in span.reason

    def test_a_lone_match_is_not_boosted(self):
        document = sample()
        plain = Retriever(document, RetrievalConfig(cooccurrence_boost=0.0)).search("revenue")
        boosted = Retriever(document, RetrievalConfig(cooccurrence_boost=0.5)).search("revenue")
        assert boosted[0].score == pytest.approx(plain[0].score)
        assert "also matched in" not in boosted[0].reason

    def test_distant_matches_in_two_modalities_do_not_corroborate(self):
        # Same word, 30 seconds apart: two separate occurrences, not one corroborated moment.
        document = doc(
            transcript=[utt(0, "the login screen", 5.0, 7.0)],
            ocr=[ocr(0, "Login", 45.0, 50.0)],
        )
        for span in search(document, "login"):
            assert "also matched in" not in span.reason


class TestMerging:
    def test_adjacent_matches_in_one_modality_become_one_span(self):
        # One sentence split across three cues is one piece of evidence. Left unmerged it takes
        # three of ten result slots and pushes the corroborating on-screen text off the page.
        document = doc(
            transcript=[
                utt(0, "our pricing", 10.0, 11.0),
                utt(1, "pricing is simple", 11.5, 13.0),
                utt(2, "pricing again", 14.0, 15.0),
            ]
        )
        result = search(document, "pricing")
        assert len(result.spans) == 1
        assert result.spans[0].ref_ids == ("utt_0000", "utt_0001", "utt_0002")
        assert result.spans[0].start == 10.0
        assert result.spans[0].end == 15.0

    def test_a_gap_wider_than_the_setting_stays_separate(self):
        document = doc(
            transcript=[utt(0, "pricing", 10.0, 11.0), utt(1, "pricing", 30.0, 31.0)]
        )
        assert len(search(document, "pricing").spans) == 2

    def test_merging_never_crosses_modalities(self):
        # Merging speech into on-screen text would produce a span whose modality is a fiction.
        document = doc(
            transcript=[utt(0, "pricing", 10.0, 11.0)],
            ocr=[ocr(0, "Pricing", 11.2, 12.0)],
        )
        result = search(document, "pricing")
        assert sorted(span.modality for span in result) == ["ocr", "transcript"]

    def test_a_merged_span_keeps_the_lowest_confidence(self):
        # A merged span is only as trustworthy as its worst source; averaging would let a
        # confident fragment vouch for a doubtful one.
        document = doc(
            transcript=[
                utt(0, "pricing", 10.0, 11.0, confidence=0.9),
                utt(1, "pricing", 11.5, 12.0, confidence=0.3),
            ]
        )
        assert search(document, "pricing")[0].confidence == pytest.approx(0.3)


class TestRecords:
    def test_records_carry_the_segments_that_contain_them(self):
        records = build_records(sample())
        assert all(record.segment_ids for record in records)

    def test_an_event_type_is_searchable_without_being_displayed(self):
        # "what command was typed" has to reach a terminal_command event, but the evidence shown
        # to a user should be the command, not the taxonomy name.
        document = doc(
            events=[
                event(0, "terminal_command", 20.0, 20.0, description="npm run build",
                      refs={"ocr": ["ocr_0000"]})
            ]
        )
        span = search(document, "terminal command")[0]
        assert span.text == "npm run build"
        assert span.kind == "terminal_command"

    def test_an_event_with_no_description_still_has_readable_text(self):
        document = doc(events=[event(0, "scene_changed", 30.0, 30.0, refs={"scenes": ["s"]})])
        assert search(document, "scene changed")[0].text == "scene changed"

    def test_vision_entities_are_searchable_and_the_description_is_shown(self):
        document = doc(
            vision=[
                VisionNote(
                    id="vis_0000",
                    start=5.0,
                    end=10.0,
                    description="a slide with a bar chart",
                    entities=["chart", "projector"],
                )
            ]
        )
        span = search(document, "projector")[0]
        assert span.text == "a slide with a bar chart"

    def test_an_empty_utterance_is_not_indexed(self):
        # Whisper emits blank segments on silence; they are not searchable facts.
        assert build_records(doc(transcript=[utt(0, "   ", 1.0, 2.0)])) == []

    def test_records_are_in_timeline_order(self):
        records = build_records(sample())
        assert [r.start for r in records] == sorted(r.start for r in records)

    def test_an_unknown_modality_name_is_ignored_not_raised(self):
        # A config written against a newer format should degrade to searching what this version
        # understands rather than failing to search at all.
        records = build_records(sample(), ["transcript", "objects"])
        assert {r.modality for r in records} == {"transcript"}


class TestAt:
    def test_it_returns_everything_covering_the_instant(self):
        result = at(sample(), 10.5)
        assert {ref for span in result for ref in span.ref_ids} == {"utt_0000", "ocr_0000"}

    def test_it_excludes_what_does_not_cover_the_instant(self):
        assert at(sample(), 25.0).spans == ()

    def test_it_orders_speech_before_screen_before_events(self):
        # The output reads as a snapshot of the moment, so it has a fixed reading order rather
        # than a score.
        result = at(sample(), 40.0)
        assert [span.modality for span in result] == ["ocr", "events"]

    def test_a_window_widens_the_lookup(self):
        assert at(sample(), 22.0, window=3.0).spans != ()

    def test_the_query_is_the_timecode(self):
        assert at(sample(), 201.45).query == "00:03:21.450"

    def test_an_instant_event_is_found(self):
        # Zero-length by design; a strict `start < ts < end` test would never return one.
        assert at(sample(), 40.0, modalities=["events"]).spans != ()


class TestResult:
    def test_iterating_a_result_yields_spans(self):
        result = search(sample(), "pricing")
        assert list(result) == list(result.spans)
        assert len(result) == len(result.spans)
        assert result[0] is result.spans[0]

    def test_a_result_is_falsey_when_empty(self):
        assert not search(sample(), "kubernetes")
        assert search(sample(), "pricing")

    def test_it_serializes_to_json(self):
        # The API and the CLI both emit this; a tuple or a dataclass leaking through would fail
        # at the surface rather than here.
        payload = json.loads(json.dumps(search(sample(), "pricing").to_dict()))
        assert payload["query"] == "pricing"
        assert {ref for span in payload["spans"] for ref in span["ref_ids"]} == {
            "utt_0000",
            "ocr_0000",
        }
        for span in payload["spans"]:
            assert span["timecode"] == format_timecode(span["start"])

    def test_it_reports_the_absent_vector_retriever(self):
        # "No semantic matches" and "no semantic retriever" are different answers to the same
        # empty result, and the user needs to know which one they got.
        result = search(sample(), "pricing")
        assert any("lexical" in note for note in result.notes)

    def test_no_note_when_vectors_were_not_asked_for(self):
        config = RetrievalConfig(vector_weight=0.0)
        assert Retriever(sample(), config).search("pricing").notes == ()

    def test_it_reports_how_long_it_took(self):
        assert search(sample(), "pricing").took_ms > 0.0


class TestFusionUnits:
    def rec(self, key: str, modality: str, start: float, end: float) -> Record:
        return Record(
            key=f"{modality}:{key}",
            id=key,
            modality=modality,
            start=start,
            end=end,
            text=key,
            tokens=(key,),
        )

    def test_rrf_ranks_by_position_not_by_score(self):
        fused = rrf([(1.0, ["a", "b", "c"])])
        assert fused["a"] > fused["b"] > fused["c"]

    def test_appearing_in_two_lists_beats_appearing_in_one(self):
        fused = rrf([(1.0, ["a", "b"]), (1.0, ["b"])])
        assert fused["b"] > fused["a"]

    def test_absence_from_a_list_is_not_a_last_place_penalty(self):
        # A record a retriever never saw is not one it ranked last; penalising absence would
        # punish every match that is specific to one modality.
        one = rrf([(1.0, ["a"])])
        two = rrf([(1.0, ["a"]), (1.0, ["z"])])
        assert one["a"] == pytest.approx(two["a"])

    def test_a_zero_weighted_list_does_not_contribute(self):
        assert rrf([(0.0, ["a"])]) == {}

    def test_merging_is_stable_when_nothing_is_adjacent(self):
        candidates = [
            Candidate(records=(self.rec("a", "ocr", 0.0, 1.0),), score=1.0, matched=("x",)),
            Candidate(records=(self.rec("b", "ocr", 50.0, 51.0),), score=0.5, matched=("x",)),
        ]
        assert len(merge_adjacent(candidates, gap=2.0)) == 2

    def test_a_negative_gap_disables_merging(self):
        candidates = [
            Candidate(records=(self.rec("a", "ocr", 0.0, 1.0),), score=1.0, matched=("x",)),
            Candidate(records=(self.rec("b", "ocr", 1.1, 2.0),), score=0.5, matched=("x",)),
        ]
        assert len(merge_adjacent(candidates, gap=-1.0)) == 2

    def test_the_boost_is_flat_not_scaled_by_how_many_agree(self):
        # Rule-derived events and vision notes overlap speech as a matter of course; a
        # multiplier would let that structural overlap outrank a genuine two-way match.
        speech = Candidate(
            records=(self.rec("u", "transcript", 10.0, 12.0),), score=1.0, matched=("x",)
        )
        others = [
            Candidate(records=(self.rec(m, m, 10.0, 12.0),), score=1.0, matched=("x",))
            for m in ("ocr", "vision", "events")
        ]
        boosted = boost_cooccurrence([speech, *others], boost=0.25)
        assert boosted[0].score == pytest.approx(1.25)

    def test_a_single_candidate_is_never_boosted(self):
        lone = Candidate(records=(self.rec("u", "transcript", 0.0, 1.0),), score=1.0, matched=())
        assert boost_cooccurrence([lone], boost=0.5)[0].score == 1.0
