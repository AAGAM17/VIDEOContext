"""Lexical scoring: does the ranking behave the way the queries people ask require?

Three of the assertions below correspond to specific, deliberate departures from textbook BM25,
and each of them guards a failure that is invisible until someone runs a real query:

* **negative IDF.** With the unguarded formula, a term in more than half the records gets a
  negative weight and a record is *penalised* for containing a word the user asked for. In a
  corpus built from one video, plenty of terms cross that line.
* **cross-modality length bias.** One pooled average document length makes every 20-word
  utterance "long" and every 2-word caption "short", and on-screen text then wins every query
  regardless of relevance.
* **coverage before BM25.** In a forty-record corpus, term statistics are noisy enough that
  BM25 alone will rank a record matching one word of a five-word query above one matching all
  five. Weighting coverage by IDF is what makes ``find every occurrence of the word competitor``
  find the rare word rather than the common ones.

No fixtures, no media: this is arithmetic over strings.
"""

from __future__ import annotations

import pytest

from videocontent.retrieval.index import Record
from videocontent.retrieval.lexical import LexicalIndex, tokenize


def record(key: str, text: str, modality: str = "transcript", start: float = 0.0) -> Record:
    return Record(
        key=f"{modality}:{key}",
        id=key,
        modality=modality,
        start=start,
        end=start + 1.0,
        text=text,
        tokens=tuple(tokenize(text)),
    )


def index(*records: Record) -> LexicalIndex:
    return LexicalIndex(list(records))


def ranked(hits) -> list[str]:
    return [hit.record.id for hit in hits]


class TestTokenize:
    def test_it_casefolds(self):
        assert tokenize("Pricing PAGE") == ["pricing", "page"]

    def test_punctuation_separates(self):
        # The point of tokenizing both sides identically: the query "localhost:3000/login" and
        # the OCR string "localhost:3000/login" must produce the same tokens, and a query typed
        # without the punctuation must still match.
        assert tokenize("localhost:3000/login") == ["localhost", "3000", "login"]

    def test_underscores_separate_identifiers(self):
        assert tokenize("terminal_command") == ["terminal", "command"]

    def test_accented_text_survives(self):
        # A `[a-z0-9]+` pattern would silently shatter this into fragments.
        assert tokenize("Café Münster") == ["café", "münster"]

    def test_digits_are_kept(self):
        assert tokenize("Q4 2026 revenue") == ["q4", "2026", "revenue"]

    def test_empty_text_gives_no_tokens(self):
        assert tokenize("   \n\t ") == []


class TestIdf:
    def test_a_term_in_every_record_still_scores_positive(self):
        # The regression: the unguarded RSJ formula returns a negative weight for df > N/2, and
        # a matching record then ranks below a non-matching one.
        idx = index(*(record(f"r{i}", "the pricing page") for i in range(5)))
        assert idx.idf("pricing") > 0.0

    def test_a_rare_term_outweighs_a_common_one(self):
        idx = index(
            record("r0", "the pricing page"),
            record("r1", "the revenue chart"),
            record("r2", "the login screen"),
            record("r3", "the competitor slide"),
        )
        assert idx.idf("competitor") > idx.idf("the")

    def test_an_unseen_term_has_the_largest_weight(self):
        idx = index(record("r0", "pricing"))
        assert idx.idf("absent") > idx.idf("pricing")


class TestLengthNormalization:
    def test_a_short_caption_does_not_beat_a_relevant_utterance(self):
        # The bias this guards: pooled avgdl. The caption is 1 token, the utterances ~9, so a
        # single average makes the caption look maximally concentrated on any term it contains
        # and it wins every query. Per-modality statistics make it compete with captions only.
        idx = index(
            record("u0", "let me walk you through the pricing for each of our plans"),
            record("u1", "and here is the revenue for the last four quarters in review"),
            record("c0", "Pricing", modality="ocr"),
            record("c1", "Revenue", modality="ocr"),
        )
        assert ranked(idx.search("pricing"))[0] == "u0"

    def test_records_with_no_tokens_do_not_divide_by_zero(self):
        # A vision note with an empty description reaches the index as a zero-length record.
        idx = index(record("v0", "", modality="vision"), record("v1", "chart", modality="vision"))
        assert ranked(idx.search("chart")) == ["v1"]


class TestCoverage:
    def test_matching_more_of_the_query_ranks_higher(self):
        idx = index(
            record("r0", "the pricing page for our plans"),
            record("r1", "the weather outside is unrelated"),
        )
        hits = idx.search("pricing page weather")
        assert ranked(hits) == ["r0", "r1"]
        assert hits[0].coverage > hits[1].coverage

    def test_matching_the_rarer_term_wins_at_equal_match_counts(self):
        # Both records match exactly one query term. "the" is in every record, so IDF has
        # learned it carries almost no information; "competitor" appears once. Unweighted
        # coverage would call these two hits equal and let the timestamp decide.
        common = [record(f"c{i}", f"the item number {i}") for i in range(8)]
        idx = index(*common, record("rare", "competitor pricing"))
        hits = idx.search("the competitor")
        assert hits[0].record.id == "rare"
        assert hits[0].coverage > 0.9
        assert hits[1].coverage < 0.1

    def test_a_word_that_recurs_everywhere_is_discounted(self):
        idx = index(*(record(f"r{i}", f"the plan number {i}") for i in range(8)))
        assert idx.idf("the") < 0.1

    def test_coverage_is_a_fraction(self):
        idx = index(record("r0", "pricing"))
        hit = idx.search("pricing revenue login")[0]
        assert 0.0 < hit.coverage < 1.0

    def test_a_fully_matched_query_reaches_full_coverage(self):
        idx = index(record("r0", "pricing page"), record("r1", "something else entirely"))
        assert idx.search("pricing page")[0].coverage == pytest.approx(1.0)


class TestPhrase:
    def test_a_contiguous_phrase_outranks_scattered_words(self):
        idx = index(
            record("r0", "the pricing page is here"),
            record("r1", "page layout, and separately, pricing"),
        )
        hits = idx.search("pricing page")
        assert ranked(hits) == ["r0", "r1"]
        assert hits[0].phrase and not hits[1].phrase

    def test_word_order_matters(self):
        idx = index(record("r0", "page pricing"))
        assert not idx.search("pricing page")[0].phrase

    def test_a_single_word_query_is_not_a_phrase_match(self):
        # Otherwise every hit is a "phrase" hit and the tier carries no information.
        idx = index(record("r0", "pricing"))
        assert not idx.search("pricing")[0].phrase

    def test_a_phrase_longer_than_the_record_does_not_match(self):
        idx = index(record("r0", "pricing"))
        assert not idx.search("pricing page details")[0].phrase


class TestResults:
    def test_every_match_is_returned(self):
        # "Find every occurrence of X" is a real query with a required answer of all of them.
        # Truncation belongs to the caller, after fusion — a retriever that has already dropped
        # the tail cannot serve it.
        idx = index(*(record(f"r{i}", "pricing", start=float(i)) for i in range(25)))
        assert len(idx.search("pricing")) == 25

    def test_a_record_matching_nothing_is_absent(self):
        # Not "present with score zero": a record that matched no query term is not evidence,
        # and including it would let fusion promote it to rank 1 of its modality.
        idx = index(record("r0", "pricing"), record("r1", "entirely unrelated"))
        assert ranked(idx.search("pricing")) == ["r0"]

    def test_matched_terms_are_reported(self):
        idx = index(record("r0", "the pricing page"))
        assert set(idx.search("pricing revenue")[0].matched) == {"pricing"}

    def test_an_empty_query_finds_nothing(self):
        assert index(record("r0", "pricing")).search("   ") == []

    def test_an_empty_index_finds_nothing(self):
        assert index().search("pricing") == []

    def test_ties_break_deterministically_by_time(self):
        # Two identical captions must not swap places between runs; a result order that changes
        # under no change to the document destroys any hope of a regression test downstream.
        records = [
            record(f"r{i}", "Pricing", modality="ocr", start=float(10 - i)) for i in range(4)
        ]
        first = ranked(index(*records).search("pricing"))
        assert first == ranked(index(*reversed(records)).search("pricing"))
        assert first == ["r3", "r2", "r1", "r0"]

    def test_repeating_a_term_in_a_record_raises_its_score(self):
        idx = index(
            record("r0", "pricing pricing pricing"),
            record("r1", "pricing and some other words to match the length"),
        )
        assert ranked(idx.search("pricing"))[0] == "r0"
