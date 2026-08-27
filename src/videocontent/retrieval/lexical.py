"""Lexical retrieval — BM25 over the document's own facts.

Vector search is the wrong default for video. The queries people actually bring to a recording
are exact: a command someone typed, a product name on a slide, an error string, a URL. An
embedding of ``localhost:3000/login`` is a point in a space where every other URL is nearby,
which is precisely the wrong behaviour when the question is *which* URL was on screen. So the
lexical retriever is the floor this system always has, and embeddings are the optional
addition (ARCHITECTURE §4, Layer 4) — never the other way round.

Three departures from a textbook BM25, each because the corpus is not a corpus of articles:

**Length normalisation is per modality.** A transcript utterance runs 15-30 words; an OCR event
is often two. Pooling them gives one ``avgdl`` that makes every utterance "long" and every
caption "short", and BM25 then systematically prefers on-screen text for every query. Each
modality carries its own average length, so a hit competes against its own kind.

**Ranking is led by matched information content, not by BM25.** In a corpus of forty short
facts, term statistics are noisy enough that BM25 will rank a record matching one word of a
five-word query above one matching four. The primary sort key is therefore the share of the
query's *IDF mass* a record matched: matching more of the query ranks higher, and matching the
rarer half of it ranks higher still. BM25 then orders records of equal coverage, where its
frequency and length work is exactly what is wanted.

Two limits of that worth stating plainly, because both are easy to overclaim. IDF learns what
is common from *this document* — so it discounts a word only once that word actually recurs,
and in a short clip where every term appears once, every term weighs the same. And because
coverage rewards matching more of the query, an instruction-shaped query ("find every
occurrence of the word competitor") ranks worse than the keyword inside it ("competitor"):
records matching four common words beat the one matching the rare one. Keyword queries are
this layer's contract; turning a question into keywords is the Q&A layer's job, not the
retriever's.

**A contiguous phrase match is its own tier.** ``pricing page`` appearing as those two words in
that order is a different kind of evidence from both words appearing somewhere in the record,
and no weighting of unigram scores expresses that reliably.

There is no stopword list. It would be a per-language asset the project does not have, and IDF
already discovers what is common *in this video* — which is more accurate than any fixed list,
because in a recording about pricing, "pricing" is a stopword.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - avoids a cycle with .index at runtime
    from .index import Record

#: Word characters excluding ``_``, so identifiers split the way a person would read them and
#: accented text survives. ``localhost:3000/login`` becomes three tokens, and so does the same
#: string typed as a query — punctuation cannot cause a miss because it is dropped on both
#: sides. A run of CJK text becomes a single token, which is a known limitation: segmenting it
#: needs a per-language dictionary, and until one is configured, exact-substring queries in
#: those scripts still work while partial ones do not.
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

#: BM25 saturation and length-normalisation constants. The Lucene/Robertson defaults; there is
#: no tuning set for this corpus, so pretending to have tuned them would be dishonest (§33).
K1 = 1.2
B = 0.75


def tokenize(text: str) -> list[str]:
    """Casefolded word tokens. The single definition used for both records and queries."""
    return _TOKEN.findall(text.casefold())


@dataclass(frozen=True)
class LexicalHit:
    """One record matched, with every component of its ranking exposed.

    The components are not diagnostics — ``matched`` becomes the span's ``matched_terms`` and
    the rest becomes its ``reason``, because a retrieval system that cannot say *why* something
    ranked where it did cannot be debugged by the person whose query went wrong.
    """

    record: Record
    bm25: float
    coverage: float
    """Share of the query's total IDF mass this record matched, in [0, 1]."""

    phrase: bool
    matched: tuple[str, ...]

    @property
    def sort_key(self) -> tuple[int, float, float, float]:
        # Negated so a plain ascending sort puts the best hit first, and the record's own start
        # time breaks remaining ties: two identical captions must not swap order between runs.
        return (0 if self.phrase else 1, -self.coverage, -self.bm25, self.record.start)


class LexicalIndex:
    """BM25 statistics over a fixed set of records.

    Built once per document and reused across queries — the per-query work is proportional to
    the number of query terms, not to the size of the corpus, because the postings are
    inverted at construction.
    """

    def __init__(self, records: list[Record] | tuple[Record, ...]) -> None:
        self.records: tuple[Record, ...] = tuple(records)
        self.n = len(self.records)

        self._postings: dict[str, dict[int, int]] = {}
        self._lengths: list[int] = []
        totals: dict[str, list[int]] = {}

        for position, record in enumerate(self.records):
            self._lengths.append(len(record.tokens))
            counts: dict[str, int] = {}
            for token in record.tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                self._postings.setdefault(token, {})[position] = count
            totals.setdefault(record.modality, []).append(len(record.tokens))

        # Empty-token records would give a modality an average length of zero and divide by it
        # below; a floor of 1.0 treats them as one-token documents, which is how they behave.
        self._avgdl = {
            modality: max(1.0, sum(lengths) / len(lengths)) for modality, lengths in totals.items()
        }

    def idf(self, term: str) -> float:
        """Robertson-Spärck Jones IDF with the ``1 +`` guard.

        The guard is not cosmetic: without it, a term appearing in more than half the records
        gets a *negative* weight, and a record would be penalised for containing a word the
        user asked for. In a forty-record corpus built from one video, terms above that
        threshold are common.
        """
        df = len(self._postings.get(term, ()))
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str) -> list[LexicalHit]:
        """Every record matching at least one query term, best first.

        Deliberately not truncated to ``top_k``. "Find every occurrence of the word competitor"
        is a real query with a required answer of *all* of them, and a retriever that has
        already discarded the tail cannot serve it — truncation is the caller's decision, made
        after fusion.
        """
        terms = tokenize(query)
        if not terms or self.n == 0:
            return []

        unique = dict.fromkeys(terms)  # order-preserving dedupe
        weights = {term: self.idf(term) for term in unique}
        total_idf = sum(weights.values())

        scored: dict[int, float] = {}
        matched: dict[int, list[str]] = {}
        for term in unique:
            postings = self._postings.get(term)
            if not postings:
                continue
            weight = weights[term]
            for position, frequency in postings.items():
                length = self._lengths[position]
                avgdl = self._avgdl[self.records[position].modality]
                denominator = frequency + K1 * (1.0 - B + B * length / avgdl)
                scored[position] = scored.get(position, 0.0) + weight * frequency * (K1 + 1) / (
                    denominator or 1.0
                )
                matched.setdefault(position, []).append(term)

        phrase_terms = terms if len(terms) > 1 else []
        hits = [
            LexicalHit(
                record=self.records[position],
                bm25=score,
                coverage=(
                    sum(weights[t] for t in matched[position]) / total_idf if total_idf else 0.0
                ),
                phrase=bool(phrase_terms)
                and _has_phrase(self.records[position].tokens, phrase_terms),
                matched=tuple(matched[position]),
            )
            for position, score in scored.items()
        ]
        hits.sort(key=lambda hit: hit.sort_key)
        return hits


def _has_phrase(tokens: tuple[str, ...], phrase: list[str]) -> bool:
    """Whether ``phrase`` appears in ``tokens`` as a contiguous run."""
    width = len(phrase)
    if width > len(tokens):
        return False
    first = phrase[0]
    return any(
        tokens[start] == first and list(tokens[start : start + width]) == phrase
        for start in range(len(tokens) - width + 1)
    )


__all__ = ["K1", "B", "LexicalHit", "LexicalIndex", "tokenize"]
