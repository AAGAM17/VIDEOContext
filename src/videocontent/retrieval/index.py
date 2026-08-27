"""Turning a document into searchable records — the projection retrieval scores.

A ``.vctx`` document stores facts by modality: utterances, OCR events, vision notes, events.
Retrieval needs them in one flat list with uniform fields, and it needs to know which segment
each one belongs to so a hit can be widened into context. That projection is all this module
does; nothing here scores or ranks.

Two decisions are worth naming.

**Facts, not segments, are the retrieval unit.** Segments are the *fusion* unit (spec §12) and
they carry the co-occurrence signal, but their spans are up to 45 seconds wide. Answering "when
did the speaker mention revenue?" with a 45-second window is a worse answer than the document
can support, because the utterance's own timestamps are right there. So records are facts, and
each one carries the ids of the segments that contain it — precision from the fact, context
from the segment.

**``text`` is what a human is shown; ``tokens`` is what the matcher sees.** They differ for
vision notes and events, where the searchable surface includes entities, actions and the event
type but the displayed evidence should stay the description a model actually wrote. Keeping
them separate is what lets ``What command was typed?`` match on an event's ``terminal_command``
type without that string appearing in the quoted evidence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..schema.v1 import VideoContextDocument
from .lexical import tokenize

#: The modality names accepted in ``RetrievalConfig.modalities``. These are the document's own
#: field names, so a config typo is caught by comparing against this set rather than silently
#: searching nothing.
MODALITIES = ("transcript", "ocr", "vision", "events")


@dataclass(frozen=True)
class Record:
    """One searchable fact, copied out of the document.

    Every field here exists in the source document. Nothing is generated — in particular
    ``start``/``end`` are copied, never computed, which is what makes it impossible for a
    returned timestamp to be one the extraction never observed.
    """

    key: str
    """Unique within an index: ``modality:id``. Document ids are prefixed per modality in
    practice, but the spec does not require it, so the modality is part of the key."""

    id: str
    modality: str
    start: float
    end: float
    text: str
    tokens: tuple[str, ...]
    segment_ids: tuple[str, ...] = ()
    confidence: float | None = None
    language: str | None = None
    kind: str | None = None
    """Event type, or ``None`` for modalities that have no sub-type."""

    extra: dict[str, str] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def segment_map(doc: VideoContextDocument) -> dict[str, tuple[str, ...]]:
    """Reverse index: fact id → the segments referencing it.

    A fact can appear in more than one segment (on-screen text that spans a scene change is
    genuinely present in both), so the value is a tuple, not a single id.
    """
    owners: dict[str, list[str]] = defaultdict(list)
    for segment in doc.segments:
        for ids in (
            segment.transcript_ids,
            segment.ocr_ids,
            segment.vision_ids,
            segment.event_ids,
            segment.scene_ids,
        ):
            for fact_id in ids:
                owners[fact_id].append(segment.id)
    return {fact_id: tuple(segment_ids) for fact_id, segment_ids in owners.items()}


def _searchable(*parts: str | None) -> tuple[str, ...]:
    """Tokens of every non-empty part, in order, duplicates kept.

    Duplicates matter: BM25 uses term frequency, and a word that appears in both a vision
    description and its entity list really is more central to that record.
    """
    tokens: list[str] = []
    for part in parts:
        if part:
            tokens.extend(tokenize(part))
    return tuple(tokens)


def build_records(
    doc: VideoContextDocument,
    modalities: tuple[str, ...] | list[str] | None = None,
) -> list[Record]:
    """Flatten ``doc`` into records for the requested modalities, in timeline order.

    Unknown modality names are ignored rather than raised on: a config written against a
    newer version of the format should degrade to searching what this version understands,
    not fail to search at all. The caller learns what was actually searched from
    ``SearchResult.modalities``.
    """
    wanted = tuple(modalities) if modalities is not None else MODALITIES
    selected = tuple(name for name in MODALITIES if name in wanted)
    owners = segment_map(doc)
    records: list[Record] = []

    def add(
        modality: str,
        fact_id: str,
        start: float,
        end: float,
        text: str,
        tokens: tuple[str, ...],
        **kw: object,
    ) -> None:
        records.append(
            Record(
                key=f"{modality}:{fact_id}",
                id=fact_id,
                modality=modality,
                start=start,
                end=end,
                text=text,
                tokens=tokens,
                segment_ids=owners.get(fact_id, ()),
                **kw,  # type: ignore[arg-type]
            )
        )

    if "transcript" in selected:
        for utterance in doc.transcript:
            text = utterance.text.strip()
            if text:
                add(
                    "transcript",
                    utterance.id,
                    utterance.start,
                    utterance.end,
                    text,
                    _searchable(text),
                    confidence=utterance.confidence,
                    language=utterance.language,
                    extra={"speaker": utterance.speaker} if utterance.speaker else {},
                )

    if "ocr" in selected:
        for ocr in doc.ocr:
            text = ocr.text.strip()
            if text:
                add(
                    "ocr",
                    ocr.id,
                    ocr.start,
                    ocr.end,
                    text,
                    _searchable(text),
                    confidence=ocr.confidence,
                    language=ocr.language,
                )

    if "vision" in selected:
        for note in doc.vision:
            text = note.description.strip()
            if text or note.entities:
                add(
                    "vision",
                    note.id,
                    note.start,
                    note.end,
                    text,
                    _searchable(text, " ".join(note.entities), " ".join(note.actions)),
                    confidence=note.confidence,
                    language=note.language,
                )

    if "events" in selected:
        for event in doc.events:
            # The type is searchable but not displayed: "terminal_command" is how a query
            # finds the moment, while the description is what a human should be shown.
            description = (event.description or "").strip()
            attributes = " ".join(str(v) for v in event.attributes.values() if v is not None)
            add(
                "events",
                event.id,
                event.start,
                event.end,
                description or event.type.replace("_", " "),
                _searchable(description, event.type.replace("_", " "), attributes),
                confidence=event.confidence,
                kind=event.type,
            )

    records.sort(key=lambda r: (r.start, r.end, r.key))
    return records


__all__ = ["MODALITIES", "Record", "build_records", "segment_map"]
