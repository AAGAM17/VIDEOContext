"""Timestamp-grounded entities with cross-modal linking.

Entities are extracted with cheap, honest detectors — error signatures and shell
prompts reuse the event layer's own evidence, and generic terms are repeated
capitalized phrases or quoted strings. Every occurrence names its modality and
reference id; linking across modalities requires temporal co-occurrence, and
uncertainty is explicit (``confidence`` + ``ambiguous``) rather than averaged away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .logging import get_logger

log = get_logger("entities")

#: Two mentions in different modalities link only when their spans overlap or meet
#: within this tolerance — the same 1 s the retrieval fusion uses.
LINK_TOLERANCE_S = 1.0
_MAX_ENTITIES = 200

_QUOTED_RE = re.compile(r'"([^"]{3,80})"')
_PHRASE_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3})\b")
_TERM_STOP = frozenset([
    "The", "And", "For", "With", "From", "That", "This", "These", "Those",
    "Your", "Our", "Their", "Its",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "June", "July", "August",
    "September", "October", "November", "December",
])


@dataclass(frozen=True)
class Occurrence:
    start: float
    end: float
    modality: str
    ref_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start, "end": self.end, "modality": self.modality,
            "ref_id": self.ref_id, "text": self.text[:160],
        }


@dataclass
class Entity:
    """One concept with every timestamped sighting merged."""

    id: str
    name: str
    type: str
    occurrences: list[Occurrence] = field(default_factory=list)
    confidence: float = 0.5
    method: str = "term"
    ambiguous: bool = False

    @property
    def linked_modalities(self) -> tuple[str, ...]:
        return tuple(sorted({o.modality for o in self.occurrences}))

    @property
    def first_seen(self) -> float | None:
        return min((o.start for o in self.occurrences), default=None)

    @property
    def last_seen(self) -> float | None:
        return max((o.end for o in self.occurrences), default=None)

    def timeline(self) -> list[Occurrence]:
        return sorted(self.occurrences, key=lambda o: (o.start, o.end))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "type": self.type,
            "occurrences": [o.to_dict() for o in self.timeline()],
            "linked_modalities": list(self.linked_modalities),
            "first_seen": self.first_seen, "last_seen": self.last_seen,
            "confidence": self.confidence, "method": self.method,
            "ambiguous": self.ambiguous,
        }


def normalize_name(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s$.:/-]", "", text)).strip().lower()


def _near(a: Occurrence, b: Occurrence, tol: float = LINK_TOLERANCE_S) -> bool:
    return a.start < b.end + tol and b.start < a.end + tol


def _terms_in(text: str) -> list[str]:
    found = _QUOTED_RE.findall(text or "")
    found += _PHRASE_RE.findall(text or "")
    out: list[str] = []
    for candidate in found:
        candidate = candidate.strip()
        if len(candidate) < 3 or candidate in _TERM_STOP:
            continue
        words = candidate.split()
        if all(w in _TERM_STOP for w in words):
            continue
        out.append(candidate)
    return out


def extract_entities(doc: Any) -> list[Entity]:
    """Extract entities from a document. Deterministic for a given document.

    Detectors (no models, no network):
    - ``error_signature``: ``error_shown`` events → ERROR entities.
    - ``shell_prompt``: ``command_entered`` events → COMMAND entities.
    - ``term``: repeated capitalized/quoted phrases in OCR + transcript, linked
      across modalities on temporal co-occurrence.
    """
    entities: list[Entity] = []
    counter = 0

    def _next(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}_{counter:04d}"

    # Detector 1+2: reuse the event layer's own evidence (never re-implement patterns).
    for event in getattr(doc, "events", []):
        if event.type == "error_shown":
            name = (event.description or event.attributes.get("signal") or "error").strip()
            entities.append(Entity(
                _next("ent"), name[:80], "ERROR",
                [Occurrence(event.start, event.end, "events", event.id,
                            (event.description or "")[:160])],
                confidence=float(event.confidence) if event.confidence is not None else 0.7,
                method="error_signature"))
        elif event.type == "command_entered":
            command = str(event.attributes.get("command", "") or "").strip()
            if command:
                entities.append(Entity(
                    _next("ent"), command[:80], "COMMAND",
                    [Occurrence(event.start, event.end, "events", event.id, command[:160])],
                    confidence=0.85, method="shell_prompt"))

    # Detector 3: cross-modal term linking.
    mentions: dict[str, dict[str, Any]] = {}
    for item in list(getattr(doc, "ocr", [])):
        for term in _terms_in(item.text):
            key = normalize_name(term)
            slot = mentions.setdefault(key, {"display": term.strip(), "occ": []})
            slot["occ"].append(Occurrence(item.start, item.end, "ocr", item.id, item.text))
    for item in list(getattr(doc, "transcript", [])):
        for term in _terms_in(item.text):
            key = normalize_name(term)
            slot = mentions.setdefault(key, {"display": term.strip(), "occ": []})
            slot["occ"].append(Occurrence(item.start, item.end, "transcript", item.id,
                                          item.text))
    for item in list(getattr(doc, "vision", [])):
        for raw in list(getattr(item, "entities", []) or []):
            term = str(raw).strip()
            if len(term) < 3:
                continue
            key = normalize_name(term)
            slot = mentions.setdefault(key, {"display": term, "occ": []})
            slot["occ"].append(Occurrence(item.start, item.end, "vision", item.id,
                                          item.description or term))

    # Merge mentions that name the same concept at different granularities
    # ("Stripe" vs "Stripe API"): shared content tokens + temporal proximity.
    groups = _merge_mentions(mentions)
    for display, occ in groups:
        modalities = {o.modality for o in occ}
        linked = len(modalities) >= 2 and any(
            _near(a, b) for a in occ for b in occ if a.modality != b.modality)
        if linked:
            confidence, ambiguous = 0.85, False
        elif len(occ) >= 3:
            confidence, ambiguous = 0.6, True
        elif len(occ) == 2:
            confidence, ambiguous = 0.5, True
        else:
            continue  # a single mention is not an entity, it is a sighting
        entities.append(Entity(_next("ent"), display, "CONCEPT", occ,
                               confidence, "term", ambiguous))

    entities.sort(key=lambda e: (-e.confidence, -(e.first_seen or 0.0)))
    return entities[:_MAX_ENTITIES]


def _merge_mentions(mentions: dict[str, dict[str, Any]]) -> list[tuple[str, list[Occurrence]]]:
    """Union mentions sharing a content token that co-occur in time.

    Returns ``(display, occurrences)`` per group; display is the longest variant,
    which is the most informative ("Stripe API Dashboard", not "Stripe").
    """
    keys = list(mentions)
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def tokens(key: str) -> set[str]:
        return {t for t in key.split() if len(t) >= 3}

    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            shared = tokens(ka) & tokens(kb)
            if not shared or not any(len(t) >= 4 for t in shared):
                continue  # short words ("api", "app") collide too easily alone
            occ_a, occ_b = mentions[ka]["occ"], mentions[kb]["occ"]
            if any(_near(a, b) for a in occ_a for b in occ_b):
                parent[find(ka)] = find(kb)

    grouped: dict[str, list[str]] = {}
    for key in keys:
        grouped.setdefault(find(key), []).append(key)
    out: list[tuple[str, list[Occurrence]]] = []
    for members in grouped.values():
        occ = [o for m in members for o in mentions[m]["occ"]]
        display = max((mentions[m]["display"] for m in members), key=len)
        out.append((display.strip(), occ))
    return out


def entity_timeline(entity: Entity) -> list[Occurrence]:
    """All occurrences of an entity in time order — answers 'when was X seen?'."""
    return entity.timeline()


__all__ = ["LINK_TOLERANCE_S", "Entity", "Occurrence", "entity_timeline", "extract_entities",
           "normalize_name"]
