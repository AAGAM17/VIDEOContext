"""Temporal Semantic Compression — State Detection and Deduplication.

This module implements the core compression logic that converts per-frame
descriptions into persistent visual states with change events.

Instead of: Frame 1: Dashboard. Frame 2: Dashboard. Frame 3: Dashboard.
We produce: 00:00-00:45 — Dashboard visible (persistent elements: sidebar, header, cards)
             00:04 — Chart updates
             00:08 — Navigation opened
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema.v1 import VideoContextDocument, TimeSpan, VisualState, Event


@dataclass
class FrameDescription:
    """A single frame's semantic description."""
    ts: float
    description: str
    entities: list[str]
    actions: list[str]
    ui: dict[str, Any]
    confidence: float | None


@dataclass
class VisualStateTrack:
    """A track of a persistent visual state."""
    state_id: str
    name: str
    start: float
    end: float
    descriptions: list[FrameDescription]
    persistent_elements: list[str]
    changes: list[dict[str, Any]]


def _extract_frame_descriptions(doc: VideoContextDocument) -> list[FrameDescription]:
    """Convert vision notes into frame descriptions."""
    descriptions = []

    for vision in doc.vision:
        descriptions.append(FrameDescription(
            ts=vision.start,
            description=vision.description,
            entities=vision.entities,
            actions=vision.actions,
            ui=vision.ui,
            confidence=vision.confidence,
        ))

    # Sort by timestamp
    descriptions.sort(key=lambda d: d.ts)
    return descriptions


def _compute_similarity(desc1: FrameDescription, desc2: FrameDescription) -> float:
    """Compute similarity between two frame descriptions."""
    if desc1.description == desc2.description:
        return 1.0

    # Jaccard similarity on entities
    entities1 = set(desc1.entities)
    entities2 = set(desc2.entities)
    if entities1 or entities2:
        entity_sim = len(entities1 & entities2) / len(entities1 | entities2)
    else:
        entity_sim = 0.0

    # Jaccard similarity on actions
    actions1 = set(desc1.actions)
    actions2 = set(desc2.actions)
    if actions1 or actions2:
        action_sim = len(actions1 & actions2) / len(actions1 | actions2)
    else:
        action_sim = 0.0

    # Weighted combination
    return 0.6 * entity_sim + 0.4 * action_sim


def _extract_persistent_elements(descriptions: list[FrameDescription]) -> list[str]:
    """Find elements that persist across all descriptions in a state."""
    if not descriptions:
        return []

    # Elements that appear in >80% of frames
    element_counts: dict[str, int] = {}
    for desc in descriptions:
        for entity in desc.entities:
            element_counts[entity] = element_counts.get(entity, 0) + 1

    threshold = len(descriptions) * 0.8
    return [elem for elem, count in element_counts.items() if count >= threshold]


def _detect_changes(prev_desc: FrameDescription, curr_desc: FrameDescription) -> list[dict[str, Any]]:
    """Detect what changed between two frame descriptions."""
    changes = []

    # New entities
    new_entities = set(curr_desc.entities) - set(prev_desc.entities)
    for entity in new_entities:
        changes.append({
            "type": "element_appeared",
            "element": entity,
        })

    # Disappeared entities
    disappeared_entities = set(prev_desc.entities) - set(curr_desc.entities)
    for entity in disappeared_entities:
        changes.append({
            "type": "element_disappeared",
            "element": entity,
        })

    # New actions
    new_actions = set(curr_desc.actions) - set(prev_desc.actions)
    for action in new_actions:
        changes.append({
            "type": "action",
            "action": action,
        })

    # UI state changes
    for key in set(prev_desc.ui.keys()) | set(curr_desc.ui.keys()):
        prev_val = prev_desc.ui.get(key)
        curr_val = curr_desc.ui.get(key)
        if prev_val != curr_val:
            changes.append({
                "type": "ui_change",
                "property": key,
                "from": prev_val,
                "to": curr_val,
            })

    return changes


def compress_temporal_states(doc: VideoContextDocument, similarity_threshold: float = 0.7) -> tuple[list[VisualState], list[Event]]:
    """Compress vision descriptions into persistent visual states with change events.

    Args:
        doc: The video context document
        similarity_threshold: Minimum similarity to consider frames part of same state

    Returns:
        Tuple of (visual_states, change_events)
    """
    frame_descs = _extract_frame_descriptions(doc)
    if not frame_descs:
        return [], []

    states: list[VisualStateTrack] = []
    change_events: list[Event] = []

    current_state = VisualStateTrack(
        state_id="state_0000",
        name="Unknown",
        start=frame_descs[0].ts,
        end=frame_descs[0].ts,
        descriptions=[frame_descs[0]],
        persistent_elements=[],
        changes=[],
    )

    state_counter = 1

    for i in range(1, len(frame_descs)):
        curr = frame_descs[i]
        prev = frame_descs[i - 1]

        similarity = _compute_similarity(prev, curr)

        if similarity >= similarity_threshold:
            # Same state - extend it
            current_state.descriptions.append(curr)
            current_state.end = curr.ts
        else:
            # State changed - finalize current state and start new one
            # Compute persistent elements
            current_state.persistent_elements = _extract_persistent_elements(current_state.descriptions)

            # Generate name from first description
            first_desc = current_state.descriptions[0].description
            current_state.name = first_desc[:80] if first_desc else "State"

            states.append(current_state)

            # Detect and record changes
            changes = _detect_changes(prev, curr)
            for change in changes:
                event = Event(
                    id=f"evt_change_{len(change_events):04d}",
                    type="visual_change",
                    start=prev.ts,
                    end=curr.ts,
                    description=f"{change.get('type', 'change')}: {change.get('element', change.get('action', change.get('property', 'unknown')))}",
                    confidence=0.7,
                    source=["vision"],
                    detector="temporal_compression",
                    attributes=change,
                )
                change_events.append(event)

            # Start new state
            current_state = VisualStateTrack(
                state_id=f"state_{state_counter:04d}",
                name="Unknown",
                start=curr.ts,
                end=curr.ts,
                descriptions=[curr],
                persistent_elements=[],
                changes=[],
            )
            state_counter += 1

    # Finalize last state
    current_state.persistent_elements = _extract_persistent_elements(current_state.descriptions)
    first_desc = current_state.descriptions[0].description
    current_state.name = first_desc[:80] if first_desc else "State"
    states.append(current_state)

    # Convert to VisualState models
    visual_states = []
    for state in states:
        visual_states.append(VisualState(
            state_id=state.state_id,
            name=state.name,
            start=state.start,
            end=state.end,
            persistent_elements=state.persistent_elements,
            changes=state.changes,
            confidence=0.7,
        ))

    return visual_states, change_events


def compress_interaction_states(doc: VideoContextDocument) -> tuple[list[VisualState], list[Event]]:
    """Compress based on interaction events (screen changes, navigation, etc.)."""
    # Use events as primary state boundaries
    boundary_events = [
        e for e in doc.events
        if e.type in ["screen_changed", "slide_changed", "scene_changed", "button_clicked", "command_entered"]
    ]

    if not boundary_events:
        return compress_temporal_states(doc)

    states: list[VisualStateTrack] = []
    change_events: list[Event] = []

    # Create states between boundary events
    boundaries = sorted(boundary_events, key=lambda e: e.start)
    all_boundaries = [0.0] + [e.start for e in boundaries] + [doc.video.duration]

    state_counter = 0

    for i in range(len(all_boundaries) - 1):
        start = all_boundaries[i]
        end = all_boundaries[i + 1]

        # Find vision notes in this interval
        interval_visions = [
            v for v in doc.vision
            if v.start >= start and v.end <= end
        ]

        if not interval_visions:
            continue

        # Merge descriptions
        descriptions = " ".join(v.description for v in interval_visions if v.description)

        state = VisualStateTrack(
            state_id=f"state_{state_counter:04d}",
            name=descriptions[:80] if descriptions else f"State {state_counter}",
            start=start,
            end=end,
            descriptions=[],
            persistent_elements=[],
            changes=[],
        )

        # Add boundary event as change
        if i < len(boundaries):
            event = boundaries[i]
            change_events.append(Event(
                id=f"evt_state_{state_counter:04d}",
                type="state_transition",
                start=event.start,
                end=event.end,
                description=event.description or f"Transition: {event.type}",
                confidence=event.confidence,
                source=["events"],
                detector="interaction_compression",
                attributes={"transition_type": event.type},
            ))

        states.append(state)
        state_counter += 1

    visual_states = []
    for state in states:
        visual_states.append(VisualState(
            state_id=state.state_id,
            name=state.name,
            start=state.start,
            end=state.end,
            persistent_elements=state.persistent_elements,
            changes=state.changes,
            confidence=0.7,
        ))

    return visual_states, change_events


__all__ = [
    "compress_temporal_states",
    "compress_interaction_states",
    "FrameDescription",
    "VisualStateTrack",
]