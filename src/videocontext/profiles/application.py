"""Application Profile Builder.

Analyzes screen recordings of software applications to understand
screen hierarchy, user flows, state transitions, and interactions.
"""

from __future__ import annotations

from typing import Any

from ..schema.v1 import VideoContextDocument, ApplicationProfile, TimeSpan
from .base import SemanticProfile, ProfileContext


class ApplicationProfileBuilder(SemanticProfile):
    """Builds an application understanding profile from video evidence."""

    name = "application"
    display_name = "Application Understanding"
    description = "Screen hierarchy, user flows, state transitions, and key interactions"

    def supports(self, context: ProfileContext) -> bool:
        """Check if video contains application content."""
        for vision in context.doc.vision:
            desc = vision.description.lower()
            if any(kw in desc for kw in ["application", "app", "software", "interface", "dashboard", "tool", "platform", "system"]):
                return True
        return False

    def build(self, context: ProfileContext) -> ApplicationProfile:
        """Build the application profile."""
        doc = context.doc

        overview = self._generate_overview(doc)
        screen_hierarchy = self._analyze_screen_hierarchy(doc)
        user_flows = self._analyze_user_flows(doc)
        state_transitions = self._analyze_state_transitions(doc)
        important_interactions = self._analyze_important_interactions(doc)

        evidence = self._collect_evidence(doc)

        return ApplicationProfile(
            overview=overview,
            screen_hierarchy=screen_hierarchy,
            user_flows=user_flows,
            state_transitions=state_transitions,
            important_interactions=important_interactions,
            evidence=evidence,
        )

    def _generate_overview(self, doc: VideoContextDocument) -> str | None:
        """Generate a high-level overview of the application."""
        if not doc.vision:
            return None

        # Combine vision descriptions to create overview
        descriptions = [v.description for v in doc.vision if v.description]
        if not descriptions:
            return None

        # Simple heuristic: find the most descriptive vision note
        longest = max(descriptions, key=len)
        return longest[:500]

    def _analyze_screen_hierarchy(self, doc: VideoContextDocument) -> list[dict[str, Any]]:
        """Analyze the hierarchy of screens/views in the application."""
        hierarchy = []
        seen_screens = set()

        # Use vision descriptions and OCR to identify screens
        for vision in doc.vision:
            desc = vision.description.lower()
            # Try to identify screen names
            screen_name = self._extract_screen_name(desc)
            if screen_name and screen_name not in seen_screens:
                seen_screens.add(screen_name)
                hierarchy.append({
                    "screen": screen_name,
                    "first_seen": vision.start,
                    "last_seen": vision.end,
                    "description": vision.description[:200],
                })

        # Also check events for screen changes
        for event in doc.events:
            if event.type in ["screen_changed", "slide_changed"]:
                from_screen = event.attributes.get("from", "unknown")
                to_screen = event.attributes.get("to", "unknown")
                if to_screen not in seen_screens:
                    seen_screens.add(to_screen)
                    hierarchy.append({
                        "screen": to_screen,
                        "first_seen": event.start,
                        "last_seen": event.end,
                        "transition_from": from_screen,
                    })

        return hierarchy

    def _extract_screen_name(self, description: str) -> str | None:
        """Try to extract a screen name from a description."""
        # Common screen indicators
        screen_indicators = [
            "dashboard", "login", "settings", "profile", "home", "landing",
            "onboarding", "setup", "configuration", "admin", "analytics",
            "reports", "users", "projects", "tasks", "calendar", "messages",
            "notifications", "search", "results", "details", "editor",
            "builder", "creator", "designer", "viewer", "preview",
        ]

        for indicator in screen_indicators:
            if indicator in description:
                return indicator.title()

        # Try to extract from "X screen" or "X page" patterns
        import re
        match = re.search(r'(\w+(?:\s+\w+)?)\s+(screen|page|view)', description)
        if match:
            return match.group(1).title()

        return None

    def _analyze_user_flows(self, doc: VideoContextDocument) -> list[dict[str, Any]]:
        """Analyze user flows through the application."""
        flows = []

        # Use events to detect flows
        flow_events = [e for e in doc.events if e.type in [
            "screen_changed", "slide_changed", "button_clicked", "command_entered",
            "person_entered", "person_left"
        ]]

        if not flow_events:
            return flows

        # Group sequential events into flows
        current_flow = []
        for event in flow_events:
            current_flow.append({
                "action": event.type.replace("_", " "),
                "time": event.start,
                "description": event.description or event.type,
            })

            # If there's a gap > 30s, start a new flow
            if len(current_flow) > 1:
                prev_time = flow_events[flow_events.index(event) - 1].end
                if event.start - prev_time > 30:
                    if current_flow:
                        flows.append({
                            "start": current_flow[0]["time"],
                            "end": current_flow[-1]["time"],
                            "steps": current_flow,
                        })
                    current_flow = [current_flow[-1]]

        if current_flow:
            flows.append({
                "start": current_flow[0]["time"],
                "end": current_flow[-1]["time"],
                "steps": current_flow,
            })

        return flows

    def _analyze_state_transitions(self, doc: VideoContextDocument) -> list[dict[str, Any]]:
        """Analyze state transitions in the application."""
        transitions = []

        for event in doc.events:
            if event.type in ["screen_changed", "slide_changed", "person_entered", "person_left"]:
                transitions.append({
                    "type": event.type,
                    "from_state": event.attributes.get("from", "unknown"),
                    "to_state": event.attributes.get("to", "unknown"),
                    "time": event.start,
                    "description": event.description or event.type,
                    "confidence": event.confidence,
                })

        return transitions

    def _analyze_important_interactions(self, doc: VideoContextDocument) -> list[dict[str, Any]]:
        """Identify important interactions."""
        interactions = []

        important_event_types = [
            "button_clicked", "command_entered", "error_shown",
            "text_appeared", "text_changed", "object_appeared"
        ]

        for event in doc.events:
            if event.type in important_event_types:
                interactions.append({
                    "type": event.type,
                    "time": event.start,
                    "description": event.description or event.type,
                    "confidence": event.confidence,
                })

        # Also check vision for interaction descriptions
        for vision in doc.vision:
            desc = vision.description.lower()
            interaction_keywords = [
                "click", "press", "tap", "drag", "drop", "scroll",
                "hover", "focus", "select", "submit", "save", "delete",
                "create", "edit", "update", "navigate"
            ]
            if any(kw in desc for kw in interaction_keywords):
                interactions.append({
                    "type": "vision_detected",
                    "time": vision.start,
                    "description": vision.description[:200],
                    "confidence": vision.confidence,
                })

        return interactions

    def _collect_evidence(self, doc: VideoContextDocument) -> list[TimeSpan]:
        """Collect all relevant evidence spans."""
        from ..schema.v1 import TimeSpan

        evidence = []

        for vision in doc.vision:
            evidence.append(TimeSpan(start=vision.start, end=vision.end))

        for event in doc.events:
            if event.type in ["screen_changed", "slide_changed", "button_clicked", "command_entered"]:
                evidence.append(TimeSpan(start=event.start, end=event.end))

        return evidence


__all__ = ["ApplicationProfileBuilder"]