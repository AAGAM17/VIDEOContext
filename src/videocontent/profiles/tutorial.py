"""Tutorial Profile Builder.

Analyzes educational/tutorial videos to extract learning objectives,
steps, key concepts, and structure.
"""

from __future__ import annotations

from typing import Any

from ..schema.v1 import VideoContextDocument, TutorialProfile, TimeSpan
from .base import SemanticProfile, ProfileContext


class TutorialProfileBuilder(SemanticProfile):
    """Builds a tutorial/educational profile from video evidence."""

    name = "tutorial"
    display_name = "Tutorial / Educational"
    description = "Learning objectives, steps, key concepts, and structure"

    def supports(self, context: ProfileContext) -> bool:
        """Check if video is a tutorial."""
        for vision in context.doc.vision:
            desc = vision.description.lower()
            if any(kw in desc for kw in ["tutorial", "how to", "learn", "teach", "lesson", "course", "training", "guide", "walkthrough", "step by step"]):
                return True
        for transcript in context.doc.transcript:
            text = transcript.text.lower()
            if any(kw in text for kw in ["tutorial", "how to", "learn", "teach", "lesson", "course", "training", "guide", "step by step", "in this video", "today we"]):
                return True
        return False

    def build(self, context: ProfileContext) -> TutorialProfile:
        """Build the tutorial profile."""
        doc = context.doc

        topic = self._extract_topic(doc)
        learning_objectives = self._extract_learning_objectives(doc)
        steps = self._extract_steps(doc)
        key_concepts = self._extract_key_concepts(doc)
        evidence = self._collect_evidence(doc)

        return TutorialProfile(
            topic=topic,
            learning_objectives=learning_objectives,
            steps=steps,
            key_concepts=key_concepts,
            evidence=evidence,
        )

    def _extract_topic(self, doc: VideoContextDocument) -> str | None:
        """Extract the main topic of the tutorial."""
        # Look at the beginning of the transcript
        for t in doc.transcript[:5]:
            text = t.text.lower()
            topic_keywords = ["how to", "learn", "tutorial", "guide", "introduction to", "getting started with"]
            for kw in topic_keywords:
                if kw in text:
                    # Extract the phrase after the keyword
                    import re
                    match = re.search(rf"{kw}\s+(.+?)[\.\?\!]", text)
                    if match:
                        return match.group(1).strip().title()

        # Fallback: use first vision description
        for v in doc.vision[:3]:
            if v.description:
                return v.description[:100]

        return None

    def _extract_learning_objectives(self, doc: VideoContextDocument) -> list[str]:
        """Extract learning objectives from the tutorial."""
        objective_keywords = [
            "you will learn", "you'll learn", "we will learn", "we'll learn",
            "in this video", "in this tutorial", "by the end",
            "objective", "goal", "aim", "outcome",
            "understand", "master", "be able to",
        ]

        objectives = set()

        for t in doc.transcript:
            text = t.text.lower()
            for kw in objective_keywords:
                if kw in text:
                    sentences = text.split(".")
                    for sent in sentences:
                        if kw in sent and len(sent.strip()) > 10:
                            objectives.add(sent.strip()[:150])

        return list(objectives)[:10]

    def _extract_steps(self, doc: VideoContextDocument) -> list[dict[str, Any]]:
        """Extract tutorial steps."""
        step_keywords = [
            "step", "first", "second", "third", "next", "then", "finally",
            "now", "after that", "once you", "when you",
        ]

        steps = []
        step_number = 0

        for t in doc.transcript:
            text = t.text.lower()
            for kw in step_keywords:
                if kw in text:
                    sentences = text.split(".")
                    for sent in sentences:
                        if kw in sent and len(sent.strip()) > 10:
                            step_number += 1
                            steps.append({
                                "step": step_number,
                                "time": t.start,
                                "description": sent.strip(),
                            })
                            break

        # If no explicit steps, create from vision events
        if not steps:
            for i, vision in enumerate(doc.vision):
                if vision.description:
                    steps.append({
                        "step": i + 1,
                        "time": vision.start,
                        "description": vision.description[:200],
                    })

        return steps[:20]

    def _extract_key_concepts(self, doc: VideoContextDocument) -> list[str]:
        """Extract key concepts/terms from the tutorial."""
        # Look for technical terms, tool names, method names
        concept_indicators = [
            "called", "named", "known as", "referred to as",
            "the key", "the main", "the important", "the core",
            "concept", "principle", "pattern", "technique", "method",
            "approach", "strategy", "best practice",
        ]

        concepts = set()

        for t in doc.transcript:
            text = t.text
            for kw in concept_indicators:
                if kw in text.lower():
                    # Find capitalized words nearby
                    import re
                    # Look for capitalized phrases
                    matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
                    for match in matches:
                        if len(match) > 2:
                            concepts.add(match)

        # Also check vision for technical terms
        for v in doc.vision:
            if v.description:
                matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', v.description)
                for match in matches:
                    if len(match) > 2:
                        concepts.add(match)

        return list(concepts)[:20]

    def _collect_evidence(self, doc: VideoContextDocument) -> list[TimeSpan]:
        """Collect evidence spans."""
        from ..schema.v1 import TimeSpan

        evidence = []

        for t in doc.transcript:
            evidence.append(TimeSpan(start=t.start, end=t.end))

        for v in doc.vision:
            evidence.append(TimeSpan(start=v.start, end=v.end))

        return evidence


__all__ = ["TutorialProfileBuilder"]