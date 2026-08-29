"""Product Demo Profile Builder.

Analyzes product demonstration videos to extract product overview,
features shown, use cases, and UI walkthrough.
"""

from __future__ import annotations

from typing import Any

from ..schema.v1 import VideoContextDocument, ProductDemoProfile, TimeSpan
from .base import SemanticProfile, ProfileContext


class ProductDemoProfileBuilder(SemanticProfile):
    """Builds a product demonstration profile from video evidence."""

    name = "product_demo"
    display_name = "Product Demo Analysis"
    description = "Product overview, features, use cases, and UI walkthrough"

    def supports(self, context: ProfileContext) -> bool:
        """Check if video is a product demo."""
        for vision in context.doc.vision:
            desc = vision.description.lower()
            if any(kw in desc for kw in ["demo", "demonstration", "product", "feature", "showcase", "walkthrough", "tutorial", "how to", "overview"]):
                return True
        for transcript in context.doc.transcript:
            text = transcript.text.lower()
            if any(kw in text for kw in ["demo", "demonstration", "product", "feature", "showcase", "walkthrough", "tutorial", "how to"]):
                return True
        return False

    def build(self, context: ProfileContext) -> ProductDemoProfile:
        """Build the product demo profile."""
        doc = context.doc

        product_overview = self._generate_product_overview(doc)
        features_shown = self._extract_features(doc)
        use_cases = self._extract_use_cases(doc)
        ui_walkthrough = self._extract_ui_walkthrough(doc)
        evidence = self._collect_evidence(doc)

        return ProductDemoProfile(
            product_overview=product_overview,
            features_shown=features_shown,
            use_cases=use_cases,
            ui_walkthrough=ui_walkthrough,
            evidence=evidence,
        )

    def _generate_product_overview(self, doc: VideoContextDocument) -> str | None:
        """Generate product overview from transcript and vision."""
        # Combine transcript and vision to create overview
        texts = []

        for t in doc.transcript:
            if t.text:
                texts.append(t.text)

        for v in doc.vision:
            if v.description:
                texts.append(v.description)

        if not texts:
            return None

        # Return first substantial text as overview
        combined = " ".join(texts)
        if len(combined) > 200:
            return combined[:500]
        return combined

    def _extract_features(self, doc: VideoContextDocument) -> list[str]:
        """Extract features mentioned in the demo."""
        feature_keywords = [
            "feature", "capability", "function", "functionality",
            "supports", "enables", "allows", "provides", "offers",
            "includes", "comes with", "has", "built-in",
        ]

        features = set()

        for t in doc.transcript:
            text = t.text.lower()
            for kw in feature_keywords:
                if kw in text:
                    # Extract the sentence containing the feature keyword
                    sentences = text.split(".")
                    for sent in sentences:
                        if kw in sent and len(sent.strip()) > 10:
                            features.add(sent.strip()[:100])

        for v in doc.vision:
            if v.description:
                text = v.description.lower()
                for kw in feature_keywords:
                    if kw in text:
                        sentences = text.split(".")
                        for sent in sentences:
                            if kw in sent and len(sent.strip()) > 10:
                                features.add(sent.strip()[:100])

        return list(features)[:20]

    def _extract_use_cases(self, doc: VideoContextDocument) -> list[str]:
        """Extract use cases from the demo."""
        use_case_keywords = [
            "use case", "use-case", "scenario", "workflow",
            "for example", "for instance", "when you", "if you",
            "perfect for", "ideal for", "great for", "designed for",
        ]

        use_cases = set()

        for t in doc.transcript:
            text = t.text.lower()
            for kw in use_case_keywords:
                if kw in text:
                    sentences = text.split(".")
                    for sent in sentences:
                        if kw in sent and len(sent.strip()) > 10:
                            use_cases.add(sent.strip()[:150])

        return list(use_cases)[:15]

    def _extract_ui_walkthrough(self, doc: VideoContextDocument) -> list[dict[str, Any]]:
        """Extract UI walkthrough steps."""
        walkthrough = []

        # Use events to identify walkthrough steps
        step_events = [e for e in doc.events if e.type in [
            "screen_changed", "slide_changed", "text_appeared", "button_clicked"
        ]]

        for event in step_events:
            walkthrough.append({
                "step": len(walkthrough) + 1,
                "time": event.start,
                "screen": event.attributes.get("to", "unknown"),
                "action": event.type.replace("_", " "),
                "description": event.description or event.type,
            })

        # If no events, create steps from vision
        if not walkthrough:
            for i, vision in enumerate(doc.vision):
                if vision.description:
                    walkthrough.append({
                        "step": i + 1,
                        "time": vision.start,
                        "description": vision.description[:200],
                    })

        return walkthrough

    def _collect_evidence(self, doc: VideoContextDocument) -> list[TimeSpan]:
        """Collect evidence spans."""
        from ..schema.v1 import TimeSpan

        evidence = []

        for t in doc.transcript:
            evidence.append(TimeSpan(start=t.start, end=t.end))

        for v in doc.vision:
            evidence.append(TimeSpan(start=v.start, end=v.end))

        return evidence


__all__ = ["ProductDemoProfileBuilder"]