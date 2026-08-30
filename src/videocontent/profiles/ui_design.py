"""UI Design Profile Builder.

Analyzes screen recordings of websites, applications, dashboards, and interfaces
to extract visual style, typography, layout patterns, component patterns,
interaction patterns, and motion patterns.
"""

from __future__ import annotations

from typing import Any

from ..schema.v1 import (
    VideoContextDocument,
    TimeSpan,
    UIDesignProfile,
    VisualStyleProfile,
    TypographyProfile,
    LayoutProfile,
    ComponentProfile,
    InteractionProfile,
    MotionProfile,
)
from .base import SemanticProfile, ProfileContext


class UIDesignProfileBuilder(SemanticProfile):
    """Builds a comprehensive UI/Design profile from video evidence."""

    name = "ui_design"
    display_name = "UI / Website Design"
    description = "Visual style, typography, layout, components, interactions, and motion patterns"

    def supports(self, context: ProfileContext) -> bool:
        """Check if video contains UI/interface content."""
        # Look for signals: screen recordings, browser content, application UI
        for vision in context.doc.vision:
            desc = vision.description.lower()
            if any(kw in desc for kw in ["browser", "website", "dashboard", "interface", "ui", "screen", "application", "app", "modal", "navigation", "sidebar", "card", "button"]):
                return True
        for ocr in context.doc.ocr:
            text = ocr.text.lower()
            if any(kw in text for kw in ["http", "localhost", "login", "dashboard", "menu", "navigation", "button", "input", "form"]):
                return True
        return False

    def build(self, context: ProfileContext) -> Any:
        """Build the complete UI design profile."""
        doc = context.doc

        # Analyze visual style from vision descriptions
        visual_style = self._analyze_visual_style(doc)

        # Analyze typography
        typography = self._analyze_typography(doc)

        # Analyze layout patterns
        layout = self._analyze_layout(doc)

        # Analyze components
        components = self._analyze_components(doc)

        # Analyze interactions
        interaction = self._analyze_interactions(doc)

        # Analyze motion
        motion = self._analyze_motion(doc)

        from ..schema.v1 import UIDesignProfile
        return UIDesignProfile(
            visual_style=visual_style,
            typography=typography,
            layout=layout,
            components=components,
            interaction=interaction,
            motion=motion,
        )

    def _analyze_visual_style(self, doc: VideoContextDocument) -> Any:
        """Analyze overall visual style from vision and OCR."""
        from ..schema.v1 import VisualStyleProfile, TimeSpan

        style_keywords = {
            "minimal": ["minimal", "clean", "simple", "uncluttered"],
            "dark": ["dark", "dark mode", "dark theme", "black background"],
            "light": ["light", "light mode", "white background", "bright"],
            "premium": ["premium", "luxury", "high-end", "polished", "refined"],
            "modern": ["modern", "contemporary", "current"],
            "colorful": ["colorful", "vibrant", "bright colors", "gradient"],
            "monochrome": ["monochrome", "grayscale", "black and white"],
            "gradient": ["gradient", "gradients"],
            "glassmorphism": ["glass", "glassmorphism", "frosted", "blur"],
            "neomorphism": ["neomorphism", "neumorphism", "soft ui"],
        }

        detected = []
        evidence_spans = []

        for vision in doc.vision:
            desc = vision.description.lower()
            for style, keywords in style_keywords.items():
                if any(kw in desc for kw in keywords):
                    if style not in detected:
                        detected.append(style)
                    evidence_spans.append(TimeSpan(start=vision.start, end=vision.end))

        for ocr in doc.ocr:
            text = ocr.text.lower()
            for style, keywords in style_keywords.items():
                if any(kw in text for kw in keywords):
                    if style not in detected:
                        detected.append(style)
                    evidence_spans.append(TimeSpan(start=ocr.start, end=ocr.end))

        color_characteristics = []
        if "dark" in detected:
            color_characteristics.append("dark theme")
        elif "light" in detected:
            color_characteristics.append("light theme")

        if "gradient" in detected:
            color_characteristics.append("gradients")

        surface_style = []
        if "glassmorphism" in detected:
            surface_style.append("glassmorphism")
        if "neomorphism" in detected:
            surface_style.append("neomorphism")

        return VisualStyleProfile(
            overall=detected,
            color_characteristics=color_characteristics,
            surface_style=surface_style,
            confidence=0.7 if detected else 0.3,
            evidence=evidence_spans,
        )

    def _analyze_typography(self, doc: VideoContextDocument) -> Any:
        """Analyze typography characteristics."""
        from ..schema.v1 import TypographyProfile, TimeSpan

        typography_keywords = {
            "strong_hierarchy": ["large heading", "big title", "prominent heading", "heading hierarchy"],
            "large_headings": ["large text", "big text", "huge heading", "hero text"],
            "spacious": ["generous spacing", "wide spacing", "ample whitespace", "roomy"],
            "compact": ["compact", "tight spacing", "dense text"],
            "monospace": ["monospace", "code font", "terminal font", "fixed width"],
            "serif": ["serif", "serif font"],
            "sans_serif": ["sans-serif", "sans serif", "clean font"],
        }

        detected = []
        evidence_spans = []

        for vision in doc.vision:
            desc = vision.description.lower()
            for prop, keywords in typography_keywords.items():
                if any(kw in desc for kw in keywords):
                    if prop not in detected:
                        detected.append(prop)
                    evidence_spans.append(TimeSpan(start=vision.start, end=vision.end))

        for ocr in doc.ocr:
            text = ocr.text.lower()
            for prop, keywords in typography_keywords.items():
                if any(kw in text for kw in keywords):
                    if prop not in detected:
                        detected.append(prop)
                    evidence_spans.append(TimeSpan(start=ocr.start, end=ocr.end))

        hierarchy = "strong" if "strong_hierarchy" in detected else "moderate"
        heading_style = "large" if "large_headings" in detected else "normal"
        density = "spacious" if "spacious" in detected else ("compact" if "compact" in detected else "normal")

        return TypographyProfile(
            hierarchy=hierarchy,
            heading_style=heading_style,
            density=density,
            confidence=0.6 if detected else 0.2,
            evidence=evidence_spans,
        )

    def _analyze_layout(self, doc: VideoContextDocument) -> Any:
        """Analyze layout patterns."""
        from ..schema.v1 import LayoutProfile, TimeSpan

        layout_patterns = {
            "full_width_sections": ["full width", "full-width", "edge to edge", "full bleed"],
            "hero_area": ["hero", "hero section", "banner", "landing section"],
            "card_based": ["card", "cards", "card grid", "card layout"],
            "grid_layout": ["grid", "grid layout", "grid system"],
            "sidebar": ["sidebar", "side panel", "navigation panel"],
            "navigation_bar": ["navbar", "navigation bar", "top navigation", "header navigation"],
            "footer": ["footer", "page footer"],
            "modal": ["modal", "dialog", "popup", "overlay"],
            "split_view": ["split", "split view", "two column", "multi-column"],
            "dashboard": ["dashboard", "dashboard layout", "metrics grid"],
            "centered_content": ["centered", "center-aligned", "max-width"],
        }

        detected = []
        evidence_spans = []

        for vision in doc.vision:
            desc = vision.description.lower()
            for pattern, keywords in layout_patterns.items():
                if any(kw in desc for kw in keywords):
                    if pattern not in detected:
                        detected.append(pattern)
                    evidence_spans.append(TimeSpan(start=vision.start, end=vision.end))

        return LayoutProfile(
            patterns=detected,
            confidence=0.65 if detected else 0.2,
            evidence=evidence_spans,
        )

    def _analyze_components(self, doc: VideoContextDocument) -> list[Any]:
        """Detect and analyze UI components."""
        from ..schema.v1 import ComponentProfile, TimeSpan

        component_types = {
            "navigation": ["navigation", "nav bar", "navbar", "menu", "sidebar navigation"],
            "hero": ["hero", "hero section", "banner", "landing hero"],
            "button": ["button", "btn", "cta", "call to action"],
            "card": ["card", "card component", "content card"],
            "modal": ["modal", "dialog", "popup", "overlay"],
            "sidebar": ["sidebar", "side navigation", "side panel"],
            "input": ["input", "text field", "search field", "form input"],
            "form": ["form", "form field", "form group"],
            "table": ["table", "data table", "data grid"],
            "dashboard_widget": ["widget", "metric", "kpi", "chart", "graph"],
            "footer": ["footer", "page footer"],
            "breadcrumb": ["breadcrumb", "breadcrumbs"],
            "tabs": ["tabs", "tab bar", "tab navigation"],
            "dropdown": ["dropdown", "select", "drop-down"],
            "tooltip": ["tooltip", "hint", "popover"],
            "notification": ["notification", "toast", "alert", "snackbar"],
            "avatar": ["avatar", "profile picture", "user image"],
            "badge": ["badge", "tag", "label"],
            "progress": ["progress bar", "progress indicator", "loading"],
            "accordion": ["accordion", "collapsible", "expandable"],
        }

        components = []
        seen_types = set()

        for vision in doc.vision:
            desc = vision.description.lower()
            for comp_type, keywords in component_types.items():
                if any(kw in desc for kw in keywords):
                    if comp_type not in seen_types:
                        seen_types.add(comp_type)
                        components.append(ComponentProfile(
                            component_id=f"{comp_type}_{len(components)}",
                            type=comp_type,
                            first_seen=vision.start,
                            last_seen=vision.end,
                            visual_characteristics=[],
                            content_structure=[],
                            confidence=0.7,
                            evidence=[TimeSpan(start=vision.start, end=vision.end)],
                        ))

        for ocr in doc.ocr:
            text = ocr.text.lower()
            for comp_type, keywords in component_types.items():
                if any(kw in text for kw in keywords):
                    if comp_type not in seen_types:
                        seen_types.add(comp_type)
                        components.append(ComponentProfile(
                            component_id=f"{comp_type}_{len(components)}",
                            type=comp_type,
                            first_seen=ocr.start,
                            last_seen=ocr.end,
                            visual_characteristics=[],
                            content_structure=[],
                            confidence=0.6,
                            evidence=[TimeSpan(start=ocr.start, end=ocr.end)],
                        ))

        return components

    def _analyze_interactions(self, doc: VideoContextDocument) -> Any:
        """Analyze interaction patterns."""
        from ..schema.v1 import InteractionProfile, TimeSpan

        interaction_patterns = {
            "smooth_transitions": ["smooth", "transition", "animate", "animation"],
            "hover_effects": ["hover", "hover effect", "mouse over"],
            "click_feedback": ["click", "press", "tap", "button press"],
            "scroll_behavior": ["scroll", "scrolling", "scroll behavior"],
            "drag_drop": ["drag", "drop", "drag and drop"],
            "modal_transitions": ["modal open", "modal close", "dialog open", "dialog close"],
            "page_transitions": ["page transition", "route transition", "navigation transition"],
            "loading_states": ["loading", "skeleton", "spinner", "loading state"],
            "form_validation": ["validation", "error message", "field error"],
            "toast_notifications": ["toast", "notification", "snackbar", "alert"],
        }

        detected = []
        evidence_spans = []

        for vision in doc.vision:
            desc = vision.description.lower()
            for pattern, keywords in interaction_patterns.items():
                if any(kw in desc for kw in keywords):
                    if pattern not in detected:
                        detected.append(pattern)
                    evidence_spans.append(TimeSpan(start=vision.start, end=vision.end))

        for event in doc.events:
            if event.type in ["button_clicked", "command_entered", "screen_changed", "slide_changed"]:
                pattern = event.type.replace("_", " ")
                if pattern not in detected:
                    detected.append(pattern)
                evidence_spans.append(TimeSpan(start=event.start, end=event.end))

        pattern_str = ", ".join(detected) if detected else "standard interactions"

        return InteractionProfile(
            type="ui_interaction",
            pattern=pattern_str,
            confidence=0.6 if detected else 0.3,
            evidence=evidence_spans,
        )

    def _analyze_motion(self, doc: VideoContextDocument) -> list[Any]:
        """Analyze motion/animation patterns."""
        from ..schema.v1 import MotionProfile, TimeSpan

        motion_keywords = {
            "fade": ["fade in", "fade out", "fade", "opacity"],
            "slide": ["slide in", "slide out", "slide", "translate"],
            "scale": ["scale", "zoom", "grow", "shrink"],
            "rotate": ["rotate", "rotation", "spin"],
            "sequential": ["sequential", "staggered", "cascade", "one by one"],
            "simultaneous": ["simultaneous", "together", "at once"],
            "easing": ["ease", "easing", "smooth", "spring", "bounce"],
        }

        motions = []
        seen = set()

        for vision in doc.vision:
            desc = vision.description.lower()
            for motion_type, keywords in motion_keywords.items():
                if any(kw in desc for kw in keywords):
                    key = f"{motion_type}_{vision.start:.0f}"
                    if key not in seen:
                        seen.add(key)
                        motions.append(MotionProfile(
                            motion_id=f"motion_{len(motions)}",
                            element="unknown",
                            type=motion_type,
                            direction="unknown",
                            style="unknown",
                            duration_category="unknown",
                            confidence=0.5,
                            evidence=[TimeSpan(start=vision.start, end=vision.end)],
                        ))

        for event in doc.events:
            if event.type in ["slide_changed", "screen_changed"]:
                key = f"transition_{event.start:.0f}"
                if key not in seen:
                    seen.add(key)
                    motions.append(MotionProfile(
                        motion_id=f"motion_{len(motions)}",
                        element="screen",
                        type="transition",
                        direction="unknown",
                        style="cut" if "cut" in (event.description or "").lower() else "fade",
                        duration_category="instant",
                        confidence=0.6,
                        evidence=[TimeSpan(start=event.start, end=event.end)],
                    ))

        return motions


__all__ = ["UIDesignProfileBuilder"]