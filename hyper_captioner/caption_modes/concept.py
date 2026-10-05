"""
Concept Mode contract for training specific themes, objects, and visual idea LoRAs.
"""

from typing import List, Optional

from hyper_captioner.caption_modes.base import BaseCaptionMode
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionToken,
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
    TriggerConfig,
)


class ConceptMode(BaseCaptionMode):
    """
    Mode contract for Concept, Theme, and Object LoRAs.
    Focuses on attributes defining the configured concept, interacting objects,
    materials, textures, and isolating context.
    Suppresses unrelated character identities, generic background clutter, and uncertainty.
    """

    def __init__(self, focal_concept: str = ""):
        super().__init__(
            name="concept",
            description=(
                "Focuses on a specific visual concept, theme, interacting objects, "
                "materials, and textures for training concept LoRAs."
            ),
            include_categories={
                SemanticCategory.CONCEPT,
                SemanticCategory.OBJECTS,
                SemanticCategory.MATERIAL,
                SemanticCategory.TEXTURE,
                SemanticCategory.COLOR,
            },
            exclude_categories={
                SemanticCategory.IDENTITY,
                SemanticCategory.APPEARANCE,
                SemanticCategory.UNCERTAINTY,
            },
        )
        self.focal_concept: str = focal_concept

    def build_extraction_instructions(self) -> str:
        if self.focal_concept:
            return (
                f"Focus on the visual representation of '{self.focal_concept}', including "
                "thematic visual elements, interacting objects, material characteristics, "
                "surface textures, and distinct color schemes. Suppress unrelated character "
                "identities, unrelated background noise, and uncertain visual guesses."
            )
        return (
            "Focus on the focal visual concept, thematic elements, interacting objects, "
            "material properties, surface textures, and color relationships. Suppress unrelated "
            "character identities, unrelated background noise, and uncertain visual guesses."
        )

    def filter_facts(
        self, facts: StructuredVisualFacts, trigger_cfg: TriggerConfig
    ) -> List[FactItem]:
        accepted: List[FactItem] = []
        seen_ids = set()

        for fact in facts.all_facts():
            fid = fact.id if fact.id else f"{fact.primary_category}:{fact.text}"
            if fid in seen_ids:
                continue
            seen_ids.add(fid)

            # Inviolable user locks override all exclusions
            if fact.locked:
                accepted.append(fact)
                continue

            # Strict exclusion of uncertainty
            if fact.is_uncertain or fact.primary_category == SemanticCategory.UNCERTAINTY:
                continue

            # Check category contract
            if not self.is_category_allowed(fact.primary_category):
                continue

            accepted.append(fact)

        return accepted

    def format_tokens(
        self, tokens: List[CaptionToken], format_type: CaptionFormat
    ) -> str:
        return self._default_format_tokens(tokens, format_type)
