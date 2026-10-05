"""
Style Mode contract for training visual rendering and aesthetic LoRAs.
"""

from typing import List

from hyper_captioner.caption_modes.base import BaseCaptionMode
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionToken,
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
    TriggerConfig,
)


class StyleMode(BaseCaptionMode):
    """
    Mode contract for Style, Rendering, and Aesthetic LoRAs.
    Focuses on rendering methodology, shading behavior, surface material response,
    lighting physics, line quality, edge treatment, texture fidelity, and color treatment.
    Suppresses subject content (character identity, hair/eye colors, clothing items, narrative props).
    """

    def __init__(self):
        super().__init__(
            name="style",
            description=(
                "Focuses strictly on visual rendering, shading behavior, surface response, "
                "texture fidelity, lighting, and aesthetics for training style LoRAs."
            ),
            include_categories={
                SemanticCategory.RENDERING,
                SemanticCategory.STYLE,
                SemanticCategory.LIGHTING,
                SemanticCategory.MATERIAL,
                SemanticCategory.TEXTURE,
                SemanticCategory.COLOR,
                SemanticCategory.COMPOSITION,
                SemanticCategory.CAMERA,
            },
            exclude_categories={
                SemanticCategory.IDENTITY,
                SemanticCategory.APPEARANCE,
                SemanticCategory.CLOTHING,
                SemanticCategory.OBJECTS,
                SemanticCategory.UNCERTAINTY,
            },
        )

    def build_extraction_instructions(self) -> str:
        return (
            "Focus strictly on artistic style, visual rendering techniques (e.g. line art, "
            "cel shading, watercolor wash, painterly textures), shading behavior, surface "
            "material response, lighting behavior, color treatment, and compositional framing. "
            "Strictly exclude character identity traits, hair/eye color, specific clothing garments, "
            "narrative props, and uncertain visual guesses."
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
