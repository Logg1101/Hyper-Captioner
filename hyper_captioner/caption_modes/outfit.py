"""
Outfit Mode contract for training clothing, costumes, and fashion concepts.
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


class OutfitMode(BaseCaptionMode):
    """
    Mode contract for Clothing, Costume, and Fashion LoRAs.
    Focuses on garment structure, silhouette, fabric material, texture, patterns,
    trims, collars, sleeves, fasteners, accessories, and colors.
    Suppresses character identity, unrelated background clutter, and pose details.
    """

    def __init__(self):
        super().__init__(
            name="outfit",
            description=(
                "Focuses on garment structure, silhouette, fabric materials, textures, "
                "trims, accessories, fasteners, and colors for training clothing/fashion LoRAs."
            ),
            include_categories={
                SemanticCategory.CLOTHING,
                SemanticCategory.MATERIAL,
                SemanticCategory.TEXTURE,
                SemanticCategory.COLOR,
                SemanticCategory.OBJECTS,
            },
            exclude_categories={
                SemanticCategory.IDENTITY,
                SemanticCategory.APPEARANCE,
                SemanticCategory.ENVIRONMENT,
                SemanticCategory.POSE,
                SemanticCategory.UNCERTAINTY,
            },
        )

    def build_extraction_instructions(self) -> str:
        return (
            "Focus in detail on clothing garments, costume construction, garment cuts, "
            "silhouettes, fabric materials, textile textures, trims, collars, sleeves, "
            "cuffs, fasteners, wearable accessories, and outfit colors. Disregard character "
            "facial identity traits, unrelated background environment clutter, pose nuances, "
            "and uncertain visual guesses."
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
