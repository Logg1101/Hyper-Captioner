"""
Character Mode contract for training character and identity LoRAs.
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


class CharacterMode(BaseCaptionMode):
    """
    Mode contract for Character and Identity LoRAs.
    Focuses on character identity, distinctive physical appearance, pose, framing,
    lighting context, and distinguishing clothing.
    Suppresses esoteric 3D/rendering jargon, subjective hype words, and redundant stable traits.
    """

    def __init__(self):
        super().__init__(
            name="character",
            description=(
                "Captures character identity, distinctive traits, pose, framing, "
                "lighting, and distinguishing clothing for training character/identity LoRAs."
            ),
            include_categories={
                SemanticCategory.IDENTITY,
                SemanticCategory.APPEARANCE,
                SemanticCategory.EXPRESSION,
                SemanticCategory.POSE,
                SemanticCategory.COMPOSITION,
                SemanticCategory.CAMERA,
                SemanticCategory.ENVIRONMENT,
                SemanticCategory.LIGHTING,
                SemanticCategory.CLOTHING,
            },
            exclude_categories={
                SemanticCategory.STYLE,
                SemanticCategory.RENDERING,
                SemanticCategory.QUALITY,
                SemanticCategory.UNCERTAINTY,
            },
        )

    def build_extraction_instructions(self) -> str:
        return (
            "Focus on character identity, distinctive physical appearance (hair style/color, "
            "eyes, distinctive facial features), facial expression, body pose, shot composition "
            "and camera framing, environmental context, lighting, and distinguishing clothing. "
            "Avoid esoteric rendering jargon (e.g., octane render, 3d engine tags), subjective "
            "quality hype, and uncertain visual guesses."
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

            # Absorb stable traits into trigger if configured
            if (
                trigger_cfg
                and trigger_cfg.word
                and trigger_cfg.absorb_stable_traits
                and fact.is_stable
            ):
                continue

            accepted.append(fact)

        return accepted

    def format_tokens(
        self, tokens: List[CaptionToken], format_type: CaptionFormat
    ) -> str:
        return self._default_format_tokens(tokens, format_type)
