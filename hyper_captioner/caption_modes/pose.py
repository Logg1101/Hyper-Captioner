"""
Pose Mode contract for training posture, body position, and spatial composition LoRAs.
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


class PoseMode(BaseCaptionMode):
    """
    Mode contract for Pose, Stance, and Composition LoRAs.
    Focuses on body orientation, limb positioning, hand placement, torso angle,
    head tilt, gaze direction, stance, framing scale, and camera perspective.
    Suppresses character identity, facial hair/features, clothing details, and environment clutter.
    """

    def __init__(self):
        super().__init__(
            name="pose",
            description=(
                "Focuses on body stance, limb positioning, torso angle, head tilt, "
                "gaze direction, framing, and camera angle for training pose LoRAs."
            ),
            include_categories={
                SemanticCategory.POSE,
                SemanticCategory.COMPOSITION,
                SemanticCategory.CAMERA,
                SemanticCategory.OBJECTS,
            },
            exclude_categories={
                SemanticCategory.IDENTITY,
                SemanticCategory.APPEARANCE,
                SemanticCategory.CLOTHING,
                SemanticCategory.STYLE,
                SemanticCategory.ENVIRONMENT,
                SemanticCategory.UNCERTAINTY,
            },
        )

    def build_extraction_instructions(self) -> str:
        return (
            "Focus strictly on body position, posture, limb arrangement, arm/leg placement, "
            "torso orientation, head angle, gaze direction, shot scale (e.g., cowboy shot, "
            "full body, close-up), camera perspective (e.g., low angle, dynamic view), and "
            "any immediate physical contact props (chair, floor). Disregard character facial "
            "features, clothing items, artistic styles, environment backgrounds, and uncertain guesses."
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
