"""
Attribute Fusion Engine with category-specific source priorities and contradiction prevention.
"""

import logging
import re
from typing import List, Optional, Tuple

from hyper_captioner.core.types import CharacterConfig, TagCategory, TagItem
from hyper_captioner.vocabulary.normalizer import VocabularyNormalizer

logger = logging.getLogger(__name__)


class AttributeFusionEngine:
    """
    Fuses attributes from User input, WD14 Tagger, and JoyCaption VLM.
    Enforces category-specific source priorities:
      - Character Identity: User > WD14 > JoyCaption
      - Physical Appearance: User > WD14/JoyCaption
      - Clothing: User > WD14/JoyCaption
      - Action: User > JoyCaption > WD14
      - Pose: User > JoyCaption/WD14
      - Environment: User > JoyCaption
      - Style: User/Project Configuration > Model Inference
    """

    def __init__(self, normalizer: Optional[VocabularyNormalizer] = None):
        self.normalizer = normalizer or VocabularyNormalizer()

    def parse_joycaption_phrases(self, raw_text: str) -> List[TagItem]:
        """Extracts candidate attribute phrases from JoyCaption output, respecting section headers."""
        if not raw_text or not raw_text.strip():
            return []

        header_cat_map = {
            "subject": TagCategory.CHARACTER,
            "count": TagCategory.CHARACTER,
            "appearance": TagCategory.APPEARANCE,
            "clothing": TagCategory.CLOTHING,
            "accessories": TagCategory.CLOTHING,
            "pose": TagCategory.POSE,
            "action": TagCategory.ACTION,
            "camera": TagCategory.FRAMING,
            "framing": TagCategory.FRAMING,
            "setting": TagCategory.ENVIRONMENT,
            "environment": TagCategory.ENVIRONMENT,
            "lighting": TagCategory.LIGHTING,
            "style": TagCategory.STYLE,
        }

        items: List[TagItem] = []
        lines = raw_text.strip().splitlines()

        for line in lines:
            line_str = line.strip().strip("-* ")
            if not line_str:
                continue

            current_category = TagCategory.OTHER
            content = line_str

            # Check if line begins with a section header (e.g. "Clothing & Accessories:")
            header_match = re.match(r"^([A-Za-z\s&/]+):\s*(.*)$", line_str)
            if header_match:
                header_name = header_match.group(1).lower().strip()
                content = header_match.group(2).strip()
                for key, mapped_cat in header_cat_map.items():
                    if key in header_name:
                        current_category = mapped_cat
                        break

            # If content signifies absence or is empty, omit it completely
            if content.lower() in {"none", "n/a", "unknown", "not visible", "no humans", "unclear", "none visible", "none observable", ""}:
                continue

            # Split comma-separated values within this section
            parts = re.split(r"[,;]+", content)
            for p in parts:
                clean = self.normalizer.clean_text(p)
                if not clean or len(clean) < 2 or clean.lower() in {"none", "n/a", "none visible", "unclear"}:
                    continue

                # Strip common leading articles
                clean = re.sub(r"^(a|an|the)\s+", "", clean, flags=re.IGNORECASE)

                item_cat = current_category
                if item_cat == TagCategory.OTHER:
                    item_cat = self.normalizer.categorize_tag(clean)

                conf = 0.94 if len(clean.split()) > 1 else 0.89

                items.append(
                    TagItem(
                        text=clean,
                        source="joycaption",
                        confidence=conf,
                        category=item_cat,
                        raw_text=p.strip(),
                    )
                )

        return items

    def parse_wd14_tags(self, wd14_results: List[Tuple[str, float, int]]) -> List[TagItem]:
        """Converts WD14 prediction tuples into TagItems with provenance."""
        items: List[TagItem] = []
        for tag_name, score, cat_id in wd14_results:
            clean = tag_name.replace("_", " ").strip()
            cat = TagCategory.CHARACTER if cat_id == 4 else self.normalizer.categorize_tag(clean)
            items.append(
                TagItem(
                    text=clean,
                    source="wd14",
                    confidence=score,
                    category=cat,
                    raw_text=tag_name,
                )
            )
        return items

    def fuse(
        self,
        wd14_results: List[Tuple[str, float, int]],
        joycaption_text: str,
        user_tags: Optional[List[TagItem]] = None,
        character_config: Optional[CharacterConfig] = None,
        keep_underscores: bool = False,
        filter_poisons: bool = True,
        extra_blacklist: Optional[List[str]] = None,
    ) -> List[TagItem]:
        """
        Fuses all attribute streams into a clean, non-contradictory list of TagItems.
        """
        raw_candidates: List[TagItem] = []

        # 1. User Tags (Highest priority)
        if user_tags:
            for ut in user_tags:
                ut.source = "user"
                ut.confidence = 1.0
                raw_candidates.append(ut)

        # 2. Character configuration (Trigger word & reference)
        if character_config and character_config.trigger_word:
            trigger_item = TagItem(
                text=character_config.trigger_word.strip(),
                source="user",
                confidence=1.0,
                category=TagCategory.CHARACTER,
                locked=True,
            )
            raw_candidates.append(trigger_item)

        # 3. WD14 Tags
        wd14_items = self.parse_wd14_tags(wd14_results)

        # Handle character identity rule: If character trigger is set by user, drop unconfirmed character names from WD14
        if character_config and character_config.trigger_word:
            wd14_items = [
                item for item in wd14_items
                if not (item.category == TagCategory.CHARACTER and not re.match(r"^(\d+girls?|\d+boys?|solo)$", item.text.lower()))
            ]

        raw_candidates.extend(wd14_items)

        # 4. JoyCaption attributes
        joy_items = self.parse_joycaption_phrases(joycaption_text)
        raw_candidates.extend(joy_items)

        # 5. Filter, canonicalize synonyms, and deduplicate
        canonical_items = self.normalizer.filter_and_canonicalize(
            raw_candidates,
            keep_underscores=keep_underscores,
            filter_poisons=filter_poisons,
            extra_blacklist=extra_blacklist,
        )

        # 6. Resolve Contradictions (e.g. skirt vs pants)
        final_items = self.normalizer.resolve_contradictions(canonical_items)

        return final_items
