"""
Controlled vocabulary normalizer, synonym consolidation, and contradiction resolver.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from hyper_captioner.core.types import TagCategory, TagItem

logger = logging.getLogger(__name__)

RULES_FILE = Path(__file__).resolve().parent / "default_rules.json"

# Source Priority Ranking for resolution (higher number = higher priority)
SOURCE_PRIORITY = {
    "user": 100,
    "preset": 80,
    "florence2": 60,
    "wd14": 40,
    "joycaption": 30,
    "normalizer": 10,
    "other": 0,
}


class VocabularyNormalizer:
    """
    Ensures dataset-wide vocabulary consistency.
    Canonicalizes synonyms, resolves conflicting attributes,
    and categorizes concepts for LoRA caption building.
    """

    def __init__(self, custom_rules_path: Optional[Path] = None):
        self.synonyms: Dict[str, str] = {}
        self.blacklist: Set[str] = set()
        self.contradictions: List[Tuple[str, str]] = []
        self._load_rules(custom_rules_path)

    def _load_rules(self, custom_rules_path: Optional[Path] = None):
        # Load defaults
        if RULES_FILE.exists():
            try:
                data = json.loads(RULES_FILE.read_text(encoding="utf-8"))
                self.synonyms.update(data.get("synonyms", {}))
                self.blacklist.update([t.lower().strip() for t in data.get("blacklist", [])])
                for pair in data.get("contradictions", []):
                    if len(pair) == 2:
                        self.contradictions.append((pair[0].lower().strip(), pair[1].lower().strip()))
            except Exception as e:
                logger.error(f"Failed to load default rules from {RULES_FILE}: {e}")

        # Load custom overrides if provided
        if custom_rules_path and custom_rules_path.exists():
            try:
                custom_data = json.loads(custom_rules_path.read_text(encoding="utf-8"))
                self.synonyms.update(custom_data.get("synonyms", {}))
                self.blacklist.update([t.lower().strip() for t in custom_data.get("blacklist", [])])
                for pair in custom_data.get("contradictions", []):
                    if len(pair) == 2:
                        self.contradictions.append((pair[0].lower().strip(), pair[1].lower().strip()))
            except Exception as e:
                logger.warning(f"Failed to load custom rules from {custom_rules_path}: {e}")

    def clean_text(self, text: str) -> str:
        """Strips surrounding quotes, punctuation, and repeated spaces/underscores."""
        s = text.strip().strip("`'\"!?:;~. \t\n\r").strip("_")
        s = re.sub(r"[\s_]+", " ", s).strip()
        return s

    def normalize_tag_string(self, raw_tag: str, keep_underscores: bool = False) -> str:
        """Canonicalizes a single tag using the controlled vocabulary synonym dictionary."""
        cleaned = self.clean_text(raw_tag)
        if not cleaned:
            return ""

        lower = cleaned.lower()
        # Apply synonym replacement if present
        canonical = self.synonyms.get(lower, lower)

        if keep_underscores:
            return canonical.replace(" ", "_")
        return canonical.replace("_", " ")

    def categorize_tag(self, tag_str: str) -> TagCategory:
        """Determines the semantic attribute category for a given tag."""
        t = tag_str.lower().replace("_", " ").strip()

        # Character identity and counts
        if re.match(r"^(\d+girls?|\d+boys?|solo|multiple girls|multiple boys)$", t):
            return TagCategory.CHARACTER
        if any(w in t for w in ["female", "male", "crossdressing", "twins"]):
            return TagCategory.CHARACTER

        # Physical Appearance
        if any(t.endswith(s) for s in [" hair", " eyes", " skin"]) or any(
            w in t for w in ["ponytail", "twintails", "bangs", "braid", "ahoge", "breasts", "horns", "wings", "tail", "pointed ears", "fang", "flat chest"]
        ):
            return TagCategory.APPEARANCE

        # Expression
        if any(w in t for w in ["smile", "grin", "angry", "surprised", "sad", "smug", "blush", "closed eyes", "open mouth", "frown", "pout", "neutral", "expressionless"]):
            return TagCategory.EXPRESSION

        # Pose
        if any(w in t for w in ["standing", "sitting", "kneeling", "lying", "crouching", "walking", "running", "bending", "leaning", "floating", "crossed legs", "crossed arms"]):
            return TagCategory.POSE

        # Action
        if any(t.startswith(w) or f" {w}" in t for w in ["holding", "reading", "cleaning", "cooking", "eating", "drinking", "wiping", "reaching", "pointing", "touching", "carrying", "adjusting", "looking at viewer", "looking away"]):
            return TagCategory.ACTION

        # Clothing & Accessories
        if any(
            w in t
            for w in [
                "dress", "skirt", "uniform", "shirt", "blouse", "gloves", "stockings", "thigh-highs",
                "knee-highs", "socks", "shoes", "boots", "headdress", "hat", "jacket", "coat",
                "panties", "bra", "apron", "swimsuit", "bikini", "necklace", "earrings", "jewelry",
                "choker", "ribbon", "bow", "belt", "pants", "trousers", "shorts", "hoodie", "veil",
                "sleeves", "barefoot", "bare shoulders", "collar"
            ]
        ):
            return TagCategory.CLOTHING

        # Camera & Framing
        if any(w in t for w in ["view", "angle", "shot", "close-up", "headshot", "upper body", "medium shot", "cowboy shot", "full body", "wide shot", "portrait", "profile", "overhead", "from above", "from below"]):
            return TagCategory.FRAMING

        # Lighting
        if any(w in t for w in ["sunlight", "daylight", "moonlight", "shadows", "shadow", "backlighting", "rim light", "neon", "warm lighting", "cool lighting", "dark", "glow"]):
            return TagCategory.LIGHTING

        # Environment
        if any(
            w in t
            for w in [
                "indoors", "outdoors", "room", "bedroom", "classroom", "street", "building", "sky",
                "cloud", "forest", "trees", "ocean", "beach", "window", "wall", "floor", "table",
                "chair", "desk", "night", "day", "simple background", "scenery"
            ]
        ):
            return TagCategory.ENVIRONMENT

        # Style
        if any(w in t for w in ["anime", "manga", "illustration", "photorealistic", "semi-realistic", "monochrome", "comic", "greyscale", "sketch", "pixel art"]):
            return TagCategory.STYLE

        # Quality
        if any(w in t for w in ["masterpiece", "best quality", "highly detailed", "intricate details"]):
            return TagCategory.QUALITY

        return TagCategory.OTHER

    def is_blacklisted(self, tag_str: str) -> bool:
        t = tag_str.lower().replace("_", " ").strip()
        t_under = tag_str.lower().replace(" ", "_").strip()
        return t in self.blacklist or t_under in self.blacklist

    def filter_and_canonicalize(
        self,
        tag_items: List[TagItem],
        keep_underscores: bool = False,
        filter_poisons: bool = True,
        extra_blacklist: Optional[List[str]] = None,
    ) -> List[TagItem]:
        """
        Deduplicates, normalizes synonyms, categorizes, and filters tags.
        Preserves highest-priority and locked tags when duplicates arise.
        """
        active_blacklist = set(self.blacklist)
        if extra_blacklist:
            active_blacklist.update([t.lower().strip() for t in extra_blacklist])

        seen: Dict[str, TagItem] = {}

        for item in tag_items:
            norm_text = self.normalize_tag_string(item.text, keep_underscores=keep_underscores)
            if not norm_text:
                continue

            lookup_key = norm_text.lower().replace("_", " ")

            if filter_poisons and lookup_key in active_blacklist:
                continue

            category = item.category
            if category == TagCategory.OTHER or not category:
                category = self.categorize_tag(norm_text)

            new_item = TagItem(
                text=norm_text,
                source=item.source,
                confidence=item.confidence,
                category=category,
                locked=item.locked,
                raw_text=item.raw_text or item.text,
            )

            # Deduplication: keep locked or higher priority source
            if lookup_key in seen:
                existing = seen[lookup_key]
                if not existing.locked and (
                    new_item.locked
                    or SOURCE_PRIORITY.get(new_item.source, 0) > SOURCE_PRIORITY.get(existing.source, 0)
                    or (
                        SOURCE_PRIORITY.get(new_item.source, 0) == SOURCE_PRIORITY.get(existing.source, 0)
                        and new_item.confidence > existing.confidence
                    )
                ):
                    seen[lookup_key] = new_item
            else:
                seen[lookup_key] = new_item

        return list(seen.values())

    def resolve_contradictions(self, tags: List[TagItem]) -> List[TagItem]:
        """
        Resolves contradictory attributes (e.g., 'skirt' vs 'trousers', 'barefoot' vs 'shoes').
        Keeps locked or higher priority/confidence attribute.
        """
        current_tags = list(tags)

        for pair in self.contradictions:
            item_a = None
            item_b = None
            for item in current_tags:
                norm = item.text.lower().replace("_", " ")
                if norm == pair[0]:
                    item_a = item
                elif norm == pair[1]:
                    item_b = item

            if item_a is not None and item_b is not None:
                # Contradiction detected!
                # If one is locked, keep the locked one
                if item_a.locked and not item_b.locked:
                    current_tags.remove(item_b)
                    continue
                elif item_b.locked and not item_a.locked:
                    current_tags.remove(item_a)
                    continue

                # Compare source priority
                prio_a = SOURCE_PRIORITY.get(item_a.source, 0)
                prio_b = SOURCE_PRIORITY.get(item_b.source, 0)

                if prio_a > prio_b:
                    current_tags.remove(item_b)
                elif prio_b > prio_a:
                    current_tags.remove(item_a)
                else:
                    # Compare confidence
                    if item_a.confidence >= item_b.confidence:
                        current_tags.remove(item_b)
                    else:
                        current_tags.remove(item_a)

        return current_tags

    def prune_redundancies(self, tags: List[TagItem], keep_underscores: bool = False) -> List[TagItem]:
        """
        Prunes hierarchical redundancies (e.g. 'dress' when 'lace dress' is present,
        'jewelry' when 'earrings' is present) and resolves contradictory physical attributes
        on single subjects (e.g. 'medium hair' vs 'short hair', 'medium breasts' vs 'large breasts').
        """
        if not tags:
            return []

        # Map normalized string to tag item
        tag_map = {t.text.lower().replace("_", " ").strip(): t for t in tags}
        seen_keys = set(tag_map.keys())
        to_remove = set()

        # Check if single subject
        is_solo = any(k in seen_keys for k in ["1girl", "1boy", "solo"]) and not any(
            k in seen_keys for k in ["2girls", "2boys", "3girls", "3boys", "multiple girls", "multiple boys"]
        )

        # -------------------------------------------------------------
        # 1. Mutually Exclusive Physical Attributes (Single Subjects)
        # -------------------------------------------------------------
        if is_solo:
            # A. Breast sizes: keep single highest priority/confidence
            breast_sizes = ["flat chest", "small breasts", "medium breasts", "large breasts", "huge breasts"]
            present_breasts = [tag_map[k] for k in breast_sizes if k in tag_map]
            if len(present_breasts) > 1:
                # Sort: locked first, then source priority, then confidence
                present_breasts.sort(
                    key=lambda t: (1 if t.locked else 0, SOURCE_PRIORITY.get(t.source, 0), t.confidence),
                    reverse=True
                )
                winner = present_breasts[0]
                for loser in present_breasts[1:]:
                    if not loser.locked and loser.source != "user":
                        to_remove.add(loser.text.lower().replace("_", " ").strip())

            # B. Hair lengths: keep single highest priority/confidence
            hair_lengths = ["very short hair", "short hair", "medium hair", "long hair", "very long hair", "absurdly long hair"]
            present_lengths = [tag_map[k] for k in hair_lengths if k in tag_map]
            if len(present_lengths) > 1:
                present_lengths.sort(
                    key=lambda t: (1 if t.locked else 0, SOURCE_PRIORITY.get(t.source, 0), t.confidence),
                    reverse=True
                )
                winner = present_lengths[0]
                for loser in present_lengths[1:]:
                    if not loser.locked and loser.source != "user":
                        to_remove.add(loser.text.lower().replace("_", " ").strip())

            # C. Primary body pose: keep single highest priority/confidence
            major_poses = ["standing", "sitting", "kneeling", "lying", "crouching"]
            present_poses = [tag_map[k] for k in major_poses if k in tag_map]
            if len(present_poses) > 1:
                present_poses.sort(
                    key=lambda t: (1 if t.locked else 0, SOURCE_PRIORITY.get(t.source, 0), t.confidence),
                    reverse=True
                )
                winner = present_poses[0]
                for loser in present_poses[1:]:
                    if not loser.locked and loser.source != "user":
                        to_remove.add(loser.text.lower().replace("_", " ").strip())

            # D. Primary hair colors: if not multicolored/two-tone, keep highest
            multicolor = any(k in seen_keys for k in ["two-tone hair", "multicolored hair", "gradient hair", "streaked hair"])
            if not multicolor:
                hair_colors = [
                    "black hair", "blue hair", "brown hair", "blonde hair", "green hair",
                    "grey hair", "white hair", "pink hair", "purple hair", "red hair", "orange hair"
                ]
                present_hcolors = [tag_map[k] for k in hair_colors if k in tag_map]
                if len(present_hcolors) > 1:
                    present_hcolors.sort(
                        key=lambda t: (1 if t.locked else 0, SOURCE_PRIORITY.get(t.source, 0), t.confidence),
                        reverse=True
                    )
                    winner = present_hcolors[0]
                    for loser in present_hcolors[1:]:
                        if not loser.locked and loser.source != "user":
                            to_remove.add(loser.text.lower().replace("_", " ").strip())

        # -------------------------------------------------------------
        # 2. Hierarchical Subsumption Pruning (Parent vs Child)
        # -------------------------------------------------------------
        has_specific_dress = any(k.endswith(" dress") and k != "dress" for k in seen_keys)
        if has_specific_dress and "dress" in tag_map and not tag_map["dress"].locked:
            to_remove.add("dress")

        has_specific_skirt = any(k.endswith(" skirt") and k != "skirt" for k in seen_keys)
        if has_specific_skirt and "skirt" in tag_map and not tag_map["skirt"].locked:
            to_remove.add("skirt")

        # In solo subjects, a full dress subsumes generic skirt tags
        if is_solo and has_specific_dress:
            if "skirt" in tag_map and not tag_map["skirt"].locked:
                to_remove.add("skirt")

        has_specific_shirt = any(k.endswith(" shirt") and k != "shirt" for k in seen_keys)
        if has_specific_shirt and "shirt" in tag_map and not tag_map["shirt"].locked:
            to_remove.add("shirt")

        has_specific_jacket = any(k.endswith(" jacket") and k != "jacket" for k in seen_keys)
        if has_specific_jacket and "jacket" in tag_map and not tag_map["jacket"].locked:
            to_remove.add("jacket")

        has_specific_coat = any(k.endswith(" coat") and k != "coat" for k in seen_keys)
        if has_specific_coat and "coat" in tag_map and not tag_map["coat"].locked:
            to_remove.add("coat")

        has_specific_panties = any(k.endswith(" panties") and k != "panties" for k in seen_keys)
        if has_specific_panties:
            if "panties" in tag_map and not tag_map["panties"].locked:
                to_remove.add("panties")
            if "underwear" in tag_map and not tag_map["underwear"].locked:
                to_remove.add("underwear")

        has_specific_socks = any(k.endswith(" socks") and k != "socks" for k in seen_keys) or "thigh-highs" in seen_keys or "knee-highs" in seen_keys
        if has_specific_socks and "socks" in tag_map and not tag_map["socks"].locked:
            to_remove.add("socks")

        has_specific_stockings = "thigh-highs" in seen_keys or any(k.endswith(" stockings") and k != "stockings" for k in seen_keys)
        if has_specific_stockings and "stockings" in tag_map and not tag_map["stockings"].locked:
            to_remove.add("stockings")

        has_specific_shoes = any(k.endswith(" shoes") and k != "shoes" for k in seen_keys) or any(
            k in seen_keys for k in ["boots", "high heels", "sneakers", "sandals", "loafers"]
        )
        if has_specific_shoes:
            if "shoes" in tag_map and not tag_map["shoes"].locked:
                to_remove.add("shoes")
            if "footwear" in tag_map and not tag_map["footwear"].locked:
                to_remove.add("footwear")
            if "black footwear" in tag_map and not tag_map["black footwear"].locked and "boots" in seen_keys:
                to_remove.add("black footwear")

        # Resolve contradictory "no socks" vs pantyhose/socks
        if any(k in seen_keys for k in ["pantyhose", "black pantyhose", "brown pantyhose", "socks", "thigh-highs"]):
            if "no socks" in tag_map and not tag_map["no socks"].locked:
                to_remove.add("no socks")

        has_specific_gloves = any(k.endswith(" gloves") and k != "gloves" for k in seen_keys)
        if has_specific_gloves and "gloves" in tag_map and not tag_map["gloves"].locked:
            to_remove.add("gloves")

        has_specific_sleeves = any(k.endswith(" sleeves") and k != "sleeves" for k in seen_keys)
        if has_specific_sleeves and "sleeves" in tag_map and not tag_map["sleeves"].locked:
            to_remove.add("sleeves")

        has_specific_apron = any(k.endswith(" apron") and k != "apron" for k in seen_keys)
        if has_specific_apron and "apron" in tag_map and not tag_map["apron"].locked:
            to_remove.add("apron")

        has_specific_vest = any(k.endswith(" vest") and k != "vest" for k in seen_keys)
        if has_specific_vest and "vest" in tag_map and not tag_map["vest"].locked:
            to_remove.add("vest")

        has_specific_jewelry = any(
            k in seen_keys for k in ["earrings", "necklace", "ring", "bracelet", "choker", "pendant", "brooch"]
        ) or any(k.endswith(" necklace") or k.endswith(" earrings") for k in seen_keys)
        if has_specific_jewelry and "jewelry" in tag_map and not tag_map["jewelry"].locked:
            to_remove.add("jewelry")

        has_specific_breasts = any(
            k in seen_keys for k in ["flat chest", "small breasts", "medium breasts", "large breasts", "huge breasts"]
        )
        if has_specific_breasts and "breasts" in tag_map and not tag_map["breasts"].locked:
            to_remove.add("breasts")

        has_specific_hair = any(
            k.endswith(" hair") and k != "hair" for k in seen_keys
        ) or any(k in seen_keys for k in ["bangs", "ponytail", "twintails", "braid", "ahoge"])
        if has_specific_hair and "hair" in tag_map and not tag_map["hair"].locked:
            to_remove.add("hair")

        has_specific_eyes = any(k.endswith(" eyes") and k != "eyes" for k in seen_keys)
        if has_specific_eyes and "eyes" in tag_map and not tag_map["eyes"].locked:
            to_remove.add("eyes")

        # Flower subsumption
        has_specific_flower = any(
            k in seen_keys for k in ["blue rose", "red rose", "rose", "rose (flower)", "blue flower", "red flower", "sunflower"]
        )
        if has_specific_flower and "flower" in tag_map and not tag_map["flower"].locked:
            to_remove.add("flower")

        has_color_rose = "blue rose" in seen_keys or "red rose" in seen_keys or "black rose" in seen_keys
        if has_color_rose:
            if "rose (flower)" in tag_map and not tag_map["rose (flower)"].locked:
                to_remove.add("rose (flower)")
            if "rose" in tag_map and not tag_map["rose"].locked:
                to_remove.add("rose")

        # Night sky subsumption
        if "night sky" in seen_keys:
            if "night" in tag_map and not tag_map["night"].locked:
                to_remove.add("night")
            if "sky" in tag_map and not tag_map["sky"].locked:
                to_remove.add("sky")

        # Mouth / teeth subsumption
        if "parted lips" in seen_keys:
            if "open mouth" in tag_map and not tag_map["open mouth"].locked:
                to_remove.add("open mouth")
        if any(k in seen_keys for k in ["open mouth", "smile", "parted lips", "grin", "frown"]):
            if "teeth" in tag_map and not tag_map["teeth"].locked:
                to_remove.add("teeth")

        # Legs & Thighs
        if any(k in seen_keys for k in ["bare legs", "legs together", "crossed legs", "pantyhose", "black pantyhose", "brown pantyhose"]):
            if "legs" in tag_map and not tag_map["legs"].locked:
                to_remove.add("legs")
        if any(k in seen_keys for k in ["thigh-highs", "hand on own thigh", "bare legs"]):
            if "thighs" in tag_map and not tag_map["thighs"].locked:
                to_remove.add("thighs")

        # Tableware
        if "teacup" in seen_keys or "coffee cup" in seen_keys:
            if "cup" in tag_map and not tag_map["cup"].locked:
                to_remove.add("cup")

        # Lace
        if "lace trim" in seen_keys or "lace dress" in seen_keys:
            if "lace" in tag_map and not tag_map["lace"].locked:
                to_remove.add("lace")

        # Highleg
        if "highleg dress" in seen_keys or "highleg panties" in seen_keys:
            if "highleg" in tag_map and not tag_map["highleg"].locked:
                to_remove.add("highleg")

        # Patterned & Floral print
        if any(k in seen_keys for k in ["patterned dress", "floral print dress", "patterned skirt", "floral print skirt"]):
            if "patterned clothing" in tag_map and not tag_map["patterned clothing"].locked:
                to_remove.add("patterned clothing")

        if "floral print dress" in seen_keys:
            if "patterned dress" in tag_map and not tag_map["patterned dress"].locked:
                to_remove.add("patterned dress")

        if "floral print skirt" in seen_keys:
            if "patterned skirt" in tag_map and not tag_map["patterned skirt"].locked:
                to_remove.add("patterned skirt")

        # Filter out to_remove
        pruned_results = []
        for t in tags:
            k = t.text.lower().replace("_", " ").strip()
            if k in to_remove and not t.locked and t.source != "user":
                continue
            pruned_results.append(t)

        return pruned_results
