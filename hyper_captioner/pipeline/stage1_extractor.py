"""
Stage 1 Structured Visual Fact Extractor (Approach C Hybrid Dual-Parser).

Parses JoyCaption model outputs and merges auxiliary WD14 tag predictions
into a typed StructuredVisualFacts container.
Features:
1. Primary JSON Parser with inline syntax repair (trailing commas, quotes, unquoted keys).
2. Fallback Tagged Semantic Block Parser (labeled category headers).
3. Secondary Fused Fallback Parser (comma-separated heuristic classification).
4. Authoritative Uncertainty Handling (is_uncertain=True, SemanticCategory.UNCERTAINTY).
5. WD14 tag enrichment, character/general categorization, provenance, and deduplication.
"""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from hyper_captioner.core.types import (
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
)

logger = logging.getLogger(__name__)

# Category mapping from raw headers / keys to SemanticCategory enum members
CATEGORY_HEADER_MAP: Dict[str, SemanticCategory] = {
    # Identity & Subjects
    "identity": SemanticCategory.IDENTITY,
    "character": SemanticCategory.IDENTITY,
    "characters": SemanticCategory.IDENTITY,
    "subject": SemanticCategory.IDENTITY,
    "subjects": SemanticCategory.IDENTITY,
    "count": SemanticCategory.IDENTITY,
    "subject & count": SemanticCategory.IDENTITY,
    "subject and count": SemanticCategory.IDENTITY,
    # Physical Appearance
    "appearance": SemanticCategory.APPEARANCE,
    "physical": SemanticCategory.APPEARANCE,
    "physical appearance": SemanticCategory.APPEARANCE,
    "anatomy": SemanticCategory.APPEARANCE,
    "body": SemanticCategory.APPEARANCE,
    "face": SemanticCategory.APPEARANCE,
    # Clothing & Accessories
    "clothing": SemanticCategory.CLOTHING,
    "clothes": SemanticCategory.CLOTHING,
    "clothing & accessories": SemanticCategory.CLOTHING,
    "clothing and accessories": SemanticCategory.CLOTHING,
    "outfit": SemanticCategory.CLOTHING,
    "accessories": SemanticCategory.CLOTHING,
    "garment": SemanticCategory.CLOTHING,
    "garments": SemanticCategory.CLOTHING,
    # Pose & Physical Action
    "pose": SemanticCategory.POSE,
    "pose & action": SemanticCategory.POSE,
    "pose and action": SemanticCategory.POSE,
    "posture": SemanticCategory.POSE,
    "action": SemanticCategory.POSE,
    "actions": SemanticCategory.POSE,
    # Expression
    "expression": SemanticCategory.EXPRESSION,
    "expressions": SemanticCategory.EXPRESSION,
    "facial expression": SemanticCategory.EXPRESSION,
    "facial_expression": SemanticCategory.EXPRESSION,
    "emotion": SemanticCategory.EXPRESSION,
    # Composition & Framing
    "composition": SemanticCategory.COMPOSITION,
    "framing": SemanticCategory.COMPOSITION,
    "camera & framing": SemanticCategory.COMPOSITION,
    "camera and framing": SemanticCategory.COMPOSITION,
    # Camera
    "camera": SemanticCategory.CAMERA,
    "angle": SemanticCategory.CAMERA,
    "view": SemanticCategory.CAMERA,
    "perspective": SemanticCategory.CAMERA,
    # Environment & Setting
    "environment": SemanticCategory.ENVIRONMENT,
    "setting": SemanticCategory.ENVIRONMENT,
    "setting & environment": SemanticCategory.ENVIRONMENT,
    "setting and environment": SemanticCategory.ENVIRONMENT,
    "background": SemanticCategory.ENVIRONMENT,
    "location": SemanticCategory.ENVIRONMENT,
    "scenery": SemanticCategory.ENVIRONMENT,
    # Lighting
    "lighting": SemanticCategory.LIGHTING,
    "light": SemanticCategory.LIGHTING,
    "illumination": SemanticCategory.LIGHTING,
    # Material
    "material": SemanticCategory.MATERIAL,
    "materials": SemanticCategory.MATERIAL,
    # Rendering
    "rendering": SemanticCategory.RENDERING,
    "render": SemanticCategory.RENDERING,
    # Style
    "style": SemanticCategory.STYLE,
    "art style": SemanticCategory.STYLE,
    "art_style": SemanticCategory.STYLE,
    # Objects & Props
    "objects": SemanticCategory.OBJECTS,
    "object": SemanticCategory.OBJECTS,
    "props": SemanticCategory.OBJECTS,
    "items": SemanticCategory.OBJECTS,
    # Color
    "color": SemanticCategory.COLOR,
    "colors": SemanticCategory.COLOR,
    "palette": SemanticCategory.COLOR,
    # Texture
    "texture": SemanticCategory.TEXTURE,
    "textures": SemanticCategory.TEXTURE,
    # Concept
    "concept": SemanticCategory.CONCEPT,
    "concepts": SemanticCategory.CONCEPT,
    "theme": SemanticCategory.CONCEPT,
    # Quality
    "quality": SemanticCategory.QUALITY,
    # Authoritative Uncertainty
    "uncertain": SemanticCategory.UNCERTAINTY,
    "uncertainty": SemanticCategory.UNCERTAINTY,
    "ambiguous": SemanticCategory.UNCERTAINTY,
}

# Regex to match character count / demographic tags for category 4
CHARACTER_DEMOGRAPHIC_REGEX = re.compile(
    r"^(\d+girls?|\d+boys?|solo|multiple\s+girls|multiple\s+boys|no\s+humans)$",
    re.IGNORECASE,
)


class Stage1Extractor:
    """
    Approach C Hybrid Dual-Parser Visual Fact Extractor.

    Parses JoyCaption model outputs and merges auxiliary WD14 tag predictions
    into a typed StructuredVisualFacts container.
    """

    def __init__(self):
        pass

    def extract(
        self,
        joycaption_raw: str,
        wd14_tags: Optional[List[Tuple[str, float, int]]] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
    ) -> StructuredVisualFacts:
        """
        Parses raw JoyCaption output and enriches with WD14 tag predictions.

        Args:
            joycaption_raw: Raw text output from JoyCaption VLM.
            wd14_tags: Optional list of (tag_name, confidence_score, category_id) tuples.
            image_metadata: Optional dict of image metadata (width, height, etc.).

        Returns:
            StructuredVisualFacts containing categorized FactItems and provenance metadata.
        """
        facts_by_cat, parse_method = self.parse(joycaption_raw)

        facts = StructuredVisualFacts(
            facts_by_category=facts_by_cat,
            raw_response=joycaption_raw or "",
            parse_method=parse_method,
            image_metadata=image_metadata or {},
        )

        if wd14_tags:
            facts = self.enrich_with_wd14(facts, wd14_tags)

        return facts

    def parse(self, text: str) -> Tuple[Dict[SemanticCategory, List[FactItem]], str]:
        """
        Attempts Hybrid Dual-Parsing in strict priority order:
        1. Primary JSON Parser with inline syntax repair.
        2. Fallback Tagged Category Block Parser.
        3. Secondary Fused Fallback Parser (comma-separated phrases).
        """
        if not text or not text.strip() or text.strip().lower() in {"none", "n/a", "null", "unknown"}:
            return {}, "json"

        # 1. Primary: JSON Parser
        json_facts = self._parse_json(text)
        if json_facts is not None:
            return json_facts, "json"

        # 2. Fallback: Tagged Semantic Category Block Parser
        block_facts = self._parse_tagged_blocks(text)
        if block_facts is not None and any(block_facts.values()):
            return block_facts, "tagged_block"

        # 3. Secondary Fallback: Fused Comma-Separated Parser
        fused_facts = self._parse_fused_fallback(text)
        return fused_facts, "fused_fallback"

    # ========================================================================
    # 1. JSON Parser & Syntax Repair
    # ========================================================================

    def _repair_json_syntax(self, raw: str) -> str:
        """Applies regex heuristics to repair common LLM JSON syntax errors."""
        s = raw

        # Strip markdown fences if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if fence_match:
            s = fence_match.group(1).strip()

        # Extract between outermost braces
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end > start:
            s = s[start : end + 1]

        # 1. Strip trailing commas before } or ]
        s = re.sub(r",\s*([\]}])", r"\1", s)

        # 2. Quote unquoted keys: e.g. { identity: ... } or , appearance: ...
        s = re.sub(r'([{\[,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s)

        # 3. Replace single-quoted keys: 'appearance': -> "appearance":
        s = re.sub(r"([{\[,]\s*)'([a-zA-Z_][a-zA-Z0-9_]*)'\s*:", r'\1"\2":', s)

        # 4. Replace single-quoted string values:
        s = re.sub(r":\s*'([^']*)'", r': "\1"', s)
        s = re.sub(r"\[\s*'([^']*)'", r'["\1"', s)
        s = re.sub(r",\s*'([^']*)'", r', "\1"', s)

        # 5. Remove any trailing commas that might have been revealed
        s = re.sub(r",\s*([\]}])", r"\1", s)

        return s

    def _parse_json(self, text: str) -> Optional[Dict[SemanticCategory, List[FactItem]]]:
        """Attempts to deserialize JSON with inline syntax repair and literal_eval fallback."""
        s = text.strip()

        # Strip markdown fence if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if fence_match:
            s = fence_match.group(1).strip()

        # Find outermost braces
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end <= start:
            return None

        json_candidate = s[start : end + 1]

        data: Optional[Dict[str, Any]] = None

        # Attempt 1: Direct standard json.loads
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, dict):
                data = parsed
        except Exception:
            pass

        # Attempt 2: Repaired json.loads
        if data is None:
            repaired = self._repair_json_syntax(json_candidate)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                pass

        # Attempt 3: ast.literal_eval fallback (handles python-like dict syntax)
        if data is None:
            try:
                parsed = ast.literal_eval(json_candidate)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                pass

        if data is None:
            try:
                repaired = self._repair_json_syntax(json_candidate)
                parsed = ast.literal_eval(repaired)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                pass

        if data is None or not isinstance(data, dict):
            return None

        # Build categorized FactItems
        facts_by_cat: Dict[SemanticCategory, List[FactItem]] = {}
        cat_counts: Dict[SemanticCategory, int] = {}

        for raw_key, raw_val in data.items():
            category = self._resolve_category_header(raw_key)
            is_uncertain_category = (
                category == SemanticCategory.UNCERTAINTY
                or raw_key.lower().strip() in {"uncertain", "uncertainty", "ambiguous"}
            )
            if is_uncertain_category:
                category = SemanticCategory.UNCERTAINTY

            string_items = self._extract_string_items(raw_val)
            for item_text in string_items:
                item_cat = category
                if item_cat is None:
                    item_cat = self._heuristic_categorize(item_text)
                fact = self._create_fact_item(
                    text=item_text,
                    category=item_cat,
                    source="joycaption",
                    is_uncertain=is_uncertain_category or (item_cat == SemanticCategory.UNCERTAINTY),
                    raw_text=item_text,
                    cat_counts=cat_counts,
                )
                if fact:
                    facts_by_cat.setdefault(fact.primary_category, []).append(fact)

        return facts_by_cat

    # ========================================================================
    # 2. Tagged Semantic Category Block Fallback Parser
    # ========================================================================

    def _parse_tagged_blocks(self, text: str) -> Optional[Dict[SemanticCategory, List[FactItem]]]:
        """
        Extracts labeled category blocks using regex:
        e.g. [CATEGORY]: items..., CATEGORY: items..., - Category: items...
        """
        lines = text.strip().splitlines()
        facts_by_cat: Dict[SemanticCategory, List[FactItem]] = {}
        cat_counts: Dict[SemanticCategory, int] = {}
        current_category: Optional[SemanticCategory] = None
        has_matched_blocks = False

        # Pattern for matching category headers on lines:
        # Require brackets [Category] with optional colon, or unbracketed Category followed by a colon.
        header_pattern = re.compile(
            r"^\s*(?:[-*•#]{1,3}\s*)?(?:\[([A-Za-z\s&/_]+)\]\s*:?|([A-Za-z\s&/_]{3,35})\s*:\s*)(.*)$",
            re.IGNORECASE,
        )

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            match = header_pattern.match(line_str)
            resolved_cat = None
            if match:
                raw_header = (match.group(1) or match.group(2)).strip()
                resolved_cat = self._resolve_category_header(raw_header)
                if resolved_cat is not None:
                    current_category = resolved_cat
                    has_matched_blocks = True
                    content = match.group(3).strip()
                    if content:
                        is_uncertain = (
                            current_category == SemanticCategory.UNCERTAINTY
                            or raw_header.lower() in {"uncertain", "uncertainty", "ambiguous"}
                        )
                        items = [p.strip() for p in re.split(r"[,;]+", content) if p.strip()]
                        for it in items:
                            fact = self._create_fact_item(
                                text=it,
                                category=current_category,
                                source="joycaption",
                                is_uncertain=is_uncertain,
                                raw_text=it,
                                cat_counts=cat_counts,
                            )
                            if fact:
                                facts_by_cat.setdefault(fact.primary_category, []).append(fact)
                    continue

            # Multiline continuation under the active header
            if current_category is not None and resolved_cat is None:
                cleaned_line = line_str.lstrip("-*• ").strip()
                if cleaned_line:
                    is_uncertain = current_category == SemanticCategory.UNCERTAINTY
                    items = [p.strip() for p in re.split(r"[,;]+", cleaned_line) if p.strip()]
                    for it in items:
                        fact = self._create_fact_item(
                            text=it,
                            category=current_category,
                            source="joycaption",
                            is_uncertain=is_uncertain,
                            raw_text=it,
                            cat_counts=cat_counts,
                        )
                        if fact:
                            facts_by_cat.setdefault(fact.primary_category, []).append(fact)

        if has_matched_blocks and any(facts_by_cat.values()):
            return facts_by_cat

        return None

    # ========================================================================
    # 3. Secondary Fused Fallback Parser
    # ========================================================================

    def _parse_fused_fallback(self, text: str) -> Dict[SemanticCategory, List[FactItem]]:
        """Parses comma-separated phrases with heuristic category assignment."""
        clean_text = re.sub(r"```(?:[a-zA-Z0-9_-]+)?", "", text)
        parts = [p.strip() for p in re.split(r"[,;\n]+", clean_text) if p.strip()]
        facts_by_cat: Dict[SemanticCategory, List[FactItem]] = {}
        cat_counts: Dict[SemanticCategory, int] = {}

        for p in parts:
            clean = self._clean_fact_text(p)
            if not clean or len(clean) < 2 or clean.lower() in {"none", "n/a", "unknown", "unclear", "json", "markdown"}:
                continue

            cat = self._heuristic_categorize(clean)
            is_uncertain = cat == SemanticCategory.UNCERTAINTY

            fact = self._create_fact_item(
                text=clean,
                category=cat,
                source="joycaption",
                is_uncertain=is_uncertain,
                raw_text=p,
                cat_counts=cat_counts,
            )
            if fact:
                facts_by_cat.setdefault(fact.primary_category, []).append(fact)

        return facts_by_cat

    # ========================================================================
    # 4. WD14 Auxiliary Tag Grounding & Enrichment
    # ========================================================================

    def enrich_with_wd14(
        self,
        facts: StructuredVisualFacts,
        wd14_tags: List[Tuple[str, float, int]],
    ) -> StructuredVisualFacts:
        """
        Merges auxiliary WD14 tag predictions:
        - Character tags (cat_id == 4): Mapped to IDENTITY if demographic counts (1girl/solo),
          filtered out if specific unconfirmed character names.
        - General tags (cat_id == 0): Categorized via semantic keywords.
        - Prevents duplicate tags if already extracted by JoyCaption.
        - Sets source="wd14", confidence=score.
        """
        if not wd14_tags:
            return facts

        # Collect normalized texts of all existing facts to prevent duplicates
        seen_texts: Set[str] = {
            self._normalize_for_dedup(f.text) for f in facts.all_facts()
        }

        cat_counts: Dict[SemanticCategory, int] = {}
        for cat, items in facts.facts_by_category.items():
            cat_counts[cat] = len(items)

        for tag_name, score, cat_id in wd14_tags:
            clean_tag = self._clean_fact_text(tag_name)
            if not clean_tag:
                continue

            norm_tag = self._normalize_for_dedup(clean_tag)

            # Deduplication: Skip if already extracted
            if norm_tag in seen_texts:
                continue

            # Character tags (Category 4)
            if cat_id == 4:
                # Only admit general demographic counts (1girl, solo, etc.)
                if CHARACTER_DEMOGRAPHIC_REGEX.match(clean_tag):
                    category = SemanticCategory.IDENTITY
                else:
                    # Filter out unconfirmed character name
                    continue
            elif cat_id == 0:
                # General tags (Category 0)
                category = self._heuristic_categorize(clean_tag)
            else:
                # Skip artist (1), copyright (3), meta (5/9)
                continue

            fact = self._create_fact_item(
                text=clean_tag,
                category=category,
                source="wd14",
                confidence=float(score),
                is_uncertain=False,
                raw_text=tag_name,
                cat_counts=cat_counts,
            )

            if fact:
                facts.facts_by_category.setdefault(fact.primary_category, []).append(fact)
                seen_texts.add(norm_tag)

        return facts

    # ========================================================================
    # 5. Helper Methods & Cleaners
    # ========================================================================

    def _clean_fact_text(self, text: str) -> str:
        """Strips quotes, leading/trailing punctuation, underscores, and leading articles."""
        if not text:
            return ""

        s = text.strip().strip("`'\"~ \t\n\r")
        # Handle escaped quotes
        s = s.strip('\\"').strip("\\'")
        # Strip leading bullet dashes
        s = re.sub(r"^[-*•]\s*", "", s)
        # Replace underscores with spaces
        s = s.replace("_", " ")
        # Strip common leading English articles: 'a ', 'an ', 'the '
        s = re.sub(r"^(a|an|the)\s+", "", s, flags=re.IGNORECASE)
        # Collapse multiple spaces
        s = re.sub(r"\s+", " ", s).strip()
        # Strip any residual quotes
        s = s.strip("`'\"")
        return s

    def _normalize_for_dedup(self, text: str) -> str:
        """Canonical comparison string for deduplication."""
        return text.lower().replace("_", " ").strip()

    def _create_fact_item(
        self,
        text: str,
        category: SemanticCategory,
        source: str = "joycaption",
        confidence: Optional[float] = None,
        is_uncertain: bool = False,
        raw_text: str = "",
        cat_counts: Optional[Dict[SemanticCategory, int]] = None,
    ) -> Optional[FactItem]:
        """Constructs a validated, typed FactItem with provenance and unique ID."""
        cleaned = self._clean_fact_text(text)
        if not cleaned or len(cleaned) < 2 or cleaned.lower() in {"none", "n/a", "unknown", "not visible", "json", "markdown"}:
            return None
        if not re.search(r"[a-zA-Z0-9]", cleaned):
            return None

        # Fallback to OBJECTS if category is None
        if category is None:
            category = SemanticCategory.OBJECTS

        # Authoritative uncertainty check
        if category == SemanticCategory.UNCERTAINTY or is_uncertain:
            is_uncertain = True
            category = SemanticCategory.UNCERTAINTY

        if cat_counts is not None:
            cat_counts[category] = cat_counts.get(category, 0) + 1
            idx = cat_counts[category]
        else:
            idx = 1

        item_id = f"{category.value}_{idx}"

        if confidence is None:
            # 0.92 for multi-word, 0.88 for single-word
            confidence = 0.92 if len(cleaned.split()) > 1 else 0.88

        return FactItem(
            id=item_id,
            text=cleaned,
            primary_category=category,
            categories=[category],
            confidence=confidence,
            source=source,
            is_uncertain=is_uncertain,
            raw_text=raw_text or text,
        )

    def _extract_string_items(self, val: Any) -> List[str]:
        """Safely extracts a flat list of strings from parsed JSON values."""
        if isinstance(val, list):
            items: List[str] = []
            for x in val:
                if isinstance(x, str):
                    items.append(x)
                elif isinstance(x, dict):
                    items.extend(self._extract_string_items(list(x.values())))
            return items
        elif isinstance(val, str):
            parts = [p.strip() for p in val.split(",") if p.strip()]
            return parts if parts else [val]
        elif isinstance(val, dict):
            return self._extract_string_items(list(val.values()))
        return []

    def _resolve_category_header(self, header: str) -> Optional[SemanticCategory]:
        """Maps a category string or header to a SemanticCategory enum member."""
        h = header.lower().strip().strip("[]:- ")
        h_norm = re.sub(r"[\s&/_]+", " ", h).strip()

        if h in CATEGORY_HEADER_MAP:
            return CATEGORY_HEADER_MAP[h]
        if h_norm in CATEGORY_HEADER_MAP:
            return CATEGORY_HEADER_MAP[h_norm]

        for k, v in sorted(CATEGORY_HEADER_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(rf"\b{re.escape(k)}\b", h_norm):
                return v

        try:
            return SemanticCategory(h_norm)
        except ValueError:
            pass

        return None

    def _heuristic_categorize(self, text: str) -> SemanticCategory:
        """Determines semantic category based on keywords for WD14 and fallback parsing."""
        t = text.lower().strip()

        # Uncertainty indicators
        if any(w in t for w in ["unclear", "uncertain", "possibly", "maybe", "obscured", "unknown object"]):
            return SemanticCategory.UNCERTAINTY

        # Identity & Subject counts
        if CHARACTER_DEMOGRAPHIC_REGEX.match(t) or any(
            w in t for w in ["1girl", "2girls", "1boy", "2boys", "solo", "female", "male", "character"]
        ):
            return SemanticCategory.IDENTITY

        # Physical Appearance
        if any(t.endswith(s) for s in [" hair", " eyes", " skin", " horn", " horns", " tail", " tails", " wings", " ears"]) or any(
            w in t
            for w in [
                "hair", "eyes", "skin", "ponytail", "twintails", "twin tails", "bangs", "braid", "ahoge",
                "breasts", "cleavage", "horns", "wings", "tail", "pointed ears", "fang", "flat chest",
                "navel", "abs", "legs", "thighs", "feet", "arms", "hands", "face", "freckles",
                "collarbone", "bare shoulders", "bare arms", "bare legs", "barefoot"
            ]
        ):
            return SemanticCategory.APPEARANCE

        # Expression
        if any(
            w in t
            for w in [
                "smile", "grin", "angry", "surprised", "sad", "smug", "blush", "closed eyes", "open mouth",
                "frown", "pout", "neutral", "expressionless", "wink", "tears", "parted lips"
            ]
        ):
            return SemanticCategory.EXPRESSION

        # Pose & Action
        if any(
            w in t
            for w in [
                "standing", "sitting", "kneeling", "lying", "crouching", "walking", "running", "bending",
                "leaning", "floating", "crossed legs", "crossed arms", "holding", "reading", "cleaning",
                "cooking", "eating", "drinking", "wiping", "reaching", "pointing", "touching", "carrying",
                "adjusting", "looking at viewer", "looking away", "looking to the side", "head tilt",
                "arms behind back", "arms up", "hand on hip", "pose", "action"
            ]
        ):
            return SemanticCategory.POSE

        # Clothing & Garments
        if any(
            w in t
            for w in [
                "dress", "skirt", "uniform", "shirt", "blouse", "gloves", "stockings", "thigh-highs", "thighhighs",
                "knee-highs", "kneehighs", "socks", "shoes", "boots", "headdress", "hat", "jacket", "coat",
                "panties", "bra", "apron", "swimsuit", "bikini", "necklace", "earrings", "jewelry",
                "choker", "ribbon", "bow", "belt", "pants", "trousers", "shorts", "hoodie", "veil",
                "sleeves", "collar", "cape", "cloak", "robe", "kimono", "yukata", "suit", "tuxedo", "armor",
                "glasses", "sunglasses", "goggles", "sash", "footwear", "sandals", "sneakers", "heels", "loafers"
            ]
        ):
            return SemanticCategory.CLOTHING

        # Composition & Framing
        if any(
            w in t
            for w in [
                "cowboy shot", "close-up", "closeup", "headshot", "upper body", "medium shot", "full body",
                "wide shot", "portrait", "profile", "view", "angle", "dutch angle", "from above", "from below",
                "overhead", "shot", "framing", "rule of thirds"
            ]
        ):
            return SemanticCategory.COMPOSITION

        # Camera
        if any(w in t for w in ["depth of field", "bokeh", "lens flare", "motion blur", "fisheye", "focal length", "macro"]):
            return SemanticCategory.CAMERA

        # Lighting
        if any(
            w in t
            for w in [
                "sunlight", "daylight", "moonlight", "shadows", "shadow", "backlighting", "rim light",
                "neon", "warm lighting", "cool lighting", "lighting", "illumination", "light rays", "god rays", "glow"
            ]
        ):
            return SemanticCategory.LIGHTING

        # Environment
        if any(
            w in t
            for w in [
                "indoors", "outdoors", "room", "bedroom", "classroom", "street", "building", "sky",
                "cloud", "clouds", "forest", "trees", "tree", "ocean", "beach", "window", "wall", "floor",
                "table", "chair", "desk", "night", "day", "simple background", "scenery", "nature", "bookshelves", "library"
            ]
        ):
            return SemanticCategory.ENVIRONMENT

        # Objects
        if any(
            w in t
            for w in [
                "sword", "weapon", "blade", "gun", "knife", "book", "cup", "bottle", "food", "flower",
                "rose", "umbrella", "guitar", "phone", "camera", "vehicle", "car", "backpack", "bag", "staff", "wand"
            ]
        ):
            return SemanticCategory.OBJECTS

        # Style
        if any(
            w in t
            for w in [
                "anime", "manga", "illustration", "photorealistic", "semi-realistic", "monochrome", "comic",
                "greyscale", "sketch", "pixel art", "lineart", "ukiyo-e", "traditional"
            ]
        ):
            return SemanticCategory.STYLE

        # Material
        if any(w in t for w in ["leather", "silk", "metal", "metallic", "denim", "velvet", "latex", "wool", "lace"]):
            return SemanticCategory.MATERIAL

        # Rendering
        if any(w in t for w in ["cel shading", "soft shading", "ambient occlusion", "shading"]):
            return SemanticCategory.RENDERING

        # Color
        if any(w in t for w in ["monochrome", "greyscale", "sepia", "colorful", "palette"]):
            return SemanticCategory.COLOR

        # Quality buzzwords
        if any(w in t for w in ["masterpiece", "best quality", "highly detailed"]):
            return SemanticCategory.QUALITY

        # Default fallback
        return SemanticCategory.OBJECTS
