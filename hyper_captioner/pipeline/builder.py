"""
Caption Builder supporting general-purpose semantic mode contracts,
lineage-tracked CaptionTokens, diffusion-friendly category progression,
trigger word placement, and multi-format synthesizers (TAGS, STRUCTURED, NATURAL),
while maintaining full backward compatibility for legacy training pipelines.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from hyper_captioner.caption_modes.base import BaseCaptionMode
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionMode,
    CaptionResult,
    CaptionToken,
    CharacterConfig,
    FactItem,
    LoRAStrategy,
    SemanticCategory,
    TagCategory,
    TagItem,
    TriggerConfig,
    TriggerPlacement,
)
from hyper_captioner.pipeline.semantic_filter import SemanticFilterResult

logger = logging.getLogger(__name__)

# Diffusion-friendly category progression hierarchy:
# IDENTITY (10) ➔ APPEARANCE (20) ➔ EXPRESSION (30) ➔ CLOTHING (40) ➔ POSE (50) ➔
# COMPOSITION (60) ➔ CAMERA (70) ➔ OBJECTS (80) ➔ ENVIRONMENT (90) ➔ LIGHTING (100) ➔
# MATERIAL (110) ➔ TEXTURE (120) ➔ COLOR (130) ➔ RENDERING (140) ➔ STYLE (150) ➔
# CONCEPT (160) ➔ QUALITY (170)
CATEGORY_PROGRESSION: Dict[SemanticCategory, int] = {
    SemanticCategory.IDENTITY: 10,
    SemanticCategory.APPEARANCE: 20,
    SemanticCategory.EXPRESSION: 30,
    SemanticCategory.CLOTHING: 40,
    SemanticCategory.POSE: 50,
    SemanticCategory.COMPOSITION: 60,
    SemanticCategory.CAMERA: 70,
    SemanticCategory.OBJECTS: 80,
    SemanticCategory.ENVIRONMENT: 90,
    SemanticCategory.LIGHTING: 100,
    SemanticCategory.MATERIAL: 110,
    SemanticCategory.TEXTURE: 120,
    SemanticCategory.COLOR: 130,
    SemanticCategory.RENDERING: 140,
    SemanticCategory.STYLE: 150,
    SemanticCategory.CONCEPT: 160,
    SemanticCategory.QUALITY: 170,
    SemanticCategory.UNCERTAINTY: 999,
}

# Legacy category ordering for backward compatibility
CATEGORY_ORDER = {
    TagCategory.CHARACTER: 10,
    TagCategory.APPEARANCE: 20,
    TagCategory.EXPRESSION: 30,
    TagCategory.CLOTHING: 40,
    TagCategory.POSE: 50,
    TagCategory.ACTION: 60,
    TagCategory.FRAMING: 70,
    TagCategory.CAMERA: 75,
    TagCategory.COMPOSITION: 80,
    TagCategory.OBJECTS: 90,
    TagCategory.ENVIRONMENT: 100,
    TagCategory.LIGHTING: 110,
    TagCategory.STYLE: 120,
    TagCategory.QUALITY: 130,
    TagCategory.OTHER: 140,
    TagCategory.META: 999,
}


class CaptionBuilder:
    """
    Constructs final training captions according to selected CaptionFormat
    (TAGS, STRUCTURED, NATURAL) and mode contract, or legacy CaptionMode.
    """

    def build(
        self,
        mode_or_tags: Union[BaseCaptionMode, List[TagItem], None] = None,
        filter_result: Optional[SemanticFilterResult] = None,
        trigger_cfg: Optional[TriggerConfig] = None,
        format_type: CaptionFormat = CaptionFormat.TAGS,
        keep_underscores: bool = False,
        raw_joycaption: str = "",
        execution_time: float = 0.0,
        # Legacy parameters
        caption_mode: CaptionMode = CaptionMode.HYBRID,
        lora_strategy: LoRAStrategy = LoRAStrategy.CHARACTER,
        character_config: Optional[CharacterConfig] = None,
        quality_boosters: bool = False,
        # Keyword aliases
        mode: Optional[BaseCaptionMode] = None,
        fused_tags: Optional[List[TagItem]] = None,
    ) -> Union[Tuple[str, List[CaptionToken]], CaptionResult]:
        """
        Constructs the final caption and packages either:
        - Tuple[str, List[CaptionToken]] when called with BaseCaptionMode and SemanticFilterResult.
        - CaptionResult when called with legacy fused_tags List[TagItem].
        """
        # Resolve target mode or tags
        if mode_or_tags is None:
            if mode is not None:
                mode_or_tags = mode
            elif fused_tags is not None:
                mode_or_tags = fused_tags

        # Handle legacy dispatch
        if isinstance(mode_or_tags, list) or (
            mode_or_tags is not None and not isinstance(mode_or_tags, BaseCaptionMode)
        ):
            if isinstance(filter_result, CaptionMode):
                caption_mode = filter_result
            if isinstance(trigger_cfg, LoRAStrategy):
                lora_strategy = trigger_cfg
            if isinstance(format_type, CharacterConfig):
                character_config = format_type
            return self._build_legacy(
                fused_tags=mode_or_tags or [],
                caption_mode=caption_mode,
                lora_strategy=lora_strategy,
                character_config=character_config,
                quality_boosters=quality_boosters,
                keep_underscores=keep_underscores,
                raw_joycaption=raw_joycaption,
                execution_time=execution_time,
            )

        # General-purpose mode contract pathway
        if isinstance(format_type, str):
            format_type = CaptionFormat(format_type)

        return self._build_general(
            mode=mode_or_tags,
            filter_result=filter_result,
            trigger_cfg=trigger_cfg,
            format_type=format_type,
            keep_underscores=keep_underscores,
            raw_joycaption=raw_joycaption,
            execution_time=execution_time,
        )

    def _build_general(
        self,
        mode: Optional[BaseCaptionMode],
        filter_result: Optional[SemanticFilterResult],
        trigger_cfg: Optional[TriggerConfig],
        format_type: CaptionFormat,
        keep_underscores: bool,
        raw_joycaption: str,
        execution_time: float,
    ) -> Tuple[str, List[CaptionToken]]:
        """Assembles CaptionToken objects, sorts them, and formats final caption."""
        accepted_facts = filter_result.accepted if filter_result else []

        # 1. Deduplicate accepted facts while preserving insertion order (preferring locked=True)
        unique_facts_map: Dict[str, FactItem] = {}
        for fact in accepted_facts:
            raw_t = fact.text.strip().lower()
            if not raw_t:
                continue
            if raw_t not in unique_facts_map or (fact.locked and not unique_facts_map[raw_t].locked):
                unique_facts_map[raw_t] = fact
        unique_facts = list(unique_facts_map.values())

        # 2. Token Construction & Lineage
        tokens: List[CaptionToken] = []
        for fact in unique_facts:
            text = fact.text.strip()
            if keep_underscores:
                text = text.replace(" ", "_")
            else:
                text = text.replace("_", " ")

            categories = (
                list(fact.categories) if fact.categories else [fact.primary_category]
            )
            transformation = "locked_override" if fact.locked else "direct"
            source_ids = [fact.id] if fact.id else []

            token = CaptionToken(
                text=text,
                primary_category=fact.primary_category,
                categories=categories,
                source_fact_ids=source_ids,
                confidence=fact.confidence,
                transformation=transformation,
                locked=fact.locked,
            )
            tokens.append(token)

        # 3. Logical Category Progression Ordering
        def _token_sort_key(t: CaptionToken) -> Tuple[int, float]:
            cat_order = CATEGORY_PROGRESSION.get(t.primary_category, 180)
            return (cat_order, -t.confidence)

        tokens.sort(key=_token_sort_key)

        # 4. Trigger Word Injection & Deduplication
        if trigger_cfg and trigger_cfg.word and trigger_cfg.word.strip():
            raw_trigger = trigger_cfg.word.strip()
            trigger_text = (
                raw_trigger.replace(" ", "_")
                if keep_underscores
                else raw_trigger.replace("_", " ")
            )

            placement = (
                trigger_cfg.placement
                if isinstance(trigger_cfg.placement, TriggerPlacement)
                else TriggerPlacement(str(trigger_cfg.placement))
            )

            if placement != TriggerPlacement.OMIT:
                # Remove duplicate tokens matching trigger word
                if trigger_cfg.case_sensitive:
                    tokens = [
                        t
                        for t in tokens
                        if t.text.strip() != trigger_text
                        and t.text.strip() != raw_trigger
                    ]
                else:
                    tokens = [
                        t
                        for t in tokens
                        if t.text.strip().lower() != trigger_text.lower()
                        and t.text.strip().lower() != raw_trigger.lower()
                    ]

                def _make_trigger_token() -> CaptionToken:
                    return CaptionToken(
                        text=trigger_text,
                        primary_category=SemanticCategory.IDENTITY,
                        categories=[SemanticCategory.IDENTITY],
                        source_fact_ids=[],
                        confidence=1.0,
                        transformation="trigger_injected",
                        locked=True,
                    )

                if placement == TriggerPlacement.PREPEND:
                    tokens = [_make_trigger_token()] + tokens
                elif placement == TriggerPlacement.APPEND:
                    tokens = tokens + [_make_trigger_token()]
                elif placement == TriggerPlacement.WRAP:
                    tokens = [_make_trigger_token()] + tokens + [_make_trigger_token()]

        # 5. Output Formatters
        if format_type == CaptionFormat.STRUCTURED:
            caption_text = self._format_structured(tokens)
        elif format_type == CaptionFormat.NATURAL:
            caption_text = self._format_natural(tokens, raw_joycaption, trigger_cfg)
        else:
            # Default: CaptionFormat.TAGS
            caption_text = self._format_tags(tokens)

        return caption_text, tokens

    def _format_tags(self, tokens: List[CaptionToken]) -> str:
        """Comma-separated tag list."""
        return ", ".join(t.text.strip() for t in tokens if t.text and t.text.strip())

    def _format_structured(self, tokens: List[CaptionToken]) -> str:
        """
        Information-dense grouped phrasing preserving category blocks
        or semicolon-separated clauses.
        """
        if not tokens:
            return ""

        clauses: List[str] = []
        current_clause: List[str] = []
        current_cat: Optional[SemanticCategory] = None

        for t in tokens:
            text = t.text.strip()
            if not text:
                continue
            cat = t.primary_category
            if current_cat is None:
                current_cat = cat
                current_clause.append(text)
            elif cat == current_cat:
                current_clause.append(text)
            else:
                clauses.append(", ".join(current_clause))
                current_clause = [text]
                current_cat = cat

        if current_clause:
            clauses.append(", ".join(current_clause))

        return "; ".join(clauses)

    def _format_natural(
        self,
        tokens: List[CaptionToken],
        raw_joycaption: str,
        trigger_cfg: Optional[TriggerConfig],
    ) -> str:
        """
        Visually grounded declarative sentence ending with a period without subjective filler prose.
        If raw_joycaption is present and clean, trigger can be merged naturally.
        """
        clean_joy = raw_joycaption.strip() if raw_joycaption else ""
        is_clean_prose = (
            bool(clean_joy)
            and not clean_joy.startswith(("[", "{", "```"))
            and len(clean_joy) > 5
        )

        trigger_word = (
            trigger_cfg.word.strip() if (trigger_cfg and trigger_cfg.word) else ""
        )
        placement = (
            trigger_cfg.placement
            if (trigger_cfg and isinstance(trigger_cfg.placement, TriggerPlacement))
            else (
                TriggerPlacement(str(trigger_cfg.placement))
                if trigger_cfg
                else TriggerPlacement.PREPEND
            )
        )

        if is_clean_prose:
            prose = clean_joy
            if trigger_word and placement != TriggerPlacement.OMIT:
                case_sensitive = trigger_cfg.case_sensitive if trigger_cfg else False
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = rf"\b{re.escape(trigger_word)}\b"
                has_trigger = bool(re.search(pattern, prose, flags=flags))

                if placement == TriggerPlacement.PREPEND:
                    if not has_trigger:
                        prose = f"{trigger_word}, {prose}"
                elif placement == TriggerPlacement.APPEND:
                    if not has_trigger:
                        if prose.endswith("."):
                            prose = prose[:-1].rstrip()
                        prose = f"{prose}, {trigger_word}"
                elif placement == TriggerPlacement.WRAP:
                    if not has_trigger:
                        if prose.endswith("."):
                            prose = prose[:-1].rstrip()
                        prose = f"{trigger_word}, {prose}, {trigger_word}"

            if not prose.endswith("."):
                prose += "."
            return prose

        # Fallback / Direct synthesis from tokens
        if not tokens:
            return ""

        # Separate trigger tokens injected at extremities if present
        tokens_to_process = list(tokens)
        prepend_trigger = ""
        append_trigger = ""

        if (
            tokens_to_process
            and tokens_to_process[0].transformation == "trigger_injected"
            and placement in (TriggerPlacement.PREPEND, TriggerPlacement.WRAP)
        ):
            prepend_trigger = tokens_to_process[0].text
            tokens_to_process = tokens_to_process[1:]

        if (
            tokens_to_process
            and tokens_to_process[-1].transformation == "trigger_injected"
            and placement in (TriggerPlacement.APPEND, TriggerPlacement.WRAP)
        ):
            append_trigger = tokens_to_process[-1].text
            tokens_to_process = tokens_to_process[:-1]

        # Group tokens by semantic category
        by_cat: Dict[SemanticCategory, List[str]] = {}
        for t in tokens_to_process:
            txt = t.text.strip()
            if not txt:
                continue
            by_cat.setdefault(t.primary_category, []).append(txt)

        identity = by_cat.get(SemanticCategory.IDENTITY, [])
        appearance = by_cat.get(SemanticCategory.APPEARANCE, [])
        expression = by_cat.get(SemanticCategory.EXPRESSION, [])
        clothing = by_cat.get(SemanticCategory.CLOTHING, [])
        pose = by_cat.get(SemanticCategory.POSE, [])
        composition = by_cat.get(SemanticCategory.COMPOSITION, []) + by_cat.get(
            SemanticCategory.CAMERA, []
        )
        environment = by_cat.get(SemanticCategory.ENVIRONMENT, [])
        lighting = by_cat.get(SemanticCategory.LIGHTING, [])

        other_cats = [
            SemanticCategory.OBJECTS,
            SemanticCategory.MATERIAL,
            SemanticCategory.TEXTURE,
            SemanticCategory.COLOR,
            SemanticCategory.RENDERING,
            SemanticCategory.STYLE,
            SemanticCategory.CONCEPT,
            SemanticCategory.QUALITY,
        ]
        other_tokens: List[str] = []
        for cat in other_cats:
            other_tokens.extend(by_cat.get(cat, []))

        # Check if subject/person context exists
        has_subject = bool(identity or appearance or clothing or pose or expression)

        clauses: List[str] = []
        if has_subject:
            if identity:
                subject = ", ".join(identity)
            else:
                subject = "A character"

            if appearance:
                app_str = "with " + ", ".join(appearance)
                subject = f"{subject} {app_str}"

            if expression:
                subject = f"{subject}, {', '.join(expression)}"

            clauses.append(subject)

            if clothing:
                clauses.append("wearing " + ", ".join(clothing))
            if pose:
                clauses.append(", ".join(pose))
            if composition:
                clauses.append(", ".join(composition))
            if environment:
                env_str = ", ".join(environment)
                if not any(
                    env_str.lower().startswith(p)
                    for p in ("in ", "at ", "on ", "indoors", "outdoors")
                ):
                    clauses.append("in " + env_str)
                else:
                    clauses.append(env_str)
            if lighting:
                light_str = ", ".join(lighting)
                if not any(
                    light_str.lower().startswith(p) for p in ("with ", "under ", "in ")
                ):
                    clauses.append("with " + light_str)
                else:
                    clauses.append(light_str)
            if other_tokens:
                clauses.append(", ".join(other_tokens))
        else:
            # Non-character context (e.g. style, environment, concept, object mode)
            all_texts = [t.text.strip() for t in tokens_to_process if t.text.strip()]
            if all_texts:
                clauses.append(", ".join(all_texts))

        body = ", ".join(clauses) if clauses else ""

        # Stitch triggers
        if prepend_trigger and body:
            body = f"{prepend_trigger}, {body}"
        elif prepend_trigger and not body:
            body = prepend_trigger

        if append_trigger and body:
            body = f"{body}, {append_trigger}"
        elif append_trigger and not body:
            body = append_trigger

        body = body.strip()
        if body and not body.endswith("."):
            body += "."

        return body

    # ========================================================================
    # Legacy Pipeline Implementation
    # ========================================================================

    def _build_legacy(
        self,
        fused_tags: List[TagItem],
        caption_mode: CaptionMode = CaptionMode.HYBRID,
        lora_strategy: LoRAStrategy = LoRAStrategy.CHARACTER,
        character_config: Optional[CharacterConfig] = None,
        quality_boosters: bool = False,
        keep_underscores: bool = False,
        raw_joycaption: str = "",
        execution_time: float = 0.0,
    ) -> CaptionResult:
        """Constructs legacy CaptionResult for backward compatibility."""
        tags = list(fused_tags)

        # 1. Filter attributes based on LoRA Strategy
        if lora_strategy == LoRAStrategy.STYLE:
            tags = [
                t
                for t in tags
                if t.category not in {TagCategory.APPEARANCE, TagCategory.CLOTHING}
                or t.source == "user"
                or t.locked
            ]
        elif lora_strategy == LoRAStrategy.CONCEPT:
            tags = [
                t
                for t in tags
                if t.category != TagCategory.APPEARANCE
                or t.source == "user"
                or t.locked
            ]
        elif lora_strategy == LoRAStrategy.CHARACTER and character_config:
            if (
                character_config.prune_reference_from_caption
                and character_config.reference_description
            ):
                ref_parts = {
                    p.strip().lower().replace("_", " ")
                    for p in character_config.reference_description.split(",")
                    if p.strip()
                }
                tags = [
                    t
                    for t in tags
                    if t.text.lower().replace("_", " ") not in ref_parts or t.locked
                ]

        # 2. Add Quality boosters if enabled
        if quality_boosters and lora_strategy in {
            LoRAStrategy.CHARACTER,
            LoRAStrategy.GENERAL,
        }:
            for q_tag in ["masterpiece", "best quality"]:
                if not any(t.text.lower() == q_tag for t in tags):
                    tags.append(
                        TagItem(
                            text=q_tag,
                            source="preset",
                            confidence=1.0,
                            category=TagCategory.QUALITY,
                        )
                    )

        # Standardize tag text casing/underscores
        for t in tags:
            if keep_underscores:
                t.text = t.text.replace(" ", "_")
            else:
                t.text = t.text.replace("_", " ")

        # 3. Format according to Caption Mode
        if caption_mode == CaptionMode.NATURAL:
            caption_text = self._build_legacy_natural_caption(
                tags, raw_joycaption, character_config
            )
        elif caption_mode == CaptionMode.TAG:
            caption_text = self._build_legacy_tag_caption(
                tags, character_config, keep_underscores=keep_underscores
            )
        else:
            caption_text = self._build_legacy_hybrid_caption(
                tags, character_config, keep_underscores=keep_underscores
            )

        return CaptionResult(
            caption=caption_text,
            mode=caption_mode,
            tags=tags,
            raw_joycaption=raw_joycaption,
            execution_time=execution_time,
            metadata={
                "lora_strategy": (
                    lora_strategy.value
                    if isinstance(lora_strategy, LoRAStrategy)
                    else str(lora_strategy)
                ),
                "character_trigger": (
                    character_config.trigger_word if character_config else ""
                ),
                "keep_underscores": keep_underscores,
            },
        )

    def _sort_legacy_tags(
        self, tags: List[TagItem], character_config: Optional[CharacterConfig]
    ) -> List[TagItem]:
        """Sorts legacy tags into LoRA-optimal progression."""
        trigger = (
            character_config.trigger_word.lower().strip()
            if (character_config and character_config.trigger_word)
            else ""
        )

        def sort_key(item: TagItem):
            is_trigger = (
                0 if (trigger and item.text.lower().strip() == trigger) else 1
            )
            cat_order = CATEGORY_ORDER.get(item.category, 140)
            return (is_trigger, cat_order, -item.confidence)

        return sorted(tags, key=sort_key)

    def _build_legacy_hybrid_caption(
        self,
        tags: List[TagItem],
        character_config: Optional[CharacterConfig],
        keep_underscores: bool = False,
    ) -> str:
        sorted_items = self._sort_legacy_tags(tags, character_config)
        tag_texts = [
            (
                item.text.replace(" ", "_")
                if keep_underscores
                else item.text.replace("_", " ")
            )
            for item in sorted_items
        ]
        return ", ".join(tag_texts)

    def _build_legacy_tag_caption(
        self,
        tags: List[TagItem],
        character_config: Optional[CharacterConfig],
        keep_underscores: bool = False,
    ) -> str:
        sorted_items = self._sort_legacy_tags(tags, character_config)
        tag_texts = [
            (
                item.text.replace(" ", "_")
                if keep_underscores
                else item.text.replace("_", " ")
            )
            for item in sorted_items
        ]
        return ", ".join(tag_texts)

    def _build_legacy_natural_caption(
        self,
        tags: List[TagItem],
        raw_joycaption: str,
        character_config: Optional[CharacterConfig],
    ) -> str:
        trigger = character_config.trigger_word.strip() if character_config else ""

        if raw_joycaption and not raw_joycaption.startswith("["):
            prose = raw_joycaption.strip()
            if trigger and trigger.lower() not in prose.lower():
                return f"{trigger}, {prose}"
            return prose

        sorted_items = self._sort_legacy_tags(tags, character_config)
        tag_texts = [t.text for t in sorted_items]
        return ", ".join(tag_texts)
