"""
Unit tests for CaptionBuilder refactoring, output formatters (TAGS, STRUCTURED, NATURAL),
token lineage, trigger word injection, category progression, and backward compatibility.
"""

import pytest

from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionMode,
    CaptionResult,
    CaptionToken,
    FactItem,
    LoRAStrategy,
    SemanticCategory,
    TagCategory,
    TagItem,
    TriggerConfig,
    TriggerPlacement,
)
from hyper_captioner.pipeline.builder import CaptionBuilder
from hyper_captioner.pipeline.semantic_filter import SemanticFilterResult


def test_builder_tag_format():
    """Test TAGS format with prepended trigger word."""
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    trigger = TriggerConfig(word="Heroine", placement=TriggerPlacement.PREPEND)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            FactItem(id="f2", text="blue eyes", primary_category=SemanticCategory.APPEARANCE),
            FactItem(id="f3", text="standing", primary_category=SemanticCategory.POSE),
        ],
        rejected=[],
    )
    caption, tokens = builder.build(mode, filter_result, trigger, CaptionFormat.TAGS)
    assert caption.startswith("Heroine, ")
    assert "1girl" in caption
    assert "blue eyes" in caption
    assert "standing" in caption
    assert len(tokens) == 4
    assert tokens[0].text == "Heroine"
    assert tokens[0].transformation == "trigger_injected"
    assert tokens[0].source_fact_ids == []
    assert tokens[0].locked is True
    assert tokens[1].source_fact_ids == ["f1"]


def test_builder_trigger_append():
    """Test trigger placed at the end with APPEND."""
    builder = CaptionBuilder()
    mode = get_caption_mode("style")
    trigger = TriggerConfig(word="AestheticV1", placement=TriggerPlacement.APPEND)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f1", text="cel shading", primary_category=SemanticCategory.RENDERING),
            FactItem(id="f2", text="vibrant colors", primary_category=SemanticCategory.COLOR),
        ],
        rejected=[],
    )
    caption, tokens = builder.build(mode, filter_result, trigger, CaptionFormat.TAGS)
    assert caption.endswith(", AestheticV1")
    assert tokens[-1].text == "AestheticV1"
    assert tokens[-1].transformation == "trigger_injected"
    assert tokens[-1].source_fact_ids == []


def test_builder_trigger_wrap():
    """Test trigger wrapped at both index 0 and end with WRAP."""
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    trigger = TriggerConfig(word="Heroine", placement=TriggerPlacement.WRAP)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            FactItem(id="f2", text="standing", primary_category=SemanticCategory.POSE),
        ],
        rejected=[],
    )
    caption, tokens = builder.build(mode, filter_result, trigger, CaptionFormat.TAGS)
    assert caption.startswith("Heroine, ")
    assert caption.endswith(", Heroine")
    assert len(tokens) == 4
    assert tokens[0].text == "Heroine"
    assert tokens[-1].text == "Heroine"
    assert tokens[0].transformation == "trigger_injected"
    assert tokens[-1].transformation == "trigger_injected"


def test_builder_trigger_omit():
    """Test trigger omitted when placement is OMIT."""
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    trigger = TriggerConfig(word="Heroine", placement=TriggerPlacement.OMIT)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            FactItem(id="f2", text="blue eyes", primary_category=SemanticCategory.APPEARANCE),
        ],
        rejected=[],
    )
    caption, tokens = builder.build(mode, filter_result, trigger, CaptionFormat.TAGS)
    assert "Heroine" not in caption
    assert not any(t.text == "Heroine" for t in tokens)
    assert len(tokens) == 2


def test_builder_trigger_deduplication():
    """Test case-insensitive deduplication when trigger word matches an accepted fact."""
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    trigger = TriggerConfig(word="Heroine", placement=TriggerPlacement.PREPEND, case_sensitive=False)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f0", text="heroine", primary_category=SemanticCategory.IDENTITY),
            FactItem(id="f1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            FactItem(id="f2", text="blue eyes", primary_category=SemanticCategory.APPEARANCE),
        ],
        rejected=[],
    )
    caption, tokens = builder.build(mode, filter_result, trigger, CaptionFormat.TAGS)
    # Duplicate 'heroine' token should be removed, leaving only the injected 'Heroine' trigger
    assert caption.count("Heroine") == 1
    assert caption.count("heroine") == 0
    heroine_tokens = [t for t in tokens if t.text.lower() == "heroine"]
    assert len(heroine_tokens) == 1
    assert heroine_tokens[0].transformation == "trigger_injected"


def test_builder_structured_format():
    """Test STRUCTURED format grouped by category blocks with semicolons."""
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    trigger = TriggerConfig(word="Heroine", placement=TriggerPlacement.PREPEND)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            FactItem(id="f2", text="blue eyes", primary_category=SemanticCategory.APPEARANCE),
            FactItem(id="f3", text="long hair", primary_category=SemanticCategory.APPEARANCE),
            FactItem(id="f4", text="sailor uniform", primary_category=SemanticCategory.CLOTHING),
            FactItem(id="f5", text="standing", primary_category=SemanticCategory.POSE),
        ],
        rejected=[],
    )
    caption, tokens = builder.build(mode, filter_result, trigger, CaptionFormat.STRUCTURED)
    assert ";" in caption
    assert "Heroine, 1girl" in caption
    assert "blue eyes, long hair" in caption
    assert "sailor uniform" in caption
    assert "standing" in caption
    assert len(tokens) == 6


def test_builder_natural_format_synthesized():
    """Test NATURAL format synthesized declarative sentence ending with a period."""
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    trigger = TriggerConfig(word="Heroine", placement=TriggerPlacement.PREPEND)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            FactItem(id="f2", text="blue eyes", primary_category=SemanticCategory.APPEARANCE),
            FactItem(id="f3", text="sailor uniform", primary_category=SemanticCategory.CLOTHING),
            FactItem(id="f4", text="standing", primary_category=SemanticCategory.POSE),
            FactItem(id="f5", text="indoors", primary_category=SemanticCategory.ENVIRONMENT),
            FactItem(id="f6", text="soft lighting", primary_category=SemanticCategory.LIGHTING),
        ],
        rejected=[],
    )
    caption, tokens = builder.build(mode, filter_result, trigger, CaptionFormat.NATURAL)
    assert caption.endswith(".")
    assert caption.startswith("Heroine, ")
    assert "1girl" in caption
    assert "blue eyes" in caption
    assert "sailor uniform" in caption
    assert "standing" in caption
    assert "indoors" in caption
    assert "soft lighting" in caption


def test_builder_natural_format_with_raw_joycaption():
    """Test NATURAL format with clean raw_joycaption and trigger merging."""
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    trigger = TriggerConfig(word="Heroine", placement=TriggerPlacement.PREPEND)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f1", text="1girl", primary_category=SemanticCategory.IDENTITY),
        ],
        rejected=[],
    )
    raw_joycaption = "A girl standing in a tranquil garden with soft lighting."
    caption, _ = builder.build(
        mode,
        filter_result,
        trigger,
        format_type=CaptionFormat.NATURAL,
        raw_joycaption=raw_joycaption,
    )
    assert caption.startswith("Heroine, ")
    assert "tranquil garden" in caption
    assert caption.endswith(".")


def test_builder_keep_underscores_toggle():
    """Test underscores replacement toggle (True = underscores, False = spaces)."""
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f1", text="blue eyes", primary_category=SemanticCategory.APPEARANCE),
            FactItem(id="f2", text="sailor uniform", primary_category=SemanticCategory.CLOTHING),
        ],
        rejected=[],
    )

    caption_spaces, tokens_spaces = builder.build(
        mode, filter_result, format_type=CaptionFormat.TAGS, keep_underscores=False
    )
    assert "blue eyes" in caption_spaces
    assert "sailor uniform" in caption_spaces
    assert "blue_eyes" not in caption_spaces
    assert tokens_spaces[0].text == "blue eyes"

    caption_underscores, tokens_underscores = builder.build(
        mode, filter_result, format_type=CaptionFormat.TAGS, keep_underscores=True
    )
    assert "blue_eyes" in caption_underscores
    assert "sailor_uniform" in caption_underscores
    assert "blue eyes" not in caption_underscores
    assert tokens_underscores[0].text == "blue_eyes"


def test_builder_token_provenance_and_lineage():
    """Test complete lineage and provenance tracking on CaptionToken objects."""
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    trigger = TriggerConfig(word="Heroine", placement=TriggerPlacement.PREPEND)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(
                id="f1",
                text="1girl",
                primary_category=SemanticCategory.IDENTITY,
                confidence=0.99,
                locked=False,
            ),
            FactItem(
                id="f2",
                text="silver hair",
                primary_category=SemanticCategory.APPEARANCE,
                confidence=0.95,
                locked=True,  # Inviolable user lock
            ),
        ],
        rejected=[],
    )
    caption, tokens = builder.build(mode, filter_result, trigger, CaptionFormat.TAGS)
    assert len(tokens) == 3

    # Trigger token
    tok_trigger = tokens[0]
    assert tok_trigger.text == "Heroine"
    assert tok_trigger.transformation == "trigger_injected"
    assert tok_trigger.source_fact_ids == []
    assert tok_trigger.confidence == 1.0
    assert tok_trigger.locked is True

    # Direct token
    tok_direct = tokens[1]
    assert tok_direct.text == "1girl"
    assert tok_direct.transformation == "direct"
    assert tok_direct.source_fact_ids == ["f1"]
    assert tok_direct.confidence == 0.99
    assert tok_direct.locked is False

    # Locked override token
    tok_locked = tokens[2]
    assert tok_locked.text == "silver hair"
    assert tok_locked.transformation == "locked_override"
    assert tok_locked.source_fact_ids == ["f2"]
    assert tok_locked.confidence == 0.95
    assert tok_locked.locked is True


def test_builder_category_progression_sorting():
    """Test logical diffusion-friendly category progression and confidence descending sorting."""
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f_light", text="soft lighting", primary_category=SemanticCategory.LIGHTING, confidence=0.8),
            FactItem(id="f_id", text="1girl", primary_category=SemanticCategory.IDENTITY, confidence=0.9),
            FactItem(id="f_cloth", text="leather jacket", primary_category=SemanticCategory.CLOTHING, confidence=0.85),
            FactItem(id="f_app2", text="blue eyes", primary_category=SemanticCategory.APPEARANCE, confidence=0.75),
            FactItem(id="f_app1", text="blonde hair", primary_category=SemanticCategory.APPEARANCE, confidence=0.95),
            FactItem(id="f_pose", text="standing", primary_category=SemanticCategory.POSE, confidence=0.9),
        ],
        rejected=[],
    )
    caption, tokens = builder.build(mode, filter_result, format_type=CaptionFormat.TAGS)
    token_texts = [t.text for t in tokens]

    # IDENTITY (10) ➔ APPEARANCE (20) ➔ CLOTHING (40) ➔ POSE (50) ➔ LIGHTING (100)
    assert token_texts[0] == "1girl"
    # Within APPEARANCE: blonde hair (0.95) comes before blue eyes (0.75)
    assert token_texts[1] == "blonde hair"
    assert token_texts[2] == "blue eyes"
    assert token_texts[3] == "leather jacket"
    assert token_texts[4] == "standing"
    assert token_texts[5] == "soft lighting"


def test_builder_legacy_backward_compatibility():
    """Test backward compatibility when called with legacy fused_tags List[TagItem]."""
    builder = CaptionBuilder()
    legacy_tags = [
        TagItem(text="1girl", source="wd14", confidence=0.98, category=TagCategory.CHARACTER),
        TagItem(text="black_hair", source="wd14", confidence=0.90, category=TagCategory.APPEARANCE),
        TagItem(text="school_uniform", source="wd14", confidence=0.85, category=TagCategory.CLOTHING),
    ]

    result = builder.build(
        fused_tags=legacy_tags,
        caption_mode=CaptionMode.HYBRID,
        lora_strategy=LoRAStrategy.CHARACTER,
        keep_underscores=False,
    )

    assert isinstance(result, CaptionResult)
    assert result.mode == CaptionMode.HYBRID
    assert "1girl" in result.caption
    assert "black hair" in result.caption
    assert "school uniform" in result.caption
    assert len(result.tags) == 3
