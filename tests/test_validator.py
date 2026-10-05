"""
Tests for SemanticValidator: Quality Control & Semantic Validator with Safe Auto-Repair Boundary.
"""

import pytest
from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import (
    CaptionToken,
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
    TriggerConfig,
    TriggerPlacement,
    ValidationStatus,
)
from hyper_captioner.pipeline.validator import SemanticValidator


# ============================================================================
# 1. HARD Contradiction Tests (Mutually Exclusive Physical States -> ERROR -> REJECTED)
# ============================================================================


def test_validator_detects_hard_contradiction_standing_sitting():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    tokens = [
        CaptionToken(text="standing", primary_category=SemanticCategory.POSE),
        CaptionToken(text="sitting", primary_category=SemanticCategory.POSE),
    ]
    report = validator.validate(
        caption="standing, sitting",
        tokens=tokens,
        mode=mode,
        facts=StructuredVisualFacts({}),
        trigger_cfg=TriggerConfig(),
    )
    assert any(
        i.severity == "ERROR" and "contradiction" in i.code.lower()
        for i in report.issues
    )
    assert report.status == ValidationStatus.REJECTED


def test_validator_detects_hard_contradiction_lying_standing():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    report = validator.validate(
        caption="1girl, lying on bed, standing",
        mode=mode,
        facts=StructuredVisualFacts({}),
        trigger_cfg=TriggerConfig(),
    )
    assert any(
        i.severity == "ERROR" and "contradiction" in i.code.lower()
        for i in report.issues
    )
    assert report.status == ValidationStatus.REJECTED


def test_validator_detects_hard_contradiction_eyes_open_closed():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    report = validator.validate(
        caption="1girl, eyes open, closed eyes",
        mode=mode,
        facts=StructuredVisualFacts({}),
        trigger_cfg=TriggerConfig(),
    )
    assert any(
        i.severity == "ERROR" and "contradiction" in i.code.lower()
        for i in report.issues
    )
    assert report.status == ValidationStatus.REJECTED


def test_validator_detects_hard_contradiction_indoors_outdoors():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    report = validator.validate(
        caption="1girl, indoors, sitting in room, outdoors",
        mode=mode,
        facts=StructuredVisualFacts({}),
        trigger_cfg=TriggerConfig(),
    )
    assert any(
        i.severity == "ERROR" and "contradiction" in i.code.lower()
        for i in report.issues
    )
    assert report.status == ValidationStatus.REJECTED


def test_validator_detects_hard_contradiction_day_night():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    report = validator.validate(
        caption="sunny day, night sky, 1girl",
        mode=mode,
        facts=StructuredVisualFacts({}),
        trigger_cfg=TriggerConfig(),
    )
    assert any(
        i.severity == "ERROR" and "contradiction" in i.code.lower()
        for i in report.issues
    )
    assert report.status == ValidationStatus.REJECTED


def test_validator_detects_hard_contradiction_against_source_facts():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.POSE: [
                FactItem(id="pose_1", text="sitting", primary_category=SemanticCategory.POSE)
            ]
        }
    )
    report = validator.validate(
        caption="1girl, standing",
        mode=mode,
        facts=facts,
        trigger_cfg=TriggerConfig(),
    )
    assert any(
        i.severity == "ERROR" and "contradiction" in i.code.lower()
        for i in report.issues
    )
    assert report.status == ValidationStatus.REJECTED


# ============================================================================
# 2. POTENTIAL Contradiction Tests (Nuanced Coexisting States -> WARNING -> Retained)
# ============================================================================


def test_validator_warns_on_potential_contradiction_without_deleting():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    tokens = [
        CaptionToken(text="smiling", primary_category=SemanticCategory.EXPRESSION),
        CaptionToken(text="neutral expression", primary_category=SemanticCategory.EXPRESSION),
    ]
    report = validator.validate(
        caption="smiling, neutral expression",
        tokens=tokens,
        mode=mode,
        facts=StructuredVisualFacts({}),
        trigger_cfg=TriggerConfig(),
    )
    assert any(i.severity == "WARNING" for i in report.issues)
    assert not any(i.severity == "ERROR" for i in report.issues)
    assert "smiling" in report.repaired_caption
    assert "neutral expression" in report.repaired_caption
    assert report.status != ValidationStatus.REJECTED


def test_validator_potential_contradiction_looking_viewer_closed_eyes():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    report = validator.validate(
        caption="1girl, looking at viewer, closed eyes",
        mode=mode,
        facts=StructuredVisualFacts({}),
        trigger_cfg=TriggerConfig(),
    )
    assert any(i.severity == "WARNING" for i in report.issues)
    assert not any(i.severity == "ERROR" for i in report.issues)
    assert "looking at viewer" in report.repaired_caption
    assert "closed eyes" in report.repaired_caption


def test_validator_potential_contradiction_front_side_view():
    validator = SemanticValidator()
    mode = get_caption_mode("pose")
    report = validator.validate(
        caption="front view, side view",
        mode=mode,
        facts=StructuredVisualFacts({}),
        trigger_cfg=TriggerConfig(),
    )
    assert any(i.severity == "WARNING" for i in report.issues)
    assert not any(i.severity == "ERROR" for i in report.issues)
    assert "front view" in report.repaired_caption
    assert "side view" in report.repaired_caption


# ============================================================================
# 3. Trigger Word Deduplication & Placement Enforcement
# ============================================================================


def test_validator_safe_auto_repairs_duplicate_case_insensitive_triggers():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    trigger_cfg = TriggerConfig(word="MyChar", case_sensitive=False, placement=TriggerPlacement.PREPEND)
    raw_caption = "mychar, MYCHAR, 1girl, standing, MyChar"
    report = validator.validate(
        caption=raw_caption,
        tokens=[],
        mode=mode,
        facts=StructuredVisualFacts({}),
        trigger_cfg=trigger_cfg,
    )
    assert report.status == ValidationStatus.REPAIRED
    parts = [p.strip() for p in report.repaired_caption.split(",")]
    assert parts.count("MyChar") == 1
    assert parts[0] == "MyChar"
    assert "1girl" in parts
    assert "standing" in parts


def test_validator_trigger_placement_prepend():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    trigger_cfg = TriggerConfig(word="Heroine", placement=TriggerPlacement.PREPEND)
    raw_caption = "1girl, standing, Heroine"
    report = validator.validate(caption=raw_caption, trigger_cfg=trigger_cfg, mode=mode)
    assert report.status == ValidationStatus.REPAIRED
    assert any(i.code == "TRIGGER_REPAIRED" for i in report.issues)
    assert report.repaired_caption.startswith("Heroine, ")
    parts = [p.strip() for p in report.repaired_caption.split(",")]
    assert parts.count("Heroine") == 1


def test_validator_trigger_placement_append():
    validator = SemanticValidator()
    mode = get_caption_mode("style")
    trigger_cfg = TriggerConfig(word="AestheticV1", placement=TriggerPlacement.APPEND)
    raw_caption = "AestheticV1, cel shading, vibrant colors"
    report = validator.validate(caption=raw_caption, trigger_cfg=trigger_cfg, mode=mode)
    assert report.status == ValidationStatus.REPAIRED
    assert any(i.code == "TRIGGER_REPAIRED" for i in report.issues)
    assert report.repaired_caption.endswith(", AestheticV1")
    parts = [p.strip() for p in report.repaired_caption.split(",")]
    assert parts.count("AestheticV1") == 1


def test_validator_trigger_placement_wrap():
    validator = SemanticValidator()
    mode = get_caption_mode("concept")
    trigger_cfg = TriggerConfig(word="ConceptTag", placement=TriggerPlacement.WRAP)
    raw_caption = "1girl, floating in space"
    report = validator.validate(caption=raw_caption, trigger_cfg=trigger_cfg, mode=mode)
    assert report.status == ValidationStatus.REPAIRED
    assert any(i.code == "TRIGGER_REPAIRED" for i in report.issues)
    parts = [p.strip() for p in report.repaired_caption.split(",")]
    assert parts[0] == "ConceptTag"
    assert parts[-1] == "ConceptTag"
    assert parts.count("ConceptTag") == 2


def test_validator_trigger_placement_omit():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    trigger_cfg = TriggerConfig(word="MyChar", placement=TriggerPlacement.OMIT)
    raw_caption = "MyChar, 1girl, standing, MyChar"
    report = validator.validate(caption=raw_caption, trigger_cfg=trigger_cfg, mode=mode)
    assert report.status == ValidationStatus.REPAIRED
    assert any(i.code == "TRIGGER_REPAIRED" for i in report.issues)
    parts = [p.strip() for p in report.repaired_caption.split(",")]
    assert "MyChar" not in parts
    assert "1girl" in parts
    assert "standing" in parts


def test_validator_structured_tokens_quality_category_auto_repaired():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    tokens = [
        CaptionToken(text="masterpiece", primary_category=SemanticCategory.QUALITY, categories=[SemanticCategory.QUALITY], source_fact_ids=["f1"], confidence=0.9, transformation="direct"),
        CaptionToken(text="1girl", primary_category=SemanticCategory.IDENTITY, categories=[SemanticCategory.IDENTITY], source_fact_ids=["f2"], confidence=0.99, transformation="direct"),
        CaptionToken(text="standing", primary_category=SemanticCategory.POSE, categories=[SemanticCategory.POSE], source_fact_ids=["f3"], confidence=0.95, transformation="direct"),
    ]
    raw_caption = "masterpiece, 1girl, standing"
    report = validator.validate(caption=raw_caption, tokens=tokens, mode=mode)
    assert report.status == ValidationStatus.REPAIRED
    assert not any(i.code == "CATEGORY_LEAKAGE" for i in report.issues)
    assert any(i.code == "SUBJECTIVE_HYPE_REMOVED" for i in report.issues)
    assert report.repaired_caption == "1girl, standing"


def test_validator_case_sensitive_trigger_respects_flag():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    trigger_cfg = TriggerConfig(word="MyChar", case_sensitive=True, placement=TriggerPlacement.PREPEND)
    raw_caption = "mychar, 1girl, MyChar"
    report = validator.validate(caption=raw_caption, trigger_cfg=trigger_cfg, mode=mode)
    # Under case_sensitive=True, "mychar" is treated as distinct from "MyChar"
    parts = [p.strip() for p in report.repaired_caption.split(",")]
    assert parts[0] == "MyChar"
    assert "mychar" in parts


# ============================================================================
# 4. Auto-Repair: Subjective Hype & Quality Buzzwords
# ============================================================================


def test_validator_subjective_hype_auto_repair_standalone_and_modified():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    raw_caption = "masterpiece, breathtaking girl, stunning dress, 1girl, best quality"
    report = validator.validate(caption=raw_caption, mode=mode)
    assert report.status == ValidationStatus.REPAIRED
    parts = [p.strip() for p in report.repaired_caption.split(",")]
    assert "masterpiece" not in parts
    assert "best quality" not in parts
    assert "breathtaking" not in report.repaired_caption.lower()
    assert "stunning" not in report.repaired_caption.lower()
    # Meaningful nouns preserved
    assert any("girl" in p for p in parts)
    assert any("dress" in p for p in parts)


# ============================================================================
# 5. Inviolable User Locks (locked=True Tokens Preserved)
# ============================================================================


def test_validator_preserves_locked_tokens_even_if_hype():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.QUALITY: [
                FactItem(id="q1", text="masterpiece", primary_category=SemanticCategory.QUALITY, locked=True)
            ]
        }
    )
    tokens = [
        CaptionToken(
            text="masterpiece",
            primary_category=SemanticCategory.QUALITY,
            source_fact_ids=["q1"],
        ),
        CaptionToken(
            text="1girl",
            primary_category=SemanticCategory.IDENTITY,
        ),
    ]
    report = validator.validate(
        caption="masterpiece, 1girl",
        tokens=tokens,
        mode=mode,
        facts=facts,
    )
    # Masterpiece was locked by user, so it MUST NOT be removed
    assert "masterpiece" in report.repaired_caption


def test_validator_preserves_locked_tokens_in_style_mode():
    validator = SemanticValidator()
    mode = get_caption_mode("style")
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.APPEARANCE: [
                FactItem(id="app_1", text="pink hair", primary_category=SemanticCategory.APPEARANCE, locked=True)
            ]
        }
    )
    tokens = [
        CaptionToken(
            text="pink hair",
            primary_category=SemanticCategory.APPEARANCE,
            source_fact_ids=["app_1"],
        ),
        CaptionToken(
            text="cel shading",
            primary_category=SemanticCategory.RENDERING,
        ),
    ]
    report = validator.validate(
        caption="pink hair, cel shading",
        tokens=tokens,
        mode=mode,
        facts=facts,
    )
    # Locked user fact pink hair is NOT rejected
    assert report.status != ValidationStatus.REJECTED
    assert "pink hair" in report.repaired_caption


# ============================================================================
# 6. Mode Contract & Category Leakage Detection
# ============================================================================


def test_validator_detects_content_leak_in_style_mode():
    validator = SemanticValidator()
    mode = get_caption_mode("style")
    raw_caption = "pink hair, blue eyes, maid dress, cel shading"
    report = validator.validate(caption=raw_caption, mode=mode)
    assert any(
        i.severity == "ERROR" and "content" in i.code.lower() or "leak" in i.code.lower()
        for i in report.issues
    )
    assert report.status == ValidationStatus.REJECTED


def test_validator_allows_valid_style_rendering_in_style_mode():
    validator = SemanticValidator()
    mode = get_caption_mode("style")
    raw_caption = "clean line art, soft cel shading, diffused specular highlights"
    report = validator.validate(caption=raw_caption, mode=mode)
    assert not any(i.severity == "ERROR" for i in report.issues)
    assert report.status == ValidationStatus.VALID


# ============================================================================
# 7. Conversational Filler Prose & Punctuation Auto-Repair
# ============================================================================


def test_validator_repairs_conversational_filler_prose():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    raw_caption = "the picture shows 1girl standing in a room, here is an image of a blue sky"
    report = validator.validate(caption=raw_caption, mode=mode)
    assert report.status == ValidationStatus.REPAIRED
    assert "the picture shows" not in report.repaired_caption.lower()
    assert "here is an image of" not in report.repaired_caption.lower()


def test_validator_repairs_malformed_punctuation_and_empty_sections():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    raw_caption = ",, 1girl ,   , standing ,, sitting in chair ,,"
    # Wait, sitting and standing in chair is hard contradiction if both poses, so test with non-contradictory:
    raw_clean_tokens = ",, 1girl ,   , blue eyes ,, , standing ,,"
    report = validator.validate(caption=raw_clean_tokens, mode=mode)
    assert report.status == ValidationStatus.REPAIRED
    assert report.repaired_caption == "1girl, blue eyes, standing"


def test_validator_deduplicates_duplicate_tokens():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    raw_caption = "1girl, standing, 1girl, blue eyes, standing"
    report = validator.validate(caption=raw_caption, mode=mode)
    assert report.status == ValidationStatus.REPAIRED
    parts = [p.strip() for p in report.repaired_caption.split(",")]
    assert parts == ["1girl", "standing", "blue eyes"]


def test_validator_valid_caption_clean_status():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    raw_caption = "1girl, solo, standing, blue eyes"
    report = validator.validate(caption=raw_caption, mode=mode)
    assert report.status == ValidationStatus.VALID
    assert len(report.issues) == 0
    assert report.repaired_caption == raw_caption


def test_validator_does_not_invent_visual_facts():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    # Uncertain or removed items do not get hallucinated replacements
    raw_caption = "masterpiece, 1girl, standing"
    report = validator.validate(caption=raw_caption, mode=mode)
    assert report.repaired_caption == "1girl, standing"
    # Never invents colors, backgrounds, or items not in source
    assert "blue eyes" not in report.repaired_caption
    assert "dress" not in report.repaired_caption
