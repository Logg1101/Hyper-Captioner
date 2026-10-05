"""
Unit tests for hyper_captioner/core/types.py.
Covers core semantic types, multi-category FactItem, TriggerConfig,
CaptionToken, StructuredVisualFacts, and validation types.
"""

import pytest
from hyper_captioner.core.types import (
    # New types
    CaptionFormat,
    CaptionModeType,
    CaptionToken,
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
    TriggerConfig,
    TriggerPlacement,
    ValidationIssue,
    ValidationReport,
    ValidationStatus,
    # Backward compatibility types
    CaptionMode,
    CaptionResult,
    ImageRecord,
    LoRAStrategy,
    PresetConfig,
    TagCategory,
    TagItem,
    VRAMMode,
    WD14Device,
)


def test_semantic_categories_enumeration():
    expected = {
        "identity",
        "appearance",
        "clothing",
        "pose",
        "expression",
        "composition",
        "camera",
        "environment",
        "lighting",
        "material",
        "rendering",
        "style",
        "objects",
        "color",
        "texture",
        "concept",
        "quality",
        "uncertainty",
    }
    actual = {c.value for c in SemanticCategory}
    assert expected == actual
    assert len(SemanticCategory) == 18


def test_caption_mode_type_enumeration():
    expected = {"character", "style", "outfit", "pose", "concept"}
    actual = {c.value for c in CaptionModeType}
    assert expected == actual


def test_caption_format_enumeration():
    expected = {"tags", "structured", "natural"}
    actual = {c.value for c in CaptionFormat}
    assert expected == actual


def test_trigger_placement_enumeration():
    expected = {"prepend", "append", "wrap", "omit"}
    actual = {c.value for c in TriggerPlacement}
    assert expected == actual


def test_trigger_config_defaults_and_custom():
    cfg = TriggerConfig(word="MyTrigger")
    assert cfg.word == "MyTrigger"
    assert cfg.placement == TriggerPlacement.PREPEND
    assert cfg.case_sensitive is False
    assert cfg.deduplicate is True
    assert cfg.absorb_stable_traits is False

    custom_cfg = TriggerConfig(
        word="AltTrigger",
        placement=TriggerPlacement.APPEND,
        case_sensitive=True,
        deduplicate=False,
        absorb_stable_traits=True,
    )
    assert custom_cfg.word == "AltTrigger"
    assert custom_cfg.placement == TriggerPlacement.APPEND
    assert custom_cfg.case_sensitive is True
    assert custom_cfg.deduplicate is False
    assert custom_cfg.absorb_stable_traits is True

    # Roundtrip serialization
    data = custom_cfg.to_dict()
    assert data["word"] == "AltTrigger"
    assert data["placement"] == "append"
    assert data["case_sensitive"] is True
    assert data["deduplicate"] is False
    assert data["absorb_stable_traits"] is True

    restored = TriggerConfig.from_dict(data)
    assert restored == custom_cfg


def test_fact_item_multi_category_and_uncertainty():
    item = FactItem(
        id="fact_01",
        text="leather jacket",
        primary_category=SemanticCategory.CLOTHING,
        categories=[SemanticCategory.CLOTHING, SemanticCategory.MATERIAL],
        confidence=0.95,
        source="joycaption",
        is_stable=False,
        is_uncertain=False,
        locked=True,
        raw_text="black leather jacket",
    )
    assert item.id == "fact_01"
    assert item.text == "leather jacket"
    assert item.primary_category == SemanticCategory.CLOTHING
    assert SemanticCategory.MATERIAL in item.categories
    assert SemanticCategory.CLOTHING in item.categories
    assert item.confidence == 0.95
    assert item.source == "joycaption"
    assert item.is_stable is False
    assert item.is_uncertain is False
    assert item.locked is True
    assert item.raw_text == "black leather jacket"


def test_fact_item_serialization_roundtrip():
    item = FactItem(
        id="fact_02",
        text="mysterious aura",
        primary_category=SemanticCategory.CONCEPT,
        categories=[SemanticCategory.CONCEPT, SemanticCategory.STYLE],
        confidence=0.88,
        source="wd14",
        is_stable=True,
        is_uncertain=True,
        locked=False,
        raw_text="glowing mysterious aura",
    )
    data = item.to_dict()
    assert data["id"] == "fact_02"
    assert data["primary_category"] == "concept"
    assert data["categories"] == ["concept", "style"]
    assert data["is_uncertain"] is True
    assert data["is_stable"] is True
    assert data["locked"] is False

    restored = FactItem.from_dict(data)
    assert restored.id == item.id
    assert restored.text == item.text
    assert restored.primary_category == SemanticCategory.CONCEPT
    assert restored.categories == [SemanticCategory.CONCEPT, SemanticCategory.STYLE]
    assert restored.confidence == pytest.approx(0.88)
    assert restored.source == "wd14"
    assert restored.is_stable is True
    assert restored.is_uncertain is True
    assert restored.locked is False
    assert restored.raw_text == "glowing mysterious aura"


def test_caption_token_traceability_and_serialization():
    token = CaptionToken(
        text="maid dress",
        primary_category=SemanticCategory.CLOTHING,
        categories=[SemanticCategory.CLOTHING],
        source_fact_ids=["fact_01", "fact_02"],
        confidence=0.92,
        transformation="direct",
    )
    assert token.text == "maid dress"
    assert token.primary_category == SemanticCategory.CLOTHING
    assert token.categories == [SemanticCategory.CLOTHING]
    assert token.source_fact_ids == ["fact_01", "fact_02"]
    assert token.confidence == 0.92
    assert token.transformation == "direct"

    data = token.to_dict()
    assert data["text"] == "maid dress"
    assert data["primary_category"] == "clothing"
    assert data["source_fact_ids"] == ["fact_01", "fact_02"]

    restored = CaptionToken.from_dict(data)
    assert restored.text == token.text
    assert restored.primary_category == SemanticCategory.CLOTHING
    assert restored.source_fact_ids == ["fact_01", "fact_02"]
    assert restored.confidence == pytest.approx(0.92)


def test_structured_visual_facts_helpers_and_roundtrip():
    fact_app = FactItem(
        id="f1",
        text="silver hair",
        primary_category=SemanticCategory.APPEARANCE,
    )
    fact_clo = FactItem(
        id="f2",
        text="school uniform",
        primary_category=SemanticCategory.CLOTHING,
    )
    fact_ren = FactItem(
        id="f3",
        text="cel shading",
        primary_category=SemanticCategory.RENDERING,
    )

    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.APPEARANCE: [fact_app],
            SemanticCategory.CLOTHING: [fact_clo],
            SemanticCategory.RENDERING: [fact_ren],
        },
        raw_response='{"appearance": ["silver hair"]}',
        parse_method="json",
        image_metadata={"width": 1024, "height": 1024},
    )

    # get_category
    assert facts.get_category(SemanticCategory.APPEARANCE) == [fact_app]
    assert facts.get_category(SemanticCategory.POSE) == []

    # filter_categories
    allowed = {SemanticCategory.APPEARANCE, SemanticCategory.RENDERING}
    filtered = facts.filter_categories(allowed)
    assert len(filtered) == 2
    filtered_texts = {f.text for f in filtered}
    assert filtered_texts == {"silver hair", "cel shading"}

    # roundtrip
    data = facts.to_dict()
    restored = StructuredVisualFacts.from_dict(data)
    assert len(restored.get_category(SemanticCategory.APPEARANCE)) == 1
    assert restored.get_category(SemanticCategory.APPEARANCE)[0].text == "silver hair"
    assert restored.parse_method == "json"
    assert restored.image_metadata["width"] == 1024


def test_validation_types():
    assert ValidationStatus.VALID.value == "valid"
    assert ValidationStatus.REPAIRED.value == "repaired"
    assert ValidationStatus.REJECTED.value == "rejected"

    issue = ValidationIssue(
        severity="WARNING",
        code="SUSPECT_TAG",
        message="Potentially subjective tag",
        token="masterpiece",
        suggested_repair="",
    )
    assert issue.severity == "WARNING"
    assert issue.code == "SUSPECT_TAG"
    assert issue.token == "masterpiece"

    report = ValidationReport(
        status=ValidationStatus.REPAIRED,
        issues=[issue],
        repaired_caption="1girl, solo, silver hair",
    )
    assert report.status == ValidationStatus.REPAIRED
    assert len(report.issues) == 1
    assert report.repaired_caption == "1girl, solo, silver hair"

    data = report.to_dict()
    assert data["status"] == "repaired"
    assert len(data["issues"]) == 1
    assert data["repaired_caption"] == "1girl, solo, silver hair"

    restored = ValidationReport.from_dict(data)
    assert restored.status == ValidationStatus.REPAIRED
    assert len(restored.issues) == 1
    assert restored.issues[0].code == "SUSPECT_TAG"
    assert restored.repaired_caption == "1girl, solo, silver hair"


def test_backward_compatibility():
    # Existing Enums and Classes should still be intact
    assert CaptionMode.TAG.value == "tag"
    assert CaptionMode.NATURAL.value == "natural"
    assert CaptionMode.HYBRID.value == "hybrid"

    assert LoRAStrategy.CHARACTER.value == "character"
    assert VRAMMode.BALANCED.value == "balanced"
    assert WD14Device.AUTO.value == "auto"

    tag = TagItem(text="1girl", source="wd14", category=TagCategory.CHARACTER)
    assert tag.text == "1girl"
    assert tag.source == "wd14"
    assert tag.category == TagCategory.CHARACTER

    cfg = PresetConfig()
    assert cfg.caption_mode == CaptionMode.HYBRID
    assert cfg.lora_strategy == LoRAStrategy.CHARACTER
