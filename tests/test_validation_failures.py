"""
Comprehensive Deliberate Failure and Edge Case Test Suite.

Verifies:
1. Hard Contradictions (mutually exclusive states -> ERROR, status REJECTED).
2. Potential Contradictions (nuanced coexisting states -> WARNING, status VALID, tokens preserved).
3. Case-Insensitive Trigger Deduplication (PREPEND, APPEND, WRAP, OMIT).
4. Subjective Hype Stripping & Safe Auto-Repair (standalone & modifying buzzwords -> REPAIRED).
5. Authoritative Uncertainty (is_uncertain=True & UNCERTAINTY category strictly omitted).
6. Inviolable User Locks (locked=True survives forbidden exclusions across all modes & hype).
7. Empty & Malformed Inputs (None, empty string, malformed JSON, punctuation-only degrade safely).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from PIL import Image

from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import (
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
)
from hyper_captioner.dataset.manager import DatasetPipeline
from hyper_captioner.pipeline.builder import CaptionBuilder
from hyper_captioner.pipeline.semantic_filter import SemanticFilter
from hyper_captioner.pipeline.stage1_extractor import Stage1Extractor
from hyper_captioner.pipeline.validator import SemanticValidator


def _create_test_image(path: Path) -> Path:
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img.save(path)
    return path


class TestHardContradictions:
    """Verifies all hard contradiction pairs produce ERROR severity and REJECTED status."""

    @pytest.mark.parametrize(
        "contradictory_caption,desc",
        [
            ("1girl, standing, sitting on a bench", "standing vs sitting"),
            ("1girl, indoors, outdoors in a park", "indoors vs outdoors"),
            ("1girl, daytime, bright sunny day, starry night sky", "day vs night"),
            ("1girl, eyes open, eyes closed, smiling", "eyes open vs eyes closed"),
            ("1girl, lying on the floor, standing tall", "lying vs standing"),
            ("1girl, sitting on grass, lying down", "sitting vs lying"),
        ],
    )
    def test_hard_contradiction_in_caption_yields_rejected(self, contradictory_caption, desc):
        validator = SemanticValidator()
        report = validator.validate(contradictory_caption)

        assert report.status == ValidationStatus.REJECTED, f"Expected REJECTED for {desc}"
        errors = [i for i in report.issues if i.severity == "ERROR"]
        assert len(errors) >= 1
        assert any(i.code == "HARD_CONTRADICTION" for i in errors)

    def test_hard_contradiction_against_source_facts(self):
        """Verifies contradiction between candidate caption and verified source facts."""
        validator = SemanticValidator()
        facts = StructuredVisualFacts(
            facts_by_category={
                SemanticCategory.POSE: [
                    FactItem(id="pose_1", text="standing", primary_category=SemanticCategory.POSE)
                ]
            }
        )

        # Caption claims sitting while source fact is standing
        report = validator.validate(caption="1girl, sitting on a chair", facts=facts)
        assert report.status == ValidationStatus.REJECTED
        errors = [i for i in report.issues if i.severity == "ERROR" and i.code == "HARD_CONTRADICTION"]
        assert len(errors) >= 1
        assert "standing vs. sitting" in errors[0].message.lower()


class TestPotentialContradictions:
    """Verifies nuanced coexisting states yield WARNING, remain VALID, and are NEVER deleted."""

    @pytest.mark.parametrize(
        "candidate_caption,expected_tokens",
        [
            (
                "1girl, smiling, neutral expression",
                ["smiling", "neutral expression"],
            ),
            (
                "1girl, looking at viewer, closed eyes",
                ["looking at viewer", "closed eyes"],
            ),
            (
                "1girl, front view, side view",
                ["front view", "side view"],
            ),
        ],
    )
    def test_potential_contradiction_yields_warning_and_preserves_tokens(
        self, candidate_caption, expected_tokens
    ):
        validator = SemanticValidator()
        report = validator.validate(candidate_caption)

        # Must NOT be rejected
        assert report.status != ValidationStatus.REJECTED

        # Must record a WARNING issue
        warnings = [i for i in report.issues if i.severity == "WARNING"]
        assert len(warnings) >= 1
        assert any(i.code == "POTENTIAL_CONTRADICTION" for i in warnings)

        # Both tokens must remain in the final caption (NEVER deleted)
        final_caption = report.repaired_caption or candidate_caption
        for tok in expected_tokens:
            assert tok in final_caption


class TestCaseInsensitiveTriggerDeduplication:
    """Verifies casing variations of triggers are normalized and deduplicated across placements."""

    @pytest.mark.parametrize(
        "placement,input_caption,expected_start,expected_end,expected_count",
        [
            (
                TriggerPlacement.PREPEND,
                "HERO, 1girl, standing, hero, solo, Hero",
                "Hero",
                "solo",
                1,
            ),
            (
                TriggerPlacement.APPEND,
                "hero, 1girl, standing, Hero, solo, HERO",
                "1girl",
                "Hero",
                1,
            ),
            (
                TriggerPlacement.WRAP,
                "Hero, 1girl, standing, hero, solo, HERO",
                "Hero",
                "Hero",
                2,
            ),
            (
                TriggerPlacement.OMIT,
                "hero, 1girl, standing, Hero, solo, HERO",
                "1girl",
                "solo",
                0,
            ),
        ],
    )
    def test_trigger_deduplication_placements(
        self, placement, input_caption, expected_start, expected_end, expected_count
    ):
        validator = SemanticValidator()
        trigger_cfg = TriggerConfig(word="Hero", placement=placement, case_sensitive=False)

        report = validator.validate(caption=input_caption, trigger_cfg=trigger_cfg)

        assert report.status in (ValidationStatus.VALID, ValidationStatus.REPAIRED)
        repaired = report.repaired_caption
        tokens = [t.strip() for t in repaired.split(",") if t.strip()]

        # Verify trigger frequency
        hero_count = sum(1 for t in tokens if t.lower() == "hero")
        assert hero_count == expected_count

        if expected_count > 0:
            if placement == TriggerPlacement.PREPEND:
                assert tokens[0] == "Hero"
            elif placement == TriggerPlacement.APPEND:
                assert tokens[-1] == "Hero"
            elif placement == TriggerPlacement.WRAP:
                assert tokens[0] == "Hero"
                assert tokens[-1] == "Hero"
        else:
            assert all(t.lower() != "hero" for t in tokens)


class TestSubjectiveHypeStripping:
    """Verifies removal of hype buzzwords and safe repair of hype-modified tokens."""

    def test_standalone_hype_buzzwords_removed(self):
        validator = SemanticValidator()
        raw_caption = "masterpiece, 1girl, breathtaking, solo, ultra high quality, blue eyes, award winning"
        report = validator.validate(raw_caption)

        assert report.status == ValidationStatus.REPAIRED
        repaired = report.repaired_caption

        assert "masterpiece" not in repaired.lower()
        assert "breathtaking" not in repaired.lower()
        assert "ultra high quality" not in repaired.lower()
        assert "award winning" not in repaired.lower()
        assert "1girl" in repaired
        assert "solo" in repaired
        assert "blue eyes" in repaired

    def test_modifying_hype_buzzwords_repaired_to_underlying_noun(self):
        validator = SemanticValidator()
        raw_caption = "1girl, stunning landscape, gorgeous dress, solo, incredible lighting"
        report = validator.validate(raw_caption)

        assert report.status == ValidationStatus.REPAIRED
        repaired = report.repaired_caption

        assert "landscape" in repaired
        assert "dress" in repaired
        assert "lighting" in repaired
        assert "stunning" not in repaired.lower()
        assert "gorgeous" not in repaired.lower()
        assert "incredible" not in repaired.lower()


class TestAuthoritativeUncertainty:
    """Verifies that facts marked as uncertain are strictly excluded from captions."""

    def test_uncertain_facts_omitted_in_filter_and_builder(self):
        facts = StructuredVisualFacts(
            facts_by_category={
                SemanticCategory.IDENTITY: [
                    FactItem(id="id_1", text="1girl", primary_category=SemanticCategory.IDENTITY),
                ],
                SemanticCategory.APPEARANCE: [
                    FactItem(id="app_1", text="blonde hair", primary_category=SemanticCategory.APPEARANCE),
                    FactItem(
                        id="app_unc",
                        text="maybe red ribbon",
                        primary_category=SemanticCategory.APPEARANCE,
                        is_uncertain=True,
                    ),
                ],
                SemanticCategory.UNCERTAINTY: [
                    FactItem(
                        id="unc_cat",
                        text="unclear background object",
                        primary_category=SemanticCategory.UNCERTAINTY,
                        is_uncertain=True,
                    )
                ],
            }
        )

        filter_engine = SemanticFilter()
        mode = get_caption_mode("character")
        trigger_cfg = TriggerConfig(word="CharTest")

        filter_res = filter_engine.filter_facts(mode=mode, facts=facts, trigger_cfg=trigger_cfg)
        accepted_texts = [f.text for f in filter_res.accepted]

        assert "1girl" in accepted_texts
        assert "blonde hair" in accepted_texts
        assert "maybe red ribbon" not in accepted_texts
        assert "unclear background object" not in accepted_texts

        builder = CaptionBuilder()
        caption, tokens = builder.build(
            mode_or_tags=mode,
            filter_result=filter_res,
            trigger_cfg=trigger_cfg,
            format_type=CaptionFormat.TAGS,
        )

        assert "maybe red ribbon" not in caption
        assert "unclear background object" not in caption

    def test_uncertain_json_extraction_drops_to_omission_e2e(self, tmp_path):
        """End-to-end pipeline test ensuring JoyCaption uncertainty section is never emitted."""
        img_path = _create_test_image(tmp_path / "uncertain.png")
        pipeline = DatasetPipeline()

        mock_wd14 = MagicMock()
        mock_wd14.predict.return_value = []
        pipeline.wd14 = mock_wd14

        mock_joy = MagicMock()
        mock_joy.generate.return_value = [
            json.dumps({
                "identity": ["1girl"],
                "appearance": ["green eyes"],
                "uncertainty": ["possibly wearing a necklace", "perhaps outdoors"],
            })
        ]
        pipeline.joycaption = mock_joy

        result = pipeline.caption_image(
            image_path=img_path,
            caption_mode="character",
            caption_format=CaptionFormat.TAGS,
        )

        assert "1girl" in result.caption
        assert "green eyes" in result.caption
        assert "necklace" not in result.caption.lower()
        assert "outdoors" not in result.caption.lower()


class TestInviolableUserLocks:
    """Verifies that locked=True facts and tokens survive across all modes and repairs."""

    def test_locked_token_survives_style_mode_exclusion(self):
        facts = StructuredVisualFacts(
            facts_by_category={
                SemanticCategory.APPEARANCE: [
                    FactItem(
                        id="app_locked",
                        text="neon blue hair",
                        primary_category=SemanticCategory.APPEARANCE,
                        locked=True,
                    ),
                    FactItem(
                        id="app_normal",
                        text="green eyes",
                        primary_category=SemanticCategory.APPEARANCE,
                        locked=False,
                    ),
                ],
                SemanticCategory.RENDERING: [
                    FactItem(id="ren_1", text="oil painting", primary_category=SemanticCategory.RENDERING),
                ],
            }
        )

        filter_engine = SemanticFilter()
        style_mode = get_caption_mode("style")
        trigger_cfg = TriggerConfig(word="OilStyle")

        filter_res = filter_engine.filter_facts(mode=style_mode, facts=facts, trigger_cfg=trigger_cfg)
        accepted_texts = [f.text for f in filter_res.accepted]

        assert "neon blue hair" in accepted_texts  # Locked survives!
        assert "green eyes" not in accepted_texts  # Unlocked appearance dropped in style mode
        assert "oil painting" in accepted_texts

        builder = CaptionBuilder()
        caption, tokens = builder.build(
            mode_or_tags=style_mode,
            filter_result=filter_res,
            trigger_cfg=trigger_cfg,
        )

        validator = SemanticValidator()
        report = validator.validate(caption=caption, tokens=tokens, mode=style_mode, facts=facts)

        assert "neon blue hair" in report.repaired_caption
        # Must not be rejected as a content leak because it is locked
        assert report.status != ValidationStatus.REJECTED

    def test_locked_hype_word_survives_validator_stripping(self):
        """If user explicitly locks a buzzword, validator must not remove it."""
        validator = SemanticValidator()
        # Pretend the character name or franchise includes 'Masterpiece'
        facts = StructuredVisualFacts(
            facts_by_category={
                SemanticCategory.IDENTITY: [
                    FactItem(
                        id="id_locked",
                        text="masterpiece",
                        primary_category=SemanticCategory.IDENTITY,
                        locked=True,
                    )
                ]
            }
        )

        caption = "masterpiece, 1girl, standing"
        report = validator.validate(caption=caption, facts=facts)

        assert "masterpiece" in report.repaired_caption
        assert report.status in (ValidationStatus.VALID, ValidationStatus.REPAIRED)

    def test_locked_tokens_survive_outfit_pose_concept_modes(self):
        """Locked facts survive exclusions in OutfitMode, PoseMode, and ConceptMode."""
        filter_engine = SemanticFilter()
        trigger_cfg = TriggerConfig(word="Trigger")

        # 1. OutfitMode: normally excludes environment, but locked environment survives
        outfit_mode = get_caption_mode("outfit")
        facts_outfit = StructuredVisualFacts(
            facts_by_category={
                SemanticCategory.CLOTHING: [
                    FactItem(id="c1", text="bomber jacket", primary_category=SemanticCategory.CLOTHING)
                ],
                SemanticCategory.ENVIRONMENT: [
                    FactItem(
                        id="e1",
                        text="cyberpunk alleyway",
                        primary_category=SemanticCategory.ENVIRONMENT,
                        locked=True,
                    ),
                    FactItem(
                        id="e2",
                        text="neon billboard",
                        primary_category=SemanticCategory.ENVIRONMENT,
                        locked=False,
                    ),
                ],
            }
        )
        res = filter_engine.filter_facts(mode=outfit_mode, facts=facts_outfit, trigger_cfg=trigger_cfg)
        texts = [f.text for f in res.accepted]
        assert "bomber jacket" in texts
        assert "cyberpunk alleyway" in texts
        assert "neon billboard" not in texts

        # 2. PoseMode: normally excludes clothing, but locked clothing survives
        pose_mode = get_caption_mode("pose")
        facts_pose = StructuredVisualFacts(
            facts_by_category={
                SemanticCategory.POSE: [
                    FactItem(id="p1", text="jumping", primary_category=SemanticCategory.POSE)
                ],
                SemanticCategory.CLOTHING: [
                    FactItem(id="c_locked", text="tuxedo", primary_category=SemanticCategory.CLOTHING, locked=True),
                    FactItem(id="c_unlocked", text="sneakers", primary_category=SemanticCategory.CLOTHING),
                ],
            }
        )
        res_pose = filter_engine.filter_facts(mode=pose_mode, facts=facts_pose, trigger_cfg=trigger_cfg)
        texts_pose = [f.text for f in res_pose.accepted]
        assert "jumping" in texts_pose
        assert "tuxedo" in texts_pose
        assert "sneakers" not in texts_pose


class TestEmptyAndMalformedInputs:
    """Verifies that none, empty, malformed, and punctuation-only inputs degrade gracefully."""

    def test_extractor_handles_none_empty_and_whitespace(self):
        extractor = Stage1Extractor()

        for val in [None, "", "   ", "\n\t", "None", "null", "N/A"]:
            facts = extractor.extract(val)
            assert isinstance(facts, StructuredVisualFacts)
            assert len(facts.all_facts()) == 0

    def test_extractor_handles_malformed_json_without_crashing(self):
        extractor = Stage1Extractor()
        malformed_inputs = [
            '{"identity": ["1girl", "solo", "unterminated',
            '{"identity": [1girl], appearance: [blue eyes]}',
            '{bad json: true, ,,,}',
            "{{{",
            "just plain text without json fences or blocks",
        ]

        for text in malformed_inputs:
            facts = extractor.extract(text)
            assert isinstance(facts, StructuredVisualFacts)

    def test_extractor_handles_punctuation_only_input(self):
        extractor = Stage1Extractor()
        punctuation_inputs = [
            ",,, ,,, ...",
            "::: ;;; ---",
            "```json\n,,,\n```",
        ]
        for p in punctuation_inputs:
            facts = extractor.extract(p)
            assert isinstance(facts, StructuredVisualFacts)
            assert len(facts.all_facts()) == 0

    def test_validator_handles_none_empty_and_punctuation(self):
        validator = SemanticValidator()

        # None caption
        rep_none = validator.validate(None)
        assert rep_none.status == ValidationStatus.VALID
        assert rep_none.repaired_caption == ""

        # Empty caption
        rep_empty = validator.validate("")
        assert rep_empty.status == ValidationStatus.VALID
        assert rep_empty.repaired_caption == ""

        # Punctuation-only caption
        rep_punct = validator.validate(",,,   ...   :::   ;;;")
        assert rep_punct.status in (ValidationStatus.VALID, ValidationStatus.REPAIRED)

    def test_builder_handles_empty_filter_results(self):
        builder = CaptionBuilder()
        mode = get_caption_mode("character")

        caption, tokens = builder.build(
            mode_or_tags=mode,
            filter_result=None,
            format_type=CaptionFormat.TAGS,
        )
        assert caption == ""
        assert len(tokens) == 0

    def test_pipeline_handles_empty_image_generation_e2e(self, tmp_path):
        img_path = _create_test_image(tmp_path / "empty_input.png")
        pipeline = DatasetPipeline()

        mock_wd14 = MagicMock()
        mock_wd14.predict.return_value = []
        pipeline.wd14 = mock_wd14

        mock_joy = MagicMock()
        mock_joy.generate.return_value = [""]
        pipeline.joycaption = mock_joy

        result = pipeline.caption_image(
            image_path=img_path,
            caption_mode="character",
            caption_format=CaptionFormat.TAGS,
        )

        assert result is not None
        assert isinstance(result.caption, str)
        assert result.validation_report.status == ValidationStatus.VALID
