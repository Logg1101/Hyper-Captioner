"""
Comprehensive End-to-End Tests for all 5 Core Caption Modes:
CharacterMode, StyleMode, OutfitMode, PoseMode, and ConceptMode.

Verifies end-to-end execution through DatasetPipeline across all 3 output formats:
TAGS, STRUCTURED, and NATURAL, ensuring strict category contract compliance,
rendering jargon suppression, identity stripping, environment isolation,
and sidecar persistence.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from PIL import Image

from hyper_captioner.caption_modes.concept import ConceptMode
from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionModeType,
    TriggerConfig,
    TriggerPlacement,
    ValidationStatus,
)
from hyper_captioner.dataset.manager import DatasetPipeline


def _create_test_image(path: Path) -> Path:
    """Creates a small dummy image for pipeline testing."""
    img = Image.new("RGB", (64, 64), color=(120, 140, 160))
    img.save(path)
    return path


class TestCharacterModeE2E:
    """E2E verification of CharacterMode contract."""

    def test_character_mode_preserves_character_traits_and_suppresses_3d_jargon(self, tmp_path):
        img_path = _create_test_image(tmp_path / "char_test.png")
        pipeline = DatasetPipeline()

        # Mock WD14
        mock_wd14 = MagicMock()
        mock_wd14.predict.return_value = [
            ("solo", 0.99, 4),
            ("1girl", 0.99, 4),
            ("long hair", 0.90, 0),
        ]
        pipeline.wd14 = mock_wd14

        # Mock JoyCaption with visual character traits plus 3D jargon
        mock_joy = MagicMock()
        mock_joy.generate.return_value = [
            json.dumps({
                "identity": ["1girl", "solo"],
                "appearance": ["crimson hair", "amber eyes"],
                "expression": ["confident smirk"],
                "clothing": ["leather pilot jacket"],
                "pose": ["standing with hands on hips"],
                "composition": ["medium close-up"],
                "camera": ["eye level"],
                "lighting": ["sunset golden hour lighting"],
                "environment": ["hangar runway"],
                "rendering": ["unreal engine 5", "octane render", "ray tracing"],
            })
        ]
        pipeline.joycaption = mock_joy

        trigger_cfg = TriggerConfig(word="SkyCaptain", placement=TriggerPlacement.PREPEND)

        for fmt in [CaptionFormat.TAGS, CaptionFormat.STRUCTURED, CaptionFormat.NATURAL]:
            result = pipeline.caption_image(
                image_path=img_path,
                caption_mode="character",
                caption_format=fmt,
                trigger_config=trigger_cfg,
                write_txt=True,
                write_audit=True,
            )

            assert result is not None
            caption = result.caption

            # 1. Trigger present
            assert "SkyCaptain" in caption

            # 2. Character traits, clothing, pose, lighting, environment preserved
            assert "crimson hair" in caption or "amber eyes" in caption
            assert "leather pilot jacket" in caption
            assert "standing with hands on hips" in caption
            assert "sunset golden hour lighting" in caption
            assert "hangar runway" in caption

            # 3. 3D / rendering jargon strictly suppressed
            assert "unreal engine" not in caption.lower()
            assert "octane render" not in caption.lower()
            assert "ray tracing" not in caption.lower()

            # 4. Format-specific verification
            if fmt == CaptionFormat.TAGS:
                assert "," in caption
                assert ";" not in caption
            elif fmt == CaptionFormat.STRUCTURED:
                assert ";" in caption
            elif fmt == CaptionFormat.NATURAL:
                assert caption.endswith(".")

            # 5. Validation and Audit Sidecars
            assert result.validation_report.status in (ValidationStatus.VALID, ValidationStatus.REPAIRED)
            txt_file = img_path.with_suffix(".txt")
            assert txt_file.exists()
            assert txt_file.read_text(encoding="utf-8") == caption

            audit_file = img_path.parent / f"{img_path.stem}.audit.json"
            assert audit_file.exists()
            audit_json = json.loads(audit_file.read_text(encoding="utf-8"))
            assert audit_json["mode"] == "character"
            assert audit_json["format"] == fmt.value


class TestStyleModeE2E:
    """E2E verification of StyleMode contract."""

    def test_style_mode_drops_identity_and_clothing_preserves_rendering_and_lighting(self, tmp_path):
        img_path = _create_test_image(tmp_path / "style_test.png")
        pipeline = DatasetPipeline()

        # Mock WD14
        mock_wd14 = MagicMock()
        mock_wd14.predict.return_value = [
            ("solo", 0.95, 4),
            ("1girl", 0.95, 4),
            ("oil painting", 0.92, 0),
        ]
        pipeline.wd14 = mock_wd14

        # Mock JoyCaption with character content mixed with artistic style
        mock_joy = MagicMock()
        mock_joy.generate.return_value = [
            json.dumps({
                "identity": ["1girl", "solo", "hero"],
                "appearance": ["golden blonde hair", "sapphire eyes", "fair skin"],
                "clothing": ["red silk dress", "black leather gloves"],
                "rendering": ["oil painting", "impasto brushwork"],
                "style": ["baroque painterly style"],
                "lighting": ["chiaroscuro dramatic side lighting"],
                "material": ["oil on canvas"],
                "texture": ["rough canvas grain"],
                "color": ["muted earthy palette"],
            })
        ]
        pipeline.joycaption = mock_joy

        trigger_cfg = TriggerConfig(word="BaroqueOil", placement=TriggerPlacement.PREPEND)

        for fmt in [CaptionFormat.TAGS, CaptionFormat.STRUCTURED, CaptionFormat.NATURAL]:
            result = pipeline.caption_image(
                image_path=img_path,
                caption_mode="style",
                caption_format=fmt,
                trigger_config=trigger_cfg,
                write_txt=True,
                write_audit=True,
            )

            assert result is not None
            caption = result.caption

            # 1. Trigger present
            assert "BaroqueOil" in caption

            # 2. Rendering, lighting, texture, color strictly preserved
            assert "oil painting" in caption
            assert "impasto brushwork" in caption
            assert "chiaroscuro dramatic side lighting" in caption or "chiaroscuro" in caption
            assert "rough canvas grain" in caption or "canvas grain" in caption

            # 3. Character identity, hair, eyes, and clothing strictly dropped
            assert "1girl" not in caption.lower()
            assert "solo" not in caption.lower()
            assert "blonde hair" not in caption.lower()
            assert "sapphire eyes" not in caption.lower()
            assert "red silk dress" not in caption.lower()
            assert "black leather gloves" not in caption.lower()

            # 4. Format-specific verification
            if fmt == CaptionFormat.TAGS:
                assert "," in caption
            elif fmt == CaptionFormat.STRUCTURED:
                assert ";" in caption
            elif fmt == CaptionFormat.NATURAL:
                assert caption.endswith(".")

            # 5. Validation status clean
            assert result.validation_report.status in (ValidationStatus.VALID, ValidationStatus.REPAIRED)


class TestOutfitModeE2E:
    """E2E verification of OutfitMode contract."""

    def test_outfit_mode_isolates_garments_and_materials_suppresses_environment(self, tmp_path):
        img_path = _create_test_image(tmp_path / "outfit_test.png")
        pipeline = DatasetPipeline()

        mock_wd14 = MagicMock()
        mock_wd14.predict.return_value = [
            ("solo", 0.95, 4),
            ("1girl", 0.95, 4),
            ("jacket", 0.92, 0),
        ]
        pipeline.wd14 = mock_wd14

        mock_joy = MagicMock()
        mock_joy.generate.return_value = [
            json.dumps({
                "identity": ["1girl", "solo"],
                "appearance": ["green eyes", "twin tails hair"],
                "clothing": [
                    "oversized bomber jacket",
                    "pleated plaid skirt",
                    "thigh-high socks",
                    "combat boots",
                ],
                "material": ["distressed leather", "wool fabric"],
                "objects": ["silver zipper pull", "studded belt"],
                "environment": ["crowded neo-tokyo subway station", "neon signs"],
                "pose": ["walking forward"],
            })
        ]
        pipeline.joycaption = mock_joy

        trigger_cfg = TriggerConfig(word="PunkUniform", placement=TriggerPlacement.PREPEND)

        for fmt in [CaptionFormat.TAGS, CaptionFormat.STRUCTURED, CaptionFormat.NATURAL]:
            result = pipeline.caption_image(
                image_path=img_path,
                caption_mode="outfit",
                caption_format=fmt,
                trigger_config=trigger_cfg,
                write_txt=True,
                write_audit=True,
            )

            assert result is not None
            caption = result.caption

            # 1. Trigger present
            assert "PunkUniform" in caption

            # 2. Garments, footwear, materials, accessories preserved
            assert "oversized bomber jacket" in caption
            assert "pleated plaid skirt" in caption
            assert "combat boots" in caption
            assert "distressed leather" in caption or "wool fabric" in caption

            # 3. Environment and character identity suppressed
            assert "subway station" not in caption.lower()
            assert "neon signs" not in caption.lower()
            assert "1girl" not in caption.lower()
            assert "green eyes" not in caption.lower()

            # 4. Format-specific verification
            if fmt == CaptionFormat.TAGS:
                assert "," in caption
            elif fmt == CaptionFormat.STRUCTURED:
                assert ";" in caption
            elif fmt == CaptionFormat.NATURAL:
                assert caption.endswith(".")


class TestPoseModeE2E:
    """E2E verification of PoseMode contract."""

    def test_pose_mode_isolates_posture_and_camera_suppresses_clothing_and_face(self, tmp_path):
        img_path = _create_test_image(tmp_path / "pose_test.png")
        pipeline = DatasetPipeline()

        mock_wd14 = MagicMock()
        mock_wd14.predict.return_value = []
        pipeline.wd14 = mock_wd14

        mock_joy = MagicMock()
        mock_joy.generate.return_value = [
            json.dumps({
                "identity": ["1dancer", "solo"],
                "appearance": ["hazel eyes", "flowing brown hair"],
                "expression": ["joyful smile"],
                "clothing": ["elaborate sequin leotard", "silk cape"],
                "pose": ["dynamic mid-air leap", "arched back", "extended right arm"],
                "composition": ["full body shot", "diagonal action line"],
                "camera": ["low angle perspective"],
                "environment": ["sunlit rehearsal studio"],
            })
        ]
        pipeline.joycaption = mock_joy

        trigger_cfg = TriggerConfig(word="AerialLeap", placement=TriggerPlacement.PREPEND)

        for fmt in [CaptionFormat.TAGS, CaptionFormat.STRUCTURED, CaptionFormat.NATURAL]:
            result = pipeline.caption_image(
                image_path=img_path,
                caption_mode="pose",
                caption_format=fmt,
                trigger_config=trigger_cfg,
                write_txt=True,
                write_audit=True,
            )

            assert result is not None
            caption = result.caption

            # 1. Trigger present
            assert "AerialLeap" in caption

            # 2. Physical posture, action, framing, camera angle preserved
            assert "dynamic mid-air leap" in caption
            assert "arched back" in caption
            assert "full body shot" in caption
            assert "low angle perspective" in caption

            # 3. Detailed clothing and facial features suppressed
            assert "sequin leotard" not in caption.lower()
            assert "silk cape" not in caption.lower()
            assert "hazel eyes" not in caption.lower()
            assert "joyful smile" not in caption.lower()
            assert "rehearsal studio" not in caption.lower()

            # 4. Format-specific checks
            if fmt == CaptionFormat.TAGS:
                assert "," in caption
            elif fmt == CaptionFormat.STRUCTURED:
                assert ";" in caption
            elif fmt == CaptionFormat.NATURAL:
                assert caption.endswith(".")


class TestConceptModeE2E:
    """E2E verification of ConceptMode contract."""

    def test_concept_mode_isolates_thematic_elements_and_interacting_objects(self, tmp_path):
        img_path = _create_test_image(tmp_path / "concept_test.png")
        pipeline = DatasetPipeline()

        mock_wd14 = MagicMock()
        mock_wd14.predict.return_value = []
        pipeline.wd14 = mock_wd14

        mock_joy = MagicMock()
        mock_joy.generate.return_value = [
            json.dumps({
                "identity": ["1girl", "sorceress"],
                "appearance": ["purple eyes", "long silver hair"],
                "clothing": ["embroidered velvet robe"],
                "concept": ["elemental summoning ritual", "arcane energy vortex"],
                "objects": ["levitating spellbook", "glowing runic circle"],
                "material": ["translucent crystalline ether"],
                "environment": ["dark stone chamber"],
            })
        ]
        pipeline.joycaption = mock_joy

        trigger_cfg = TriggerConfig(word="ArcaneRitual", placement=TriggerPlacement.PREPEND)

        for fmt in [CaptionFormat.TAGS, CaptionFormat.STRUCTURED, CaptionFormat.NATURAL]:
            result = pipeline.caption_image(
                image_path=img_path,
                caption_mode="concept",
                caption_format=fmt,
                trigger_config=trigger_cfg,
                write_txt=True,
                write_audit=True,
            )

            assert result is not None
            caption = result.caption

            # 1. Trigger present
            assert "ArcaneRitual" in caption

            # 2. Thematic concepts, interacting objects, materials preserved
            assert "elemental summoning ritual" in caption
            assert "arcane energy vortex" in caption
            assert "levitating spellbook" in caption
            assert "glowing runic circle" in caption

            # 3. Character identity and background noise suppressed
            assert "sorceress" not in caption.lower()
            assert "purple eyes" not in caption.lower()
            assert "velvet robe" not in caption.lower()
            assert "dark stone chamber" not in caption.lower()

            # 4. Format checks
            if fmt == CaptionFormat.TAGS:
                assert "," in caption
            elif fmt == CaptionFormat.STRUCTURED:
                assert ";" in caption
            elif fmt == CaptionFormat.NATURAL:
                assert caption.endswith(".")

    def test_concept_mode_custom_focal_concept_instructions(self):
        """Verifies ConceptMode with focal concept parameter."""
        custom_mode = ConceptMode(focal_concept="crystallization decay")
        assert custom_mode.focal_concept == "crystallization decay"
        instructions = custom_mode.build_extraction_instructions()
        assert "crystallization decay" in instructions


class TestDatasetBatchE2E:
    """Verifies batch processing across dataset images with mixed modes."""

    def test_batch_dataset_processing_e2e(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        img1 = _create_test_image(dataset_dir / "img1.png")
        img2 = _create_test_image(dataset_dir / "img2.png")

        pipeline = DatasetPipeline()
        mock_wd14 = MagicMock()
        mock_wd14.predict.return_value = [("solo", 0.9, 4), ("1girl", 0.9, 4)]
        pipeline.wd14 = mock_wd14

        mock_joy = MagicMock()
        mock_joy.generate.side_effect = [
            [json.dumps({"identity": ["1girl"], "appearance": ["black hair"], "pose": ["sitting"]})],
            [json.dumps({"identity": ["1girl"], "appearance": ["blonde hair"], "pose": ["standing"]})],
        ]
        pipeline.joycaption = mock_joy

        progress = pipeline.process_dataset(
            dataset_path=dataset_dir,
            caption_mode="character",
            caption_format=CaptionFormat.TAGS,
            write_audit=True,
            overwrite=True,
        )

        assert progress.total == 2
        assert progress.completed == 2
        assert progress.valid_count == 2
        assert (dataset_dir / "img1.txt").exists()
        assert (dataset_dir / "img2.txt").exists()
        assert (dataset_dir / "img1.audit.json").exists()
        assert (dataset_dir / "img2.audit.json").exists()
