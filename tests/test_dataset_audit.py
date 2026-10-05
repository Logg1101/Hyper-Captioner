"""
Tests for Task 7: Dataset Pipeline, Audit Metadata Persistence, and Batch Traceability.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from PIL import Image

from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionMode,
    CaptionToken,
    PresetConfig,
    TagItem,
    TriggerConfig,
    TriggerPlacement,
    ValidationReport,
    ValidationStatus,
)
from hyper_captioner.dataset.exporter import write_audit_sidecar, write_sidecar
from hyper_captioner.dataset.manager import BatchProgress, DatasetPipeline


def _create_dummy_image(path: Path) -> Path:
    """Helper to create a small valid image file."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(path)
    return path


def test_sidecar_and_audit_separation(tmp_path):
    """Verifies that clean .txt sidecars and .audit.json sidecars are properly separated."""
    img_file = tmp_path / "sample_001.png"
    _create_dummy_image(img_file)

    caption = "MyTrigger, 1girl, solo, standing, blue eyes"
    audit_data = {
        "status": "valid",
        "stage1_facts_count": 8,
        "accepted_count": 4,
        "rejected_count": 4,
        "traceability": [{"token": "blue eyes", "source": "app_01"}],
    }

    # 1. Write clean training sidecar
    txt_path = write_sidecar(img_file, caption)
    assert txt_path.exists()
    assert txt_path.read_text(encoding="utf-8") == caption

    # 2. Write audit sidecar
    audit_path = write_audit_sidecar(img_file, audit_data)
    assert audit_path.name == "sample_001.audit.json"
    assert audit_path.exists()

    loaded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert loaded_audit["status"] == "valid"
    assert loaded_audit["stage1_facts_count"] == 8

    # 3. Ensure training sidecar is completely untouched and contains no JSON/audit tokens
    txt_content = txt_path.read_text(encoding="utf-8")
    assert "{" not in txt_content
    assert "}" not in txt_content
    assert "audit" not in txt_content
    assert txt_content == caption


def test_caption_image_end_to_end_with_mocks(tmp_path):
    """
    Tests end-to-end two-stage caption_image pipeline with mocked VLM and tagger engines.
    Verifies fact extraction, filtering, building, validation, and sidecar persistence.
    """
    img_path = tmp_path / "char_01.png"
    _create_dummy_image(img_path)

    pipeline = DatasetPipeline()

    # Mock WD14 engine
    mock_wd14 = MagicMock()
    mock_wd14.predict.return_value = [
        ("solo", 0.98, 4),
        ("1girl", 0.98, 4),
        ("blue eyes", 0.92, 0),
        ("standing", 0.88, 0),
        ("masterpiece", 0.95, 0),  # Hype tag to be filtered or cleaned
    ]
    pipeline.wd14 = mock_wd14

    # Mock JoyCaption engine returning standardized JSON
    mock_joy = MagicMock()
    mock_joy.generate.return_value = [
        json.dumps({
            "identity": ["1girl", "solo"],
            "appearance": ["blue eyes", "blonde hair"],
            "pose": ["standing"],
            "clothing": ["school uniform"],
        })
    ]
    pipeline.joycaption = mock_joy

    trigger_cfg = TriggerConfig(word="AsukaLangley", placement=TriggerPlacement.PREPEND)

    result = pipeline.caption_image(
        image_path=img_path,
        caption_mode="character",
        caption_format=CaptionFormat.TAGS,
        trigger_config=trigger_cfg,
        write_txt=True,
        write_audit=True,
    )

    # Verify result structure
    assert result is not None
    assert "AsukaLangley" in result.caption
    assert "blue eyes" in result.caption
    assert "masterpiece" not in result.caption  # Filtered out

    # Verify tokens and validation report
    assert hasattr(result, "tokens")
    assert len(result.tokens) > 0
    assert any(t.text == "AsukaLangley" for t in result.tokens)

    assert hasattr(result, "validation_report")
    assert result.validation_report is not None
    assert result.validation_report.status in (ValidationStatus.VALID, ValidationStatus.REPAIRED)

    # Verify sidecar files
    txt_path = img_path.with_suffix(".txt")
    assert txt_path.exists()
    assert txt_path.read_text(encoding="utf-8") == result.caption

    audit_path = img_path.parent / f"{img_path.stem}.audit.json"
    assert audit_path.exists()
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_data["image"] == "char_01.png"
    assert audit_data["caption"] == result.caption
    assert audit_data["mode"] == "character"
    assert audit_data["format"] == "tags"
    assert "validation" in audit_data
    assert "facts" in audit_data
    assert "tokens" in audit_data
    assert "execution_time" in audit_data
    assert audit_data["validation"]["status"] in ("valid", "repaired")


def test_batch_progress_validation_counters(tmp_path):
    """
    Verifies that process_dataset tracks valid_count, repaired_count, and rejected_count
    on BatchProgress.
    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create 3 test images
    img1 = _create_dummy_image(dataset_dir / "valid.png")
    img2 = _create_dummy_image(dataset_dir / "repaired.png")
    img3 = _create_dummy_image(dataset_dir / "rejected.png")

    pipeline = DatasetPipeline()

    mock_wd14 = MagicMock()
    mock_wd14.predict.return_value = []
    pipeline.wd14 = mock_wd14

    mock_joy = MagicMock()

    # Image 1: Valid clean facts
    resp_valid = json.dumps({
        "identity": ["1girl", "solo"],
        "appearance": ["black hair"],
    })

    # Image 2: Contains conversational filler that requires auto-repair
    resp_repair = json.dumps({
        "identity": ["in the image, 1girl"],
        "appearance": ["black hair"],
    })

    # Image 3: Mutually exclusive physical states (standing vs sitting) -> HARD contradiction -> REJECTED
    resp_reject = json.dumps({
        "identity": ["1girl"],
        "pose": ["standing", "sitting"],
    })

    # Sequence return for the 3 images
    mock_joy.generate.side_effect = [
        [resp_valid],
        [resp_repair],
        [resp_reject],
    ]
    pipeline.joycaption = mock_joy

    progress = pipeline.process_dataset(
        dataset_path=dataset_dir,
        caption_mode="character",
        caption_format=CaptionFormat.TAGS,
        skip_existing=False,
        overwrite=True,
        write_audit=True,
    )

    assert progress.total == 3
    assert progress.completed == 3
    assert progress.valid_count == 1
    assert progress.repaired_count == 1
    assert progress.rejected_count == 1

    progress_dict = progress.to_dict()
    assert progress_dict["valid_count"] == 1
    assert progress_dict["repaired_count"] == 1
    assert progress_dict["rejected_count"] == 1


def test_user_locked_tags_in_pipeline(tmp_path):
    """
    Verifies that user locked tags make it through the entire pipeline:
    injected into facts with locked=True, accepted by filter, prioritized by builder,
    preserved by validator, and written to audit metadata.
    """
    img_path = tmp_path / "locked_sample.png"
    _create_dummy_image(img_path)

    pipeline = DatasetPipeline()

    mock_wd14 = MagicMock()
    mock_wd14.predict.return_value = [("1girl", 0.95, 4)]
    pipeline.wd14 = mock_wd14

    mock_joy = MagicMock()
    # JoyCaption does NOT mention "cybernetic arm"
    mock_joy.generate.return_value = [
        json.dumps({
            "identity": ["1girl"],
            "appearance": ["brown hair"],
        })
    ]
    pipeline.joycaption = mock_joy

    result = pipeline.caption_image(
        image_path=img_path,
        caption_mode="character",
        caption_format=CaptionFormat.TAGS,
        locked_tags=["cybernetic arm"],
        write_audit=True,
    )

    # Check that locked tag is in final caption
    assert "cybernetic arm" in result.caption

    # Check that locked token is recorded as locked=True
    locked_tokens = [t for t in result.tokens if "cybernetic arm" in t.text]
    assert len(locked_tokens) >= 1
    assert all(t.locked for t in locked_tokens)

    # Check audit sidecar
    audit_path = img_path.parent / f"{img_path.stem}.audit.json"
    assert audit_path.exists()
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))

    # In audit tokens
    audit_locked_tokens = [t for t in audit_data["tokens"] if "cybernetic arm" in t["text"]]
    assert len(audit_locked_tokens) >= 1
    assert all(t["locked"] is True for t in audit_locked_tokens)


def test_caption_image_legacy_preset_compatibility(tmp_path):
    """
    Ensures that calling caption_image with legacy PresetConfig still executes
    correctly without regressions.
    """
    img_path = tmp_path / "legacy.png"
    _create_dummy_image(img_path)

    pipeline = DatasetPipeline()

    mock_wd14 = MagicMock()
    mock_wd14.predict.return_value = [("solo", 0.9, 4), ("1girl", 0.9, 4)]
    pipeline.wd14 = mock_wd14

    mock_joy = MagicMock()
    mock_joy.generate.return_value = [
        json.dumps({
            "identity": ["1girl", "solo"],
            "appearance": ["green eyes"],
        })
    ]
    pipeline.joycaption = mock_joy

    preset = PresetConfig(
        caption_mode=CaptionMode.TAG,
        keep_underscores=False,
    )

    result = pipeline.caption_image(
        image_path=img_path,
        preset=preset,
        write_txt=True,
    )

    assert result is not None
    assert "1girl" in result.caption
    assert "green eyes" in result.caption
    txt_file = img_path.with_suffix(".txt")
    assert txt_file.exists()
    assert txt_file.read_text(encoding="utf-8") == result.caption
