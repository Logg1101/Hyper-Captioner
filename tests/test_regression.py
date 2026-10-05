"""
Regression Test Suite for HyperCaptioner.

Verifies:
1. scan_dataset: accurate counts of total, with captions, missing captions, orphan captions,
   empty captions, and integrity corruption detection across nested folders.
2. DatasetVocabularyIndex: dataset-wide indexing, top tag frequencies, variant search,
   atomic batch tag replacement, and tag deletion across sidecar files.
3. export_master_metadata: CSV, JSON, TXT, and ZIP compilations with custom filenames.
4. create_backup: timestamped backup creation preserving directory trees and data integrity.
5. Legacy preset compatibility: PresetConfig serialization, CaptionMode enum backward
   compatibility, and legacy preset pipeline execution.
"""

import csv
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from PIL import Image

from hyper_captioner.core.types import (
    CaptionMode,
    CharacterConfig,
    LoRAStrategy,
    PresetConfig,
)
from hyper_captioner.dataset.exporter import create_backup, export_master_metadata
from hyper_captioner.dataset.manager import DatasetPipeline
from hyper_captioner.dataset.scanner import DatasetScanResult, scan_dataset
from hyper_captioner.vocabulary.database import DatasetVocabularyIndex


def _create_dummy_image(path: Path, valid: bool = True) -> Path:
    """Creates a valid or corrupted image for regression testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid:
        img = Image.new("RGB", (32, 32), color=(80, 120, 160))
        img.save(path)
    else:
        # Corrupted / invalid image bytes
        path.write_bytes(b"NOT_A_VALID_IMAGE_HEADER_CORRUPTED_BYTES")
    return path


class TestScanDatasetRegression:
    """Verifies scanner accuracy across dataset configurations."""

    def test_scan_dataset_counts_and_categories(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        sub_dir = dataset_dir / "subfolder"
        sub_dir.mkdir()

        # 1. Image with valid caption
        img1 = _create_dummy_image(dataset_dir / "img1.png")
        (dataset_dir / "img1.txt").write_text("1girl, solo, standing", encoding="utf-8")

        # 2. Image with empty caption
        img2 = _create_dummy_image(dataset_dir / "img2.jpg")
        (dataset_dir / "img2.txt").write_text("   \n  ", encoding="utf-8")

        # 3. Image missing caption
        img3 = _create_dummy_image(dataset_dir / "img3.webp")

        # 4. Nested image with valid caption
        img4 = _create_dummy_image(sub_dir / "nested_img.png")
        (sub_dir / "nested_img.txt").write_text("landscape, mountains", encoding="utf-8")

        # 5. Orphan caption (no corresponding image)
        orphan_txt = dataset_dir / "orphan_without_image.txt"
        orphan_txt.write_text("orphan caption content", encoding="utf-8")

        scan = scan_dataset(dataset_dir, verify_integrity=False)

        assert isinstance(scan, DatasetScanResult)
        assert scan.total_images == 4
        assert len(scan.with_captions) == 2
        assert img1 in scan.with_captions
        assert img4 in scan.with_captions

        assert len(scan.missing_captions) == 2
        assert img2 in scan.missing_captions
        assert img3 in scan.missing_captions

        assert len(scan.empty_captions) == 1
        assert img2 in scan.empty_captions

        assert len(scan.orphan_captions) == 1
        assert orphan_txt in scan.orphan_captions

        # Verify serialization dict
        d = scan.to_dict()
        assert d["total_images"] == 4
        assert d["with_captions_count"] == 2
        assert d["missing_captions_count"] == 2
        assert d["empty_captions_count"] == 1
        assert d["orphan_captions_count"] == 1

    def test_scan_dataset_verify_integrity_detects_corrupted(self, tmp_path):
        dataset_dir = tmp_path / "dataset_integrity"
        dataset_dir.mkdir()

        valid_img = _create_dummy_image(dataset_dir / "valid.png", valid=True)
        corrupted_img = _create_dummy_image(dataset_dir / "broken.png", valid=False)

        scan = scan_dataset(dataset_dir, verify_integrity=True)

        assert scan.total_images == 2
        assert len(scan.corrupted_images) == 1
        assert corrupted_img in scan.corrupted_images
        assert valid_img not in scan.corrupted_images

    def test_scan_dataset_nonexistent_path_raises_filenotfound(self, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            scan_dataset(missing_dir)


class TestDatasetVocabularyIndexRegression:
    """Verifies vocabulary index extraction, frequency analysis, and batch updates."""

    @pytest.fixture
    def populated_dataset(self, tmp_path):
        d = tmp_path / "vocab_dataset"
        d.mkdir()

        (d / "img1.txt").write_text("1girl, solo, blue eyes, long hair, school uniform", encoding="utf-8")
        (d / "img2.txt").write_text("1girl, solo, blue eyes, short hair, leather jacket", encoding="utf-8")
        (d / "img3.txt").write_text("1boy, solo, brown hair, leather jacket", encoding="utf-8")
        (d / "img4.txt").write_text("thigh high socks, thigh-highs, striped thighhighs", encoding="utf-8")

        return d

    def test_index_and_top_tags(self, populated_dataset):
        vocab = DatasetVocabularyIndex(populated_dataset)
        counts = vocab.index()

        assert counts["1girl"] == 2
        assert counts["solo"] == 3
        assert counts["blue eyes"] == 2
        assert counts["leather jacket"] == 2
        assert counts["1boy"] == 1

        top_tags = vocab.get_top_tags(limit=3)
        assert len(top_tags) == 3
        assert top_tags[0] == ("solo", 3)

    def test_find_variants(self, populated_dataset):
        vocab = DatasetVocabularyIndex(populated_dataset)
        vocab.index()

        variants = vocab.find_variants("thigh")
        variant_tags = [tag for tag, count in variants]

        assert "thigh high socks" in variant_tags
        assert "thigh-highs" in variant_tags
        assert "striped thighhighs" in variant_tags

    def test_replace_tag(self, populated_dataset):
        vocab = DatasetVocabularyIndex(populated_dataset)
        vocab.index()

        modified_count = vocab.replace_tag(old_tag="leather jacket", new_tag="biker jacket")

        assert modified_count == 2
        assert "biker jacket" in (populated_dataset / "img2.txt").read_text(encoding="utf-8")
        assert "leather jacket" not in (populated_dataset / "img2.txt").read_text(encoding="utf-8")
        assert "biker jacket" in (populated_dataset / "img3.txt").read_text(encoding="utf-8")

        # Verify index was refreshed
        assert vocab.tag_counts["biker jacket"] == 2
        assert vocab.tag_counts["leather jacket"] == 0

    def test_delete_tag(self, populated_dataset):
        vocab = DatasetVocabularyIndex(populated_dataset)
        vocab.index()

        modified_count = vocab.delete_tag("solo")

        assert modified_count == 3
        for fname in ["img1.txt", "img2.txt", "img3.txt"]:
            content = (populated_dataset / fname).read_text(encoding="utf-8")
            assert "solo" not in content

        assert vocab.tag_counts["solo"] == 0

    def test_batch_normalize(self, populated_dataset):
        vocab = DatasetVocabularyIndex(populated_dataset)
        vocab.index()

        mapping = {
            "blue eyes": "sapphire eyes",
            "1girl": "female",
        }
        modified_count = vocab.batch_normalize(mapping)

        assert modified_count == 2
        c1 = (populated_dataset / "img1.txt").read_text(encoding="utf-8")
        assert "sapphire eyes" in c1
        assert "female" in c1
        assert "blue eyes" not in c1
        assert "1girl" not in c1


class TestExportMasterMetadataRegression:
    """Verifies export of dataset captions into CSV, JSON, TXT, and ZIP formats."""

    @pytest.fixture
    def export_dataset(self, tmp_path):
        d = tmp_path / "export_ds"
        d.mkdir()

        _create_dummy_image(d / "sample_a.png")
        (d / "sample_a.txt").write_text("caption a, test a", encoding="utf-8")

        _create_dummy_image(d / "sample_b.jpg")
        (d / "sample_b.txt").write_text("caption b, test b", encoding="utf-8")

        return d

    def test_export_csv(self, export_dataset):
        out_csv = export_master_metadata(export_dataset, export_format="csv")
        assert out_csv.exists()
        assert out_csv.name == "captions.csv"

        with open(out_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert reader[0] == ["image", "caption"]
            rows = {row[0]: row[1] for row in reader[1:]}
            assert "sample_a.png" in rows
            assert rows["sample_a.png"] == "caption a, test a"
            assert "sample_b.jpg" in rows
            assert rows["sample_b.jpg"] == "caption b, test b"

    def test_export_json(self, export_dataset):
        out_json = export_master_metadata(
            export_dataset, export_format="json", output_filename="custom_metadata.json"
        )
        assert out_json.exists()
        assert out_json.name == "custom_metadata.json"

        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert "sample_a.png" in data
        assert data["sample_a.png"] == "caption a, test a"
        assert "sample_b.jpg" in data
        assert data["sample_b.jpg"] == "caption b, test b"

    def test_export_txt(self, export_dataset):
        out_txt = export_master_metadata(export_dataset, export_format="txt")
        assert out_txt.exists()
        assert out_txt.name == "captions.txt"

        content = out_txt.read_text(encoding="utf-8")
        assert "sample_a.png: caption a, test a" in content
        assert "sample_b.jpg: caption b, test b" in content

    def test_export_zip(self, export_dataset):
        out_zip = export_master_metadata(export_dataset, export_format="zip")
        assert out_zip.exists()
        assert out_zip.name == "captions_txt_sidecars.zip"

        with zipfile.ZipFile(out_zip, "r") as z:
            names = z.namelist()
            assert "sample_a.txt" in names
            assert "sample_b.txt" in names
            assert z.read("sample_a.txt").decode("utf-8") == "caption a, test a"


class TestCreateBackupRegression:
    """Verifies intact directory backup creation."""

    def test_create_backup_preserves_tree_and_contents(self, tmp_path):
        source = tmp_path / "original_dataset"
        source.mkdir()
        sub = source / "nested"
        sub.mkdir()

        _create_dummy_image(source / "img1.png")
        (source / "img1.txt").write_text("img1 caption", encoding="utf-8")
        _create_dummy_image(sub / "sub_img.png")
        (sub / "sub_img.txt").write_text("nested caption", encoding="utf-8")

        backup_path = create_backup(source)

        assert backup_path.exists()
        assert backup_path.is_dir()
        assert "_BACKUP_" in backup_path.name

        # Verify files in backup
        assert (backup_path / "img1.png").exists()
        assert (backup_path / "img1.txt").read_text(encoding="utf-8") == "img1 caption"
        assert (backup_path / "nested" / "sub_img.png").exists()
        assert (backup_path / "nested" / "sub_img.txt").read_text(encoding="utf-8") == "nested caption"

        # Verify source remained intact
        assert (source / "img1.txt").exists()
        assert (source / "nested" / "sub_img.txt").exists()


class TestLegacyPresetCompatibilityRegression:
    """Verifies backward compatibility for PresetConfig and legacy pipeline invocations."""

    def test_preset_config_serialization_roundtrip(self):
        preset = PresetConfig(
            caption_mode=CaptionMode.NATURAL,
            keep_underscores=True,
            character=CharacterConfig(
                name="TestHero",
                trigger_word="Hero",
                prune_reference_from_caption=True,
            ),
        )

        d = preset.to_dict()
        assert d["caption_mode"] == "natural"
        assert d["keep_underscores"] is True
        assert d["character"]["trigger_word"] == "Hero"

        restored = PresetConfig.from_dict(d)
        assert restored.caption_mode == CaptionMode.NATURAL
        assert restored.keep_underscores is True
        assert restored.character.trigger_word == "Hero"
        assert restored.character.prune_reference_from_caption is True

    def test_pipeline_caption_image_with_legacy_preset(self, tmp_path):
        img_path = _create_dummy_image(tmp_path / "legacy_test.png")
        pipeline = DatasetPipeline()

        mock_wd14 = MagicMock()
        mock_wd14.predict.return_value = [("1girl", 0.95, 4), ("solo", 0.95, 4)]
        pipeline.wd14 = mock_wd14

        mock_joy = MagicMock()
        mock_joy.generate.return_value = [
            json.dumps({
                "identity": ["1girl", "solo"],
                "appearance": ["purple eyes"],
                "pose": ["sitting"],
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
        assert "purple eyes" in result.caption
        assert (img_path.with_suffix(".txt")).exists()
        assert (img_path.with_suffix(".txt")).read_text(encoding="utf-8") == result.caption
