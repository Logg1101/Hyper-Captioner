"""
Unit tests for API v2 endpoints and updates:
- GET /api/modes (mode contracts, formats, trigger placements)
- GET /api/review/item (validation fields, .audit.json loading)
- GET /api/batch/progress (validation counters)
- POST /api/batch/preview (modes, formats, validation status, tokens)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hyper_captioner.api.app import app, batch_state
from hyper_captioner.core.types import (
    CaptionMode,
    CaptionResult,
    CaptionToken,
    SemanticCategory,
    ValidationIssue,
    ValidationReport,
    ValidationStatus,
)
from hyper_captioner.dataset.manager import BatchProgress


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def temp_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    img_path = dataset_dir / "test_01.png"
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(img_path)
    txt_path = dataset_dir / "test_01.txt"
    txt_path.write_text("solo, blue background", encoding="utf-8")
    return dataset_dir, img_path, txt_path


def test_api_modes_endpoint(client):
    """Checks GET /api/modes returns the 5 mode contracts, formats, and placements."""
    response = client.get("/api/modes")
    assert response.status_code == 200
    data = response.json()

    assert "modes" in data
    assert "formats" in data
    assert "trigger_placements" in data

    # Modes must include character, style, outfit, pose, concept
    mode_names = [m["name"] if isinstance(m, dict) else m for m in data["modes"]]
    assert set(mode_names) == {"character", "style", "outfit", "pose", "concept"}

    # Formats must be tags, structured, natural
    assert data["formats"] == ["tags", "structured", "natural"]

    # Trigger placements must be prepend, append, wrap, omit
    assert data["trigger_placements"] == ["prepend", "append", "wrap", "omit"]


def test_api_review_item_contains_validation_fields(client, temp_dataset):
    """Checks GET /api/review/item always includes validation_status and traceability."""
    _, img_path, _ = temp_dataset
    response = client.get(f"/api/review/item?image_path={img_path.as_posix()}")
    assert response.status_code == 200
    data = response.json()

    assert "validation_status" in data
    assert data["validation_status"] in ["valid", "unvalidated"]
    assert "traceability" in data
    assert isinstance(data["traceability"], list)


def test_api_review_item_loads_audit_json(client, temp_dataset):
    """Checks GET /api/review/item deserializes adjacent <stem>.audit.json metadata."""
    dataset_dir, img_path, _ = temp_dataset
    audit_path = dataset_dir / f"{img_path.stem}.audit.json"
    audit_data = {
        "image": img_path.name,
        "caption": "masterpiece, 1girl, solo",
        "mode": "character",
        "format": "tags",
        "validation": {
            "status": "repaired",
            "issues": [
                {
                    "severity": "WARNING",
                    "code": "SUSPECT_HYPE",
                    "message": "Removed hype tag",
                    "token": "masterpiece",
                    "suggested_repair": None,
                }
            ],
            "repaired_caption": "1girl, solo",
        },
        "tokens": [
            {
                "text": "1girl",
                "primary_category": "identity",
                "categories": ["identity"],
                "source_fact_ids": ["f1"],
                "confidence": 1.0,
                "transformation": "direct",
                "locked": True,
            },
            {
                "text": "solo",
                "primary_category": "composition",
                "categories": ["composition"],
                "source_fact_ids": ["f2"],
                "confidence": 0.95,
                "transformation": "direct",
                "locked": False,
            },
        ],
        "execution_time": 0.25,
    }
    audit_path.write_text(json.dumps(audit_data), encoding="utf-8")

    response = client.get(f"/api/review/item?image_path={img_path.as_posix()}")
    assert response.status_code == 200
    data = response.json()

    assert data["validation_status"] == "repaired"
    assert data["validation_report"]["status"] == "repaired"
    assert len(data["validation_report"]["issues"]) == 1
    assert data["mode"] == "character"
    assert data["format"] == "tags"
    assert len(data["traceability"]) == 2
    assert data["traceability"][0]["text"] == "1girl"
    assert data["traceability"][0]["locked"] is True


def test_api_batch_progress_validation_counters(client):
    """Checks GET /api/batch/progress includes valid_count, repaired_count, and rejected_count."""
    # Set known state in batch_state
    prog = BatchProgress(
        current=10,
        total=10,
        completed=10,
        valid_count=7,
        repaired_count=2,
        rejected_count=1,
    )
    batch_state["progress"] = prog

    response = client.get("/api/batch/progress")
    assert response.status_code == 200
    data = response.json()

    assert data["valid_count"] == 7
    assert data["repaired_count"] == 2
    assert data["rejected_count"] == 1


def test_api_preview_with_modes_and_formats(client, temp_dataset):
    """Tests POST /api/batch/preview payload handling with mode/format and validation fields."""
    dataset_dir, img_path, _ = temp_dataset

    # Mock pipeline generate_preview to test API response shape cleanly
    token = CaptionToken(
        text="oil painting",
        primary_category=SemanticCategory.STYLE,
        categories=[SemanticCategory.STYLE],
        locked=False,
    )
    val_report = ValidationReport(
        status=ValidationStatus.VALID,
        issues=[],
        repaired_caption="oil painting",
    )
    cap_res = CaptionResult(
        caption="oil painting",
        mode=CaptionMode.NATURAL,
        tags=[],
        execution_time=0.12,
        metadata={"validation_status": "valid"},
    )
    cap_res.tokens = [token]
    cap_res.validation_report = val_report

    with patch("hyper_captioner.api.app.get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()
        mock_pipeline.generate_preview.return_value = [(img_path, cap_res)]
        mock_get_pipeline.return_value = mock_pipeline

        payload = {
            "dataset_path": str(dataset_dir),
            "sample_count": 1,
            "caption_mode": "style",
            "caption_format": "natural",
            "trigger_placement": "prepend",
            "write_audit": True,
        }
        response = client.post("/api/batch/preview", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "previews" in data
        assert len(data["previews"]) == 1
        item = data["previews"][0]
        assert item["caption"] == "oil painting"
        assert item["validation_status"] == "valid"
        assert "tokens" in item
        assert len(item["tokens"]) == 1
        assert item["tokens"][0]["text"] == "oil painting"
