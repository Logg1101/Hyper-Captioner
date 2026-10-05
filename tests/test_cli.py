"""
Tests for CLI Parser, Mode Contracts, Format Separation, and Backward Compatibility.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from main import build_parser, parse_args_helper, cmd_caption
from hyper_captioner.caption_modes.base import BaseCaptionMode
from hyper_captioner.core.types import CaptionFormat, TriggerPlacement, TriggerConfig
from hyper_captioner.dataset.manager import BatchProgress


def test_cli_mode_and_format_aliases():
    """Test legacy hybrid, tag, and natural mode aliasing to character mode and respective formats."""
    # 1. Legacy hybrid -> character mode, tags format
    args_hybrid = parse_args_helper(["caption", "--dataset", "test_ds", "--mode", "hybrid"])
    assert args_hybrid.mode == "character"
    assert args_hybrid.format == "tags"

    # 2. Legacy tag -> character mode, tags format
    args_tag = parse_args_helper(["caption", "--dataset", "test_ds", "-m", "tag"])
    assert args_tag.mode == "character"
    assert args_tag.format == "tags"

    # 3. Legacy natural -> character mode, natural format
    args_nat = parse_args_helper(["caption", "--dataset", "test_ds", "--mode", "natural"])
    assert args_nat.mode == "character"
    assert args_nat.format == "natural"

    args_nat_short = parse_args_helper(["caption", "--dataset", "test_ds", "-m", "natural"])
    assert args_nat_short.mode == "character"
    assert args_nat_short.format == "natural"


def test_cli_new_modes_and_trigger_placement():
    """Test general-purpose mode contracts, trigger placement policies, and --audit flag."""
    modes = ["character", "style", "outfit", "pose", "concept"]
    for m in modes:
        args = parse_args_helper(["caption", "--dataset", "test_ds", "--mode", m])
        assert args.mode == m
        assert args.format == "tags"

    # Trigger placements
    placements = ["prepend", "append", "wrap", "omit"]
    for p in placements:
        args = parse_args_helper([
            "caption",
            "--dataset", "test_ds",
            "--mode", "style",
            "--trigger", "civitai_style",
            "--trigger-placement", p,
        ])
        assert args.mode == "style"
        assert args.trigger == "civitai_style"
        assert args.trigger_placement == p

    # Default audit is False
    args_no_audit = parse_args_helper(["caption", "--dataset", "test_ds"])
    assert args_no_audit.audit is False

    # Explicit --audit
    args_audit = parse_args_helper(["caption", "--dataset", "test_ds", "--audit"])
    assert args_audit.audit is True


def test_cli_explicit_format_override():
    """Test that explicit --format overrides defaults and legacy mode aliases."""
    # hybrid with structured format
    args = parse_args_helper(["caption", "--dataset", "test_ds", "--mode", "hybrid", "--format", "structured"])
    assert args.mode == "character"
    assert args.format == "structured"

    # hybrid with natural format
    args_hn = parse_args_helper(["caption", "--dataset", "test_ds", "--mode", "hybrid", "-f", "natural"])
    assert args_hn.mode == "character"
    assert args_hn.format == "natural"

    # tag with structured format
    args_ts = parse_args_helper(["caption", "--dataset", "test_ds", "--mode", "tag", "-f", "structured"])
    assert args_ts.mode == "character"
    assert args_ts.format == "structured"

    # natural mode with tags format override
    args_nt = parse_args_helper(["caption", "--dataset", "test_ds", "--mode", "natural", "--format", "tags"])
    assert args_nt.mode == "character"
    assert args_nt.format == "tags"

    # natural mode with structured format override
    args_ns = parse_args_helper(["caption", "--dataset", "test_ds", "-m", "natural", "-f", "structured"])
    assert args_ns.mode == "character"
    assert args_ns.format == "structured"

    # style mode with natural format
    args_sn = parse_args_helper(["caption", "--dataset", "test_ds", "--mode", "style", "--format", "natural"])
    assert args_sn.mode == "style"
    assert args_sn.format == "natural"


def test_cli_debug_pipeline_alias():
    """Test that --debug-pipeline sets audit=True as an alias."""
    args = parse_args_helper(["caption", "--dataset", "test_ds", "--debug-pipeline"])
    assert args.audit is True


def test_cli_default_caption_arguments():
    """Test default values when invoking caption command without optional flags."""
    args = parse_args_helper(["caption", "--dataset", "my_data"])
    assert args.dataset == "my_data"
    assert args.mode == "character"
    assert args.format == "tags"
    assert args.trigger_placement == "prepend"
    assert args.audit is False


def test_cmd_caption_execution_with_mocks(tmp_path, capsys):
    """Test cmd_caption wires mode contracts, formats, trigger configs, and audit flags to pipeline."""
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()

    fake_progress = BatchProgress(
        current=2,
        total=2,
        completed=2,
        skipped=0,
        failed=0,
        valid_count=1,
        repaired_count=1,
        rejected_count=0,
    )

    args = parse_args_helper([
        "caption",
        "--dataset", str(dataset_dir),
        "--mode", "style",
        "--format", "structured",
        "--trigger", "oil_painting",
        "--trigger-placement", "append",
        "--audit",
    ])

    with patch("main.DatasetPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_pipeline.process_dataset.return_value = fake_progress

        cmd_caption(args)

        mock_pipeline.process_dataset.assert_called_once()
        call_kwargs = mock_pipeline.process_dataset.call_args[1]

        # Verify resolved objects passed to process_dataset
        assert isinstance(call_kwargs["caption_mode"], BaseCaptionMode)
        assert call_kwargs["caption_mode"].name == "style"
        assert call_kwargs["caption_format"] == CaptionFormat.STRUCTURED
        assert isinstance(call_kwargs["trigger_config"], TriggerConfig)
        assert call_kwargs["trigger_config"].word == "oil_painting"
        assert call_kwargs["trigger_config"].placement == TriggerPlacement.APPEND
        assert call_kwargs["write_audit"] is True

        captured = capsys.readouterr()
        assert "BATCH RUN COMPLETED" in captured.out
        assert "Valid Captions    : 1" in captured.out
        assert "Repaired Captions : 1" in captured.out
        assert "Rejected Captions : 0" in captured.out
