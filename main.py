"""
Hyper Captioner - Command-Line Interface and Production Runner.
Provides fully functional headless dataset captioning, scanning, vocabulary management, and export.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Enforce UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Offline defaults
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")

# Ensure package is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hyper_captioner.caption_modes.base import BaseCaptionMode
from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionMode,
    CaptionModeType,
    CharacterConfig,
    LoRAStrategy,
    ModelSource,
    PresetConfig,
    TriggerConfig,
    TriggerPlacement,
    VRAMMode,
    WD14Device,
)
from hyper_captioner.dataset.exporter import create_backup, export_master_metadata
from hyper_captioner.dataset.manager import DatasetPipeline
from hyper_captioner.dataset.scanner import scan_dataset
from hyper_captioner.pipeline.presets import PresetManager
from hyper_captioner.vocabulary.database import DatasetVocabularyIndex


def cmd_scan(args):
    dataset_path = Path(args.dataset).resolve()
    print("=" * 65)
    print(f"SCANNING DATASET: {dataset_path}")
    print("=" * 65)
    scan = scan_dataset(dataset_path, verify_integrity=args.verify)
    print(f"Total Images        : {scan.total_images}")
    print(f"With Captions (.txt): {len(scan.with_captions)}")
    print(f"Missing Captions    : {len(scan.missing_captions)}")
    print(f"Orphan Captions     : {len(scan.orphan_captions)}")
    print(f"Empty Captions      : {len(scan.empty_captions)}")
    if args.verify:
        print(f"Corrupted Images    : {len(scan.corrupted_images)}")
    print("=" * 65)


def cmd_caption(args):
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    preset_mgr = PresetManager()
    preset_name = args.preset or "Character LoRA"
    # Find matching preset case-insensitively
    preset_found = None
    for name in preset_mgr.list_presets():
        if name.lower() == preset_name.lower():
            preset_found = preset_mgr.get_preset(name)
            break
    preset = preset_found or preset_mgr.get_preset("Character LoRA")

    # 1. Resolve Active Caption Mode Contract
    mode_name = getattr(args, "mode", "character") or "character"
    if mode_name.lower() in ("hybrid", "tag", "natural"):
        mode_name = "character"
    caption_mode_obj = get_caption_mode(mode_name)

    # 2. Resolve Caption Format
    format_name = getattr(args, "format", "tags") or "tags"
    caption_format_obj = CaptionFormat(format_name.lower())

    # 3. Resolve Trigger Configuration
    trigger_word = getattr(args, "trigger", "") or ""
    trigger_placement_val = getattr(args, "trigger_placement", "prepend") or "prepend"
    trigger_placement_enum = TriggerPlacement(trigger_placement_val.lower())
    absorb_traits = (
        getattr(preset.character, "prune_reference_from_caption", False)
        if preset and preset.character
        else False
    )
    trigger_config = TriggerConfig(
        word=trigger_word,
        placement=trigger_placement_enum,
        absorb_stable_traits=absorb_traits,
    )

    # 4. Resolve Audit Mode
    write_audit = bool(getattr(args, "audit", False))

    # Override preset settings with CLI flags if provided
    if caption_format_obj == CaptionFormat.NATURAL:
        preset.caption_mode = CaptionMode.NATURAL
    else:
        preset.caption_mode = CaptionMode.HYBRID

    if args.strategy:
        preset.lora_strategy = LoRAStrategy(args.strategy.lower())
    if trigger_word:
        preset.character.trigger_word = trigger_word
    if args.character_name:
        preset.character.name = args.character_name
    if args.ref_desc:
        preset.character.reference_description = args.ref_desc
    if args.wd14_threshold:
        preset.wd14_general_threshold = float(args.wd14_threshold)
    if args.wd14_device:
        preset.wd14_device = WD14Device(args.wd14_device.lower())
    if args.vram_mode:
        preset.vram_mode = VRAMMode(args.vram_mode.lower())
    if args.model_source:
        preset.model_source = ModelSource(args.model_source.lower())
    if args.underscores:
        preset.keep_underscores = True

    if args.backup:
        print("[*] Creating pre-run dataset backup...")
        create_backup(dataset_path)

    pipeline = DatasetPipeline(
        model_source=preset.model_source,
        wd14_device=preset.wd14_device,
    )

    # 1. Preview Mode
    if args.preview:
        sample_count = int(args.preview)
        scan = scan_dataset(dataset_path)
        sample_pool = scan.missing_captions or scan.with_captions
        if not sample_pool:
            print("No images found for preview.")
            return

        print("\n" + "=" * 65)
        print(f"GENERATING PREVIEW ({min(sample_count, len(sample_pool))} images)")
        print(
            f"Mode: {caption_mode_obj.name.upper()} | "
            f"Format: {caption_format_obj.value.upper()} | "
            f"LoRA: {preset.lora_strategy.value.upper()} | "
            f"Trigger: '{trigger_config.word}' ({trigger_config.placement.value})"
        )
        print("=" * 65)

        previews = pipeline.generate_preview(
            sample_pool,
            preset,
            sample_count=sample_count,
            caption_mode=caption_mode_obj,
            caption_format=caption_format_obj,
            trigger_config=trigger_config,
            write_audit=write_audit,
        )
        for idx, (path, res) in enumerate(previews, 1):
            print(f"\n[{idx}/{len(previews)}] Image: {path.name}")
            print(f"Caption: \"{res.caption}\"")
            if hasattr(res, "validation_report") and res.validation_report:
                vr = res.validation_report
                print(
                    f"Validation: status={vr.status.value} "
                    f"(repairs={len(vr.repairs)}, violations={len(vr.violations)})"
                )
            print(f"Provenance ({len(res.tags)} tags):")
            for t in res.tags[:8]:
                print(f"  - {t.text:<20} [{t.source} | {t.confidence:.2f}]")

        print("\n" + "=" * 65)
        print("Preview completed. Run without --preview to caption the entire dataset.")
        return

    # 2. Full Batch Execution
    print("\n" + "=" * 65)
    print("HYPER CAPTIONER - BATCH DATASET CAPTIONING")
    print(f"Dataset Path : {dataset_path}")
    print(f"Preset       : {preset.name}")
    print(f"Caption Mode : {caption_mode_obj.name}")
    print(f"Format       : {caption_format_obj.value}")
    print(f"LoRA Target  : {preset.lora_strategy.value}")
    print(f"Trigger Word : {trigger_config.word or '(None)'} ({trigger_config.placement.value})")
    print(f"Audit Mode   : {'Enabled (.audit.json)' if write_audit else 'Disabled'}")
    print(f"Resume Mode  : {'Skip Existing' if not args.overwrite else 'Overwrite All'}")
    print("=" * 65 + "\n")

    def progress_printer(p):
        print(f"[{p.current}/{p.total}] ({p.percent}%) {p.current_file} -> Status: {p.status}")
        if p.last_caption and p.status != "idle":
            snippet = (p.last_caption[:80] + "...") if len(p.last_caption) > 80 else p.last_caption
            print(f"    Caption: \"{snippet}\"")

    progress = pipeline.process_dataset(
        dataset_path,
        preset=preset,
        skip_existing=not args.overwrite,
        overwrite=args.overwrite,
        progress_callback=progress_printer,
        caption_mode=caption_mode_obj,
        caption_format=caption_format_obj,
        trigger_config=trigger_config,
        write_audit=write_audit,
    )

    print("\n" + "=" * 65)
    print("BATCH RUN COMPLETED")
    print(f"Total Processed   : {progress.completed}")
    print(f"Skipped           : {progress.skipped}")
    print(f"Failed            : {progress.failed}")
    print(f"Valid Captions    : {progress.valid_count}")
    print(f"Repaired Captions : {progress.repaired_count}")
    print(f"Rejected Captions : {progress.rejected_count}")
    print("=" * 65)


def cmd_vocab(args):
    dataset_path = Path(args.dataset).resolve()
    vocab = DatasetVocabularyIndex()
    vocab.index(dataset_path)

    if args.replace:
        old_tag, new_tag = args.replace[0], args.replace[1]
        print(f"Replacing '{old_tag}' -> '{new_tag}' across dataset...")
        modified = vocab.replace_tag(old_tag, new_tag, dataset_path=dataset_path)
        print(f"[OK] Replaced in {modified} files.")
        return

    if args.delete:
        target = args.delete
        print(f"Removing tag '{target}' across dataset...")
        modified = vocab.delete_tag(target, dataset_path=dataset_path)
        print(f"[OK] Removed from {modified} files.")
        return

    if args.search:
        print(f"\nSearching tags matching '{args.search}':")
        matches = vocab.find_variants(args.search)
        for tag, count in matches[:30]:
            print(f"  {tag:<30} -> {count}")
        return

    # Default: Top tags
    top_limit = int(args.top) if args.top else 50
    print(f"\nTOP {top_limit} TAGS IN DATASET:")
    print("-" * 45)
    for tag, count in vocab.get_top_tags(top_limit):
        print(f"  {tag:<32} {count:>5}")
    print("-" * 45)


def cmd_export(args):
    dataset_path = Path(args.dataset).resolve()
    fmt = args.format or "csv"
    out = export_master_metadata(dataset_path, export_format=fmt, output_filename=args.output)
    print(f"[OK] Master metadata exported to: {out}")


def cmd_backup(args):
    dataset_path = Path(args.dataset).resolve()
    dest = create_backup(dataset_path)
    print(f"[OK] Backup saved to: {dest}")


def cmd_ui(args):
    import uvicorn
    from hyper_captioner.api.app import app
    port = int(args.port or 7860)
    host = args.host or "127.0.0.1"
    print(f"\n[*] Launching Hyper Captioner UI on http://{host}:{port} ...")
    uvicorn.run(app, host=host, port=port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hyper Captioner - AI Dataset Captioning for LoRA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # SCAN
    scan_p = subparsers.add_parser("scan", help="Scan dataset health and status")
    scan_p.add_argument("--dataset", "-d", required=True, help="Path to dataset directory")
    scan_p.add_argument("--verify", action="store_true", help="Verify image file integrity")
    scan_p.set_defaults(func=cmd_scan)

    # CAPTION
    cap_p = subparsers.add_parser("caption", help="Run dataset captioning pipeline")
    cap_p.add_argument("--dataset", "-d", required=True, help="Path to dataset directory")
    cap_p.add_argument("--preset", "-p", default="Character LoRA", help="Preset name")
    cap_p.add_argument(
        "--mode",
        "-m",
        choices=["character", "style", "outfit", "pose", "concept", "hybrid", "tag", "natural"],
        default="character",
        help="Caption mode contract (or legacy alias: hybrid, tag, natural)",
    )
    cap_p.add_argument(
        "--format",
        "-f",
        choices=["tags", "structured", "natural"],
        default="tags",
        help="Caption output format",
    )
    cap_p.add_argument(
        "--trigger-placement",
        choices=["prepend", "append", "wrap", "omit"],
        default="prepend",
        help="Trigger word placement policy",
    )
    cap_p.add_argument(
        "--audit",
        "--debug-pipeline",
        dest="audit",
        action="store_true",
        help="Persist rich .audit.json sidecars alongside training captions and output validation trace",
    )
    cap_p.add_argument("--strategy", "-s", choices=["character", "style", "concept", "general"], help="LoRA strategy")
    cap_p.add_argument("--trigger", "-t", default="", help="Trigger word for main subject")
    cap_p.add_argument("--character-name", default="", help="Character name")
    cap_p.add_argument("--ref-desc", default="", help="Reference description")
    cap_p.add_argument("--preview", type=int, nargs="?", const=5, help="Generate preview on N images (default 5)")
    cap_p.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt sidecars")
    cap_p.add_argument("--wd14-threshold", type=float, help="WD14 confidence threshold")
    cap_p.add_argument("--wd14-device", choices=["auto", "cuda", "cpu"], help="WD14 device")
    cap_p.add_argument("--vram-mode", choices=["balanced", "max_speed", "min_vram"], help="VRAM execution mode")
    cap_p.add_argument("--model-source", choices=["local_only", "download_missing"], help="Model source mode")
    cap_p.add_argument("--underscores", action="store_true", help="Format tags with underscores instead of spaces")
    cap_p.add_argument("--backup", action="store_true", help="Create backup before processing")
    cap_p.set_defaults(func=cmd_caption)

    # VOCAB
    vocab_p = subparsers.add_parser("vocab", help="Dataset vocabulary analysis and batch normalization")
    vocab_p.add_argument("--dataset", "-d", required=True, help="Path to dataset directory")
    vocab_p.add_argument("--top", type=int, default=50, help="Print top N tags")
    vocab_p.add_argument("--search", help="Search tag occurrences")
    vocab_p.add_argument("--replace", nargs=2, metavar=("OLD", "NEW"), help="Replace tag across dataset")
    vocab_p.add_argument("--delete", help="Remove tag across dataset")
    vocab_p.set_defaults(func=cmd_vocab)

    # EXPORT
    exp_p = subparsers.add_parser("export", help="Export dataset captions (txt, zip, sidecars, csv, json)")
    exp_p.add_argument("--dataset", "-d", required=True, help="Path to dataset directory")
    exp_p.add_argument("--format", choices=["txt", "zip", "sidecars", "csv", "json"], default="txt", help="Export format")
    exp_p.add_argument("--output", "-o", help="Custom output filename")
    exp_p.set_defaults(func=cmd_export)

    # BACKUP
    bak_p = subparsers.add_parser("backup", help="Create dataset backup folder")
    bak_p.add_argument("--dataset", "-d", required=True, help="Path to dataset directory")
    bak_p.set_defaults(func=cmd_backup)

    # UI
    ui_p = subparsers.add_parser("ui", help="Launch Web UI CyberDeck")
    ui_p.add_argument("--port", type=int, default=7860, help="Server port")
    ui_p.add_argument("--host", default="127.0.0.1", help="Server host")
    ui_p.set_defaults(func=cmd_ui)

    return parser


def parse_args_helper(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    if args_list is not None:
        args = parser.parse_args(args_list)
        raw_tokens = list(args_list)
    else:
        args = parser.parse_args()
        raw_tokens = list(sys.argv[1:])

    if getattr(args, "command", None) == "caption":
        # Check if format was explicitly specified
        format_specified = any(
            token in ("-f", "--format") or token.startswith("--format=") or token.startswith("-f=")
            for token in raw_tokens
        )

        mode_val = (getattr(args, "mode", None) or "").lower()
        if mode_val == "hybrid":
            args.mode = "character"
            if not format_specified:
                args.format = "tags"
        elif mode_val == "tag":
            args.mode = "character"
            if not format_specified:
                args.format = "tags"
        elif mode_val == "natural":
            args.mode = "character"
            if not format_specified:
                args.format = "natural"

        audit_flag = bool(getattr(args, "audit", False))
        args.audit = audit_flag
        args.debug_pipeline = audit_flag

    return args


def main():
    if len(sys.argv) == 1:
        # Default with no args: launch UI
        sys.argv.append("ui")

    args = parse_args_helper()
    if hasattr(args, "func"):
        args.func(args)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()