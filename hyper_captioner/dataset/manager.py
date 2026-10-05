"""
Dataset Processing Pipeline: Resumable batch runner, preview generation, and single-image captioning.
Wires end-to-end two-stage architecture:
Stage1Extractor -> SemanticFilter -> CaptionBuilder -> SemanticValidator -> Sidecar Exporters.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from hyper_captioner.caption_modes.base import BaseCaptionMode
from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionMode,
    CaptionModeType,
    CaptionResult,
    CaptionToken,
    FactItem,
    ImageRecord,
    ModelSource,
    PresetConfig,
    SemanticCategory,
    StructuredVisualFacts,
    TagCategory,
    TagItem,
    TriggerConfig,
    TriggerPlacement,
    ValidationReport,
    ValidationStatus,
    VRAMMode,
    WD14Device,
)
from hyper_captioner.core.vram import handle_vram_lifecycle
from hyper_captioner.dataset.exporter import write_audit_sidecar, write_sidecar
from hyper_captioner.dataset.scanner import DatasetScanResult, scan_dataset
from hyper_captioner.engines.joycaption import (
    JoyCaptionEngine,
    build_stage1_extraction_prompt,
    build_system_prompt,
)
from hyper_captioner.engines.wd14 import WD14Engine
from hyper_captioner.pipeline.builder import CaptionBuilder
from hyper_captioner.pipeline.fusion import AttributeFusionEngine
from hyper_captioner.pipeline.semantic_filter import SemanticFilter
from hyper_captioner.pipeline.stage1_extractor import Stage1Extractor
from hyper_captioner.pipeline.validator import SemanticValidator
from hyper_captioner.vocabulary.normalizer import VocabularyNormalizer

logger = logging.getLogger(__name__)


@dataclass
class BatchProgress:
    current: int = 0
    total: int = 0
    current_file: str = ""
    status: str = "idle"  # "idle", "processing", "paused", "completed", "cancelled"
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    last_caption: str = ""
    records: List[ImageRecord] = field(default_factory=list)
    valid_count: int = 0
    repaired_count: int = 0
    rejected_count: int = 0

    def to_dict(self) -> Dict:
        pct = round((self.current / max(1, self.total)) * 100, 1) if self.total > 0 else 0.0
        return {
            "current": self.current,
            "total": self.total,
            "percent": pct,
            "current_file": self.current_file,
            "status": self.status,
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": self.failed,
            "last_caption": self.last_caption,
            "valid_count": self.valid_count,
            "repaired_count": self.repaired_count,
            "rejected_count": self.rejected_count,
        }


class DatasetPipeline:
    """
    Coordinates model loading, inference, Stage 1 visual fact extraction,
    meaning-based semantic filtering, format building, semantic validation,
    and sidecar generation (clean .txt training captions and .audit.json metadata).
    """

    def __init__(
        self,
        model_source: ModelSource = ModelSource.LOCAL_ONLY,
        wd14_device: WD14Device = WD14Device.AUTO,
    ):
        self.model_source = model_source
        self.wd14_device = wd14_device

        self.normalizer = VocabularyNormalizer()
        self.fusion = AttributeFusionEngine(normalizer=self.normalizer)
        self.builder = CaptionBuilder()

        # Two-Stage Redesign Components
        self.extractor = Stage1Extractor()
        self.semantic_filter = SemanticFilter()
        self.validator = SemanticValidator()

        self.wd14: Optional[WD14Engine] = None
        self.joycaption: Optional[JoyCaptionEngine] = None

    def _ensure_models(self, preset: Optional[PresetConfig] = None):
        """Lazily loads models if not already initialized."""
        if preset is None:
            preset = PresetConfig()

        if self.wd14 is None:
            self.wd14 = WD14Engine(
                device_choice=preset.wd14_device or self.wd14_device,
                model_source=preset.model_source or self.model_source,
            )
            self.wd14.load()

        if self.joycaption is None:
            self.joycaption = JoyCaptionEngine(
                model_source=preset.model_source or self.model_source,
                load_in_4bit=True,
            )
            self.joycaption.load()

    def caption_image(
        self,
        image_path: Path,
        preset: Optional[PresetConfig] = None,
        user_tags: Optional[List[TagItem]] = None,
        locked_tags: Optional[List[str]] = None,
        write_txt: bool = True,
        caption_mode: Optional[Union[BaseCaptionMode, str]] = None,
        caption_format: Optional[Union[CaptionFormat, str]] = None,
        trigger_config: Optional[TriggerConfig] = None,
        write_audit: bool = False,
        save_audit: bool = False,
        **kwargs,
    ) -> CaptionResult:
        """
        Processes a single image through the two-stage general-purpose captioning pipeline:
        Stage 1 Extraction ➔ Semantic Filtering ➔ Format Building ➔ Validation & Repair ➔ Sidecar persistence.
        """
        if preset is None:
            preset = PresetConfig()

        self._ensure_models(preset)
        t0 = time.time()

        should_write_audit = write_audit or save_audit or kwargs.get("audit", False)

        # 1. Resolve Active Caption Mode Contract
        target_mode_spec = caption_mode
        if target_mode_spec is None:
            target_mode_spec = getattr(preset, "caption_mode", None)

        mode: BaseCaptionMode
        if isinstance(target_mode_spec, BaseCaptionMode):
            mode = target_mode_spec
        elif isinstance(target_mode_spec, (str, CaptionModeType)):
            try:
                mode = get_caption_mode(str(target_mode_spec).lower())
            except KeyError:
                spec_lower = str(target_mode_spec).lower()
                if "style" in spec_lower:
                    mode = get_caption_mode("style")
                elif "outfit" in spec_lower:
                    mode = get_caption_mode("outfit")
                elif "pose" in spec_lower:
                    mode = get_caption_mode("pose")
                elif "concept" in spec_lower:
                    mode = get_caption_mode("concept")
                else:
                    mode = get_caption_mode("character")
        else:
            mode = get_caption_mode("character")

        # 2. Resolve Caption Format
        if caption_format is not None:
            c_format = CaptionFormat(caption_format) if isinstance(caption_format, str) else caption_format
        else:
            preset_mode = getattr(preset, "caption_mode", None)
            if preset_mode == CaptionMode.NATURAL or (isinstance(caption_mode, str) and caption_mode.lower() == "natural"):
                c_format = CaptionFormat.NATURAL
            else:
                c_format = CaptionFormat.TAGS

        # 3. Resolve Trigger Configuration
        if trigger_config is not None:
            trigger_cfg = trigger_config
        elif preset.character and preset.character.trigger_word:
            trigger_cfg = TriggerConfig(
                word=preset.character.trigger_word,
                absorb_stable_traits=preset.character.prune_reference_from_caption,
            )
        else:
            trigger_cfg = TriggerConfig()

        # 4. Load Image & Extract Basic Metadata
        img = Image.open(image_path).convert("RGB")
        image_metadata = {"width": img.width, "height": img.height}

        # 5. WD14 Auxiliary Tag Predictions
        wd14_tags = self.wd14.predict(
            img,
            general_threshold=preset.wd14_general_threshold,
            character_threshold=preset.wd14_character_threshold,
        )

        # 6. JoyCaption VLM Generation using build_stage1_extraction_prompt()
        mode_instructions = mode.build_extraction_instructions() if hasattr(mode, "build_extraction_instructions") else ""
        prompt = build_stage1_extraction_prompt(mode_instructions=mode_instructions)
        joy_texts = self.joycaption.generate(
            [img],
            prompt=prompt,
            max_new_tokens=400,
            temperature=0.0,
        )
        joy_text = joy_texts[0] if joy_texts else ""

        # 7. Stage 1 Extraction: Standardized JSON / Blocks / Fused Fallback ➔ StructuredVisualFacts
        facts = self.extractor.extract(
            joycaption_raw=joy_text,
            wd14_tags=wd14_tags,
            image_metadata=image_metadata,
        )

        # 8. Inject User Locked Tags into Facts (inviolable user lock preservation)
        if locked_tags:
            for lt in locked_tags:
                clean_lt = self.extractor._clean_fact_text(lt)
                if not clean_lt:
                    continue
                matched = False
                for f in facts.all_facts():
                    if f.text.lower() == clean_lt.lower():
                        f.locked = True
                        matched = True
                if not matched:
                    cat = self.extractor._heuristic_categorize(clean_lt)
                    new_fact = FactItem(
                        id=f"user_locked_{len(facts.all_facts()) + 1}",
                        text=clean_lt,
                        primary_category=cat,
                        categories=[cat],
                        confidence=1.0,
                        source="user",
                        locked=True,
                        raw_text=lt,
                    )
                    facts.facts_by_category.setdefault(cat, []).append(new_fact)

        if user_tags:
            for ut in user_tags:
                clean_ut = self.extractor._clean_fact_text(ut.text)
                if not clean_ut:
                    continue
                matched = False
                for f in facts.all_facts():
                    if f.text.lower() == clean_ut.lower():
                        if ut.locked:
                            f.locked = True
                        matched = True
                if not matched:
                    cat = self.extractor._heuristic_categorize(clean_ut)
                    new_fact = FactItem(
                        id=f"user_{len(facts.all_facts()) + 1}",
                        text=clean_ut,
                        primary_category=cat,
                        categories=[cat],
                        confidence=ut.confidence if ut.confidence is not None else 1.0,
                        source="user",
                        locked=ut.locked,
                        raw_text=ut.text,
                    )
                    facts.facts_by_category.setdefault(cat, []).append(new_fact)

        # 9. Semantic Filter: Meaning-based gating, lock priority, uncertainty & hype pruning
        filter_result = self.semantic_filter.filter_facts(
            mode=mode,
            facts=facts,
            trigger_cfg=trigger_cfg,
        )

        # 10. Caption Builder: Category progression hierarchy & format assembly
        elapsed = time.time() - t0
        raw_caption, tokens = self.builder.build(
            mode_or_tags=mode,
            filter_result=filter_result,
            trigger_cfg=trigger_cfg,
            format_type=c_format,
            keep_underscores=preset.keep_underscores,
            raw_joycaption=joy_text,
            execution_time=elapsed,
        )

        # 11. Semantic Validator: Contradiction checks, safe auto-repair boundary
        val_report = self.validator.validate(
            caption=raw_caption,
            tokens=tokens,
            mode=mode,
            facts=facts,
            trigger_cfg=trigger_cfg,
        )

        # 12. Resolve Final Caption & Final Tokens
        final_caption = (
            val_report.repaired_caption
            if (val_report and val_report.repaired_caption)
            else raw_caption
        )
        repaired_toks = getattr(val_report, "repaired_tokens", None)
        if repaired_toks is not None:
            final_tokens = repaired_toks
        elif val_report.repaired_caption and val_report.repaired_caption != raw_caption:
            final_tokens = [
                t for t in tokens
                if t.text.lower() in val_report.repaired_caption.lower()
            ]
        else:
            final_tokens = tokens

        # 13. Write Clean .txt Sidecar (Training caption only, never contains JSON or audit metadata)
        if write_txt:
            write_sidecar(image_path, final_caption)

        # 14. Write .audit.json Sidecar if requested
        if should_write_audit:
            audit_data = {
                "image": image_path.name,
                "caption": final_caption,
                "mode": mode.name if hasattr(mode, "name") else str(mode),
                "format": c_format.value if hasattr(c_format, "value") else str(c_format),
                "validation": val_report.to_dict() if hasattr(val_report, "to_dict") else val_report,
                "facts": facts.to_dict() if hasattr(facts, "to_dict") else facts,
                "tokens": [t.to_dict() for t in final_tokens] if final_tokens else [],
                "execution_time": round(elapsed, 4),
            }
            write_audit_sidecar(image_path, audit_data)

        # 15. Package CaptionResult
        tags = [
            TagItem(
                text=t.text,
                source=t.transformation,
                confidence=t.confidence,
                category=TagCategory.OTHER,
                locked=t.locked,
                raw_text=t.text,
            )
            for t in final_tokens
        ]
        metadata = {
            "format": c_format.value if hasattr(c_format, "value") else str(c_format),
            "trigger": trigger_cfg.to_dict() if hasattr(trigger_cfg, "to_dict") else trigger_cfg,
            "facts_count": len(facts.all_facts()) if hasattr(facts, "all_facts") else 0,
            "parse_method": getattr(facts, "parse_method", ""),
            "validation_status": (
                val_report.status.value
                if hasattr(val_report.status, "value")
                else str(val_report.status)
            ),
        }
        resolved_mode = (
            caption_mode
            if caption_mode is not None
            else (preset.caption_mode if preset else CaptionMode.HYBRID)
        )
        caption_result = CaptionResult(
            caption=final_caption,
            mode=resolved_mode,
            tags=tags,
            raw_joycaption=joy_text,
            raw_wd14_tags=wd14_tags or [],
            execution_time=elapsed,
            metadata=metadata,
        )
        caption_result.mode_name = mode.name if hasattr(mode, "name") else str(mode)
        caption_result.tokens = final_tokens
        caption_result.validation_report = val_report
        caption_result.facts = facts

        handle_vram_lifecycle(preset.vram_mode, step_name="after_joycaption")
        return caption_result

    def generate_preview(
        self,
        image_paths: List[Path],
        preset: Optional[PresetConfig] = None,
        sample_count: int = 5,
        caption_mode: Optional[Union[BaseCaptionMode, str]] = None,
        caption_format: Optional[Union[CaptionFormat, str]] = None,
        trigger_config: Optional[TriggerConfig] = None,
        write_audit: bool = False,
        **kwargs,
    ) -> List[Tuple[Path, CaptionResult]]:
        """
        Generates preview captions on a small sample of images without
        writing sidecar files to disk.
        """
        if preset is None:
            preset = PresetConfig()
        samples = image_paths[:sample_count]
        results: List[Tuple[Path, CaptionResult]] = []

        for p in samples:
            res = self.caption_image(
                p,
                preset=preset,
                write_txt=False,
                caption_mode=caption_mode,
                caption_format=caption_format,
                trigger_config=trigger_config,
                write_audit=write_audit,
                **kwargs,
            )
            results.append((p, res))

        return results

    def process_dataset(
        self,
        dataset_path: Path,
        preset: Optional[PresetConfig] = None,
        skip_existing: bool = True,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
        caption_mode: Optional[Union[BaseCaptionMode, str]] = None,
        caption_format: Optional[Union[CaptionFormat, str]] = None,
        trigger_config: Optional[TriggerConfig] = None,
        write_audit: bool = False,
        save_audit: bool = False,
        **kwargs,
    ) -> BatchProgress:
        """
        Executes resumable batch processing across an entire dataset directory.
        Tracks valid_count, repaired_count, and rejected_count validation statistics on BatchProgress.
        """
        if preset is None:
            preset = PresetConfig()

        should_write_audit = write_audit or save_audit or kwargs.get("audit", False)

        scan = scan_dataset(dataset_path)
        all_images = (
            scan.with_captions + scan.missing_captions
            if (overwrite or not skip_existing)
            else scan.missing_captions
        )
        all_images = sorted(all_images)

        progress = BatchProgress(
            current=0,
            total=len(all_images),
            status="processing",
            completed=0,
            skipped=len(scan.with_captions) if skip_existing and not overwrite else 0,
            failed=0,
            valid_count=0,
            repaired_count=0,
            rejected_count=0,
        )

        if not all_images:
            progress.status = "completed"
            if progress_callback:
                progress_callback(progress)
            return progress

        self._ensure_models(preset)

        for idx, img_path in enumerate(all_images, 1):
            if cancel_requested and cancel_requested():
                progress.status = "cancelled"
                break

            progress.current = idx
            progress.current_file = img_path.name

            # Check if skipping existing
            txt_path = img_path.with_suffix(".txt")
            if skip_existing and not overwrite and txt_path.exists():
                try:
                    if txt_path.stat().st_size > 0:
                        progress.skipped += 1
                        if progress_callback:
                            progress_callback(progress)
                        continue
                except Exception:
                    pass

            record = ImageRecord(image_path=img_path, caption_path=txt_path)

            try:
                result = self.caption_image(
                    image_path=img_path,
                    preset=preset,
                    write_txt=True,
                    caption_mode=caption_mode,
                    caption_format=caption_format,
                    trigger_config=trigger_config,
                    write_audit=should_write_audit,
                )
                record.status = "completed"
                record.caption = result.caption
                record.tags = result.tags
                progress.completed += 1
                progress.last_caption = result.caption

                # Track batch validation counters
                val_report = getattr(result, "validation_report", None)
                if val_report:
                    status_val = (
                        val_report.status.value
                        if isinstance(val_report.status, ValidationStatus)
                        else str(val_report.status).lower()
                    )
                    if status_val == "valid":
                        progress.valid_count += 1
                    elif status_val == "repaired":
                        progress.repaired_count += 1
                    elif status_val == "rejected":
                        progress.rejected_count += 1
                    else:
                        progress.valid_count += 1
                else:
                    progress.valid_count += 1

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                record.status = "failed"
                record.error = str(e)
                progress.failed += 1

            progress.records.append(record)

            if progress_callback:
                progress_callback(progress)

            handle_vram_lifecycle(preset.vram_mode, step_name="after_batch")

        if progress.status != "cancelled":
            progress.status = "completed"

        if progress_callback:
            progress_callback(progress)

        return progress
