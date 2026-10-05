"""
FastAPI application for Hyper Captioner CyberDeck production UI and REST API.
"""

import json
import logging
import os
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hyper_captioner.config import (
    AVAILABLE_WD14_MODELS,
    DEFAULT_JOYCAPTION_MODEL,
    DEFAULT_WD14_MODEL,
    load_app_config,
    save_app_config,
)
from hyper_captioner.caption_modes.registry import get_caption_mode, list_caption_modes
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionMode,
    CaptionModeType,
    CharacterConfig,
    LoRAStrategy,
    ModelSource,
    PresetConfig,
    TagCategory,
    TagItem,
    TriggerConfig,
    TriggerPlacement,
    ValidationReport,
    ValidationStatus,
    VRAMMode,
    WD14Device,
)
from hyper_captioner.core.vram import get_vram_info, is_cuda_available
from hyper_captioner.dataset.exporter import create_backup, export_master_metadata, write_sidecar
from hyper_captioner.dataset.manager import BatchProgress, DatasetPipeline
from hyper_captioner.dataset.scanner import IMAGE_EXTENSIONS, scan_dataset
from hyper_captioner.pipeline.presets import PresetManager
from hyper_captioner.vocabulary.database import DatasetVocabularyIndex
from hyper_captioner.vocabulary.normalizer import VocabularyNormalizer

logger = logging.getLogger(__name__)

app = FastAPI(title="Hyper Captioner", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global App State
preset_manager = PresetManager()
normalizer = VocabularyNormalizer()
vocab_index = DatasetVocabularyIndex()

global_pipeline: Optional[DatasetPipeline] = None
pipeline_lock = threading.Lock()

batch_state = {
    "active": False,
    "cancel_requested": False,
    "progress": BatchProgress(),
    "thread": None,
    "dataset_path": "",
    "log_history": [],
}


def get_pipeline(model_source=ModelSource.LOCAL_ONLY, wd14_device=WD14Device.AUTO) -> DatasetPipeline:
    global global_pipeline
    with pipeline_lock:
        if global_pipeline is None:
            global_pipeline = DatasetPipeline(model_source=model_source, wd14_device=wd14_device)
        return global_pipeline


# -------------------------------------------------------------
# Request Models
# -------------------------------------------------------------
class ScanRequest(BaseModel):
    dataset_path: str
    verify: bool = False


class PresetSaveRequest(BaseModel):
    name: str
    caption_mode: str = "character"
    caption_format: str = "tags"
    trigger_placement: str = "prepend"
    write_audit: bool = False
    lora_strategy: str = "character"
    trigger_word: str = ""
    character_name: str = ""
    reference_description: str = ""
    prune_reference: bool = False
    wd14_general_threshold: float = 0.35
    wd14_character_threshold: float = 0.60
    keep_underscores: bool = False
    filter_poisons: bool = True
    quality_boosters: bool = False
    vram_mode: str = "balanced"
    wd14_device: str = "auto"
    model_source: str = "local_only"
    custom_tags: str = ""
    blacklist: List[str] = []


class BatchStartRequest(BaseModel):
    dataset_path: str
    preset_name: str = "Character LoRA"
    caption_mode: Optional[str] = "character"
    caption_format: Optional[str] = "tags"
    trigger_placement: Optional[str] = "prepend"
    write_audit: bool = False
    lora_strategy: Optional[str] = None
    trigger_word: Optional[str] = None
    character_name: Optional[str] = None
    reference_description: Optional[str] = None
    wd14_threshold: Optional[float] = None
    keep_underscores: Optional[bool] = None
    vram_mode: Optional[str] = None
    wd14_device: Optional[str] = None
    model_source: Optional[str] = None
    overwrite: bool = False
    do_backup: bool = False


class PreviewRequest(BaseModel):
    dataset_path: str
    preset_name: str = "Character LoRA"
    sample_count: int = 5
    caption_mode: Optional[str] = "character"
    caption_format: Optional[str] = "tags"
    trigger_placement: Optional[str] = "prepend"
    write_audit: bool = False
    lora_strategy: Optional[str] = None
    trigger_word: Optional[str] = None
    keep_underscores: Optional[bool] = None


class SaveCaptionRequest(BaseModel):
    image_path: str
    caption: str
    locked_tags: List[str] = []


class VerifyTagRequest(BaseModel):
    image_path: str
    tag: str


class RegenerateRequest(BaseModel):
    image_path: str
    preset_name: str = "Character LoRA"
    locked_tags: List[str] = []
    user_tags: List[str] = []
    caption_mode: Optional[str] = "character"
    caption_format: Optional[str] = "tags"
    trigger_placement: Optional[str] = "prepend"
    write_audit: bool = False
    lora_strategy: Optional[str] = None
    trigger_word: Optional[str] = None
    keep_underscores: Optional[bool] = None



class VocabReplaceRequest(BaseModel):
    dataset_path: str
    old_tag: str
    new_tag: str


class VocabBatchNormalizeRequest(BaseModel):
    dataset_path: str
    mappings: Dict[str, str]


class VocabDeleteRequest(BaseModel):
    dataset_path: str
    target_tag: str


class ExportRequest(BaseModel):
    dataset_path: str
    format: str = "csv"
    output_filename: Optional[str] = None


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/api/health")
def api_health():
    vram = get_vram_info()
    return {
        "status": "online",
        "cuda_available": is_cuda_available(),
        "vram": vram,
        "batch_active": batch_state["active"],
    }


@app.post("/api/browse")
def api_browse_folder():
    """Opens a native Windows directory picker dialogue."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory()
        root.destroy()
        return {"path": folder or ""}
    except Exception as e:
        logger.warning(f"Native browse failed: {e}")
        return {"path": ""}


@app.post("/api/scan")
def api_scan(req: ScanRequest):
    p = Path(req.dataset_path.strip())
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Invalid dataset folder path.")

    scan = scan_dataset(p, verify_integrity=req.verify)
    return scan.to_dict()


@app.get("/api/modes")
def api_get_modes():
    mode_names = ["character", "style", "outfit", "pose", "concept"]
    modes_list = []
    details_dict = {}
    for name in mode_names:
        try:
            m = get_caption_mode(name)
            inc_cats = [c.value if hasattr(c, "value") else str(c) for c in sorted(m.include_categories, key=lambda x: str(x))]
            exc_cats = [c.value if hasattr(c, "value") else str(c) for c in sorted(m.exclude_categories, key=lambda x: str(x))]
            info = {
                "name": m.name,
                "description": m.description,
                "include_categories": inc_cats,
                "exclude_categories": exc_cats,
            }
        except Exception:
            info = {"name": name, "description": "", "include_categories": [], "exclude_categories": []}
        modes_list.append(info)
        details_dict[name] = info

    return {
        "modes": modes_list,
        "mode_names": mode_names,
        "details": details_dict,
        "formats": ["tags", "structured", "natural"],
        "trigger_placements": ["prepend", "append", "wrap", "omit"],
    }


@app.get("/api/presets")
def api_list_presets():
    return {
        "presets": preset_manager.list_presets(),
        "details": {name: preset_manager.get_preset(name).to_dict() for name in preset_manager.list_presets()},
    }


@app.post("/api/presets")
def api_save_preset(req: PresetSaveRequest):
    char_cfg = CharacterConfig(
        name=req.character_name,
        trigger_word=req.trigger_word,
        reference_description=req.reference_description,
        prune_reference_from_caption=req.prune_reference,
    )
    try:
        c_mode = CaptionMode(req.caption_mode)
    except ValueError:
        c_mode = req.caption_mode

    try:
        l_strat = LoRAStrategy(req.lora_strategy)
    except ValueError:
        l_strat = LoRAStrategy.CHARACTER

    preset = PresetConfig(
        name=req.name,
        caption_mode=c_mode,
        caption_format=req.caption_format,
        trigger_placement=req.trigger_placement,
        write_audit=req.write_audit,
        lora_strategy=l_strat,
        character=char_cfg,
        wd14_general_threshold=req.wd14_general_threshold,
        wd14_character_threshold=req.wd14_character_threshold,
        keep_underscores=req.keep_underscores,
        filter_poisons=req.filter_poisons,
        quality_boosters=req.quality_boosters,
        vram_mode=VRAMMode(req.vram_mode),
        wd14_device=WD14Device(req.wd14_device),
        model_source=ModelSource(req.model_source),
        custom_tags=req.custom_tags,
        blacklist=req.blacklist,
    )
    saved_path = preset_manager.save_preset(preset)
    return {"status": "ok", "preset": preset.to_dict(), "path": str(saved_path)}


@app.delete("/api/presets/{preset_name}")
def api_delete_preset(preset_name: str):
    success = preset_manager.delete_preset(preset_name)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete preset (either built-in or not found).")
    return {"status": "deleted", "preset": preset_name}


@app.get("/api/gallery")
def api_gallery(path: str = Query(...)):
    p = Path(path.strip())
    if not p.exists() or not p.is_dir():
        return {"images": []}

    files = []
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            txt_path = f.with_suffix(".txt")
            has_caption = txt_path.exists() and txt_path.stat().st_size > 0
            files.append({
                "path": str(f.as_posix()),
                "filename": f.name,
                "has_caption": has_caption,
            })
    return {"images": files}


@app.get("/api/image")
def api_serve_image(path: str = Query(...)):
    decoded = urllib.parse.unquote(path)
    p = Path(decoded)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(p)


@app.get("/api/review/item")
def api_review_item(image_path: str = Query(...)):
    img_p = Path(urllib.parse.unquote(image_path))
    if not img_p.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    txt_p = img_p.with_suffix(".txt")
    caption = txt_p.read_text(encoding="utf-8").strip() if txt_p.exists() else ""

    # Parse individual tags from caption
    tags = []
    if caption:
        for raw_tag in caption.split(","):
            t = raw_tag.strip()
            if t:
                cat = normalizer.categorize_tag(t)
                tags.append({
                    "text": t,
                    "category": cat.value,
                    "locked": False,
                    "source": "sidecar",
                    "confidence": 1.0,
                })

    # Inspect adjacent .audit.json
    audit_p = img_p.parent / f"{img_p.stem}.audit.json"
    validation_status = "valid"
    validation_report = None
    traceability = []
    mode = None
    format_val = None

    if audit_p.exists():
        try:
            audit_data = json.loads(audit_p.read_text(encoding="utf-8"))
            val_obj = audit_data.get("validation", {})
            if isinstance(val_obj, dict):
                validation_status = val_obj.get("status", "valid")
                validation_report = val_obj
            elif isinstance(val_obj, str):
                validation_status = val_obj
                validation_report = {"status": val_obj, "issues": [], "repaired_caption": ""}
            traceability = audit_data.get("tokens", [])
            mode = audit_data.get("mode")
            format_val = audit_data.get("format")
        except Exception as e:
            logger.warning(f"Error reading audit file {audit_p}: {e}")

    return {
        "image_path": str(img_p.as_posix()),
        "filename": img_p.name,
        "caption": caption,
        "tags": tags,
        "validation_status": validation_status,
        "validation_report": validation_report,
        "traceability": traceability,
        "tokens": traceability,
        "mode": mode,
        "format": format_val,
    }


@app.post("/api/review/save")
def api_review_save(req: SaveCaptionRequest):
    img_p = Path(req.image_path)
    if not img_p.exists():
        raise HTTPException(status_code=404, detail="Target image not found")

    txt_path = write_sidecar(img_p, req.caption)
    return {"status": "ok", "saved_path": str(txt_path), "caption": req.caption}


@app.post("/api/review/verify")
def api_review_verify(req: VerifyTagRequest):
    img_p = Path(req.image_path)
    if not img_p.exists():
        raise HTTPException(status_code=404, detail="Target image not found")

    from PIL import Image
    from hyper_captioner.pipeline.verifier import VisualGroundingVerifier

    verifier = VisualGroundingVerifier()
    try:
        img = Image.open(img_p).convert("RGB")
        res = verifier.verify_grounding(img, req.tag)
        return res
    finally:
        verifier.unload()


@app.post("/api/review/regenerate")
def api_review_regenerate(req: RegenerateRequest):
    img_p = Path(req.image_path)
    if not img_p.exists():
        raise HTTPException(status_code=404, detail="Target image not found")

    preset = preset_manager.get_preset(req.preset_name)
    if req.lora_strategy:
        try:
            preset.lora_strategy = LoRAStrategy(req.lora_strategy)
        except ValueError:
            pass
    if req.trigger_word is not None:
        preset.character.trigger_word = req.trigger_word
    if req.keep_underscores is not None:
        preset.keep_underscores = req.keep_underscores

    user_tag_items = [
        TagItem(text=t, source="user", confidence=1.0, category=normalizer.categorize_tag(t))
        for t in req.user_tags if t.strip()
    ]

    trigger_word = req.trigger_word if req.trigger_word is not None else preset.character.trigger_word
    try:
        placement = TriggerPlacement(req.trigger_placement) if req.trigger_placement else TriggerPlacement.PREPEND
    except ValueError:
        placement = TriggerPlacement.PREPEND

    trigger_cfg = TriggerConfig(
        word=trigger_word,
        placement=placement,
        absorb_stable_traits=preset.character.prune_reference_from_caption,
    )

    pipeline = get_pipeline(model_source=preset.model_source, wd14_device=preset.wd14_device)
    result = pipeline.caption_image(
        img_p,
        preset=preset,
        user_tags=user_tag_items,
        locked_tags=req.locked_tags,
        write_txt=True,
        caption_mode=req.caption_mode,
        caption_format=req.caption_format,
        trigger_config=trigger_cfg,
        write_audit=req.write_audit,
    )

    val_report = getattr(result, "validation_report", None)
    val_report_dict = (
        val_report.to_dict()
        if hasattr(val_report, "to_dict")
        else (val_report if isinstance(val_report, dict) else None)
    )
    val_status = (
        val_report.status.value
        if hasattr(val_report, "status") and hasattr(val_report.status, "value")
        else (
            str(val_report.status)
            if hasattr(val_report, "status")
            else result.metadata.get("validation_status", "valid")
        )
    )
    tokens_list = [
        t.to_dict() for t in getattr(result, "tokens", [])
    ] if getattr(result, "tokens", None) else [t.to_dict() for t in result.tags]

    return {
        "status": "ok",
        "caption": result.caption,
        "tags": [t.to_dict() for t in result.tags],
        "tokens": tokens_list,
        "traceability": tokens_list,
        "validation_status": val_status,
        "validation_report": val_report_dict,
        "execution_time": result.execution_time,
    }


@app.post("/api/batch/preview")
def api_batch_preview(req: PreviewRequest):
    p = Path(req.dataset_path.strip())
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Invalid dataset directory.")

    scan = scan_dataset(p)
    candidates = scan.missing_captions or scan.with_captions
    if not candidates:
        return {"previews": []}

    preset = preset_manager.get_preset(req.preset_name)
    if req.lora_strategy:
        try:
            preset.lora_strategy = LoRAStrategy(req.lora_strategy)
        except ValueError:
            pass
    if req.trigger_word is not None:
        preset.character.trigger_word = req.trigger_word
    if req.keep_underscores is not None:
        preset.keep_underscores = req.keep_underscores

    trigger_word = req.trigger_word if req.trigger_word is not None else preset.character.trigger_word
    try:
        placement = TriggerPlacement(req.trigger_placement) if req.trigger_placement else TriggerPlacement.PREPEND
    except ValueError:
        placement = TriggerPlacement.PREPEND

    trigger_cfg = TriggerConfig(
        word=trigger_word,
        placement=placement,
        absorb_stable_traits=preset.character.prune_reference_from_caption,
    )

    pipeline = get_pipeline(model_source=preset.model_source, wd14_device=preset.wd14_device)
    previews = pipeline.generate_preview(
        candidates,
        preset=preset,
        sample_count=req.sample_count,
        caption_mode=req.caption_mode,
        caption_format=req.caption_format,
        trigger_config=trigger_cfg,
        write_audit=req.write_audit,
    )

    output = []
    for path, res in previews:
        val_report = getattr(res, "validation_report", None)
        val_status = (
            val_report.status.value
            if hasattr(val_report, "status") and hasattr(val_report.status, "value")
            else (
                str(val_report.status)
                if hasattr(val_report, "status")
                else res.metadata.get("validation_status", "valid")
            )
        )
        tokens_list = [
            t.to_dict() for t in getattr(res, "tokens", [])
        ] if getattr(res, "tokens", None) else [t.to_dict() for t in res.tags]

        output.append({
            "image_path": str(path.as_posix()),
            "filename": path.name,
            "caption": res.caption,
            "tags": [t.to_dict() for t in res.tags],
            "tokens": tokens_list,
            "traceability": tokens_list,
            "validation_status": val_status,
            "validation_report": val_report.to_dict() if hasattr(val_report, "to_dict") else None,
            "execution_time": res.execution_time,
        })
    return {"previews": output}


def _run_batch_worker(req: BatchStartRequest):
    batch_state["active"] = True
    batch_state["cancel_requested"] = False
    batch_state["dataset_path"] = req.dataset_path
    batch_state["log_history"] = []

    dataset_path = Path(req.dataset_path)

    try:
        if req.do_backup:
            create_backup(dataset_path)

        preset = preset_manager.get_preset(req.preset_name)
        if req.lora_strategy:
            try:
                preset.lora_strategy = LoRAStrategy(req.lora_strategy)
            except ValueError:
                pass
        if req.trigger_word is not None:
            preset.character.trigger_word = req.trigger_word
        if req.character_name:
            preset.character.name = req.character_name
        if req.reference_description:
            preset.character.reference_description = req.reference_description
        if req.wd14_threshold:
            preset.wd14_general_threshold = req.wd14_threshold
        if req.keep_underscores is not None:
            preset.keep_underscores = req.keep_underscores
        if req.vram_mode:
            preset.vram_mode = VRAMMode(req.vram_mode)
        if req.wd14_device:
            preset.wd14_device = WD14Device(req.wd14_device)
        if req.model_source:
            preset.model_source = ModelSource(req.model_source)

        trigger_word = req.trigger_word if req.trigger_word is not None else preset.character.trigger_word
        try:
            placement = TriggerPlacement(req.trigger_placement) if req.trigger_placement else TriggerPlacement.PREPEND
        except ValueError:
            placement = TriggerPlacement.PREPEND

        trigger_cfg = TriggerConfig(
            word=trigger_word,
            placement=placement,
            absorb_stable_traits=preset.character.prune_reference_from_caption,
        )

        pipeline = get_pipeline(model_source=preset.model_source, wd14_device=preset.wd14_device)

        def progress_cb(prog: BatchProgress):
            batch_state["progress"] = prog
            log_line = f"[{prog.current}/{prog.total}] {prog.current_file} -> {prog.status}"
            batch_state["log_history"].append(log_line)
            if len(batch_state["log_history"]) > 200:
                batch_state["log_history"].pop(0)

        def cancel_chk() -> bool:
            return batch_state["cancel_requested"]

        pipeline.process_dataset(
            dataset_path=dataset_path,
            preset=preset,
            skip_existing=not req.overwrite,
            overwrite=req.overwrite,
            progress_callback=progress_cb,
            cancel_requested=cancel_chk,
            caption_mode=req.caption_mode,
            caption_format=req.caption_format,
            trigger_config=trigger_cfg,
            write_audit=req.write_audit,
        )

    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        batch_state["progress"].status = "error"
        batch_state["log_history"].append(f"ERROR: {e}")
    finally:
        batch_state["active"] = False


@app.post("/api/batch/start")
def api_batch_start(req: BatchStartRequest):
    if batch_state["active"]:
        raise HTTPException(status_code=400, detail="A batch job is already running.")

    p = Path(req.dataset_path.strip())
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Invalid dataset directory.")

    th = threading.Thread(target=_run_batch_worker, args=(req,), daemon=True)
    batch_state["thread"] = th
    th.start()

    return {"status": "started", "dataset": req.dataset_path}


@app.post("/api/batch/cancel")
def api_batch_cancel():
    if not batch_state["active"]:
        return {"status": "not_running"}
    batch_state["cancel_requested"] = True
    return {"status": "cancelling"}


@app.get("/api/batch/progress")
def api_batch_progress():
    prog = batch_state["progress"]
    return prog.to_dict()


@app.get("/api/batch/status")
def api_batch_status():
    prog = batch_state["progress"]
    records_summary = [
        {"image": r.image_path.name, "status": r.status, "caption": r.caption}
        for r in prog.records[-50:]
    ]
    return {
        "active": batch_state["active"],
        "progress": prog.to_dict(),
        "valid_count": prog.valid_count,
        "repaired_count": prog.repaired_count,
        "rejected_count": prog.rejected_count,
        "logs": batch_state["log_history"][-30:],
        "recent_records": records_summary,
    }


# -------------------------------------------------------------
# Vocabulary Endpoints
# -------------------------------------------------------------
@app.get("/api/vocab/index")
def api_vocab_index(dataset_path: str = Query(...), top: int = 100, query: Optional[str] = None):
    p = Path(dataset_path.strip())
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Invalid dataset path")

    vocab_index.index(p)
    if query:
        matches = vocab_index.find_variants(query)
        return {"results": [{"tag": t, "count": c} for t, c in matches[:top]]}

    top_tags = vocab_index.get_top_tags(limit=top)
    return {
        "total_unique_tags": len(vocab_index.tag_counts),
        "results": [{"tag": t, "count": c} for t, c in top_tags],
    }


@app.post("/api/vocab/replace")
def api_vocab_replace(req: VocabReplaceRequest):
    p = Path(req.dataset_path.strip())
    modified = vocab_index.replace_tag(req.old_tag, req.new_tag, dataset_path=p)
    return {"status": "ok", "modified_files": modified, "old_tag": req.old_tag, "new_tag": req.new_tag}


@app.post("/api/vocab/normalize_all")
def api_vocab_normalize_all(req: VocabBatchNormalizeRequest):
    p = Path(req.dataset_path.strip())
    modified = vocab_index.batch_normalize(req.mappings, dataset_path=p)
    return {"status": "ok", "modified_files": modified}


@app.post("/api/vocab/delete")
def api_vocab_delete(req: VocabDeleteRequest):
    p = Path(req.dataset_path.strip())
    modified = vocab_index.delete_tag(req.target_tag, dataset_path=p)
    return {"status": "ok", "modified_files": modified, "target_tag": req.target_tag}


# -------------------------------------------------------------
# Export & Backup Endpoints
# -------------------------------------------------------------
@app.post("/api/export")
def api_export(req: ExportRequest):
    p = Path(req.dataset_path.strip())
    out_file = export_master_metadata(p, export_format=req.format, output_filename=req.output_filename)
    return {"status": "ok", "exported_file": str(out_file), "filename": out_file.name}


@app.get("/api/export/download")
def api_download_export(path: str = Query(...)):
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Exported file not found.")
    return FileResponse(p, filename=p.name, media_type="application/octet-stream")


@app.post("/api/backup")
def api_backup(dataset_path: str = Query(...)):
    p = Path(dataset_path.strip())
    dest = create_backup(p)
    return {"status": "ok", "backup_folder": str(dest)}


# -------------------------------------------------------------
# Single-Page UI Mount
# -------------------------------------------------------------
UI_DIR = Path(__file__).resolve().parent.parent / "ui"
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
def serve_ui():
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Hyper Captioner UI not found</h1>")
