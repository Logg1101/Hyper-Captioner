"""
Safe dataset exporters: Sidecar .txt writer, timestamped backups, and master metadata compilation.
"""

import csv
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hyper_captioner.dataset.scanner import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def write_sidecar(image_path: Path, caption: str) -> Path:
    """Safely writes a clean UTF-8 .txt sidecar file next to the image."""
    txt_path = image_path.with_suffix(".txt")
    clean_caption = caption.strip()
    txt_path.write_text(clean_caption, encoding="utf-8")
    return txt_path


def write_audit_sidecar(image_path: Path, audit_data: Dict[str, Any]) -> Path:
    """
    Safely writes an audit metadata sidecar (<image_stem>.audit.json) alongside the image.
    Uses UTF-8 encoding, indented JSON (indent=2), and ensure_ascii=False.
    """
    path = Path(image_path).resolve()
    audit_path = path.parent / f"{path.stem}.audit.json"
    audit_json = json.dumps(audit_data, indent=2, ensure_ascii=False)
    audit_path.write_text(audit_json, encoding="utf-8")
    return audit_path



def create_backup(dataset_path: Path) -> Path:
    """Creates a timestamped backup folder of the entire dataset."""
    src = Path(dataset_path).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = src.parent / f"{src.name}_BACKUP_{timestamp}"

    logger.info(f"Creating dataset backup: {src} -> {backup_folder}")
    shutil.copytree(src, backup_folder)
    logger.info(f"Backup created successfully: {backup_folder}")
    return backup_folder


def export_master_metadata(
    dataset_path: Path,
    export_format: str = "csv",
    output_filename: Optional[str] = None
) -> Path:
    """
    Gathers all image sidecars across the dataset and compiles them into
    a single master file ('captions.csv' or 'captions.json').
    """
    path = Path(dataset_path).resolve()
    records: Dict[str, str] = {}

    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            txt_path = file_path.with_suffix(".txt")
            if txt_path.exists() and txt_path.is_file():
                rel_img = str(file_path.relative_to(path).as_posix())
                try:
                    records[rel_img] = txt_path.read_text(encoding="utf-8").strip()
                except Exception as e:
                    logger.warning(f"Error reading {txt_path}: {e}")

    fmt = export_format.lower().strip()
    if fmt == "json":
        fname = output_filename or "captions.json"
        out_path = path / fname
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path
    elif fmt == "csv":
        fname = output_filename or "captions.csv"
        out_path = path / fname
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "caption"])
            for img_name, cap in records.items():
                writer.writerow([img_name, cap])
        return out_path
    elif fmt == "zip":
        import zipfile
        fname = output_filename or "captions_txt_sidecars.zip"
        out_path = path / fname
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in sorted(path.rglob("*")):
                if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    txt_path = file_path.with_suffix(".txt")
                    if txt_path.exists() and txt_path.is_file():
                        arcname = txt_path.relative_to(path).as_posix()
                        zipf.write(txt_path, arcname=arcname)
        return out_path
    elif fmt == "sidecars":
        # Ensure all existing image records have their .txt file safely verified
        fname = output_filename or "sidecars_synced.log"
        out_path = path / fname
        out_path.write_text(f"Verified {len(records)} .txt sidecar files next to image files.\n", encoding="utf-8")
        return out_path
    else:
        # Default: Combined TXT file (one line per image: 'image.png: caption')
        fname = output_filename or "captions.txt"
        out_path = path / fname
        lines = [f"{img_name}: {cap}" for img_name, cap in records.items()]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
