"""
Fast, robust dataset scanner and health verifier.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


@dataclass
class DatasetScanResult:
    dataset_path: Path
    total_images: int = 0
    with_captions: List[Path] = field(default_factory=list)
    missing_captions: List[Path] = field(default_factory=list)
    orphan_captions: List[Path] = field(default_factory=list)
    corrupted_images: List[Path] = field(default_factory=list)
    empty_captions: List[Path] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "dataset_path": str(self.dataset_path),
            "total_images": self.total_images,
            "with_captions_count": len(self.with_captions),
            "missing_captions_count": len(self.missing_captions),
            "orphan_captions_count": len(self.orphan_captions),
            "corrupted_images_count": len(self.corrupted_images),
            "empty_captions_count": len(self.empty_captions),
            "missing_images": [str(p) for p in self.missing_captions],
            "with_caption_images": [str(p) for p in self.with_captions],
            "orphan_captions": [str(p) for p in self.orphan_captions],
        }


def scan_dataset(dataset_path: Path, verify_integrity: bool = False) -> DatasetScanResult:
    """
    Recursively scans dataset directory for images and matching sidecar captions (.txt).
    """
    path = Path(dataset_path).resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    all_images: List[Path] = []
    image_stem_map: Dict[str, Path] = {}
    corrupted: List[Path] = []

    # 1. Discover all images
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            all_images.append(file_path)
            # Use relative stem to prevent clashes across subfolders
            rel_stem = str(file_path.relative_to(path).with_suffix(""))
            image_stem_map[rel_stem] = file_path

            if verify_integrity:
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception:
                    corrupted.append(file_path)

    all_images.sort()

    # 2. Check for sidecar .txt files
    with_captions: List[Path] = []
    missing_captions: List[Path] = []
    empty_captions: List[Path] = []

    for img_path in all_images:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and txt_path.is_file():
            try:
                content = txt_path.read_text(encoding="utf-8").strip()
                if len(content) > 0:
                    with_captions.append(img_path)
                else:
                    empty_captions.append(img_path)
                    missing_captions.append(img_path)
            except Exception:
                empty_captions.append(img_path)
                missing_captions.append(img_path)
        else:
            missing_captions.append(img_path)

    # 3. Find orphan captions
    orphan_captions: List[Path] = []
    for txt_path in path.rglob("*.txt"):
        if not txt_path.is_file():
            continue
        rel_stem = str(txt_path.relative_to(path).with_suffix(""))
        if rel_stem not in image_stem_map:
            orphan_captions.append(txt_path)

    orphan_captions.sort()

    return DatasetScanResult(
        dataset_path=path,
        total_images=len(all_images),
        with_captions=with_captions,
        missing_captions=missing_captions,
        orphan_captions=orphan_captions,
        corrupted_images=corrupted,
        empty_captions=empty_captions,
    )
