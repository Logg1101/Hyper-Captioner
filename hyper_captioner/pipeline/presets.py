"""
Dataset Presets Manager: Character LoRA, Style LoRA, Concept LoRA, and Custom presets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from hyper_captioner.config import PRESETS_DIR
from hyper_captioner.core.types import (
    CaptionMode,
    CharacterConfig,
    LoRAStrategy,
    ModelSource,
    PresetConfig,
    VRAMMode,
    WD14Device,
)

logger = logging.getLogger(__name__)

DEFAULT_PRESETS: Dict[str, PresetConfig] = {
    "Character LoRA": PresetConfig(
        name="Character LoRA",
        caption_mode=CaptionMode.HYBRID,
        lora_strategy=LoRAStrategy.CHARACTER,
        wd14_general_threshold=0.35,
        wd14_character_threshold=0.60,
        keep_underscores=False,
        filter_poisons=True,
        quality_boosters=False,
        vram_mode=VRAMMode.BALANCED,
        wd14_device=WD14Device.AUTO,
    ),
    "Style LoRA": PresetConfig(
        name="Style LoRA",
        caption_mode=CaptionMode.HYBRID,
        lora_strategy=LoRAStrategy.STYLE,
        wd14_general_threshold=0.40,
        wd14_character_threshold=0.70,
        keep_underscores=False,
        filter_poisons=True,
        quality_boosters=False,
        vram_mode=VRAMMode.BALANCED,
    ),
    "Concept LoRA": PresetConfig(
        name="Concept LoRA",
        caption_mode=CaptionMode.HYBRID,
        lora_strategy=LoRAStrategy.CONCEPT,
        wd14_general_threshold=0.35,
        wd14_character_threshold=0.65,
        keep_underscores=False,
        filter_poisons=True,
        quality_boosters=False,
        vram_mode=VRAMMode.BALANCED,
    ),
    "Danbooru Tag Only": PresetConfig(
        name="Danbooru Tag Only",
        caption_mode=CaptionMode.TAG,
        lora_strategy=LoRAStrategy.GENERAL,
        wd14_general_threshold=0.35,
        wd14_character_threshold=0.60,
        keep_underscores=True,
        filter_poisons=True,
        quality_boosters=False,
        vram_mode=VRAMMode.BALANCED,
    ),
    "Natural Language (Flux)": PresetConfig(
        name="Natural Language (Flux)",
        caption_mode=CaptionMode.NATURAL,
        lora_strategy=LoRAStrategy.CHARACTER,
        wd14_general_threshold=0.40,
        wd14_character_threshold=0.65,
        keep_underscores=False,
        filter_poisons=True,
        quality_boosters=False,
        vram_mode=VRAMMode.BALANCED,
    ),
}


class PresetManager:
    """Manages built-in and user-saved presets."""

    def __init__(self, presets_dir: Path = PRESETS_DIR):
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self.custom_presets: Dict[str, PresetConfig] = {}
        self.reload()

    def reload(self):
        """Loads custom presets from disk."""
        self.custom_presets.clear()
        for p_file in self.presets_dir.glob("*.json"):
            try:
                data = json.loads(p_file.read_text(encoding="utf-8"))
                char_data = data.get("character", {})
                char_cfg = CharacterConfig(
                    name=char_data.get("name", ""),
                    trigger_word=char_data.get("trigger_word", ""),
                    reference_description=char_data.get("reference_description", ""),
                    prune_reference_from_caption=char_data.get("prune_reference_from_caption", False),
                )
                raw_mode = data.get("caption_mode", "hybrid")
                try:
                    c_mode = CaptionMode(raw_mode)
                except ValueError:
                    c_mode = raw_mode

                preset = PresetConfig(
                    name=data.get("name", p_file.stem),
                    caption_mode=c_mode,
                    caption_format=data.get("caption_format", "tags"),
                    trigger_placement=data.get("trigger_placement", "prepend"),
                    write_audit=bool(data.get("write_audit", False)),
                    lora_strategy=LoRAStrategy(data.get("lora_strategy", "character")),
                    character=char_cfg,
                    wd14_general_threshold=data.get("wd14_general_threshold", 0.35),
                    wd14_character_threshold=data.get("wd14_character_threshold", 0.60),
                    keep_underscores=data.get("keep_underscores", False),
                    filter_poisons=data.get("filter_poisons", True),
                    quality_boosters=data.get("quality_boosters", False),
                    vram_mode=VRAMMode(data.get("vram_mode", "balanced")),
                    wd14_device=WD14Device(data.get("wd14_device", "auto")),
                    model_source=ModelSource(data.get("model_source", "local_only")),
                    custom_tags=data.get("custom_tags", ""),
                    blacklist=data.get("blacklist", []),
                )
                self.custom_presets[preset.name] = preset
            except Exception as e:
                logger.warning(f"Failed to load preset {p_file}: {e}")

    def list_presets(self) -> List[str]:
        names = list(DEFAULT_PRESETS.keys())
        for custom_name in sorted(self.custom_presets.keys()):
            if custom_name not in names:
                names.append(custom_name)
        return names

    def get_preset(self, name: str) -> PresetConfig:
        if name in self.custom_presets:
            return self.custom_presets[name]
        if name in DEFAULT_PRESETS:
            return DEFAULT_PRESETS[name]
        return DEFAULT_PRESETS["Character LoRA"]

    def save_preset(self, preset: PresetConfig) -> Path:
        safe_name = "".join(c for c in preset.name if c.isalnum() or c in " _-").strip()
        out_file = self.presets_dir / f"{safe_name}.json"
        out_file.write_text(json.dumps(preset.to_dict(), indent=2), encoding="utf-8")
        self.custom_presets[preset.name] = preset
        return out_file

    def delete_preset(self, name: str) -> bool:
        if name in DEFAULT_PRESETS:
            logger.warning("Cannot delete built-in preset.")
            return False
        safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
        p_file = self.presets_dir / f"{safe_name}.json"
        if p_file.exists():
            p_file.unlink()
            self.custom_presets.pop(name, None)
            return True
        return False
