"""
Configuration and model management for Hyper Captioner.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hyper_captioner.core.types import ModelSource, PresetConfig, VRAMMode, WD14Device

logger = logging.getLogger(__name__)

# Default model identifiers on Hugging Face / local cache
DEFAULT_JOYCAPTION_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"
DEFAULT_WD14_MODEL = "SmilingWolf/wd-swinv2-tagger-v3"
DEFAULT_FLORENCE_MODEL = "microsoft/Florence-2-large"

# Available WD14 Models
AVAILABLE_WD14_MODELS = {
    "WD14 SwinV2 v3 (Recommended)": "SmilingWolf/wd-swinv2-tagger-v3",
    "WD14 ViT v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
    "WD14 ConvNeXt v3": "SmilingWolf/wd-convnext-tagger-v3",
}

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT_DIR / "config.json"
PRESETS_DIR = ROOT_DIR / "presets"


class ModelMissingError(RuntimeError):
    """Raised when a model is missing and ModelSource is set to LOCAL_ONLY."""
    pass


def load_hf_resource(
    loader_fn: Callable[..., Any],
    model_name_or_path: str,
    model_source: ModelSource = ModelSource.LOCAL_ONLY,
    **kwargs
) -> Any:
    """
    Loads Hugging Face models/processors honoring local-only rules.
    If model_source is LOCAL_ONLY, strictly loads from local cache.
    """
    if model_source == ModelSource.LOCAL_ONLY:
        try:
            return loader_fn(model_name_or_path, local_files_only=True, **kwargs)
        except Exception as e:
            msg = (
                f"Model '{model_name_or_path}' not found in local cache and Model Source is 'Local Only'.\n"
                f"To resolve: Ensure model is cached or switch Model Source to 'Local + Download Missing Models'.\n"
                f"Original error: {e}"
            )
            logger.error(msg)
            raise ModelMissingError(msg) from e
    else:
        # Download missing allowed: try local first to avoid unnecessary network queries, then fallback to hub
        try:
            return loader_fn(model_name_or_path, local_files_only=True, **kwargs)
        except Exception:
            logger.info(f"Downloading missing model resource '{model_name_or_path}' from Hugging Face...")
            return loader_fn(model_name_or_path, local_files_only=False, **kwargs)


def load_app_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read {CONFIG_FILE}: {e}")
    return {}


def save_app_config(cfg: Dict[str, Any]):
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to save {CONFIG_FILE}: {e}")
