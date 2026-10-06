"""
VRAM and Hardware Device Manager for Hyper Captioner.
"""

import gc
import logging
from typing import Dict, Tuple

import torch

from hyper_captioner.core.types import VRAMMode, WD14Device

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_vram_info() -> Dict[str, float]:
    """Returns allocated, reserved, and free VRAM in GiB for current CUDA device."""
    if not is_cuda_available():
        return {"total_gib": 0.0, "free_gib": 0.0, "allocated_gib": 0.0, "reserved_gib": 0.0}

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated_bytes = torch.cuda.memory_allocated()
        reserved_bytes = torch.cuda.memory_reserved()
        return {
            "total_gib": round(total_bytes / (1024**3), 2),
            "free_gib": round(free_bytes / (1024**3), 2),
            "allocated_gib": round(allocated_bytes / (1024**3), 2),
            "reserved_gib": round(reserved_bytes / (1024**3), 2),
            "device_name": torch.cuda.get_device_name(0),
        }
    except Exception as e:
        logger.warning(f"Failed to query CUDA VRAM info: {e}")
        return {"total_gib": 0.0, "free_gib": 0.0, "allocated_gib": 0.0, "reserved_gib": 0.0}


def resolve_torch_device() -> torch.device:
    if is_cuda_available():
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


def resolve_wd14_provider(device_choice: WD14Device = WD14Device.AUTO) -> str:
    """
    Determines ONNX Runtime execution provider for WD14 tagger.
    'CUDAExecutionProvider' or 'CPUExecutionProvider'.
    """
    if device_choice == WD14Device.CPU:
        return "CPUExecutionProvider"

    if device_choice == WD14Device.CUDA:
        if is_cuda_available():
            return "CUDAExecutionProvider"
        logger.warning("CUDA requested for WD14 but CUDA is not available. Falling back to CPU.")
        return "CPUExecutionProvider"

    # AUTO MODE: Check free VRAM. If > 4.0 GiB free, can safely use CUDA; otherwise CPU to keep VRAM for LLM.
    if is_cuda_available():
        vram = get_vram_info()
        if vram.get("free_gib", 0.0) >= 4.0:
            return "CUDAExecutionProvider"

    return "CPUExecutionProvider"


def clean_vram():
    """Aggressively frees unreferenced PyTorch memory."""
    gc.collect()
    if is_cuda_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def handle_vram_lifecycle(vram_mode: VRAMMode, step_name: str = ""):
    """Applies VRAM cleanup policies based on the configured VRAM mode."""
    if vram_mode == VRAMMode.MIN_VRAM:
        clean_vram()
    elif vram_mode == VRAMMode.BALANCED and step_name in {"after_joycaption", "after_batch"}:
        clean_vram()
