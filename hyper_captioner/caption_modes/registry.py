"""
Dynamic registry for Caption Modes with case-insensitive discovery.
"""

from typing import Dict, List, Type, Union

from hyper_captioner.caption_modes.base import BaseCaptionMode
from hyper_captioner.caption_modes.character import CharacterMode
from hyper_captioner.caption_modes.concept import ConceptMode
from hyper_captioner.caption_modes.outfit import OutfitMode
from hyper_captioner.caption_modes.pose import PoseMode
from hyper_captioner.caption_modes.style import StyleMode
from hyper_captioner.core.types import CaptionModeType


_REGISTRY: Dict[str, BaseCaptionMode] = {}


def register_caption_mode(mode: Union[BaseCaptionMode, Type[BaseCaptionMode]]) -> None:
    """
    Register a caption mode instance or class into the dynamic registry.
    """
    if isinstance(mode, type) and issubclass(mode, BaseCaptionMode):
        instance = mode()
    elif isinstance(mode, BaseCaptionMode):
        instance = mode
    else:
        raise TypeError(
            f"Expected BaseCaptionMode instance or subclass, got {type(mode)}"
        )

    key = instance.name.lower().strip()
    _REGISTRY[key] = instance


def get_caption_mode(name: Union[str, CaptionModeType]) -> BaseCaptionMode:
    """
    Retrieve a caption mode by name or CaptionModeType (case-insensitive).
    Raises KeyError if the mode is not registered.
    """
    if isinstance(name, CaptionModeType):
        key = name.value.lower().strip()
    elif isinstance(name, str):
        key = name.lower().strip()
    else:
        raise ValueError(f"Invalid mode identifier type: {type(name)}")

    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown caption mode: '{name}'. Available modes: [{available}]")

    return _REGISTRY[key]


def list_caption_modes() -> List[str]:
    """
    List all registered caption mode names.
    """
    return list(_REGISTRY.keys())


def _init_default_registry() -> None:
    """
    Initialize the default core 5 mode contracts.
    """
    register_caption_mode(CharacterMode())
    register_caption_mode(StyleMode())
    register_caption_mode(OutfitMode())
    register_caption_mode(PoseMode())
    register_caption_mode(ConceptMode())


# Initialize core registry on module load
_init_default_registry()
