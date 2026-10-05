"""
Caption Mode contracts and dynamic registry for Hyper Captioner.
"""

from hyper_captioner.caption_modes.base import BaseCaptionMode
from hyper_captioner.caption_modes.character import CharacterMode
from hyper_captioner.caption_modes.concept import ConceptMode
from hyper_captioner.caption_modes.outfit import OutfitMode
from hyper_captioner.caption_modes.pose import PoseMode
from hyper_captioner.caption_modes.registry import (
    get_caption_mode,
    list_caption_modes,
    register_caption_mode,
)
from hyper_captioner.caption_modes.style import StyleMode

__all__ = [
    "BaseCaptionMode",
    "CharacterMode",
    "StyleMode",
    "OutfitMode",
    "PoseMode",
    "ConceptMode",
    "get_caption_mode",
    "list_caption_modes",
    "register_caption_mode",
]
