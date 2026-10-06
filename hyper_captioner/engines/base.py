"""
Abstract base classes for vision models, taggers, and grounders.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image


class BaseTagger(ABC):
    """Abstract interface for Danbooru/WD14 taggers."""

    @abstractmethod
    def load(self, model_id: Optional[str] = None):
        """Loads model weights/sessions into memory."""
        pass

    @abstractmethod
    def unload(self):
        """Unloads weights and frees memory."""
        pass

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        general_threshold: float = 0.35,
        character_threshold: float = 0.60
    ) -> List[Tuple[str, float, int]]:
        """
        Runs tagger inference on a PIL Image.
        Returns list of (tag_name, confidence_score, category_id) tuples.
        Category 0 = General, Category 4 = Character, Category 9 = Rating.
        """
        pass


class BaseCaptioner(ABC):
    """Abstract interface for Vision-Language Captioning models."""

    @abstractmethod
    def load(self, model_id: Optional[str] = None):
        """Loads model weights and processors."""
        pass

    @abstractmethod
    def unload(self):
        """Unloads model from GPU memory."""
        pass

    @abstractmethod
    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 350,
        temperature: float = 0.0,
        **kwargs
    ) -> List[str]:
        """Runs batch captioning generation."""
        pass


class BaseGrounding(ABC):
    """Abstract interface for visual grounding and object verification."""

    @abstractmethod
    def load(self, model_id: Optional[str] = None):
        pass

    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Returns detected objects with bounding boxes and labels."""
        pass

    @abstractmethod
    def verify_attributes(
        self, image: Image.Image, candidate_attributes: List[str]
    ) -> Dict[str, bool]:
        """Verifies if candidate attributes are physically grounded in image."""
        pass
