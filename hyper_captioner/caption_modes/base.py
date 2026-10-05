"""
Abstract base class for caption modes defining category contracts,
extraction instructions, fact filtering, and token formatting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionToken,
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
    TriggerConfig,
)


class BaseCaptionMode(ABC):
    """
    Abstract contract for a caption mode.
    Decouples semantic LoRA objective (what to capture/exclude) from output format.
    """

    def __init__(
        self,
        name: str,
        description: str,
        include_categories: Optional[Set[SemanticCategory]] = None,
        exclude_categories: Optional[Set[SemanticCategory]] = None,
    ):
        self.name: str = name
        self.description: str = description
        self.include_categories: Set[SemanticCategory] = (
            include_categories if include_categories is not None else set()
        )
        self.exclude_categories: Set[SemanticCategory] = (
            exclude_categories if exclude_categories is not None else set()
        )

    def is_category_allowed(self, category: Union[SemanticCategory, str]) -> bool:
        """
        Check whether a semantic category is permitted under this mode contract.
        """
        if isinstance(category, str):
            try:
                category = SemanticCategory(category)
            except ValueError:
                return False
        if category in self.exclude_categories:
            return False
        return category in self.include_categories

    def validate_categories(
        self, categories: List[Union[SemanticCategory, str]]
    ) -> List[SemanticCategory]:
        """
        Filter a list of categories to only those permitted under this mode contract.
        """
        result: List[SemanticCategory] = []
        for cat in categories:
            normalized = (
                SemanticCategory(cat)
                if isinstance(cat, str) and cat in SemanticCategory._value2member_map_
                else cat
            )
            if isinstance(normalized, SemanticCategory) and self.is_category_allowed(normalized):
                result.append(normalized)
        return result

    @abstractmethod
    def build_extraction_instructions(self) -> str:
        """
        Return mode-specific extraction guidelines for vision models (JoyCaption).
        """
        pass

    @abstractmethod
    def filter_facts(
        self, facts: StructuredVisualFacts, trigger_cfg: TriggerConfig
    ) -> List[FactItem]:
        """
        Filter extracted visual facts according to mode contract and trigger rules.
        Inviolable user locks (locked=True) must be strictly preserved.
        Uncertain facts (is_uncertain=True) must be strictly excluded.
        """
        pass

    @abstractmethod
    def format_tokens(
        self, tokens: List[CaptionToken], format_type: CaptionFormat
    ) -> str:
        """
        Format assembled caption tokens into the requested output format
        (TAGS, STRUCTURED, NATURAL).
        """
        pass

    def _default_format_tokens(
        self, tokens: List[CaptionToken], format_type: CaptionFormat
    ) -> str:
        """
        Standard token formatting fallback used across modes.
        """
        token_texts = [t.text.strip() for t in tokens if t.text and t.text.strip()]
        if not token_texts:
            return ""

        if format_type == CaptionFormat.TAGS:
            return ", ".join(token_texts)
        elif format_type == CaptionFormat.STRUCTURED:
            return ", ".join(token_texts)
        elif format_type == CaptionFormat.NATURAL:
            sentence = ", ".join(token_texts)
            if not sentence.endswith("."):
                sentence += "."
            return sentence
        return ", ".join(token_texts)
