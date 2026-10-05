"""
Core data types and contracts for Hyper Captioner.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# Core Semantic Categories & Enums (18 Categories)
# ============================================================================


class SemanticCategory(str, Enum):
    IDENTITY = "identity"
    APPEARANCE = "appearance"
    CLOTHING = "clothing"
    POSE = "pose"
    EXPRESSION = "expression"
    COMPOSITION = "composition"
    CAMERA = "camera"
    ENVIRONMENT = "environment"
    LIGHTING = "lighting"
    MATERIAL = "material"
    RENDERING = "rendering"
    STYLE = "style"
    OBJECTS = "objects"
    COLOR = "color"
    TEXTURE = "texture"
    CONCEPT = "concept"
    QUALITY = "quality"
    UNCERTAINTY = "uncertainty"


class CaptionModeType(str, Enum):
    CHARACTER = "character"
    STYLE = "style"
    OUTFIT = "outfit"
    POSE = "pose"
    CONCEPT = "concept"


class CaptionFormat(str, Enum):
    TAGS = "tags"
    STRUCTURED = "structured"
    NATURAL = "natural"


class TriggerPlacement(str, Enum):
    PREPEND = "prepend"
    APPEND = "append"
    WRAP = "wrap"
    OMIT = "omit"


# ============================================================================
# Trigger Configuration
# ============================================================================


@dataclass
class TriggerConfig:
    word: str = ""
    placement: TriggerPlacement = TriggerPlacement.PREPEND
    case_sensitive: bool = False  # Mandatory case-insensitive default
    deduplicate: bool = True
    absorb_stable_traits: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "placement": (
                self.placement.value
                if isinstance(self.placement, TriggerPlacement)
                else str(self.placement)
            ),
            "case_sensitive": self.case_sensitive,
            "deduplicate": self.deduplicate,
            "absorb_stable_traits": self.absorb_stable_traits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriggerConfig":
        placement_raw = data.get("placement", TriggerPlacement.PREPEND)
        placement = (
            TriggerPlacement(placement_raw)
            if isinstance(placement_raw, str)
            else placement_raw
        )
        return cls(
            word=data.get("word", ""),
            placement=placement,
            case_sensitive=bool(data.get("case_sensitive", False)),
            deduplicate=bool(data.get("deduplicate", True)),
            absorb_stable_traits=bool(data.get("absorb_stable_traits", False)),
        )


# ============================================================================
# Multi-Category Fact Item & Lineage Tracking
# ============================================================================


@dataclass
class FactItem:
    id: str
    text: str
    primary_category: SemanticCategory
    categories: List[SemanticCategory] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "joycaption"
    is_stable: bool = False
    is_uncertain: bool = False  # Authoritative uncertainty flag
    locked: bool = False  # Inviolable user lock
    raw_text: str = ""

    def __post_init__(self):
        if isinstance(self.primary_category, str):
            self.primary_category = SemanticCategory(self.primary_category)
        self.categories = [
            SemanticCategory(c) if isinstance(c, str) else c for c in self.categories
        ]
        if not self.categories:
            self.categories = [self.primary_category]
        elif self.primary_category not in self.categories:
            self.categories.insert(0, self.primary_category)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "primary_category": (
                self.primary_category.value
                if isinstance(self.primary_category, SemanticCategory)
                else str(self.primary_category)
            ),
            "categories": [
                c.value if isinstance(c, SemanticCategory) else str(c)
                for c in self.categories
            ],
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "is_stable": self.is_stable,
            "is_uncertain": self.is_uncertain,
            "locked": self.locked,
            "raw_text": self.raw_text or self.text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactItem":
        prim_raw = data.get("primary_category", SemanticCategory.CONCEPT)
        primary_category = (
            SemanticCategory(prim_raw) if isinstance(prim_raw, str) else prim_raw
        )
        categories = [
            SemanticCategory(c) if isinstance(c, str) else c
            for c in data.get("categories", [])
        ]
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            primary_category=primary_category,
            categories=categories,
            confidence=float(data.get("confidence", 1.0)),
            source=data.get("source", "joycaption"),
            is_stable=bool(data.get("is_stable", False)),
            is_uncertain=bool(data.get("is_uncertain", False)),
            locked=bool(data.get("locked", False)),
            raw_text=data.get("raw_text", ""),
        )


@dataclass
class CaptionToken:
    text: str
    primary_category: SemanticCategory
    categories: List[SemanticCategory] = field(default_factory=list)
    source_fact_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0
    transformation: str = "direct"
    locked: bool = False

    def __post_init__(self):
        if isinstance(self.primary_category, str):
            self.primary_category = SemanticCategory(self.primary_category)
        self.categories = [
            SemanticCategory(c) if isinstance(c, str) else c for c in self.categories
        ]
        if not self.categories:
            self.categories = [self.primary_category]
        elif self.primary_category not in self.categories:
            self.categories.insert(0, self.primary_category)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "primary_category": (
                self.primary_category.value
                if isinstance(self.primary_category, SemanticCategory)
                else str(self.primary_category)
            ),
            "categories": [
                c.value if isinstance(c, SemanticCategory) else str(c)
                for c in self.categories
            ],
            "source_fact_ids": self.source_fact_ids,
            "confidence": round(self.confidence, 4),
            "transformation": self.transformation,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaptionToken":
        prim_raw = data.get("primary_category", SemanticCategory.CONCEPT)
        primary_category = (
            SemanticCategory(prim_raw) if isinstance(prim_raw, str) else prim_raw
        )
        categories = [
            SemanticCategory(c) if isinstance(c, str) else c
            for c in data.get("categories", [])
        ]
        return cls(
            text=data.get("text", ""),
            primary_category=primary_category,
            categories=categories,
            source_fact_ids=data.get("source_fact_ids", []),
            confidence=float(data.get("confidence", 1.0)),
            transformation=data.get("transformation", "direct"),
            locked=bool(data.get("locked", False)),
        )


# ============================================================================
# Structured Visual Facts Representation
# ============================================================================


@dataclass
class StructuredVisualFacts:
    facts_by_category: Dict[SemanticCategory, List[FactItem]] = field(
        default_factory=dict
    )
    raw_response: str = ""
    parse_method: str = "json"
    image_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        normalized = {}
        for k, v in self.facts_by_category.items():
            key = SemanticCategory(k) if isinstance(k, str) else k
            normalized[key] = v
        self.facts_by_category = normalized

    def get_category(self, cat: SemanticCategory) -> List[FactItem]:
        if isinstance(cat, str):
            try:
                cat = SemanticCategory(cat)
            except ValueError:
                pass
        return self.facts_by_category.get(cat, [])

    def filter_categories(self, allowed: Set[SemanticCategory]) -> List[FactItem]:
        allowed_set = {
            SemanticCategory(c) if isinstance(c, str) else c for c in allowed
        }
        result: List[FactItem] = []
        seen_ids = set()
        for cat in self.facts_by_category:
            if cat in allowed_set:
                for item in self.facts_by_category[cat]:
                    key = item.id if item.id else id(item)
                    if key not in seen_ids:
                        seen_ids.add(key)
                        result.append(item)
        return result

    def all_facts(self) -> List[FactItem]:
        result: List[FactItem] = []
        seen_ids = set()
        for items in self.facts_by_category.values():
            for item in items:
                key = item.id if item.id else id(item)
                if key not in seen_ids:
                    seen_ids.add(key)
                    result.append(item)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts_by_category": {
                (k.value if isinstance(k, SemanticCategory) else str(k)): [
                    f.to_dict() for f in v
                ]
                for k, v in self.facts_by_category.items()
            },
            "raw_response": self.raw_response,
            "parse_method": self.parse_method,
            "image_metadata": self.image_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredVisualFacts":
        raw_facts = data.get("facts_by_category", {})
        facts_by_category: Dict[SemanticCategory, List[FactItem]] = {}
        for k, v in raw_facts.items():
            cat = SemanticCategory(k) if isinstance(k, str) else k
            facts_by_category[cat] = [
                FactItem.from_dict(item) if isinstance(item, dict) else item
                for item in v
            ]
        return cls(
            facts_by_category=facts_by_category,
            raw_response=data.get("raw_response", ""),
            parse_method=data.get("parse_method", "json"),
            image_metadata=data.get("image_metadata", {}),
        )


# ============================================================================
# Validation Types
# ============================================================================


class ValidationStatus(str, Enum):
    VALID = "valid"
    REPAIRED = "repaired"
    REJECTED = "rejected"


@dataclass
class ValidationIssue:
    severity: str  # "ERROR", "WARNING", "INFO"
    code: str
    message: str
    token: Optional[str] = None
    suggested_repair: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "token": self.token,
            "suggested_repair": self.suggested_repair,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationIssue":
        return cls(
            severity=data.get("severity", "INFO"),
            code=data.get("code", ""),
            message=data.get("message", ""),
            token=data.get("token"),
            suggested_repair=data.get("suggested_repair"),
        )


@dataclass
class ValidationReport:
    status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    repaired_caption: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": (
                self.status.value
                if isinstance(self.status, ValidationStatus)
                else str(self.status)
            ),
            "issues": [i.to_dict() for i in self.issues],
            "repaired_caption": self.repaired_caption,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationReport":
        st = data.get("status", ValidationStatus.VALID)
        if isinstance(st, str):
            st = ValidationStatus(st)
        issues = [
            ValidationIssue.from_dict(i) if isinstance(i, dict) else i
            for i in data.get("issues", [])
        ]
        return cls(
            status=st,
            issues=issues,
            repaired_caption=data.get("repaired_caption", ""),
        )


# ============================================================================
# Legacy & Backward-Compatible Types
# ============================================================================


class CaptionMode(str, Enum):
    TAG = "tag"
    NATURAL = "natural"
    HYBRID = "hybrid"


class LoRAStrategy(str, Enum):
    CHARACTER = "character"
    STYLE = "style"
    CONCEPT = "concept"
    GENERAL = "general"
    CUSTOM = "custom"


class VRAMMode(str, Enum):
    MAX_SPEED = "max_speed"
    BALANCED = "balanced"
    MIN_VRAM = "min_vram"


class ModelSource(str, Enum):
    LOCAL_ONLY = "local_only"
    DOWNLOAD_MISSING = "download_missing"


class WD14Device(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class TagCategory(str, Enum):
    CHARACTER = "character"
    APPEARANCE = "appearance"
    CLOTHING = "clothing"
    POSE = "pose"
    ACTION = "action"
    EXPRESSION = "expression"
    CAMERA = "camera"
    FRAMING = "framing"
    COMPOSITION = "composition"
    ENVIRONMENT = "environment"
    OBJECTS = "objects"
    LIGHTING = "lighting"
    STYLE = "style"
    QUALITY = "quality"
    META = "meta"
    OTHER = "other"


@dataclass
class TagItem:
    text: str
    source: str  # "user", "wd14", "joycaption", "florence2", "normalizer", "preset"
    confidence: float = 1.0
    category: str = TagCategory.OTHER
    locked: bool = False
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "confidence": round(self.confidence, 4),
            "category": self.category,
            "locked": self.locked,
            "raw_text": self.raw_text or self.text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TagItem":
        return cls(
            text=data.get("text", ""),
            source=data.get("source", "user"),
            confidence=data.get("confidence", 1.0),
            category=data.get("category", TagCategory.OTHER),
            locked=data.get("locked", False),
            raw_text=data.get("raw_text", ""),
        )


@dataclass
class CaptionResult:
    caption: str
    mode: CaptionMode
    tags: List[TagItem] = field(default_factory=list)
    raw_joycaption: str = ""
    raw_wd14_tags: List[Tuple[str, float, int]] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "caption": self.caption,
            "mode": self.mode.value if isinstance(self.mode, CaptionMode) else str(self.mode),
            "tags": [t.to_dict() for t in self.tags],
            "raw_joycaption": self.raw_joycaption,
            "raw_wd14_count": len(self.raw_wd14_tags),
            "execution_time": round(self.execution_time, 3),
            "metadata": self.metadata,
        }


@dataclass
class ImageRecord:
    image_path: Path
    caption_path: Path
    status: str = "pending"  # "pending", "processing", "completed", "failed", "skipped"
    caption: str = ""
    tags: List[TagItem] = field(default_factory=list)
    error: Optional[str] = None
    locked_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "caption_path": str(self.caption_path),
            "status": self.status,
            "caption": self.caption,
            "tags": [t.to_dict() for t in self.tags],
            "error": self.error,
            "locked_tags": self.locked_tags,
            "metadata": self.metadata,
        }


@dataclass
class CharacterConfig:
    name: str = ""
    trigger_word: str = ""
    reference_description: str = ""  # e.g., "black long hair, red eyes, maid headdress"
    prune_reference_from_caption: bool = False  # for Character LoRAs where trigger absorbs them


@dataclass
class PresetConfig:
    name: str = "Character LoRA"
    caption_mode: Any = CaptionMode.HYBRID
    caption_format: str = "tags"
    trigger_placement: str = "prepend"
    write_audit: bool = False
    lora_strategy: LoRAStrategy = LoRAStrategy.CHARACTER
    character: CharacterConfig = field(default_factory=CharacterConfig)
    wd14_general_threshold: float = 0.35
    wd14_character_threshold: float = 0.60
    keep_underscores: bool = False
    filter_poisons: bool = True
    quality_boosters: bool = False
    vram_mode: VRAMMode = VRAMMode.BALANCED
    wd14_device: WD14Device = WD14Device.AUTO
    model_source: ModelSource = ModelSource.LOCAL_ONLY
    custom_tags: str = ""
    blacklist: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "caption_mode": (
                self.caption_mode.value
                if hasattr(self.caption_mode, "value")
                else str(self.caption_mode)
            ),
            "caption_format": self.caption_format,
            "trigger_placement": self.trigger_placement,
            "write_audit": self.write_audit,
            "lora_strategy": (
                self.lora_strategy.value
                if hasattr(self.lora_strategy, "value")
                else str(self.lora_strategy)
            ),
            "character": {
                "name": self.character.name,
                "trigger_word": self.character.trigger_word,
                "reference_description": self.character.reference_description,
                "prune_reference_from_caption": self.character.prune_reference_from_caption,
            },
            "wd14_general_threshold": self.wd14_general_threshold,
            "wd14_character_threshold": self.wd14_character_threshold,
            "keep_underscores": self.keep_underscores,
            "filter_poisons": self.filter_poisons,
            "quality_boosters": self.quality_boosters,
            "vram_mode": self.vram_mode.value,
            "wd14_device": self.wd14_device.value,
            "model_source": self.model_source.value,
            "custom_tags": self.custom_tags,
            "blacklist": self.blacklist,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PresetConfig":
        """Reconstructs a PresetConfig from a dictionary."""
        char_data = data.get("character", {})
        if isinstance(char_data, dict):
            char_cfg = CharacterConfig(
                name=char_data.get("name", ""),
                trigger_word=char_data.get("trigger_word", ""),
                reference_description=char_data.get("reference_description", ""),
                prune_reference_from_caption=bool(char_data.get("prune_reference_from_caption", False)),
            )
        elif isinstance(char_data, CharacterConfig):
            char_cfg = char_data
        else:
            char_cfg = CharacterConfig()

        raw_mode = data.get("caption_mode", CaptionMode.HYBRID)
        if isinstance(raw_mode, str):
            try:
                c_mode = CaptionMode(raw_mode)
            except ValueError:
                c_mode = raw_mode
        else:
            c_mode = raw_mode

        raw_lora = data.get("lora_strategy", LoRAStrategy.CHARACTER)
        if isinstance(raw_lora, str):
            try:
                lora_strat = LoRAStrategy(raw_lora)
            except ValueError:
                lora_strat = LoRAStrategy.CHARACTER
        else:
            lora_strat = raw_lora

        raw_vram = data.get("vram_mode", VRAMMode.BALANCED)
        if isinstance(raw_vram, str):
            try:
                vram = VRAMMode(raw_vram)
            except ValueError:
                vram = VRAMMode.BALANCED
        else:
            vram = raw_vram

        raw_device = data.get("wd14_device", WD14Device.AUTO)
        if isinstance(raw_device, str):
            try:
                device = WD14Device(raw_device)
            except ValueError:
                device = WD14Device.AUTO
        else:
            device = raw_device

        raw_source = data.get("model_source", ModelSource.LOCAL_ONLY)
        if isinstance(raw_source, str):
            try:
                source = ModelSource(raw_source)
            except ValueError:
                source = ModelSource.LOCAL_ONLY
        else:
            source = raw_source

        return cls(
            name=data.get("name", "Character LoRA"),
            caption_mode=c_mode,
            caption_format=data.get("caption_format", "tags"),
            trigger_placement=data.get("trigger_placement", "prepend"),
            write_audit=bool(data.get("write_audit", False)),
            lora_strategy=lora_strat,
            character=char_cfg,
            wd14_general_threshold=float(data.get("wd14_general_threshold", 0.35)),
            wd14_character_threshold=float(data.get("wd14_character_threshold", 0.60)),
            keep_underscores=bool(data.get("keep_underscores", False)),
            filter_poisons=bool(data.get("filter_poisons", True)),
            quality_boosters=bool(data.get("quality_boosters", False)),
            vram_mode=vram,
            wd14_device=device,
            model_source=source,
            custom_tags=str(data.get("custom_tags", "")),
            blacklist=list(data.get("blacklist", [])),
        )

