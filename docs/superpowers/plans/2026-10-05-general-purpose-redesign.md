# HyperCaptioner General-Purpose Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign HyperCaptioner into a modular, multi-stage, general-purpose image captioning system with decoupled mode contracts (Character, Style, Outfit, Pose, Concept), an 18-category semantic fact extractor, meaning-based content vs. style discrimination, generic case-insensitive trigger handling, conservative semantic validation, and full audit traceability.

**Architecture:** Two-stage generation pipeline: Stage 1 extracts structured visual facts into an 18-category schema via a hybrid dual-parser (JSON + block fallback) augmented with WD14 tags; Stage 2 applies active mode contracts and semantic filtering to produce conditioning tokens; Stage 3 runs conservative validation (hard vs. potential contradictions, safe auto-repair, facts-rebuild fallback) before writing clean training `.txt` sidecars and separate `.audit.json` metadata.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, BitsAndBytes (NF4), ONNX Runtime, FastAPI, Uvicorn, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-general-purpose-redesign.md`

## Global Constraints

- Never hardcode Loggreal, any specific LoRA, character, artist, or style-specific vocabulary into core architecture.
- Distinguish between what an image contains (content) and how it renders that content (style); never infer rendering property merely because an object/subject is present.
- `locked=True` user tags are strictly inviolable across all filtering, pruning, and validation stages.
- `is_uncertain: bool = True` is the authoritative uncertainty representation; any uncertain fact is strictly excluded from final training captions.
- Trigger word matching and deduplication is case-insensitive by default.
- Training `.txt` sidecars contain ONLY clean caption text; audit metadata is stored in separate `.audit.json` sidecars.

---

### Task 1: Core Semantic Types & Multi-Category Representation

**Files:**
- Modify: `hyper_captioner/core/types.py`
- Create: `tests/test_core_types.py`

**Interfaces:**
- Consumes: None (base types).
- Produces: `SemanticCategory`, `FactItem`, `CaptionToken`, `StructuredVisualFacts`, `CaptionModeType`, `CaptionFormat`, `TriggerPlacement`, `TriggerConfig`, `ValidationStatus`, `ValidationIssue`, `ValidationReport`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_types.py
import pytest
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionModeType,
    CaptionToken,
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
    TriggerConfig,
    TriggerPlacement,
    ValidationIssue,
    ValidationReport,
    ValidationStatus,
)

def test_semantic_categories_enumeration():
    expected = {
        "identity", "appearance", "clothing", "pose", "expression",
        "composition", "camera", "environment", "lighting", "material",
        "rendering", "style", "objects", "color", "texture", "concept",
        "quality", "uncertainty"
    }
    actual = {c.value for c in SemanticCategory}
    assert expected == actual

def test_fact_item_multi_category_and_uncertainty():
    item = FactItem(
        id="fact_01",
        text="leather jacket",
        primary_category=SemanticCategory.CLOTHING,
        categories=[SemanticCategory.CLOTHING, SemanticCategory.MATERIAL],
        confidence=0.95,
        source="joycaption",
        is_uncertain=False,
        locked=True
    )
    assert item.primary_category == SemanticCategory.CLOTHING
    assert SemanticCategory.MATERIAL in item.categories
    assert item.locked is True
    assert item.is_uncertain is False

def test_trigger_config_case_insensitive_default():
    cfg = TriggerConfig(word="MyTrigger")
    assert cfg.case_sensitive is False
    assert cfg.placement == TriggerPlacement.PREPEND
    assert cfg.deduplicate is True

def test_validation_report_status():
    report = ValidationReport(
        status=ValidationStatus.REPAIRED,
        issues=[ValidationIssue(severity="WARNING", code="TEST", message="Test issue")],
        repaired_caption="Trigger, test caption"
    )
    assert report.status == ValidationStatus.REPAIRED
    assert len(report.issues) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_core_types.py -v`  
Expected: FAIL (missing types / attribute errors).

- [ ] **Step 3: Write minimal implementation in `hyper_captioner/core/types.py`**

Update `hyper_captioner/core/types.py` to define:
- `SemanticCategory` (18 categories: `IDENTITY`, `APPEARANCE`, `CLOTHING`, `POSE`, `EXPRESSION`, `COMPOSITION`, `CAMERA`, `ENVIRONMENT`, `LIGHTING`, `MATERIAL`, `RENDERING`, `STYLE`, `OBJECTS`, `COLOR`, `TEXTURE`, `CONCEPT`, `QUALITY`, `UNCERTAINTY`).
- `CaptionModeType` (`CHARACTER = "character"`, `STYLE = "style"`, `OUTFIT = "outfit"`, `POSE = "pose"`, `CONCEPT = "concept"`).
- `CaptionFormat` (`TAGS = "tags"`, `STRUCTURED = "structured"`, `NATURAL = "natural"`).
- `TriggerPlacement` (`PREPEND = "prepend"`, `APPEND = "append"`, `WRAP = "wrap"`, `OMIT = "omit"`).
- `TriggerConfig` with `word: str = ""`, `placement: TriggerPlacement = TriggerPlacement.PREPEND`, `case_sensitive: bool = False`, `deduplicate: bool = True`, `absorb_stable_traits: bool = False`.
- `FactItem` with `id: str`, `text: str`, `primary_category: SemanticCategory`, `categories: List[SemanticCategory] = field(default_factory=list)`, `confidence: float = 1.0`, `source: str = "joycaption"`, `is_stable: bool = False`, `is_uncertain: bool = False`, `locked: bool = False`, `raw_text: str = ""`, plus `to_dict()` and `from_dict()`.
- `CaptionToken` with `text: str`, `primary_category: SemanticCategory`, `categories: List[SemanticCategory]`, `source_fact_ids: List[str]`, `confidence: float`, `transformation: str`.
- `StructuredVisualFacts` with `facts_by_category: Dict[SemanticCategory, List[FactItem]]`, `raw_response: str = ""`, `parse_method: str = "json"`, `image_metadata: Dict[str, Any] = field(default_factory=dict)`.
- `ValidationStatus` (`VALID = "valid"`, `REPAIRED = "repaired"`, `REJECTED = "rejected"`).
- `ValidationIssue` with `severity: str`, `code: str`, `message: str`, `token: Optional[str] = None`, `suggested_repair: Optional[str] = None`.
- `ValidationReport` with `status: ValidationStatus`, `issues: List[ValidationIssue] = field(default_factory=list)`, `repaired_caption: str = ""`.
- Preserve existing `CaptionMode` (as alias to `CaptionFormat` for backward compatibility), `LoRAStrategy`, `VRAMMode`, `WD14Device`, `ImageRecord`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_core_types.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hyper_captioner/core/types.py tests/test_core_types.py
git commit -m "feat(core): add multi-category FactItem, TriggerConfig, and validation types"
```

---

### Task 2: Mode Contract System (`BaseCaptionMode` & 5 Core Implementations)

**Files:**
- Create: `hyper_captioner/caption_modes/base.py`
- Create: `hyper_captioner/caption_modes/character.py`
- Create: `hyper_captioner/caption_modes/style.py`
- Create: `hyper_captioner/caption_modes/outfit.py`
- Create: `hyper_captioner/caption_modes/pose.py`
- Create: `hyper_captioner/caption_modes/concept.py`
- Create: `hyper_captioner/caption_modes/registry.py`
- Create: `hyper_captioner/caption_modes/__init__.py`
- Create: `tests/test_caption_modes.py`

**Interfaces:**
- Consumes: `hyper_captioner.core.types` (`SemanticCategory`, `FactItem`, `StructuredVisualFacts`, `CaptionToken`, `CaptionFormat`, `TriggerConfig`).
- Produces: `BaseCaptionMode`, `CharacterMode`, `StyleMode`, `OutfitMode`, `PoseMode`, `ConceptMode`, `get_caption_mode(name: str) -> BaseCaptionMode`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_caption_modes.py
import pytest
from hyper_captioner.caption_modes.registry import get_caption_mode, list_caption_modes
from hyper_captioner.core.types import CaptionModeType, SemanticCategory

def test_registry_lists_all_five_modes():
    modes = list_caption_modes()
    assert set(modes) == {"character", "style", "outfit", "pose", "concept"}

def test_style_mode_excludes_character_and_clothing():
    mode = get_caption_mode("style")
    assert SemanticCategory.APPEARANCE in mode.exclude_categories
    assert SemanticCategory.CLOTHING in mode.exclude_categories
    assert SemanticCategory.IDENTITY in mode.exclude_categories
    assert SemanticCategory.RENDERING in mode.include_categories
    assert SemanticCategory.STYLE in mode.include_categories

def test_outfit_mode_focuses_on_garments():
    mode = get_caption_mode("outfit")
    assert SemanticCategory.CLOTHING in mode.include_categories
    assert SemanticCategory.MATERIAL in mode.include_categories
    assert SemanticCategory.ENVIRONMENT in mode.exclude_categories
    assert SemanticCategory.IDENTITY in mode.exclude_categories

def test_pose_mode_focuses_on_body_and_camera():
    mode = get_caption_mode("pose")
    assert SemanticCategory.POSE in mode.include_categories
    assert SemanticCategory.COMPOSITION in mode.include_categories
    assert SemanticCategory.CAMERA in mode.include_categories
    assert SemanticCategory.CLOTHING in mode.exclude_categories
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_caption_modes.py -v`  
Expected: FAIL (ModuleNotFoundError `hyper_captioner.caption_modes`).

- [ ] **Step 3: Write implementations of `BaseCaptionMode` and the 5 modes**

1. `hyper_captioner/caption_modes/base.py`:
   - `BaseCaptionMode(ABC)` defining `name`, `description`, `include_categories: Set[SemanticCategory]`, `exclude_categories: Set[SemanticCategory]`.
   - Abstract methods: `build_extraction_instructions(self) -> str`, `filter_facts(self, facts: StructuredVisualFacts, trigger_cfg: TriggerConfig) -> List[FactItem]`, `format_tokens(self, tokens: List[CaptionToken], format_type: CaptionFormat) -> str`.
2. `hyper_captioner/caption_modes/character.py`: Character mode contract implementation.
3. `hyper_captioner/caption_modes/style.py`: Style mode contract implementation.
4. `hyper_captioner/caption_modes/outfit.py`: Outfit mode contract implementation.
5. `hyper_captioner/caption_modes/pose.py`: Pose mode contract implementation.
6. `hyper_captioner/caption_modes/concept.py`: Concept mode contract implementation (configurable concept focus).
7. `hyper_captioner/caption_modes/registry.py`: Registry mapping string names / `CaptionModeType` to instances, with `get_caption_mode(name)` and `list_caption_modes()`.
8. `hyper_captioner/caption_modes/__init__.py`: Package exports.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_caption_modes.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hyper_captioner/caption_modes/ tests/test_caption_modes.py
git commit -m "feat(modes): implement decoupled BaseCaptionMode and 5 core mode contracts"
```

---

### Task 3: Meaning-Based Content vs. Style Discriminator & Semantic Filter

**Files:**
- Create: `hyper_captioner/pipeline/semantic_filter.py`
- Create: `tests/test_semantic_filter.py`

**Interfaces:**
- Consumes: `hyper_captioner.core.types` (`FactItem`, `SemanticCategory`, `StructuredVisualFacts`, `TriggerConfig`), `BaseCaptionMode`.
- Produces: `SemanticFilterResult`, `SemanticFilter.filter_facts(mode: BaseCaptionMode, facts: StructuredVisualFacts, trigger_cfg: TriggerConfig) -> SemanticFilterResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_semantic_filter.py
import pytest
from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import FactItem, SemanticCategory, StructuredVisualFacts, TriggerConfig
from hyper_captioner.pipeline.semantic_filter import SemanticFilter

def test_semantic_filter_style_mode_strips_content_preserves_rendering():
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.APPEARANCE: [
                FactItem(id="app_1", text="pink hair", primary_category=SemanticCategory.APPEARANCE),
                FactItem(id="app_2", text="blue eyes", primary_category=SemanticCategory.APPEARANCE),
            ],
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_1", text="maid dress", primary_category=SemanticCategory.CLOTHING),
            ],
            SemanticCategory.RENDERING: [
                FactItem(id="ren_1", text="soft cel shading", primary_category=SemanticCategory.RENDERING),
                FactItem(id="ren_2", text="clean line art", primary_category=SemanticCategory.RENDERING),
            ],
            SemanticCategory.TEXTURE: [
                FactItem(id="tex_1", text="individually detailed hair strands", primary_category=SemanticCategory.TEXTURE),
            ],
            SemanticCategory.UNCERTAINTY: [
                FactItem(id="unc_1", text="silk satin fabric", primary_category=SemanticCategory.UNCERTAINTY, is_uncertain=True),
            ]
        }
    )
    style_mode = get_caption_mode("style")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(style_mode, facts, TriggerConfig(word="MyStyle"))

    accepted_texts = [f.text for f in result.accepted]
    assert "soft cel shading" in accepted_texts
    assert "clean line art" in accepted_texts
    assert "individually detailed hair strands" in accepted_texts
    # Content and uncertain items must be rejected
    assert "pink hair" not in accepted_texts
    assert "blue eyes" not in accepted_texts
    assert "maid dress" not in accepted_texts
    assert "silk satin fabric" not in accepted_texts

def test_semantic_filter_respects_locked_priority():
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_1", text="red ribbon", primary_category=SemanticCategory.CLOTHING, locked=True),
            ]
        }
    )
    style_mode = get_caption_mode("style")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(style_mode, facts, TriggerConfig())
    accepted_texts = [f.text for f in result.accepted]
    # Inviolable lock must override normal exclusion
    assert "red ribbon" in accepted_texts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_semantic_filter.py -v`  
Expected: FAIL (ModuleNotFoundError `hyper_captioner.pipeline.semantic_filter`).

- [ ] **Step 3: Implement `SemanticFilter` in `hyper_captioner/pipeline/semantic_filter.py`**

- `SemanticFilterResult`: dataclass with `accepted: List[FactItem]`, `rejected: List[Tuple[FactItem, str]]` (where string is the rejection reason, e.g. `"content_leak_in_style"`, `"uncertainty"`, `"excluded_category"`).
- `SemanticFilter.filter_facts(...)`:
  1. If `fact.locked == True`, immediately accept (inviolable priority).
  2. If `fact.is_uncertain == True` or `fact.primary_category == SemanticCategory.UNCERTAINTY`, reject with reason `"uncertainty"`.
  3. Evaluate mode's `exclude_categories` and `include_categories`.
  4. Apply meaning-based Content vs. Style discrimination (distinguish rendering properties like `"individually detailed hair strands"` from content like `"pink hair"`).
  5. Check trigger trait redundancy: if `trigger_cfg.absorb_stable_traits` is True and fact is a stable trait absorbed by the trigger, compact/prune only if redundant.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_semantic_filter.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hyper_captioner/pipeline/semantic_filter.py tests/test_semantic_filter.py
git commit -m "feat(pipeline): implement meaning-based semantic filter with lock priority"
```

---

### Task 4: Quality Control & Semantic Validator with Safe Auto-Repair Boundary

**Files:**
- Create: `hyper_captioner/pipeline/validator.py`
- Create: `tests/test_validator.py`

**Interfaces:**
- Consumes: `hyper_captioner.core.types` (`FactItem`, `CaptionToken`, `StructuredVisualFacts`, `TriggerConfig`, `ValidationStatus`, `ValidationIssue`, `ValidationReport`), `BaseCaptionMode`.
- Produces: `SemanticValidator.validate(caption: str, tokens: List[CaptionToken], mode: BaseCaptionMode, facts: StructuredVisualFacts, trigger_cfg: TriggerConfig) -> ValidationReport`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validator.py
import pytest
from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import CaptionToken, SemanticCategory, StructuredVisualFacts, TriggerConfig, ValidationStatus
from hyper_captioner.pipeline.validator import SemanticValidator

def test_validator_detects_hard_contradiction():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    tokens = [
        CaptionToken(text="standing", primary_category=SemanticCategory.POSE, categories=[], source_fact_ids=[], confidence=0.9, transformation="direct"),
        CaptionToken(text="sitting", primary_category=SemanticCategory.POSE, categories=[], source_fact_ids=[], confidence=0.8, transformation="direct"),
    ]
    report = validator.validate("standing, sitting", tokens, mode, StructuredVisualFacts({}), TriggerConfig())
    assert any(i.severity == "ERROR" and "contradiction" in i.code.lower() for i in report.issues)
    assert report.status == ValidationStatus.REJECTED

def test_validator_warns_on_potential_contradiction_without_deleting():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    tokens = [
        CaptionToken(text="smiling", primary_category=SemanticCategory.EXPRESSION, categories=[], source_fact_ids=[], confidence=0.9, transformation="direct"),
        CaptionToken(text="neutral expression", primary_category=SemanticCategory.EXPRESSION, categories=[], source_fact_ids=[], confidence=0.8, transformation="direct"),
    ]
    report = validator.validate("smiling, neutral expression", tokens, mode, StructuredVisualFacts({}), TriggerConfig())
    # Must be WARNING, not ERROR, and not deleted
    assert any(i.severity == "WARNING" for i in report.issues)
    assert "smiling" in report.repaired_caption
    assert "neutral expression" in report.repaired_caption

def test_validator_safe_auto_repairs_duplicate_case_insensitive_triggers():
    validator = SemanticValidator()
    mode = get_caption_mode("character")
    trigger_cfg = TriggerConfig(word="MyChar", case_sensitive=False)
    raw_caption = "mychar, MYCHAR, 1girl, standing, MyChar"
    report = validator.validate(raw_caption, [], mode, StructuredVisualFacts({}), trigger_cfg)
    assert report.status == ValidationStatus.REPAIRED
    # Only one trigger word should remain, cleanly prepended
    parts = [p.strip() for p in report.repaired_caption.split(",")]
    assert parts.count("MyChar") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validator.py -v`  
Expected: FAIL (ModuleNotFoundError `hyper_captioner.pipeline.validator`).

- [ ] **Step 3: Implement `SemanticValidator` in `hyper_captioner/pipeline/validator.py`**

- Implements HARD contradictions (`standing + sitting`, `lying + standing`, `eyes open + eyes closed`, `indoors + outdoors`, `day + night`) generating `ERROR` severity.
- Implements POTENTIAL contradictions generating `WARNING` severity without automatic removal.
- Implements case-insensitive trigger deduplication and placement enforcement.
- Implements safe auto-repair for deterministic issues (duplicate tokens, punctuation cleanup, subjective hype adjectives like "masterpiece", "stunning").
- Sets `ValidationStatus`:
  - `VALID` if 0 issues.
  - `REPAIRED` if only safe low-risk issues corrected.
  - `REJECTED` if HARD contradictions or severe category leakage remain.
- Enforces safety boundary: Never invents visual facts.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validator.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hyper_captioner/pipeline/validator.py tests/test_validator.py
git commit -m "feat(pipeline): implement semantic validator with hard/potential conflicts and safe repair"
```

---

### Task 5: Stage 1 Structured Visual Fact Extractor (Approach C Hybrid Dual-Parser)

**Files:**
- Create: `hyper_captioner/pipeline/stage1_extractor.py`
- Modify: `hyper_captioner/engines/joycaption.py`
- Create: `tests/test_stage1_extractor.py`

**Interfaces:**
- Consumes: `hyper_captioner.core.types` (`SemanticCategory`, `FactItem`, `StructuredVisualFacts`), JoyCaption raw output, WD14 tuples.
- Produces: `Stage1Extractor.extract_facts(raw_joycaption: str, wd14_tags: List[Tuple[str, float, int]]) -> StructuredVisualFacts`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage1_extractor.py
import pytest
from hyper_captioner.core.types import SemanticCategory
from hyper_captioner.pipeline.stage1_extractor import Stage1Extractor

def test_extract_from_clean_json():
    json_payload = """
    ```json
    {
      "identity": ["1girl", "solo"],
      "appearance": ["black long hair", "red eyes"],
      "clothing": ["black maid dress", "white apron"],
      "pose": ["standing", "looking at viewer"],
      "rendering": ["clean linework", "soft shading"],
      "uncertain": ["lace trim"]
    }
    ```
    """
    extractor = Stage1Extractor()
    facts = extractor.extract_facts(json_payload, wd14_tags=[])
    assert facts.parse_method == "json"
    assert len(facts.get_category(SemanticCategory.IDENTITY)) == 2
    assert facts.get_category(SemanticCategory.IDENTITY)[0].text == "1girl"
    assert len(facts.get_category(SemanticCategory.APPEARANCE)) == 2
    assert facts.get_category(SemanticCategory.UNCERTAINTY)[0].is_uncertain is True

def test_extract_fallback_from_tagged_blocks():
    block_payload = """
    [IDENTITY]: 1girl, solo
    [APPEARANCE]: blue hair, green eyes
    [CLOTHING]: school uniform, pleated skirt
    [POSE]: sitting
    """
    extractor = Stage1Extractor()
    facts = extractor.extract_facts(block_payload, wd14_tags=[])
    assert facts.parse_method in {"tagged_block", "hybrid"}
    assert len(facts.get_category(SemanticCategory.APPEARANCE)) == 2
    assert len(facts.get_category(SemanticCategory.CLOTHING)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage1_extractor.py -v`  
Expected: FAIL (ModuleNotFoundError `hyper_captioner.pipeline.stage1_extractor`).

- [ ] **Step 3: Implement `Stage1Extractor` in `hyper_captioner/pipeline/stage1_extractor.py`**

- JSON parser with code fence stripping and inline regex repair for trailing commas and single quotes.
- Tagged block regex fallback parser matching `[CATEGORY]:` or `CATEGORY:`.
- Merges auxiliary WD14 tags into appropriate categories with provenance tracking (`source="wd14"`).
- Automatically marks any fact in `"uncertain"` with `is_uncertain=True` and category `UNCERTAINTY`.
- Updates `hyper_captioner/engines/joycaption.py` to add `build_stage1_extraction_prompt()` requesting the standardized JSON schema.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage1_extractor.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hyper_captioner/pipeline/stage1_extractor.py hyper_captioner/engines/joycaption.py tests/test_stage1_extractor.py
git commit -m "feat(pipeline): implement Stage 1 hybrid dual-parser fact extractor"
```

---

### Task 6: Caption Builder Refactoring & Output Formatters

**Files:**
- Modify: `hyper_captioner/pipeline/builder.py`
- Create: `tests/test_builder_formats.py`

**Interfaces:**
- Consumes: `BaseCaptionMode`, `SemanticFilterResult`, `TriggerConfig`, `CaptionFormat`.
- Produces: `CaptionBuilder.build(mode: BaseCaptionMode, filter_result: SemanticFilterResult, trigger_cfg: TriggerConfig, format_type: CaptionFormat) -> Tuple[str, List[CaptionToken]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_builder_formats.py
import pytest
from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import CaptionFormat, FactItem, SemanticCategory, TriggerConfig, TriggerPlacement
from hyper_captioner.pipeline.builder import CaptionBuilder
from hyper_captioner.pipeline.semantic_filter import SemanticFilterResult

def test_builder_tag_format():
    builder = CaptionBuilder()
    mode = get_caption_mode("character")
    trigger = TriggerConfig(word="Heroine", placement=TriggerPlacement.PREPEND)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            FactItem(id="f2", text="blue eyes", primary_category=SemanticCategory.APPEARANCE),
            FactItem(id="f3", text="standing", primary_category=SemanticCategory.POSE),
        ],
        rejected=[]
    )
    caption, tokens = builder.build(mode, filter_result, trigger, CaptionFormat.TAGS)
    assert caption.startswith("Heroine, ")
    assert "1girl" in caption
    assert "blue eyes" in caption
    assert len(tokens) >= 3
    assert tokens[0].source_fact_ids == ["f1"] or tokens[0].text == "Heroine"

def test_builder_trigger_append():
    builder = CaptionBuilder()
    mode = get_caption_mode("style")
    trigger = TriggerConfig(word="AestheticV1", placement=TriggerPlacement.APPEND)
    filter_result = SemanticFilterResult(
        accepted=[
            FactItem(id="f1", text="cel shading", primary_category=SemanticCategory.RENDERING),
            FactItem(id="f2", text="vibrant colors", primary_category=SemanticCategory.COLOR),
        ],
        rejected=[]
    )
    caption, _ = builder.build(mode, filter_result, trigger, CaptionFormat.TAGS)
    assert caption.endswith(", AestheticV1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_builder_formats.py -v`  
Expected: FAIL (Builder signature or method mismatch).

- [ ] **Step 3: Refactor `CaptionBuilder` in `hyper_captioner/pipeline/builder.py`**

- Implements structured token sorting by category hierarchy: `Trigger ➔ Identity ➔ Appearance ➔ Clothing ➔ Pose ➔ Composition/Camera ➔ Environment ➔ Lighting ➔ Rendering/Style`.
- Implements `TAGS` formatting (comma-separated tokens, underscore toggle support).
- Implements `STRUCTURED` formatting (compact grouped phrases).
- Implements `NATURAL` formatting (declarative prose synthesizer).
- Creates `CaptionToken` objects preserving full provenance: `source_fact_ids`, `primary_category`, `confidence`, and `transformation` (`"direct"`, `"trigger_injected"`, `"locked_override"`).
- Preserves backward-compatible legacy `build()` signature for existing caller code.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_builder_formats.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hyper_captioner/pipeline/builder.py tests/test_builder_formats.py
git commit -m "refactor(builder): integrate mode contracts, token traceability, and formatters"
```

---

### Task 7: Dataset Pipeline, Audit Metadata Persistence, and Batch Traceability

**Files:**
- Modify: `hyper_captioner/dataset/manager.py`
- Modify: `hyper_captioner/dataset/exporter.py`
- Create: `tests/test_dataset_audit.py`

**Interfaces:**
- Consumes: `Stage1Extractor`, `SemanticFilter`, `CaptionBuilder`, `SemanticValidator`, `write_sidecar`, `write_audit_sidecar`.
- Produces: `DatasetPipeline.caption_image(...) -> CaptionResult`, `DatasetPipeline.process_dataset(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset_audit.py
import json
import pytest
from pathlib import Path
from hyper_captioner.core.types import CaptionFormat, CaptionModeType, TriggerConfig
from hyper_captioner.dataset.exporter import write_audit_sidecar, write_sidecar

def test_sidecar_and_audit_separation(tmp_path):
    img_file = tmp_path / "sample_001.png"
    img_file.touch()

    caption = "MyTrigger, 1girl, solo, standing, blue eyes"
    audit_data = {
        "status": "valid",
        "stage1_facts_count": 8,
        "accepted_count": 4,
        "rejected_count": 4,
        "traceability": [{"token": "blue eyes", "source": "app_01"}]
    }

    # Write training sidecar
    txt_path = write_sidecar(img_file, caption)
    assert txt_path.read_text(encoding="utf-8") == caption

    # Write audit sidecar
    audit_path = write_audit_sidecar(img_file, audit_data)
    assert audit_path.name == "sample_001.audit.json"
    loaded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert loaded_audit["status"] == "valid"
    # Ensure .txt is untouched and clean
    assert "audit" not in txt_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_audit.py -v`  
Expected: FAIL (missing `write_audit_sidecar`).

- [ ] **Step 3: Implement `write_audit_sidecar` and integrate pipeline in `dataset/manager.py` and `dataset/exporter.py`**

- In `dataset/exporter.py`: implement `write_audit_sidecar(image_path: Path, audit_data: Dict) -> Path`.
- In `dataset/manager.py`: update `caption_image()` to coordinate:
  1. `Stage1Extractor.extract_facts(...)`
  2. `SemanticFilter.filter_facts(...)`
  3. `CaptionBuilder.build(...)`
  4. `SemanticValidator.validate(...)`
  5. If `REJECTED`: rebuild cleanly from facts.
  6. Write `<image>.txt` sidecar.
  7. If `save_audit` is True: write `<image>.audit.json`.
- Track batch statistics: count of `valid`, `repaired`, and `rejected` captions.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_audit.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hyper_captioner/dataset/manager.py hyper_captioner/dataset/exporter.py tests/test_dataset_audit.py
git commit -m "feat(dataset): wire end-to-end two-stage pipeline with separate audit sidecars"
```

---

### Task 8: CLI Updates & Backward Compatibility (`main.py`)

**Files:**
- Modify: `main.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: CLI args from terminal.
- Produces: Execution of `cmd_caption`, `cmd_scan`, `cmd_vocab`, `cmd_export` supporting new modes and audit flags.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import pytest
from main import parse_args_helper

def test_cli_mode_and_format_aliases():
    # Legacy --mode hybrid -> semantic mode character, format tags
    args = parse_args_helper(["caption", "--dataset", "dummy", "--mode", "hybrid"])
    assert args.mode == "character"

    # Legacy --mode natural -> format natural, semantic mode character
    args = parse_args_helper(["caption", "--dataset", "dummy", "--mode", "natural"])
    assert args.format == "natural"
    assert args.mode == "character"

    # Direct new mode
    args = parse_args_helper(["caption", "--dataset", "dummy", "--mode", "style", "--format", "tags", "--trigger-placement", "append", "--audit"])
    assert args.mode == "style"
    assert args.format == "tags"
    assert args.trigger_placement == "append"
    assert args.audit is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`  
Expected: FAIL (missing `parse_args_helper` / unrecognized args).

- [ ] **Step 3: Update argument parser and commands in `main.py`**

- Extract parser creation into `build_parser()` and `parse_args_helper()`.
- Add `--mode`: choices `character`, `style`, `outfit`, `pose`, `concept` + backward-compatible aliases `hybrid`, `tag`.
- Add `--format`: choices `tags`, `structured`, `natural`.
- Handle legacy `--mode natural` aliasing cleanly to `--format natural` with mode `character`.
- Add `--trigger-placement`: choices `prepend`, `append`, `wrap`, `omit` (default `prepend`).
- Add `--audit` / `--debug-pipeline`: flag to persist `.audit.json` alongside captions and print audit trace.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_cli.py
git commit -m "feat(cli): add mode contracts, format separation, and --audit flag to main.py"
```

---

### Task 9: REST API & Web UI Updates

**Files:**
- Modify: `hyper_captioner/api/app.py`
- Modify: `hyper_captioner/ui/index.html`
- Modify: `hyper_captioner/ui/js/app.js`
- Create: `tests/test_api_v2.py`

**Interfaces:**
- Consumes: REST API calls from Web UI.
- Produces: API endpoints with validation status and token chips.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_v2.py
import pytest
from fastapi.testclient import TestClient
from hyper_captioner.api.app import app

client = TestClient(app)

def test_api_modes_endpoint():
    res = client.get("/api/modes")
    assert res.status_code == 200
    data = res.json()
    assert "character" in data["modes"]
    assert "style" in data["modes"]
    assert "outfit" in data["modes"]
    assert "pose" in data["modes"]
    assert "concept" in data["modes"]

def test_api_review_item_contains_validation_fields(tmp_path):
    img = tmp_path / "test.png"
    img.touch()
    txt = tmp_path / "test.txt"
    txt.write_text("1girl, solo, standing", encoding="utf-8")
    res = client.get(f"/api/review/item?image_path={img}")
    assert res.status_code == 200
    data = res.json()
    assert "validation_status" in data
    assert "traceability" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_v2.py -v`  
Expected: FAIL (404 on `/api/modes` or missing fields).

- [ ] **Step 3: Update `hyper_captioner/api/app.py`, `ui/index.html`, and `ui/js/app.js`**

- In `api/app.py`:
  - Add `/api/modes` endpoint returning the 5 mode contracts and their descriptions.
  - Update `CaptionRequest`, `PresetSaveRequest`, and review endpoints with `caption_mode`, `caption_format`, `trigger_placement`, `validation_status`, and `traceability`.
- In `ui/index.html`:
  - Update Mode `<select id="caption-mode-select">` with the 5 core contracts: `Character Mode`, `Style Mode`, `Outfit Mode`, `Pose Mode`, `Concept Mode`.
  - Add Format `<select id="caption-format-select">`: `Tags (Comma-Separated)`, `Structured Phrases`, `Natural Prose`.
  - Add Trigger Placement `<select id="trigger-placement-select">`: `Prepend`, `Append`, `Wrap`, `Omit`.
  - In Review section: add validation status badge container.
- In `ui/js/app.js`:
  - Wire new selectors and display validation badge (`VALID ✓`, `REPAIRED ⚠️`, `REJECTED 🔄`).
  - Render clickable token chips with category and confidence tooltips.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_v2.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hyper_captioner/api/app.py hyper_captioner/ui/index.html hyper_captioner/ui/js/app.js tests/test_api_v2.py
git commit -m "feat(ui): update CyberDeck UI and API with 5 mode contracts and validation badges"
```

---

### Task 10: Comprehensive Test Suite (Deliberate Failures & Regression)

**Files:**
- Create: `tests/test_modes_e2e.py`
- Create: `tests/test_validation_failures.py`
- Create: `tests/test_regression.py`

**Interfaces:**
- Consumes: Full HyperCaptioner pipeline.
- Produces: Complete passing test suite.

- [ ] **Step 1: Write E2E Mode Tests (`tests/test_modes_e2e.py`)**

Verify end-to-end execution of all 5 modes across real test images in `test_data/phase1_validation/`, asserting that:
- Style Mode drops character content and keeps rendering/lighting.
- Outfit Mode drops environment clutter and keeps garments.
- Pose Mode isolates body stance and camera framing.
- Character Mode keeps character traits and environment/lighting while dropping esoteric 3D jargon.

- [ ] **Step 2: Write Deliberate Failure Tests (`tests/test_validation_failures.py`)**

Deliberate edge cases:
- Hard contradiction (`standing` + `sitting`) $\to$ triggers `ERROR` and `REJECTED`.
- Potential contradiction (`smiling` + `neutral expression`) $\to$ triggers `WARNING` and retains both.
- Case-insensitive trigger duplicate (`Loggreal, loggreal, LOGGREAL`) $\to$ single clean insertion.
- Subjective hype (`masterpiece, beautiful girl`) $\to$ semantic cleanup, status `REPAIRED`.
- Low confidence / uncertainty $\to$ strictly omitted from final caption.
- Inviolable lock $\to$ `locked=True` tag survives forbidden category exclusion.

- [ ] **Step 3: Write Regression Tests (`tests/test_regression.py`)**

Verify that:
- `scan_dataset()` correctly counts images, captions, orphans.
- Vocabulary index correctly searches variants and replaces tags.
- Sidecar export and backup creation operate correctly.

- [ ] **Step 4: Run the complete test suite**

Run: `pytest tests/ -v`  
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: add comprehensive mode E2E, deliberate failure, and regression test suites"
```
