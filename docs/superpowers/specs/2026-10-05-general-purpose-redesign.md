# HyperCaptioner: General-Purpose Dataset Captioning Architecture
**Specification Document**  
**Date:** 2026-10-05  
**Status:** Approved for Implementation Planning  
**Target Path:** `docs/superpowers/specs/2026-10-05-general-purpose-redesign.md`

---

## 1. Executive Summary & Purpose

HyperCaptioner is redesigned from a single-prompt, heuristic-cleaned image captioner into a modular, multi-stage, general-purpose vision captioning workstation engineered specifically for training diffusion LoRAs (Character, Style, Outfit, Pose, and Concept).

### Core Objectives
1. **Completely Generic Architecture**: Zero hard-coded assumptions about specific characters, styles (e.g. Loggreal), artists, models, or vocabularies.
2. **True Two-Stage Pipeline**:
   - **Stage 1**: Structured visual fact extraction into a standardized 18-category semantic schema using a hybrid dual-parser (JSON + tagged block fallback) grounded with auxiliary WD14 tagging.
   - **Stage 2**: Mode-specific conditioning transformation, category filtering, meaning-based content vs. style discrimination, and trigger word injection.
3. **Decoupled Mode Contracts**: Explicit contracts for `Character`, `Style`, `Outfit`, `Pose`, and `Concept` modes stored in modular files (`caption_modes/`), separating semantic captioning objective from output formatting style.
4. **Grounded Factuality & Anti-Hallucination**: Direct visual evidence over invented camera gear or technology; evidence-based confidence thresholds; zero character guessing from pretrained weights; semantic handling of subjective hype words.
5. **Conservative Validation & Safe Auto-Repair**: Hard contradictions vs. potential coexisting nuances; three-tier status (`VALID`, `REPAIRED`, `REJECTED`); safe repairs for deterministic formatting issues; rebuild from source facts on semantic errors.
6. **Full Traceability & Auditability**: Every token maps back to source facts, categories, confidence, and transformations; persistent audit metadata kept strictly separate from clean training `.txt` sidecars; CLI `--audit` / `--debug-pipeline` flags.
7. **Inviolable User Control**: `locked=True` user tags are strictly immune to pruning, filtering, or auto-repair across all modes.

---

## 2. Foundational Design Principle

> **HyperCaptioner must distinguish between what an image contains and how the image renders that content. It must never infer a rendering property merely because an object or subject is present.**
>
> *(Example: The presence of armor does not imply ray-traced metallic reflections unless visible specular responses actually demonstrate it; the presence of an illustrated character does not imply cel-shading unless flat color zones and hard line boundaries are visibly supported.)*

---

## 3. Semantic Categories & Schema

All visual facts extracted during Stage 1 are structured into exactly 18 semantic categories. Categories act as baseline routing hints, with semantic meaning determining final role.

```python
class SemanticCategory(str, Enum):
    IDENTITY    = "identity"     # Subject counts (1girl, solo), explicit naming
    APPEARANCE  = "appearance"   # Physical traits: hair color/length, eye color, facial features
    CLOTHING    = "clothing"     # Garments, cuts, footwear, wearable accessories
    POSE        = "pose"         # Stance, limb positioning, torso orientation, gaze direction
    EXPRESSION  = "expression"   # Facial expression: smile, neutral, parted lips, closed eyes
    COMPOSITION = "composition"  # Framing & shot scale: cowboy shot, close-up, upper body
    CAMERA      = "camera"       # Perspective & angle: Dutch angle, low angle, front view
    ENVIRONMENT = "environment"  # Setting: indoors, classroom, night sky, street
    LIGHTING    = "lighting"     # Illumination: direction, shadow quality, rim light, ambient light
    MATERIAL    = "material"     # Surface composition: leather, velvet, glass, metal, fabric
    RENDERING   = "rendering"    # Rendering traits: cel-shaded, linework treatment, soft shading
    STYLE       = "style"        # Artistic medium: watercolor, oil painting, line art, anime
    OBJECTS     = "objects"      # Props and held items: sword, book, umbrella, cup
    COLOR       = "color"        # Palette: monochrome, vibrant, desaturated, warm tones
    TEXTURE     = "texture"      # Micro-surface detail: smooth, rough, individual hair strands
    CONCEPT     = "concept"      # User-defined focal concept or thematic mechanism
    QUALITY     = "quality"      # Objective technical visual clarity (no subjective hype)
    UNCERTAINTY = "uncertainty"  # Ambiguous/inferred items excluded from final captions
```

### Fact Representation & Traceability
```python
@dataclass
class FactItem:
    id: str                      # Unique fact ID (e.g. "lighting_01")
    text: str                    # Cleaned tag or phrase
    category: SemanticCategory  # Semantic classification
    confidence: float = 1.0      # Evidence-based calibrated confidence
    source: str = "joycaption"   # "joycaption", "wd14", "user", "preset"
    is_stable: bool = False      # Part of invariant subject identity/concept
    is_uncertain: bool = False   # Below confidence threshold or ambiguous
    locked: bool = False         # User-locked in UI/CLI (inviolable)
    raw_text: str = ""           # Unprocessed text from generator

@dataclass
class CaptionToken:
    text: str
    category: SemanticCategory
    source_fact_ids: List[str]
    confidence: float
    transformation: str          # "direct", "compacted", "trigger_injected", "locked_override"
```

---

## 4. Separation of Semantic Mode from Output Format

HyperCaptioner strictly decouples **what** information is selected (**Semantic Mode**) from **how** that information is formatted (**Output Format**).

### 4.1 Semantic Modes (LoRA Objectives)
Defined in `hyper_captioner/caption_modes/`:
1. **Character Mode (`character.py`)**:
   - *Objective*: Train character/identity LoRAs.
   - *Retains*: Character identity, distinctive physical traits, expression, pose, framing, camera angle, environment context, lighting context, and distinguishing clothing.
   - *Suppresses*: Esoteric 3D/rendering jargon ("octane render", "subsurface scattering"), subjective hype words, and identity traits rendered redundant by the trigger word.
2. **Style Mode (`style.py`)**:
   - *Objective*: Train visual rendering and aesthetic LoRAs.
   - *Retains*: Rendering methodology, shading behavior, surface material response, lighting physics, line quality, edge treatment, depth separation, texture fidelity, color treatment.
   - *Suppresses*: Subject content (character identity, hair/eye colors, clothing items, narrative props) unless they are an intrinsic property of the artistic style.
3. **Outfit Mode (`outfit.py`)**:
   - *Objective*: Train clothing, costumes, and fashion concepts.
   - *Retains*: Garment structure, silhouette, cut, fabric material, texture, patterns, trims, collars, sleeves, cuffs, fasteners, layering, and accessories.
   - *Suppresses*: Character identity, unrelated background clutter, and pose details (unless explaining garment drape).
4. **Pose Mode (`pose.py`)**:
   - *Objective*: Train body position, posture, and spatial composition.
   - *Retains*: Body orientation, limb positioning, hand placement, torso angle, head tilt, gaze direction, stance, framing, and camera perspective.
   - *Suppresses*: Character facial features, detailed clothing descriptions, and unrelated environment details.
5. **Concept Mode (`concept.py`)**:
   - *Objective*: Train a specific theme, mechanism, or visual idea.
   - *Retains*: Visual attributes defining the configured concept, interacting objects, and isolating context.
   - *Suppresses*: Irrelevant character or background noise.

### 4.2 Output Formats (`CaptionFormat`)
Configurable independently of the semantic mode:
- **Comma-Separated Tags (`TAGS`)**: Danbooru-style tokens (`standing, cowboy shot, soft lighting`).
- **Structured Phrases (`STRUCTURED`)**: Compact, information-dense phrases grouped by logical progression.
- **Natural Language Prose (`NATURAL`)**: Visually grounded declarative sentences without flowery fluff.

---

## 5. Two-Stage Extraction & Transformation Pipeline

### Stage 1: Visual Fact Analysis
1. **Model Execution**: JoyCaption VLM is prompted with the JSON Extraction Schema requesting facts across all 18 semantic categories.
2. **Approach C Hybrid Dual-Parser**:
   - *Primary*: Parses JSON output using robust deserialization with inline syntax repair (handling trailing commas, missing braces, markdown fences).
   - *Fallback*: If JSON is malformed, parses labeled category blocks (`[CLOTHING]: ...`, `- clothing: ...`).
3. **WD14 Auxiliary Enrichment**: WD14 ONNX predictions provide high-confidence visual tags and category cross-verification.
4. **Fact Filtering & Grounding**: Items with confidence below threshold $\tau$ (default 0.35) or flagged as speculative are marked `is_uncertain=True` and routed to `UNCERTAINTY`.

### Stage 2: Mode-Specific Conditioning Transformation
1. **Contract Evaluation**: Active mode checks `include_categories` and `exclude_categories`.
2. **Meaning-Based Content vs. Style Filtering**: Semantic discriminator inspects tokens for content/style role.
3. **Redundancy Pruning**: Redundant stable traits are pruned only if absorbed by trigger; non-redundant traits are kept.
4. **User Lock Immunity**: Any fact with `locked=True` is preserved unconditionally.
5. **Token Assembly & Ordering**:
   - Ordering progression: `Trigger ➔ Identity ➔ Distinctive Traits ➔ Clothing/Garments ➔ Pose ➔ Framing/Camera ➔ Environment ➔ Lighting ➔ Rendering/Style`.
   - Each token retains its `CaptionToken` lineage (`source_fact_ids`, `transformation`).

---

## 6. Generic Case-Insensitive Trigger System

```python
class TriggerPlacement(str, Enum):
    PREPEND = "prepend"      # [TRIGGER], [caption]
    APPEND  = "append"       # [caption], [TRIGGER]
    WRAP    = "wrap"         # [TRIGGER], [caption], [TRIGGER]
    OMIT    = "omit"         # [caption] (no trigger injected)

@dataclass
class TriggerConfig:
    word: str = ""
    placement: TriggerPlacement = TriggerPlacement.PREPEND
    case_sensitive: bool = False         # Mandatory case-insensitive default
    deduplicate: bool = True             # Prevent trigger duplication
    absorb_stable_traits: bool = False   # Prune redundant traits defined by trigger
```

- **Case-Insensitive Matching**: Recognizes `Loggreal`, `loggreal`, `LOGGREAL` as identical, preventing stuttering.
- **Deduplication**: If the trigger word already exists within the assembled tokens, it is normalized to the specified placement without duplication.

---

## 7. Semantic Quality Control & Validation Engine

The validator operates as a semantic consistency checker against `StructuredVisualFacts`, the active `CaptionMode` contract, `TriggerConfig`, and the candidate caption. It does **not** rely on a giant static word blacklist.

### 7.1 Validation Status
```python
class ValidationStatus(str, Enum):
    VALID    = "valid"     # Passed validation without modification
    REPAIRED = "repaired"  # Minor deterministic issues safely corrected
    REJECTED = "rejected"  # Serious semantic/category violations; rebuilt from facts
```

### 7.2 Contradiction Handling
- **HARD Contradictions (ERROR)**: Mutually exclusive physical states for single subjects:
  - `standing` + `sitting`, `lying` + `standing`, `eyes open` + `eyes closed`, `indoors` + `outdoors`, `day` + `night`.
- **POTENTIAL Contradictions (WARNING)**: Nuanced coexisting states:
  - `looking at viewer` + `closed eyes` (may be facing forward with eyes shut), `smiling` + `neutral expression`, `front view` + `side view`.
  - **Rule**: Warnings are logged but **never** automatically deleted.

### 7.3 Auto-Repair Safety Boundary
- **Allowed Safe Repairs**: Duplicate triggers, duplicate tokens, malformed punctuation/commas, empty sections, subjective hype words, incorrect trigger placement, conversational filler phrases.
- **Strict Boundary**: The validator **never** invents visual replacements for uncertain semantic facts.
- **Rejection & Rebuild**: If serious category leakage (e.g. character hair/clothing leaking into Style Mode) or hard contradictions occur, status is set to `REJECTED`, and the caption is cleanly rebuilt from `StructuredVisualFacts` using the mode contract.

---

## 8. Persistence, Audit Metadata, and Sidecars

To prevent training corruption, training `.txt` sidecars and rich audit metadata are strictly separated:

1. **Training Sidecar (`<image_name>.txt`)**:
   - Contains ONLY the clean, final training caption text.
2. **Audit Metadata (`<image_name>.audit.json`)**:
   - Contains complete provenance:
     - `raw_model_response`: Unprocessed JoyCaption output.
     - `stage1_facts`: Full categorized `StructuredVisualFacts`.
     - `stage2_filtered`: Accepted vs. rejected facts with explicit reasons.
     - `validation_report`: Status (`VALID`, `REPAIRED`, `REJECTED`), issues, and repairs.
     - `traceability`: Array of `CaptionToken` objects mapping final tokens to source fact IDs.
     - `execution_time_ms`: Breakdown of Stage 1, Stage 2, and validation latency.
   - Generation of audit sidecars is controlled by `--audit` or UI setting.

---

## 9. CLI, REST API, and Web UI Integration

### 9.1 Headless CLI (`main.py`)
- `--mode`: `character` (default), `style`, `outfit`, `pose`, `concept` (with backward compatibility aliases `hybrid`, `tag`, `natural`).
- `--format`: `tags`, `structured`, `natural`.
- `--trigger`: Trigger word string.
- `--trigger-placement`: `prepend`, `append`, `wrap`, `omit`.
- `--audit` / `--debug-pipeline`: Enables full audit trail logging and `.audit.json` persistence.
- `--preview <N>`: Generates preview captions with audit reports for N images.
- Existing flags (`--dataset`, `--backup`, `--overwrite`, `--verify`, `--vram-mode`, `--wd14-device`) preserved.

### 9.2 REST API (`hyper_captioner/api/app.py`)
- Updated Pydantic request models accepting `caption_mode` (the 5 contracts), `caption_format`, `trigger_config`.
- `/api/review/item`: Returns caption, locked tags, validation status badge, and token traceability chips.
- `/api/batch/status`: Streams batch progress including validation statistics (`valid_count`, `repaired_count`, `rejected_count`).

### 9.3 CyberDeck Web UI
- Mode selector: Cleanly lists the 5 core contracts (Character, Style, Outfit, Pose, Concept).
- Format selector: Tags, Structured, Natural.
- Trigger Controls: Trigger Word input + Placement dropdown.
- Review Workstation: Shows validation status badge (`VALID ✓`, `REPAIRED ⚠️`, `REJECTED 🔄`) and clickable token chips with category/confidence tooltips.
- User Locks (`[🔒]`): Inviolable in UI and API.

---

## 10. Verification & Test Suite

The test suite is structured into three distinct layers:

1. **Internal Mode Pipeline Tests (`tests/test_modes.py`)**:
   - Runs representative images across all 5 modes.
   - Asserts that Style Mode excludes character appearance and clothing.
   - Asserts that Outfit Mode excludes background and character facial traits.
   - Asserts that Pose Mode isolates posture and framing.
2. **Deliberate Failure & Edge-Case Tests (`tests/test_validation_failures.py`)**:
   - **Hard Contradiction Test**: Injects `standing` + `sitting` $\to$ Verifies ERROR and resolution.
   - **Potential Contradiction Test**: Injects `smiling` + `neutral expression` $\to$ Verifies WARNING and preservation.
   - **Case-Insensitive Trigger Test**: Injects `LOGGREAL, loggreal, Loggreal` $\to$ Verifies clean single insertion.
   - **Subjective Hype Test**: Injects `"masterpiece, breathtaking girl"` $\to$ Verifies semantic removal and `REPAIRED` status.
   - **Uncertainty Rejection Test**: Injects facts with low confidence / uncertain flag $\to$ Verifies zero appearance in final caption.
   - **Inviolable Lock Test**: Injects a locked tag violating mode rules $\to$ Verifies `locked=True` tag is strictly preserved.
3. **Regression Tests (`tests/test_regression.py`)**:
   - Validates that existing dataset scanning, vocabulary indexing, and sidecar export continue to pass without regression.

---

## 11. Implementation File Map

- `hyper_captioner/caption_modes/`:
  - `__init__.py`: Package export & registry discovery.
  - `base.py`: `BaseCaptionMode`, `ModeConfig`, `CaptionFormat`.
  - `character.py`: Character mode contract.
  - `style.py`: Style mode contract.
  - `outfit.py`: Outfit mode contract.
  - `pose.py`: Pose mode contract.
  - `concept.py`: Concept mode contract.
  - `registry.py`: Dynamic mode registry.
- `hyper_captioner/pipeline/`:
  - `stage1_extractor.py`: Hybrid dual-parser & structured fact extraction.
  - `semantic_filter.py`: Meaning-based content vs. style discriminator & category filtering.
  - `validator.py`: Quality control, contradiction checker, and safe auto-repair.
  - `builder.py`: Refactored to coordinate Stage 2 token assembly, trigger injection, and formatters.
- `hyper_captioner/core/types.py`:
  - Enums (`SemanticCategory`, `CaptionModeType`, `CaptionFormat`, `TriggerPlacement`, `ValidationStatus`).
  - Dataclasses (`FactItem`, `CaptionToken`, `StructuredVisualFacts`, `TriggerConfig`, `ValidationReport`).
- `hyper_captioner/dataset/manager.py`:
  - Integrated two-stage pipeline, audit persistence, and batch progress tracking.
- `main.py` & `hyper_captioner/api/app.py`:
  - Integrated `--mode`, `--format`, `--trigger-placement`, `--audit` flags and API schemas.
- `tests/`:
  - `test_modes.py`, `test_validation_failures.py`, `test_regression.py`.
