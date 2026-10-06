# Hyper Captioner Rebuild — Architecture & Implementation Specification

**Date:** 2026-09-26  
**Status:** Approved  
**Target:** Local AI Dataset Captioning Tool for Image-Generation LoRA Training

---

## 1. Core Objective & Philosophy

Hyper Captioner is **not** a general-purpose vision chatbot or prose captioner.  
It is a **local dataset preparation engine** tailored to generate visually grounded, consistent, and diffusion-friendly captions for training image-generation LoRAs (Character, Style, Concept).

### Core Pipeline Flow:
```
IMAGE
  ↓
VISUAL ANALYSIS (WD14 Tagger + JoyCaption Vision-Language Model)
  ↓
STRUCTURED ATTRIBUTES WITH PROVENANCE
  ↓
ATTRIBUTE FUSION (Category-specific priority & contradiction prevention)
  ↓
CONTROLLED VOCABULARY NORMALIZATION
  ↓
CAPTION BUILDER (Tag Mode / Natural Mode / Hybrid Training Mode)
  ↓
HUMAN REVIEW & INTERACTIVE CORRECTION
  ↓
CLEAN .TXT SIDECAR FILE (Dataset-wide)
```

---

## 2. Architectural Pillars

### 2.1 Local-First Operation
- Default setting: `local_only` (`local_files_only=True`).
- No external network calls are made during normal inference.
- If a model file is missing from cache, the system raises an actionable configuration error with the exact Hugging Face path or prompts the user to switch to `download_missing`.
- Configurable model cache directories.

### 2.2 Device & VRAM Management
- **WD14 Device**: `Auto` (checks CUDA availability and free VRAM; falls back to CPU cleanly), `CUDA`, `CPU`.
- **VRAM Execution Modes**:
  - `Maximum Speed`: Keep both JoyCaption (4-bit NF4) and WD14 in GPU memory.
  - `Balanced` (Default): JoyCaption on GPU, WD14 on CPU. Florence-2 loaded on-demand only for verification, then offloaded.
  - `Minimum VRAM`: Sequential stage execution with explicit garbage collection and `torch.cuda.empty_cache()` between steps.

### 2.3 Staged Pipeline & Hallucination Prevention
- Primary pass runs **WD14** and **JoyCaption** only.
- **Florence-2** is **never** executed by default on every image. It is reserved for:
  - Disputed/contradictory attributes,
  - Low-confidence detections on critical categories,
  - User-initiated visual grounding checks (`<OD>`, `<DENSE_REGION_CAPTION>`).
- If an attribute is uncertain: **OMIT IT**. Visual accuracy precedes caption completeness.

### 2.4 Caption Provenance & Human Override
Every attribute internally tracks:
```python
@dataclass
class TagItem:
    text: str              # e.g., "maid uniform"
    source: str            # "user", "joycaption", "wd14", "florence2", "normalizer"
    confidence: float      # 0.0 - 1.0
    category: str          # "character", "appearance", "clothing", "pose", "action", etc.
    locked: bool = False   # Cannot be removed or altered during regeneration
```
- **Human Override Priority**: User-specified tags and attributes have the highest priority. If the user specifies `Character = Taihou`, models cannot override it.
- **Locked Tags**: Users can toggle `[LOCK]` on individual tags in the UI/CLI.
- **Output Cleanliness**: Exported `.txt` files contain only the final normalized caption. Provenance is saved in metadata / dataset state for the UI.

### 2.5 Attribute Fusion Priority Rules
When combining outputs from models:
1. **Character Identity**: User > WD14 > JoyCaption. (If uncertain, output generic count tag e.g. `1girl` without hallucinating names).
2. **Physical Appearance**: User > WD14 / JoyCaption.
3. **Clothing**: User > WD14 / JoyCaption.
4. **Action**: User > JoyCaption > WD14.
5. **Pose**: User > JoyCaption / WD14.
6. **Environment**: User > JoyCaption.
7. **Style**: User / Preset Configuration > Model inference.

**Contradiction Prevention**: Never output conflicting attributes (e.g. `skirt, trousers` or `short hair, long hair`) unless both are visually supported or separated into distinct subjects (`2girls, blonde girl with short hair, black-haired girl with long hair`).

---

## 3. Caption Modes & LoRA Targeting

### 3.1 Caption Modes
1. **Tag Mode**: Compact Danbooru tags.
   - Example: `Taihou, 1girl, black long hair, red eyes, maid headdress, maid uniform, form-fitting dress, thigh-highs, long gloves, standing, wiping a window, side view, indoors`
2. **Natural Mode**: Visually grounded readable description.
   - Example: `Taihou is standing indoors and wiping a large window while wearing a maid uniform, viewed from the side.`
3. **Hybrid Training Mode (Default)**: Structured comma-separated concepts grouping identity, key tags, and visual context.
   - Example: `Taihou, 1girl, black long hair, red eyes, maid headdress, maid uniform, form-fitting dress, thigh-highs, long gloves, standing, wiping a window, side view, looking toward window, indoors, large window, natural daylight`

### 3.2 LoRA Modes
- **Character LoRA**: Prioritizes variable attributes (clothing, pose, expression, camera, action) while keeping/pruning base character traits per user preference.
- **Style LoRA**: Prioritizes medium, linework, shading, lighting, and composition; strips specific character/clothing details to prevent style bleed.
- **Concept LoRA**: Isolates target concept, object, and interactions.

---

## 4. Controlled Vocabulary & Dataset Consistency

- Dataset-wide vocabulary indexing: scans all captions to compute tag frequency.
- Interactive and batch normalization:
  ```
  thigh high socks     (42)
  thigh-high stockings (17)
  thighhighs           (31)
  thigh-highs          (8)
  → [Normalize All → thigh-highs]
  ```
- Built-in canonical rules table and custom user replacement rules.

---

## 5. Dataset Presets & Preview System

### Reusable Presets
Presets store:
- Caption Mode (Hybrid, Tag, Natural)
- LoRA Strategy (Character, Style, Concept, Custom)
- Trigger Word & Reference Description
- WD14 Thresholds (General & Character)
- Vocabulary Profile & Tag Formatting (spaces vs underscores)
- Blacklist
- Model Selection & Device / VRAM Mode

### Caption Preview
- User can sample $N$ images (e.g., 5 images) $\to$ run preview $\to$ inspect in UI / terminal $\to$ tweak settings $\to$ accept $\to$ execute full batch.

---

## 6. Architecture & Package Structure

```
hyper_captioner/
├── config.py              # Configuration, model paths, defaults, local-only settings
├── core/
│   ├── types.py           # Dataclasses: TagItem, Provenance, CaptionResult, ImageRecord, VRAMMode
│   ├── vram.py            # Device and memory management (Max Speed, Balanced, Min VRAM)
│   └── exceptions.py      # Custom exceptions
├── engines/
│   ├── base.py            # Abstract Base Engine: BaseTagger, BaseCaptioner, BaseGrounding
│   ├── wd14.py            # SmilingWolf WD14 ONNX backend (Auto/CUDA/CPU, category filtering, scoring)
│   ├── joycaption.py      # JoyCaption LLaVA backend (4-bit NF4 / FP16, SDPA, prompt handling)
│   └── florence2.py       # Florence-2 grounding & verification backend (staged/on-demand)
├── vocabulary/
│   ├── normalizer.py      # Controlled vocabulary mappings, rules, and canonicalization
│   ├── database.py        # Tag frequency counter and dataset-wide vocabulary indexer
│   └── default_rules.json # Built-in synonym mappings & Danbooru blacklist/cleanup rules
├── pipeline/
│   ├── builder.py         # Multi-mode Caption Builder (Tag, Natural, Hybrid) & LoRA strategies
│   ├── fusion.py          # Attribute fusion engine (merges WD14 + JoyCaption with provenance)
│   └── verifier.py        # Florence-2 conflict resolution & visual grounding verifier
├── dataset/
│   ├── scanner.py         # Fast recursive scanner, format validator, orphan/missing detection
│   ├── manager.py         # Dataset state manager, batch runner, checkpoint/resume handler
│   └── exporter.py        # Safe sidecar .txt writer, backup generator, CSV/JSON metadata export
├── api/
│   ├── app.py             # FastAPI backend with REST endpoints & WebSocket progress events
│   └── routes/            # Modular route handlers (dataset, models, caption, vocabulary, review)
└── ui/                    # Dedicated Single-Page Application (HTML5 / Modern CSS / Vanilla JS)
```

---

## 7. Phase 1 Implementation & Validation Protocol

Before constructing secondary components, Phase 1 strictly validates end-to-end inference and fusion on real images:
- **Test 1**: Load WD14 model offline.
- **Test 2**: Run WD14 on one real image $\to$ verify tag extraction, category separation, and confidence scores.
- **Test 3**: Load JoyCaption model offline (4-bit NF4).
- **Test 4**: Run JoyCaption on the same image $\to$ verify prompt execution and generated output.
- **Test 5**: Fuse the outputs using explicit category-based source-priority rules.
- **Test 6**: Generate the final Hybrid Training caption.
- **Test 7**: Write the `.txt` sidecar and verify clean format and metadata.
- **Test 8**: Run the complete pipeline on 5 real images.
