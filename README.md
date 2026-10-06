# 🚀 Hyper Captioner 2.0

**Hyper Captioner** is a local AI dataset-preparation workstation built specifically for training image-generation LoRAs (Character, Style, and Concept LoRAs).

It replaces monolithic black-box captioners with a modular, multi-component pipeline designed for **accuracy, consistency, visual grounding, repeatable controlled vocabulary, and low hallucination**.

---

## 🏗️ Architecture

```
IMAGE
  ↓
VISUAL ANALYSIS (SmilingWolf WD14 Tagger + JoyCaption LLaVA VLM)
  ↓
STRUCTURED ATTRIBUTES WITH PROVENANCE
  ↓
ATTRIBUTE FUSION (Category-priority rules & contradiction prevention)
  ↓
CONTROLLED VOCABULARY NORMALIZATION (Synonym consolidation & frequency indexing)
  ↓
CAPTION BUILDER (Hybrid Training Mode / Tag Mode / Natural Mode)
  ↓
HUMAN REVIEW & LOCKING (Editable UI with tag chips & lock toggles)
  ↓
CLEAN .TXT SIDECAR FILES
```

---

## ✨ Key Capabilities

1. **Local-First / Offline Default**:
   - Strictly defaults to offline local-cache mode. No unexpected network calls during inference.
2. **Staged Multi-Model Pipeline**:
   - Primary pass: WD14 ONNX + JoyCaption (4-bit NF4) $\to$ Attribute Fusion $\to$ Caption Builder.
   - On-demand verification: Florence-2 is loaded only for visual grounding checks and conflict resolution.
3. **Attribute Fusion Priority Rules**:
   - Character identity: User > WD14 > JoyCaption
   - Physical appearance: User > WD14 / JoyCaption
   - Clothing & accessories: User > WD14 / JoyCaption
   - Action: User > JoyCaption > WD14
   - Pose: User > JoyCaption / WD14
   - Environment & lighting: User > JoyCaption
   - Style: User / Project preset > Model inference
4. **Caption Provenance & Attribute Locking**:
   - Tracks the source (`user`, `wd14`, `joycaption`, `florence2`) and confidence for every tag.
   - Users can lock attributes (`[🔒]`) in the UI or CLI so automated regeneration never alters them.
5. **Controlled Vocabulary Manager**:
   - Scans the entire dataset and computes exact tag frequencies (e.g. `thigh high socks: 42`, `thighhighs: 31`).
   - One-click global normalization: `[Normalize All → thigh-highs]` updates all `.txt` sidecars atomically.
6. **Decoupled Semantic Mode Contracts**:
   - **Character Mode**: Captures visual identity, distinctive physical traits, expressions, pose, and environmental context while suppressing 3D rendering jargon.
   - **Style Mode**: Captures visual rendering methodology, shading behavior, surface material response, lighting physics, and line quality while strictly excluding character identity and clothing content.
   - **Outfit Mode**: Captures garment structure, silhouettes, cuts, fabrics, patterns, and accessories while suppressing unrelated environment clutter.
   - **Pose Mode**: Captures body stance, limb positioning, torso angles, and camera framing while suppressing facial traits and clothing clutter.
   - **Concept Mode**: Focuses on thematic mechanisms, interactive objects, and isolating context.
7. **Independent Output Formats**:
   - **Tags (`--format tags`)**: Danbooru-style comma-separated tokens.
   - **Structured (`--format structured`)**: Information-dense semicolon-separated clauses grouped by logical category hierarchy.
   - **Natural (`--format natural`)**: Visually grounded declarative prose sentences without flowery fluff.
8. **Conservative Semantic Validator**:
   - Hard contradictions (`standing` + `sitting`, `indoors` + `outdoors`) trigger `REJECTED` and rebuild from source facts.
   - Potential contradictions (`smiling` + `neutral expression`) log warnings but are never deleted.
   - Deterministic safe auto-repair for duplicate tokens, case-insensitive trigger deduplication, and subjective hype buzzwords (`masterpiece`, `stunning`).
   - User locks (`locked=True` / `[🔒]`) are inviolable across all modes and validation stages.
9. **Persistent Audit Metadata**:
   - Training `.txt` sidecars stay 100% clean and free of JSON or metadata.
   - Optional rich `.audit.json` sidecars track full token lineage, source facts, confidence scores, and validation reports via `--audit`.

---

## 💻 Command-Line Interface (CLI)

Hyper Captioner runs completely headless without requiring the browser UI:

### 1. Scan Dataset
```bash
python main.py scan --dataset "D:\AI\DATASETS\MyLoRA"
```

### 2. Generate Preview with Audit Trail (5 images)
```bash
python main.py caption --dataset "D:\AI\DATASETS\MyLoRA" --mode character --format tags --trigger "taihou" --trigger-placement prepend --preview 5 --audit
```

### 3. Run Style Mode Captioning
```bash
python main.py caption --dataset "D:\AI\DATASETS\MyLoRA" --mode style --format tags --trigger "AestheticV1" --trigger-placement append
```

### 4. Run Outfit Mode in Structured Format
```bash
python main.py caption --dataset "D:\AI\DATASETS\MyLoRA" --mode outfit --format structured --trigger "BattleMaid" --backup
```

### 5. Inspect Dataset Vocabulary
```bash
python main.py vocab --dataset "D:\AI\DATASETS\MyLoRA" --top 30
```

### 6. Global Vocabulary Replacement
```bash
python main.py vocab --dataset "D:\AI\DATASETS\MyLoRA" --replace "thigh high socks" "thigh-highs"
```

### 7. Export Master Metadata
```bash
python main.py export --dataset "D:\AI\DATASETS\MyLoRA" --format csv
```

---

## 🌐 Web UI (CyberDeck)

To launch the interactive Web UI:

```bash
run.bat
# Or:
python run.py
```
Open your browser at `http://127.0.0.1:7860`.

### UI Sections:
- **DATASET**: Directory browser, recursive scan, health overview, gallery grid with status badges.
- **MODELS**: Device selection (Auto/CUDA/CPU), VRAM execution modes (Balanced/Max Speed/Min VRAM), thresholds, telemetry.
- **CAPTIONING**: Mode contracts (Character, Style, Outfit, Pose, Concept), output formats (Tags, Structured, Natural), trigger placement, preview modal, batch runner with live execution logs.
- **VOCABULARY**: Dataset-wide tag frequency table, variant search, global replacement tool (`[Normalize All]`).
- **REVIEW**: Side-by-side workstation: zoomable image, editable caption (`Ctrl+S`), validation status badge (`VALID ✓`, `REPAIRED ⚠️`, `REJECTED 🔄`), interactive tag chips with provenance badges, and `[🔒 Lock]` toggles.
- **EXPORT**: Master CSV/JSON export, dataset backups.
- **SETTINGS**: Model paths, offline toggles, and blacklist editor.

---

## 🧪 Verification & Tests

Run the comprehensive test suite (154 tests covering all 5 modes, failure cases, and regressions):

```bash
pytest -v
```

---

## 📦 Requirements & Installation

1. Activate your Python environment:
```bash
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Models used:
- `fancyfeast/llama-joycaption-beta-one-hf-llava` (4-bit NF4 quantized, ~6GB VRAM)
- `SmilingWolf/wd-swinv2-tagger-v3` (ONNX, runs on CPU or GPU)
- `microsoft/Florence-2-large` (On-demand grounding and verification)

