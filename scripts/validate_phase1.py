"""
Phase 1 Validation Script for Hyper Captioner Rebuild.
Executes Tests 1 through 8 sequentially to validate backend inference,
attribute fusion, caption building, and sidecar generation on real images.
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Enforce offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Ensure hyper_captioner is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hyper_captioner.core.types import (
    CaptionMode,
    CharacterConfig,
    LoRAStrategy,
    ModelSource,
    WD14Device,
)
from hyper_captioner.engines.wd14 import WD14Engine
from hyper_captioner.engines.joycaption import JoyCaptionEngine, build_system_prompt
from hyper_captioner.pipeline.fusion import AttributeFusionEngine
from hyper_captioner.pipeline.builder import CaptionBuilder
from hyper_captioner.vocabulary.normalizer import VocabularyNormalizer


def run_phase1_validation():
    print("=" * 70)
    print("HYPER CAPTIONER - PHASE 1 VALIDATION PROTOCOL")
    print("=" * 70)

    test_dir = ROOT / "test_data" / "phase1_validation"
    img1_path = test_dir / "image001.png"
    assert img1_path.exists(), f"Missing test image: {img1_path}"
    img1 = Image.open(img1_path).convert("RGB")

    # ---------------------------------------------------------
    # TEST 1: Load WD14 (Offline Local-First)
    # ---------------------------------------------------------
    print("\n[TEST 1] Loading WD14 Tagger (Local-First, Auto Device)...")
    t0 = time.time()
    wd14 = WD14Engine(device_choice=WD14Device.AUTO, model_source=ModelSource.LOCAL_ONLY)
    wd14.load()
    print(f"[OK] TEST 1 PASSED: WD14 loaded in {time.time() - t0:.2f}s with provider: {wd14.active_provider}")

    # ---------------------------------------------------------
    # TEST 2: Run WD14 on one real image
    # ---------------------------------------------------------
    print(f"\n[TEST 2] Running WD14 inference on {img1_path.name}...")
    t0 = time.time()
    wd14_tags = wd14.predict(img1, general_threshold=0.30, character_threshold=0.50)
    print(f"[OK] TEST 2 PASSED: Predicted {len(wd14_tags)} tags in {time.time() - t0:.3f}s")
    print("  Top tags preview:")
    for tag, score, cat in wd14_tags[:8]:
        cat_str = "Character" if cat == 4 else "General"
        print(f"    - {tag} ({cat_str}, conf: {score:.3f})")

    # ---------------------------------------------------------
    # TEST 3: Load JoyCaption (Offline Local-First, 4-bit NF4)
    # ---------------------------------------------------------
    print("\n[TEST 3] Loading JoyCaption (4-bit NF4, SDPA, Local-First)...")
    t0 = time.time()
    joycaption = JoyCaptionEngine(model_source=ModelSource.LOCAL_ONLY, load_in_4bit=True)
    joycaption.load()
    print(f"[OK] TEST 3 PASSED: JoyCaption loaded in {time.time() - t0:.2f}s on {joycaption.device}")

    # ---------------------------------------------------------
    # TEST 4: Run JoyCaption on the same image
    # ---------------------------------------------------------
    print(f"\n[TEST 4] Running JoyCaption inference on {img1_path.name}...")
    t0 = time.time()
    prompt = build_system_prompt(caption_mode=CaptionMode.HYBRID, lora_strategy=LoRAStrategy.CHARACTER)
    joy_output = joycaption.generate([img1], prompt=prompt, max_new_tokens=300, temperature=0.0)[0]
    print(f"[OK] TEST 4 PASSED: JoyCaption generated output in {time.time() - t0:.2f}s")
    print("  JoyCaption Raw Output:")
    print(f"    {joy_output}")

    # ---------------------------------------------------------
    # TEST 5: Fuse the results with category priority & provenance
    # ---------------------------------------------------------
    print("\n[TEST 5] Fusing attributes with explicit source-priority rules...")
    normalizer = VocabularyNormalizer()
    fusion = AttributeFusionEngine(normalizer=normalizer)
    char_config = CharacterConfig(name="Taihou", trigger_word="taihou")
    fused_tags = fusion.fuse(
        wd14_results=wd14_tags,
        joycaption_text=joy_output,
        character_config=char_config,
        keep_underscores=False,
    )
    print(f"[OK] TEST 5 PASSED: Fused into {len(fused_tags)} canonical attributes.")
    print("  Sample Fused Tags with Provenance:")
    for item in fused_tags[:10]:
        print(f"    - {item.text:<25} [source: {item.source:<10} | cat: {item.category:<12} | conf: {item.confidence:.2f}]")

    # ---------------------------------------------------------
    # TEST 6: Generate final Hybrid training caption
    # ---------------------------------------------------------
    print("\n[TEST 6] Building final Hybrid Training Caption...")
    builder = CaptionBuilder()
    caption_res = builder.build(
        fused_tags=fused_tags,
        caption_mode=CaptionMode.HYBRID,
        lora_strategy=LoRAStrategy.CHARACTER,
        character_config=char_config,
        raw_joycaption=joy_output,
    )
    print("[OK] TEST 6 PASSED: Generated final hybrid caption.")
    print("  Final Training Caption:")
    print(f"    \"{caption_res.caption}\"")

    # ---------------------------------------------------------
    # TEST 7: Write sidecar .txt file
    # ---------------------------------------------------------
    print("\n[TEST 7] Writing sidecar .txt file...")
    txt_path = img1_path.with_suffix(".txt")
    txt_path.write_text(caption_res.caption, encoding="utf-8")
    assert txt_path.exists() and txt_path.stat().st_size > 0
    read_back = txt_path.read_text(encoding="utf-8")
    assert read_back == caption_res.caption
    print(f"[OK] TEST 7 PASSED: Wrote clean sidecar to {txt_path.name} ({len(read_back)} chars)")

    # ---------------------------------------------------------
    # TEST 8: Run the complete pipeline on all 5 test images
    # ---------------------------------------------------------
    print("\n[TEST 8] Executing complete pipeline on 5 test images...")
    all_images = sorted(list(test_dir.glob("image*.*")))
    image_files = [p for p in all_images if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    print(f"Found {len(image_files)} test images.")
    assert len(image_files) >= 5, f"Expected 5 images, found {len(image_files)}"

    for idx, path in enumerate(image_files[:5], 1):
        print(f"\n  [{idx}/5] Processing {path.name}...")
        img = Image.open(path).convert("RGB")
        
        # 1. WD14
        tags = wd14.predict(img, general_threshold=0.35, character_threshold=0.60)
        
        # 2. JoyCaption
        prompt_i = build_system_prompt(caption_mode=CaptionMode.HYBRID, lora_strategy=LoRAStrategy.GENERAL)
        joy_out = joycaption.generate([img], prompt=prompt_i, max_new_tokens=250, temperature=0.0)[0]
        
        # 3. Fuse
        fused = fusion.fuse(wd14_results=tags, joycaption_text=joy_out, keep_underscores=False)
        
        # 4. Build
        res = builder.build(fused_tags=fused, caption_mode=CaptionMode.HYBRID, lora_strategy=LoRAStrategy.GENERAL)
        
        # 5. Sidecar
        out_txt = path.with_suffix(".txt")
        out_txt.write_text(res.caption, encoding="utf-8")
        print(f"    [OK] Caption: \"{res.caption[:90]}...\"")
        print(f"    [OK] Sidecar saved: {out_txt.name} ({out_txt.stat().st_size} bytes)")

    print("\n" + "=" * 70)
    print("ALL 8 PHASE 1 VALIDATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_phase1_validation()
