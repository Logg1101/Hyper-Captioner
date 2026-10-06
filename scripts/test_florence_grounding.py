"""
Test script for Florence-2 visual grounding engine.
"""

import os
import sys
from pathlib import Path
from PIL import Image

# Enforce offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hyper_captioner.pipeline.verifier import VisualGroundingVerifier

def test_grounding():
    print("Testing Florence-2 On-Demand Visual Grounding...")
    test_img_path = ROOT / "test_data" / "phase1_validation" / "image004.jpg"  # cheetah
    img = Image.open(test_img_path).convert("RGB")

    verifier = VisualGroundingVerifier()

    # Test phrase grounding
    print("Running verify_grounding for 'cheetah'...")
    res = verifier.verify_grounding(img, "cheetah")
    print(f"[OK] Phrase 'cheetah' grounded: {res.get('grounded')}, count: {res.get('count')}")
    assert res.get("grounded") is True

    # Test negative phrase grounding
    print("Running verify_grounding for 'airplane'...")
    res_neg = verifier.verify_grounding(img, "airplane")
    print(f"[OK] Phrase 'airplane' grounded: {res_neg.get('grounded')}, count: {res_neg.get('count')}")
    assert res_neg.get("grounded") is False

    verifier.unload()
    print("Florence-2 unloaded cleanly.")
    print("\nFLORENCE-2 VISUAL GROUNDING TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_grounding()
