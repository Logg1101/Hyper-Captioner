"""
Test script for Hyper Captioner FastAPI endpoints.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from hyper_captioner.api.app import app

def test_api():
    print("Testing Hyper Captioner API endpoints...")
    client = TestClient(app)

    # 1. Health
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    print(f"[OK] /api/health: status={data['status']}, cuda={data['cuda_available']}")

    # 2. Presets
    res = client.get("/api/presets")
    assert res.status_code == 200
    presets = res.json()["presets"]
    assert "Character LoRA" in presets
    print(f"[OK] /api/presets: found {len(presets)} presets: {presets}")

    # 3. UI index
    res = client.get("/")
    assert res.status_code == 200
    assert "HYPER" in res.text
    print("[OK] /: Root UI index page served successfully.")

    # 4. Scan
    test_dir = ROOT / "test_data" / "phase1_validation"
    res = client.post("/api/scan", json={"dataset_path": str(test_dir), "verify": False})
    assert res.status_code == 200
    scan_data = res.json()
    print(f"[OK] /api/scan: total_images={scan_data['total_images']}, with_captions={scan_data['with_captions_count']}")

    # 5. Gallery
    res = client.get(f"/api/gallery?path={test_dir}")
    assert res.status_code == 200
    gallery = res.json()["images"]
    print(f"[OK] /api/gallery: {len(gallery)} images found")

    # 6. Review Item
    first_img = gallery[0]["path"]
    res = client.get(f"/api/review/item?image_path={first_img}")
    assert res.status_code == 200
    item_data = res.json()
    print(f"[OK] /api/review/item: filename={item_data['filename']}, caption_len={len(item_data['caption'])}")

    # 7. Vocabulary Index
    res = client.get(f"/api/vocab/index?dataset_path={test_dir}&top=5")
    assert res.status_code == 200
    vocab_data = res.json()
    print(f"[OK] /api/vocab/index: top tags = {vocab_data['results']}")

    print("\nALL API ENDPOINT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_api()
