"""
Hyper Captioner 2.0 - Application Launcher
"""

import sys
from pathlib import Path
import uvicorn

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("HYPER CAPTIONER 2.0 - DATASET PRODUCTION WORKSTATION")
    print("Open your browser at: http://127.0.0.1:7860")
    print("=" * 65 + "\n")
    uvicorn.run("hyper_captioner.api.app:app", host="127.0.0.1", port=7860, reload=False)
