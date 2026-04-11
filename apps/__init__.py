from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
LEROBOT_SRC = PROJECT_ROOT.parent / "repos" / "lerobot" / "src"

for candidate in (SRC_ROOT, LEROBOT_SRC):
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
