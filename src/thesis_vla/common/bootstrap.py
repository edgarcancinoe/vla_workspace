from __future__ import annotations

import sys
from pathlib import Path

from thesis_vla.common.paths import PROJECT_ROOT, SRC_ROOT


def bootstrap_project() -> None:
    src_path = str(SRC_ROOT)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    lerobot_path = PROJECT_ROOT.parent / "repos" / "lerobot" / "src"
    if lerobot_path.exists():
        lerobot_str = str(lerobot_path)
        if lerobot_str not in sys.path:
            sys.path.insert(0, lerobot_str)


def bootstrap_from_file(file_path: str | Path) -> None:
    file_path = Path(file_path).resolve()
    project_root = next((parent for parent in file_path.parents if (parent / "src").exists()), None)
    if project_root is None:
        return

    src_root = project_root / "src"
    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    lerobot_path = project_root.parent / "repos" / "lerobot" / "src"
    if lerobot_path.exists():
        lerobot_str = str(lerobot_path)
        if lerobot_str not in sys.path:
            sys.path.insert(0, lerobot_str)
