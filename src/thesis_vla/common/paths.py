from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
APPS_ROOT = PROJECT_ROOT / "apps"
CONFIG_ROOT = PROJECT_ROOT / "config"
DOCS_ROOT = PROJECT_ROOT / "docs"
TESTS_ROOT = PROJECT_ROOT / "tests"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"

ROBOT_CONFIG_PATH = CONFIG_ROOT / "robot" / "robot_config.yaml"
LAUNCH_CLIENT_CONFIG_PATH = CONFIG_ROOT / "launch" / "launch_client.yaml"
LAUNCH_SERVER_CONFIG_PATH = CONFIG_ROOT / "launch" / "launch_server.yaml"

RUNTIME_LOGS_DIR = RUNTIME_ROOT / "logs"
RUNTIME_OUTPUTS_DIR = RUNTIME_ROOT / "outputs"
RUNTIME_CACHE_DIR = RUNTIME_ROOT / "cache"
RUNTIME_CAPTURES_DIR = RUNTIME_ROOT / "captures"
RUNTIME_TMP_DIR = RUNTIME_ROOT / "tmp"

DATASETS_OUTPUT_DIR = RUNTIME_OUTPUTS_DIR / "datasets"
TRAIN_OUTPUT_DIR = RUNTIME_OUTPUTS_DIR / "train"
GRIPPER_INSPECTION_DIR = RUNTIME_OUTPUTS_DIR / "gripper_inspection"
CAPTURED_IMAGES_DIR = RUNTIME_CAPTURES_DIR / "images"


def ensure_runtime_dirs() -> None:
    for path in (
        RUNTIME_ROOT,
        RUNTIME_LOGS_DIR,
        RUNTIME_OUTPUTS_DIR,
        RUNTIME_CACHE_DIR,
        RUNTIME_CAPTURES_DIR,
        RUNTIME_TMP_DIR,
        DATASETS_OUTPUT_DIR,
        TRAIN_OUTPUT_DIR,
        GRIPPER_INSPECTION_DIR,
        CAPTURED_IMAGES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

