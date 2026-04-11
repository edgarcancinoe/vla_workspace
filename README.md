# Thesis VLA Workspace

This repository is organized for day-to-day robotics and VLA work.

## Layout

- `src/thesis_vla/`: reusable Python modules
- `apps/train/`: training launch entrypoints
- `apps/robot/`: robot operation scripts
- `apps/dataset/`: dataset recording and repair tools
- `apps/eval/`: evaluation, inspection, and debugging tools
- `config/robot/`: robot and calibration config
- `config/launch/`: client/server launch config
- `docs/`: setup notes and architecture docs
- `tests/`: smoke, unit, and integration tests
- `runtime/`: generated logs, caches, outputs, captures

## External dependencies

This workspace expects upstream repos to live alongside it in the thesis root, especially:

- `../repos/lerobot`
- `../repos/X-VLA`
- optional robot hardware repos under `../repos/`

The repo bootstraps `../repos/lerobot/src` automatically through `sitecustomize.py` when you run commands from the workspace root.
