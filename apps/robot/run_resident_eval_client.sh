#!/bin/bash

if [ -f "$HOME/conda/etc/profile.d/conda.sh" ]; then
  source "$HOME/conda/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx vla; then
    conda activate vla
  elif conda env list | awk '{print $1}' | grep -qx thesis; then
    conda activate thesis
  fi
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
  source /opt/anaconda3/etc/profile.d/conda.sh
  if conda env list | awk '{print $1}' | grep -qx vla; then
    conda activate vla
  elif conda env list | awk '{print $1}' | grep -qx thesis; then
    conda activate thesis
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../repos/lerobot/src:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

python -m thesis_vla.inference.resident_eval --config_path "${PROJECT_ROOT}/config/launch/resident_eval_client.yaml" "$@"
