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

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../repos/lerobot/src:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

python -m thesis_vla.inference.resident_policy_server --config_path "${PROJECT_ROOT}/config/launch/resident_policy_server.yaml" "$@"
