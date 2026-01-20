#!/bin/bash
# Script to install lerobot into the 'vla' environment safely

# 1. Setup temporary space on NFS (1.7TB free) to avoid "No space left on device"
export TMPDIR=/home/jose/pip_tmp
export PIP_CACHE_DIR=/home/jose/pip_cache
mkdir -p $TMPDIR $PIP_CACHE_DIR

echo "using TMPDIR: $TMPDIR"

# 2. Use the specific python from the 'vla' environment to avoid "fighting" with system python
VLA_PYTHON=/home/jose/conda/envs/vla/bin/python

# 3. Upgrade torch and torchvision to satisfy lerobot (>=0.21.0) 
# We use the cu124 index to match your existing CUDA 12.4 installation and avoid redundant generic CUDA downloads
echo "Step 1: Upgrading Torch and Torchvision to versions compatible with lerobot..."
$VLA_PYTHON -m pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# 4. Install lerobot with extras
echo "Step 2: Installing lerobot[smolvla,feetech,async]..."
$VLA_PYTHON -m pip install 'lerobot[smolvla,feetech,async]'

echo "Done!"
