#!/bin/bash

# ============================================================================
# Resume Training Script for smolvla_finetuned_orange
# ============================================================================

# 1. Point to the EXISTING output directory providing the checkpoint
export OUTPUT_DIR="/home/jose/vla_workspace/outputs/train/smolvla_finetuned_orange_20260123_214145"

# 2. Enable Resume mode
export RESUME="true"

# 3. Set the NEW total target steps (Previous was 20,000)
export STEPS="40000"

# 4. (Optional) Force the same Job Name if you want to group them in W&B, 
#    otherwise a new timestamped one will be created.
#    Here we let it be new to distinguish the resume run.

# 5. Hand over to the main launch script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/launch_finetune.sh"
