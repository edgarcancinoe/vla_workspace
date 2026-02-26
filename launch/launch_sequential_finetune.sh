#!/bin/bash

# Path to the launch script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/launch_finetune_xvla.sh"

# ============================================================================
# SEQUENTIAL RUN CONFIGURATION
# ============================================================================
# Define each run as a (dataset, action_mode) pair using two parallel arrays.
# Action mode options:
#   so101_ee6d   - Train on EEF data only (xyz + rot6d + gripper, dims 0-9)
#   so101_joint  - Train on joint data only (6 motors, dims 10-15)
#   auto         - Auto-detect from dataset action shape (plain MSE, no slicing)
#   ee6d         - Original 20D EEF space (for original xVLA datasets)
# ============================================================================

DATASETS=(
    # "soarm101_pickplace_6d"                   # example
    # "soarm101_pickplace_orange_050e_fw_open"   # example
    # "soarm101_pickplace_10d"
    "soarm101_pickplace_10d"
)

ACTION_MODES=(
    # "auto"                                    # matching example above
    # "auto"                                    # matching example above
    # "so101_ee6d"    # EEF training on soarm101_pickplace_10d
    "so101_joint"   # Joint training on soarm101_pickplace_10d
)

# Sanity check: arrays must be the same length
if [ "${#DATASETS[@]}" -ne "${#ACTION_MODES[@]}" ]; then
    echo "ERROR: DATASETS and ACTION_MODES arrays must have the same number of entries."
    echo "  DATASETS:     ${#DATASETS[@]} entries"
    echo "  ACTION_MODES: ${#ACTION_MODES[@]} entries"
    exit 1
fi

echo "Starting sequential finetuning for ${#DATASETS[@]} runs..."

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    action_mode="${ACTION_MODES[$i]}"
    run_num=$((i + 1))

    echo "============================================================================"
    echo "Run ${run_num}/${#DATASETS[@]}"
    echo "  Dataset:     ${dataset}"
    echo "  Action Mode: ${action_mode}"
    echo "============================================================================"

    # Export both variables so the launch script picks them up
    export DATASET_NAME_STR="${dataset}"
    export ACTION_MODE="${action_mode}"

    # Run the launch script
    bash "${LAUNCH_SCRIPT}"

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Run ${run_num} failed for dataset '${dataset}' / action_mode '${action_mode}' (exit code ${EXIT_CODE})."
        echo "Aborting remaining runs."
        exit $EXIT_CODE
    fi

    echo "Run ${run_num} (${dataset} / ${action_mode}) completed successfully."
    echo ""
done

echo "All ${#DATASETS[@]} finetuning runs completed successfully."
