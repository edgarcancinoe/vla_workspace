#!/bin/bash

# Path to the launch script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/launch_finetune_xvla.sh"

# List of datasets to finetune sequentially
# Add or remove datasets from this list as needed
DATASETS=(
    # "soarm101_pickplace_6d"
    # "soarm101_pickplace_orange_050e_fw_open"
    "soarm101_pickplace_6d_240e_fw_closed"
    "soarm101_pickplace_orange_240e_fw_closed"
    # "another_dataset_here"
)

echo "Starting sequential finetuning for ${#DATASETS[@]} datasets..."

for dataset in "${DATASETS[@]}"; do
    echo "============================================================================"
    echo "Starting finetuning for dataset: ${dataset}"
    echo "============================================================================"
    
    # Export the dataset name string so the launch script uses it
    export DATASET_NAME_STR="${dataset}"
    
    # Run the launch script
    bash "${LAUNCH_SCRIPT}"
    
    # Capture exit code
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Error: Finetuning failed for ${dataset} with exit code ${EXIT_CODE}."
        echo "Aborting remaining runs."
        exit $EXIT_CODE
    fi
    
    echo "Finetuning for ${dataset} completed successfully."
    echo ""
done

echo "All finetuning runs completed successfully."
