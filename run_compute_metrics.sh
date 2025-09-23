#!/bin/bash

# Script to run model metrics computation
# Usage: ./run_compute_metrics.sh [OPTIONS]

# Default parameters
DEVICE="cuda"
QUALITY_START=1
QUALITY_END=6
MODELS="mbt2018-mean cheng2020-anchor bmshj2018-factorized mlic++"
OUTPUT_FILE="/Odyssey/private/o23gauvr/code/FASCINATION/pickle/model_metrics_computed.pkl"
DM_MLIC_PATH="/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_dm_4_157_196_256.pkl"
DM_CAE_PATH="/Odyssey/private/o23gauvr/code/FASCINATION/pickle/dm_enatl_mean_std_along_depth_4_157_240_240.pkl"
CAE_CHECKPOINTS_DIR="/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --device DEVICE                  Device to use (cuda/cpu) [default: $DEVICE]"
    echo "  --quality-start START            Start quality level [default: $QUALITY_START]"
    echo "  --quality-end END                End quality level (exclusive) [default: $QUALITY_END]"
    echo "  --models \"MODEL1 MODEL2 ...\"     Models to evaluate [default: \"$MODELS\"]"
    echo "  --output-file PATH               Output pickle file path [default: $OUTPUT_FILE]"
    echo "  --dm-mlic-path PATH              Path to MLIC datamodule pickle [default: $DM_MLIC_PATH]"
    echo "  --dm-cae-path PATH               Path to CAE datamodule pickle [default: $DM_CAE_PATH]"
    echo "  --cae-checkpoints-dir PATH       Directory with CAE checkpoints [default: $CAE_CHECKPOINTS_DIR]"
    echo "  -h, --help                       Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --device cpu --quality-start 2 --quality-end 4 --models \"mbt2018-mean cheng2020-anchor\""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --quality-start)
            QUALITY_START="$2"
            shift 2
            ;;
        --quality-end)
            QUALITY_END="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --dm-mlic-path)
            DM_MLIC_PATH="$2"
            shift 2
            ;;
        --dm-cae-path)
            DM_CAE_PATH="$2"
            shift 2
            ;;
        --cae-checkpoints-dir)
            CAE_CHECKPOINTS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Display selected parameters
echo "Running model metrics computation with:"
echo "  Device: $DEVICE"
echo "  Quality range: $QUALITY_START to $QUALITY_END"
echo "  Models: $MODELS"
echo "  Output file: $OUTPUT_FILE"
echo "  MLIC datamodule: $DM_MLIC_PATH"
echo "  CAE datamodule: $DM_CAE_PATH"
echo "  CAE checkpoints dir: $CAE_CHECKPOINTS_DIR"
echo ""

# Change to the script directory
cd /Odyssey/private/o23gauvr/code/FASCINATION/src/

# Run the Python script
python compute_model_metrics.py \
    --device "$DEVICE" \
    --quality_start "$QUALITY_START" \
    --quality_end "$QUALITY_END" \
    --models $MODELS \
    --output_file "$OUTPUT_FILE" \
    --dm_mlic_path "$DM_MLIC_PATH" \
    --dm_cae_path "$DM_CAE_PATH" \
    --cae_checkpoints_dir "$CAE_CHECKPOINTS_DIR"

echo "Script completed!"
