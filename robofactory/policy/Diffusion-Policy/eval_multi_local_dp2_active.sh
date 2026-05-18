#!/bin/bash

# Evaluation script for eval_multi_local_dp2_active.py
# Usage: ./eval_multi_local_dp2_active.sh <config_path> <data_num> <checkpoint_num> <ckpt_path> [debug_mode] [task_name]

# Check required arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <config_path> <data_num> <checkpoint_num> <ckpt_path> [debug_mode] [task_name]"
    echo "Example: $0 /path/to/config.yaml 150 300 /path/to/checkpoint.ckpt 0 two_robots_stack_cube_active"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Usage: $0 <config_path> <data_num> <checkpoint_num> <ckpt_path> [debug_mode] [task_name]"
    exit 1
fi
if [ -z "$3" ]; then
    echo "Usage: $0 <config_path> <data_num> <checkpoint_num> <ckpt_path> [debug_mode] [task_name]"
    exit 1
fi
if [ -z "$4" ]; then
    echo "Usage: $0 <config_path> <data_num> <checkpoint_num> <ckpt_path> [debug_mode] [task_name]"
    exit 1
fi

CONFIG_PATH="$1"
DATA_NUM="$2"
CHECKPOINT_NUM="$3"
CKPT_PATH="$4"
DEBUG_MODE="${5:-0}"
TASK_NAME="${6:-two_robots_stack_cube_active}"

# Generate log file with timestamp
LOG_FILE="eval_results_${TASK_NAME}_${DATA_NUM}_${CHECKPOINT_NUM}_$(date +"%Y%m%d_%H%M%S").log"

echo "Evaluating task: $TASK_NAME"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CKPT_PATH"
echo "Data num: $DATA_NUM, Checkpoint num: $CHECKPOINT_NUM"
echo "Evaluating task: $TASK_NAME" >> "$LOG_FILE"
echo "Config: $CONFIG_PATH" >> "$LOG_FILE"
echo "Checkpoint: $CKPT_PATH" >> "$LOG_FILE"
echo "Data num: $DATA_NUM, Checkpoint num: $CHECKPOINT_NUM" >> "$LOG_FILE"

TOTAL=0
SUCCESS=0
SUCCESS_RATE=0

# Set quiet flag if debug mode is disabled
QUIET_FLAG=""
if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
    QUIET_FLAG="--quiet"
fi

# Run 100 evaluations with seeds from 10000 to 10099
for SEED in {10000..10099}
do
    echo "Running evaluation with seed $SEED (${TOTAL}/100)..."
    OUTPUT=""
    
    # Run the evaluation script
    OUTPUT=$(python ./robofactory/policy/Diffusion-Policy/eval_multi_local_dp2_active.py \
            --config="$CONFIG_PATH" \
            --data-num=$DATA_NUM \
            --checkpoint-num=$CHECKPOINT_NUM \
            --ckpt-path="$CKPT_PATH" \
            --render-mode="rgb_array" \
            -o="rgb" \
            -b="auto" \
            -n 1 \
            -s $SEED \
            --max-steps=100 \
            $QUIET_FLAG 2>&1)
    
    echo "$OUTPUT"
    
    # Check if evaluation was successful
    LAST_LINE=$(echo "$OUTPUT" | tail -n 1)
    FINE=0
    
    # Check for success indicators in output
    if [[ $OUTPUT == *"Success"* ]] || [[ $LAST_LINE == *"Success"* ]]; then
        FINE=1
        SUCCESS=$((SUCCESS + 1))
    fi
    
    TOTAL=$((TOTAL + 1))
    SUCCESS_RATE=$(echo "scale=2; $SUCCESS * 100 / $TOTAL" | bc)
    
    # Log results
    echo "$SEED, $FINE, $SUCCESS_RATE%" >> "$LOG_FILE"
    echo "Seed $SEED done. Success: $FINE, Success Rate: $SUCCESS_RATE% ($SUCCESS/$TOTAL)"
done

# Final summary
echo "" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "Final Results:" >> "$LOG_FILE"
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%"
echo "Results saved in $LOG_FILE"
echo "========================================"

