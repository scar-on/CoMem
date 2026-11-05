#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
unset PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${REPO_ROOT}"

SAVE_PATH=${SAVE_PATH:-${SAVE_ROOT}/simcse_in_batch_contrast_uniform/}
export PYTHONPATH="$SAVE_PATH:${PYTHONPATH:-}"

DATASETS=${DATASETS:-"musique 2wiki hotpotqa nq"}
NUM_PROCESSES=${NUM_PROCESSES:-4}
NUM_MACHINES=${NUM_MACHINES:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
GENERATION_TOP_K=${GENERATION_TOP_K:-5}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-8192}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
ADAPTER_MODE=${ADAPTER_MODE:-none}
OUTPUT_DIR=${OUTPUT_DIR:-$SAVE_PATH}
GOLD_RETRIEVAL=${GOLD_RETRIEVAL:-1}

SCRIPT_DIR="${REPO_ROOT}/scripts"
EVAL_SCRIPT="${REPO_ROOT}/evaluation/raw_context_dataset_eval.py"

echo "Starting raw-context no-compression evaluation..."
echo "Model path: $SAVE_PATH"
echo "Datasets: $DATASETS"
echo "Adapter mode: $ADAPTER_MODE"

gold_flag=()
if [[ "$GOLD_RETRIEVAL" == "1" || "$GOLD_RETRIEVAL" == "true" || "$GOLD_RETRIEVAL" == "True" ]]; then
    gold_flag=(--gold_retrieval)
fi

for dataset in $DATASETS; do
    echo "Evaluating raw-context baseline on $dataset..."
    accelerate launch \
        --num_processes="$NUM_PROCESSES" \
        --num_machines="$NUM_MACHINES" \
        --config_file "$SCRIPT_DIR/accelerate_inference_config.yaml" \
        "$EVAL_SCRIPT" \
        --model_path "$SAVE_PATH" \
        --stage stage1 \
        --dataset "$dataset" \
        "${gold_flag[@]}" \
        --generation_top_k "$GENERATION_TOP_K" \
        --batch_size "$BATCH_SIZE" \
        --max_input_tokens "$MAX_INPUT_TOKENS" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --adapter_mode "$ADAPTER_MODE" \
        --output_dir "$OUTPUT_DIR"
done

echo "Raw-context evaluation completed."
