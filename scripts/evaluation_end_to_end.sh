#!/bin/bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
unset PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX:-compress}"

SAVE_PATH="${1:?Usage: $0 <save_path>}"
SAVE_PATH="$(cd "$(dirname "${SAVE_PATH}")" && pwd)/$(basename "${SAVE_PATH}")"
export PYTHONPATH="$SAVE_PATH:$PYTHONPATH"

# Run inference with torchrun for multinode musique 2wiki hotpotqa nq
echo "Starting inference on node $NODE_RANK of $NUM_NODES nodes..."
for i in musique 2wiki hotpotqa nq; do
    accelerate launch \
        --num_processes=2 \
        --num_machines=1 \
        --config_file "${REPO_ROOT}/scripts/accelerate_inference_config.yaml" \
        "${REPO_ROOT}/evaluation/test.py" \
        --model_path "$SAVE_PATH" \
        --stage stage1 \
        --dataset "$i" \
        --gold_retrieval
    done
echo "All steps completed successfully with torchrun multinode!"

for i in musique 2wiki hotpotqa nq; do
    accelerate launch \
        --num_processes=1 \
        --num_machines=1 \
        --mixed_precision=bf16 \
        "${REPO_ROOT}/evaluation/test.py" \
        --model_path "$SAVE_PATH" \
        --stage stage1 \
        --dataset "$i" \
        --gold_retrieval \
        --eval_only \
        --eval_file "$SAVE_PATH"
    done
