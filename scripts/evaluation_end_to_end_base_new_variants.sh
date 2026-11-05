SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_PATHS=(
    "${SAVE_ROOT}/simcse_in_batch_contrast_0.1simcse/"
)

DATASETS=(hotpotqa nq)

for SAVE_PATH in "${MODEL_PATHS[@]}"; do
    unset PYTHONPATH
    export PYTHONPATH="$SAVE_PATH:$PYTHONPATH"

    echo "Starting evaluation for $SAVE_PATH"
    echo "Starting inference on node $NODE_RANK of $NUM_NODES nodes..."

    for i in "${DATASETS[@]}"; do
        accelerate launch \
            --num_processes=4 \
            --num_machines=1 \
            --config_file "${REPO_ROOT}/scripts/accelerate_inference_config.yaml" \
            "${REPO_ROOT}/evaluation/test.py" \
            --model_path "$SAVE_PATH" \
            --stage stage1 \
            --dataset "$i" \
            --gold_retrieval
    done

    echo "All multi-process evaluation steps completed for $SAVE_PATH"

    for i in "${DATASETS[@]}"; do
        accelerate launch \
            --num_processes=1 \
            --num_machines=1 \
            --mixed_precision=bf16 \
            "${REPO_ROOT}/evaluation/test.py" \
            --model_path "$SAVE_PATH" \
            --stage stage1 \
            --dataset "$i" \
            --gold_retrieval \
            --eval_file "$SAVE_PATH"
    done

    echo "Finished evaluation for $SAVE_PATH"
done

