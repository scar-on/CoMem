
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
unset PYTHONPATH
SAVE_PATH="${1:-${SAVE_ROOT}/base_new_mem_bidirectional/}"
export PYTHONPATH="$SAVE_PATH:$PYTHONPATH"

export PYTHONPATH=$SAVE_PATH:$PYTHONPATH

# Run inference with torchrun for multinode musique 2wiki hotpotqa nq
# echo "Starting inference on node $NODE_RANK of $NUM_NODES nodes..."
# for i in  musique 2wiki hotpotqa nq; do    
#     accelerate launch \
#         --num_processes=4 \
#         --num_machines=1 \
#         --config_file /mnt/share/zhangsikui_workspace/comem/scripts/accelerate_inference_config.yaml \
#         /mnt/share/zhangsikui_workspace/comem/evaluation/test.py \
#         --model_path $SAVE_PATH \
#         --stage stage1 \
#         --dataset $i \
#         --gold_retrieval 
#     done
# echo "All steps completed successfully with torchrun multinode!"

for i in musique 2wiki hotpotqa nq; do    
    accelerate launch \
        --num_processes=1 \
        --num_machines=1 \
        --mixed_precision=bf16 \
        "${REPO_ROOT}/evaluation/test.py" \
        --model_path $SAVE_PATH \
        --stage stage1 \
        --dataset $i \
        --gold_retrieval \
        --eval_file  $SAVE_PATH 
    done

bash "${REPO_ROOT}/scripts/evaluation_end_to_end_base_new_variants.sh"
