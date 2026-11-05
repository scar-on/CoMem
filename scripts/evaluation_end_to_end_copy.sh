

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${REPO_ROOT}"

# CUDA_VISIBLE_DEVICES=3
unset PYTHONPATH
SAVE_PATH="${1:-${SAVE_ROOT}/qwen_query_context_bidirectional_queryscore_3.24/}"
export PYTHONPATH="$SAVE_PATH:$PYTHONPATH"

export PYTHONPATH=$SAVE_PATH:$PYTHONPATH



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
