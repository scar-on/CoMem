#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

# unset PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${REPO_ROOT}"

SAVE_PATH="${1:-${SAVE_ROOT}/test1/global_step7812_hf}"
SAVE_MODEL_NAME=${SAVE_PATH##*/}
# export PYTHONPATH="$SAVE_PATH:$PYTHONPATH"


echo "Starting inference on node $NODE_RANK of $NUM_NODES nodes..."
accelerate launch \
    --num_processes=8 \
    --num_machines=1 \
    "${REPO_ROOT}/evaluation/test.py" \
    --model_path "$SAVE_PATH" \
    --stage stage1 \
    --dataset musique,hotpotqa,2wiki,nq \
    --gold_retrieval
