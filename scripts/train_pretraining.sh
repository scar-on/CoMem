#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${REPO_ROOT}"

# export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH}

export CUDA_VISIBLE_DEVICES=4,5,6,7

DEBUG=${DEBUG:-0}

export WANDB_DIR="${WANDB_DIR:-${SAVE_ROOT}/wandb_logs}"

# Configuration #/data/project/scar/CoMem_pretraining/
data_path="${DATA_PATH:-${DATA_ROOT}/pretraining}"
SAVE_MODEL_NAME="${SAVE_MODEL_NAME:-base_new_mem_bidirectional}"
SAVE_PATH="${SAVE_ROOT}/${SAVE_MODEL_NAME}"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL}}"

mkdir -p $SAVE_PATH
# cp -r /mnt/conductor_data/code/comem_project/comem_training $SAVE_PATH/

# Extract distributed parameters dynamically
NCCL_DEBUG=INFO
NUM_NODES=1
MASTER=127.0.0.1
MASTER_PORT=29501
NODE_RANK=0
NUM_LOCAL_GPUS=4
WORLD_SIZE=$((NUM_LOCAL_GPUS * NUM_NODES))


echo "Number of nodes: ${NUM_NODES}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "Number of local GPUs: ${NUM_LOCAL_GPUS}"
echo "Master: ${MASTER}"
echo "Master port: ${MASTER_PORT}"
echo "Node rank: ${NODE_RANK}"

echo "Currently using $(which python)"

# Training command with torchrun # $data_path/pretrain_data.jsonl 
training_commands="openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset $data_path \
   --pretrain $MODEL_PATH \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --ckpt_path $SAVE_PATH \
   --max_samples 1000000 \
   --save_path $SAVE_PATH \
   --save_steps 3000 \
   --logging_steps 1 \
   --eval_steps 20 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-4 \
   --stage stage1 \
   --generation_top_k 1 \
   --qa_loss \
   --doc_max_length 256 \
   --compress_rate 32 \
   --gradient_checkpointing \
   --use_tensorboard $SAVE_PATH/tensorboard \
"
# 对比学习参数
#    --simcse_loss \
#    --use_mean_pool \
#    --use_proj_head \
#    --proj_hidden_ratio 1 \
# query 相似度加权参数
#    --use_query_relevance_bias \
#    --relevance_bias_scale 1.0 \


# Build distributed arguments
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_LOCAL_GPUS} --nnodes ${NUM_NODES} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint ${MASTER}:${MASTER_PORT} --master_addr ${MASTER} --master_port ${MASTER_PORT} --node_rank ${NODE_RANK}"

# Run training with torchrun for multinode
echo "Starting CoMem training on node $NODE_RANK of $NUM_NODES nodes..."
if [ $DEBUG -eq 0 ]; then
    if [ $NUM_NODES -gt 1 ]; then
        # For multinode, check if EFA is available
        if command -v fi_info >/dev/null 2>&1; then
            fi_info -p efa -t FI_EP_RDM; torchrun $DISTRIBUTED_ARGS -m $training_commands
        else
            torchrun $DISTRIBUTED_ARGS -m $training_commands
        fi
    else
        torchrun $DISTRIBUTED_ARGS -m $training_commands
    fi
else
    # Debug mode
    WORLD_SIZE=1 LOCAL_RANK=0 \
    python -m debugpy --listen 0.0.0.0:5678  \
    -m torch.distributed.launch --nproc_per_node=1 --master_port=20001 \
    -m $training_commands
fi





echo "CoMem training completed successfully!"


bash scripts/evaluation_end_to_end.sh
