#!/bin/bash
#
# SimCSE in-batch contrastive pretraining for CLaRa.
#
# Each context has 8 different questions.  micro_train_batch_size=2 on 4 GPUs
# yields a global micro-batch of 8 samples – exactly one context group.
# ContrastDistributedSampler guarantees the grouping.
#

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

DEBUG=${DEBUG:-0}

export WANDB_DIR="${WANDB_DIR:-${SAVE_ROOT}/wandb_logs}"

# Configuration
data_path="${CONTRAST_DATA}"
SAVE_MODEL_NAME="${SAVE_MODEL_NAME:-simcse_in_batch_contrast_160k}"
SAVE_PATH="${SAVE_ROOT}/${SAVE_MODEL_NAME}"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL}}"

mkdir -p $SAVE_PATH

# Distributed parameters
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

# micro_train_batch_size=2, 4 GPUs → global micro-batch = 8 = one context group
# train_batch_size=8 means no gradient accumulation (each step = one context)
# Increase train_batch_size (must be multiple of 8) for gradient accumulation.
training_commands="openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset $data_path \
   --pretrain $MODEL_PATH \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --ckpt_path $SAVE_PATH \
   --max_samples 160000 \
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
   --use_tensorboard $SAVE_PATH/tensorboard \
   --simcse_loss \
   --use_mean_pool \
   --simcse_in_batch \
   --ortho_loss_weight 0.1 \
   --contrast_num_questions 8 \
"
#    --gradient_checkpointing \
# Distributed arguments
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_LOCAL_GPUS} --nnodes ${NUM_NODES} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint ${MASTER}:${MASTER_PORT} --master_addr ${MASTER} --master_port ${MASTER_PORT} --node_rank ${NODE_RANK}"

# Run training
echo "Starting CLaRa SimCSE contrastive training on node $NODE_RANK of $NUM_NODES nodes..."
if [ $DEBUG -eq 0 ]; then
    if [ $NUM_NODES -gt 1 ]; then
        if command -v fi_info >/dev/null 2>&1; then
            fi_info -p efa -t FI_EP_RDM; torchrun $DISTRIBUTED_ARGS -m $training_commands
        else
            torchrun $DISTRIBUTED_ARGS -m $training_commands
        fi
    else
        torchrun $DISTRIBUTED_ARGS -m $training_commands
    fi
else
    WORLD_SIZE=1 LOCAL_RANK=0 \
    python -m debugpy --listen 0.0.0.0:5678  \
    -m torch.distributed.launch --nproc_per_node=1 --master_port=20001 \
    -m $training_commands
fi

echo "CLaRa SimCSE contrastive training completed successfully!"
