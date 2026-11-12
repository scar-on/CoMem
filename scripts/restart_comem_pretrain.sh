#!/bin/bash
# CoMem stage1 预训练重启脚本（与之前停掉的任务参数一致）
# 使用方式: conda activate compress && bash scripts/restart_comem_pretrain.sh
# 说明: 因中断时未保存 checkpoint，本次启动将从 --pretrain 从头训练

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
  --nproc_per_node 4 \
  --nnodes 1 \
  --rdzv_id 101 \
  --rdzv_backend c10d \
  --rdzv_endpoint 127.0.0.1:29500 \
  --master_addr 127.0.0.1 \
  --master_port 29500 \
  --node_rank 0 \
  -m openrlhf.cli.train_sft \
  --max_len 2048 \
  --dataset "${DATA_PATH:-${DATA_ROOT}/pretraining}" \
  --pretrain "${MODEL_PATH:-${BASE_MODEL}}" \
  --train_batch_size 128 \
  --micro_train_batch_size 2 \
  --ckpt_path "${SAVE_ROOT}/moco_stage1_loss_0.5" \
  --max_samples 1000000 \
  --save_path "${SAVE_ROOT}/moco_stage1_loss_0.5" \
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
  --simcse_loss \
  --use_mean_pool
