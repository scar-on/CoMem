#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
REPO="${REPO_ROOT}"
MODEL_CASUAL="${MODEL_CASUAL:-${SAVE_ROOT}/simcse_in_batch_contrast_160k_casual}"
MODEL_MEM_CONTEXT="${MODEL_MEM_CONTEXT:-${SAVE_ROOT}/simcse_in_batch_contrast_mem_context}"
OUTPUT_DIR="${OUTPUT_DIR:-${SAVE_ROOT}/mem_information_eval}"

unset PYTHONPATH
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

DATASETS=${DATASETS:-"musique,2wiki,hotpotqa,nq"}
SAMPLE_LIMIT=${SAMPLE_LIMIT:-3}
SAMPLE_STRATEGY=${SAMPLE_STRATEGY:-first}

sample_args=()
if [[ -n "$SAMPLE_LIMIT" && "$SAMPLE_LIMIT" != "all" ]]; then
    sample_args=(--sample_limit "$SAMPLE_LIMIT")
fi

conda run -n compress python "$REPO/evaluation/mem_information_eval.py" \
    --model "casual=$MODEL_CASUAL" \
    --model "mem_context=$MODEL_MEM_CONTEXT" \
    --datasets "$DATASETS" \
    --generation_top_k 5 \
    --sample_strategy "$SAMPLE_STRATEGY" \
    --output_dir "$OUTPUT_DIR" \
    --save_per_sample \
    "${sample_args[@]}"
