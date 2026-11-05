#!/usr/bin/env bash
# Shared paths for training and evaluation scripts.
# Override DATA_ROOT, MODEL_ROOT, ENV_PREFIX, or SAVE_ROOT for local machines.

COMMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${COMMON_ENV_DIR}/.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/save_dir}"
export ENV_PREFIX="${ENV_PREFIX:-compress}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# External resources can be supplied as local paths or Hugging Face IDs.
export BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
export CONTRAST_DATA="${CONTRAST_DATA:-${DATA_ROOT}/contrast}"
export INSTRUCTION_DATA="${INSTRUCTION_DATA:-${DATA_ROOT}/instruction_tuning}"
