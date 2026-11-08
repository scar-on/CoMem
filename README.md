# CoMem

CoMem contains the core training implementation and evaluation code for
context compression experiments with CLaRa-style memory tokens.

## Repository layout

- `openrlhf/`: model, dataset, loss, and training implementation.
- `evaluation/`: end-to-end and metric evaluation code.
- `scripts/`: training and evaluation entry points.
- `llm_judge/`: optional LLM-as-a-judge utilities.

The public Hugging Face model entry point is `modeling_comem.CoMem`. The
underlying `modeling_clara.py` module is retained so older checkpoints and
training scripts remain compatible.

## Setup

```bash
conda env create -f environment_compress.yml
conda activate compress
```

The scripts use paths relative to the repository by default. Set these
environment variables when using external data or model locations:

```bash
export DATA_ROOT=/path/to/data
export MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.2
export SAVE_ROOT="$PWD/save_dir"
```

See `scripts/common_env.sh` for the shared path configuration.

## Training

```bash
bash scripts/train_pretraining_contrast.sh
```

## Evaluation

```bash
bash scripts/evaluation_end_to_end.sh /path/to/checkpoint
```

Datasets, base model weights, checkpoints, logs, and generated evaluation
artifacts are intentionally kept outside the source repository.
