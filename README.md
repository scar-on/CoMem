# CoMem

CoMem contains the core training implementation and evaluation code for
context compression experiments with memory tokens.

## Repository layout

- `openrlhf/`: model, dataset, loss, and training implementation.
- `evaluation/`: end-to-end and metric evaluation code.
- `scripts/`: training and evaluation entry points.
- `llm_judge/`: optional LLM-as-a-judge utilities.

The model and configuration entry points are `CoMem` and `CoMemConfig` in
`openrlhf.models.modeling_comem`.

For a checkpoint stored in a Hugging Face subdirectory, pass its subfolder:

```python
from openrlhf.models.modeling_comem import CoMem

model = CoMem.from_pretrained(
    "ssxada/CoMem-Mistral-7B", subfolder="stage1/cr4"
)
```

Mistral checkpoints are available at compression rates 4, 8, 16, 64, and 128
for both training stages in
[CoMem-Mistral-7B](https://huggingface.co/ssxada/CoMem-Mistral-7B). The Qwen3-4B
checkpoint is available at rate 32 in
[CoMem-Qwen3-4B](https://huggingface.co/ssxada/CoMem-Qwen3-4B).

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

## Licensing and attribution

Parts of the model and training code derive from [Apple's original research
code](https://github.com/apple/ml-clara). Their Apple copyright headers remain
in the affected files. The accompanying [LICENSE](LICENSE) and
[ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) reproduce the upstream terms and third-party
notices; the CoMem name does not change those obligations. The base model
weights are distributed separately by their respective owners.
