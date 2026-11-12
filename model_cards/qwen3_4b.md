---
license: other
license_name: CoMem mixed-source terms
license_link: https://github.com/scar-on/CoMem/blob/master/LICENSE
base_model: Qwen/Qwen3-4B-Instruct-2507
---

# CoMem Qwen3-4B checkpoints

These checkpoints contain CoMem adapters and the decoder embedding/output
layers needed for the expanded tokenizer. They do not include the Qwen3-4B
base model weights.

| Stage | Compression rate | Training |
| --- | --- | --- |
| `stage1` | 32 | Encoder and decoder adapters |
| `stage2` | 32 | Frozen encoder adapter, trained decoder adapter |

Each checkpoint is in `stage{number}/cr32` and includes `config.json`,
`adapters.pth`, `decoder_first_last_layers.pth`, and tokenizer files.

```python
from openrlhf.models.modeling_comem import CoMem

model = CoMem.from_pretrained(
    "ssxada/CoMem-Qwen3-4B",
    subfolder="stage2/cr32",
    pure_inference=True,
)
```

The implementation and setup instructions are in the
[CoMem repository](https://github.com/scar-on/CoMem).

The model code includes portions derived from
[Apple's research code](https://github.com/apple/ml-clara). Its original
copyright notices remain in the affected files. See the repository's
[LICENSE](https://github.com/scar-on/CoMem/blob/master/LICENSE) and
[ACKNOWLEDGEMENTS](https://github.com/scar-on/CoMem/blob/master/ACKNOWLEDGEMENTS)
for the upstream terms and third-party notices. Base model weights are subject
to their own terms.
