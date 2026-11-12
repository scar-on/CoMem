---
license: other
license_name: CoMem mixed-source terms
license_link: https://github.com/scar-on/CoMem/blob/master/LICENSE
base_model: mistralai/Mistral-7B-Instruct-v0.2
---

# CoMem Mistral-7B checkpoints

These checkpoints contain CoMem adapters and the decoder embedding/output
layers needed for the expanded tokenizer. They do not include the Mistral base
model weights.

| Stage | Compression rates | Training |
| --- | --- | --- |
| `stage1` | 4, 8, 16, 64, 128 | Encoder and decoder adapters |
| `stage2` | 4, 8, 16, 64, 128 | Frozen encoder adapter, trained decoder adapter |

Each checkpoint is in `stage{number}/cr{rate}` and includes `config.json`,
`adapters.pth`, `decoder_first_last_layers.pth`, and tokenizer files. Use the
matching stage and rate together; the adapters are not interchangeable across
rates.

```python
from openrlhf.models.modeling_comem import CoMem

model = CoMem.from_pretrained(
    "ssxada/CoMem-Mistral-7B",
    subfolder="stage2/cr4",
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
