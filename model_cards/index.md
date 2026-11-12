---
license: other
license_name: comem-mixed-source-terms
license_link: https://github.com/scar-on/CoMem/blob/master/LICENSE
---

# CoMem

CoMem checkpoint releases are organized by base model:

| Base model | Repository | Compression rates | Stages |
| --- | --- | --- | --- |
| Mistral-7B-Instruct-v0.2 | [CoMem-Mistral-7B](https://huggingface.co/ssxada/CoMem-Mistral-7B) | 4, 8, 16, 64, 128 | 1 and 2 |
| Qwen3-4B-Instruct-2507 | [CoMem-Qwen3-4B](https://huggingface.co/ssxada/CoMem-Qwen3-4B) | 32 | 1 and 2 |

Each model repository contains the adapters, decoder layers, tokenizer, and
portable configuration files. This index repository contains no model weights.
Training and evaluation code is in the
[CoMem GitHub repository](https://github.com/scar-on/CoMem).

Parts of the code derive from
[Apple's research code](https://github.com/apple/ml-clara); its original
copyright headers and [license](https://github.com/scar-on/CoMem/blob/master/LICENSE)
remain applicable, together with the
[third-party notices](https://github.com/scar-on/CoMem/blob/master/ACKNOWLEDGEMENTS).
