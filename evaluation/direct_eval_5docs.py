#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import argparse
import contextlib
import json
import os
import re
import string
import sys
from datetime import timedelta
from typing import Dict, List, Optional

import torch
from accelerate import Accelerator, InitProcessGroupKwargs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from openrlhf.models.modeling_comem import CoMem


os.environ["NCCL_TIMEOUT"] = "5400"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_example(input_json: Optional[str], question: Optional[str], docs: Optional[List[str]]) -> Dict:
    """Load one direct evaluation example from JSON or CLI args."""
    if input_json:
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "question" not in data or "docs" not in data:
            raise ValueError("input_json must contain keys: question, docs")
        return data

    if not question:
        raise ValueError("--question is required when --input_json is not provided")
    if not docs:
        raise ValueError("--doc must be provided exactly 5 times when --input_json is not provided")

    return {"question": question, "docs": docs}


def validate_example(example: Dict) -> None:
    if not isinstance(example["question"], str) or not example["question"].strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(example["docs"], list):
        raise ValueError("docs must be a list of strings")
    if len(example["docs"]) != 5:
        raise ValueError(f"expected exactly 5 docs, got {len(example['docs'])}")
    if not all(isinstance(doc, str) and doc.strip() for doc in example["docs"]):
        raise ValueError("each doc must be a non-empty string")


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(token), gt_tokens.count(token)) for token in common)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def build_raw_context_prompt(model: CoMem, question: str, docs: List[str]) -> str:
    """Build the same QA prompt style as CoMem, but with raw document text."""
    background = "\n\n".join(f"[Document {idx}]\n{doc}" for idx, doc in enumerate(docs, start=1))
    return model._blend_standard_prompt(docs=background, query=question, answer=None)


def adapter_context(model: CoMem, adapter_mode: str):
    """Choose which LoRA adapter mode to use for raw-context decoding."""
    if adapter_mode == "none":
        if hasattr(model.decoder, "disable_adapter"):
            return model.decoder.disable_adapter()
        print("Warning: decoder does not support disable_adapter(); using current adapter state.")
        return contextlib.nullcontext()

    if adapter_mode == "decoder":
        if "decoder_adapter" not in model.adapter_keys:
            print("Warning: decoder_adapter not found; using current adapter state.")
        else:
            model.decoder.set_adapter("decoder_adapter")
        return contextlib.nullcontext()

    if adapter_mode == "all":
        if model.adapter_keys:
            model.decoder.set_adapter(model.adapter_keys)
        return contextlib.nullcontext()

    raise ValueError(f"Unknown adapter_mode: {adapter_mode}")


def run_direct_eval(args: argparse.Namespace) -> Dict:
    example = load_example(args.input_json, args.question, args.doc)
    validate_example(example)

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    if accelerator.is_main_process:
        print("Loading CoMem model...")

    model = CoMem.from_pretrained(
        args.model_path,
        training_stage=args.stage,
        generation_top_k=5,
        pure_inference=True,
    )

    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1

    model = accelerator.prepare(model)
    model.eval()
    model = model.module if hasattr(model, "module") else model

    question = example["question"]
    docs = example["docs"]

    if accelerator.is_main_process:
        print("Generating from raw 5-document context without compression...")

    tokenizer = model.decoder_tokenizer
    prompt = build_raw_context_prompt(model, question, docs)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_ids = inputs["input_ids"].to(model.decoder.device)
    attention_mask = inputs["attention_mask"].to(model.decoder.device)

    if accelerator.is_main_process and input_ids.size(1) == args.max_input_tokens:
        print(f"Warning: raw context was truncated to {args.max_input_tokens} tokens.")

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
    }
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    decode_context = adapter_context(model, args.adapter_mode)
    with torch.no_grad():
        with decode_context:
            output_ids = model.decoder.generate(**generate_kwargs)

    generated_ids = output_ids[:, input_ids.shape[-1]:]
    prediction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    result = {
        "mode": "raw_context_no_compression",
        "question": question,
        "docs": docs,
        "CoMem_normal_output": prediction,
        "prompt_tokens": input_ids.size(1),
        "adapter_mode": args.adapter_mode,
    }

    answer = args.answer or example.get("answer")
    if answer is not None:
        result["answer"] = answer
        result["metrics"] = {
            "f1": f1_score(prediction, answer),
            "em": exact_match_score(prediction, answer),
            "acc": all(token in normalize_answer(prediction).split() for token in normalize_answer(answer).split()),
            "output_length": len(prediction.split()),
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct raw-context eval: answer one question from exactly 5 docs without compression."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage1_2"])
    parser.add_argument("--question", type=str, help="Question to answer")
    parser.add_argument(
        "--doc",
        action="append",
        help="One document string. Provide this argument exactly 5 times.",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        help='Optional JSON file with {"question": str, "docs": [str x 5], optional "answer": str}.',
    )
    parser.add_argument("--answer", type=str, help="Optional ground-truth answer for simple F1/EM metrics")
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--adapter_mode",
        type=str,
        default="none",
        choices=["none", "decoder", "all"],
        help="Adapter mode for raw-context decoding. Use 'none' for the original base decoder.",
    )
    parser.add_argument("--output_path", type=str, help="Optional path to save result JSON")

    args = parser.parse_args()
    result = run_direct_eval(args)

    print("\nPrediction:")
    print(result["CoMem_normal_output"])

    if "metrics" in result:
        print("\nMetrics:")
        print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))

    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResult saved to: {args.output_path}")


if __name__ == "__main__":
    main()
