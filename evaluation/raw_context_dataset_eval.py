#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import argparse
import gc
import json
import os
import sys
from datetime import timedelta
from typing import Dict, List

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from direct_eval_5docs import (
    adapter_context,
    build_raw_context_prompt,
    exact_match_score,
    f1_score,
    normalize_answer,
)
from openrlhf.models.modeling_comem import CoMem


os.environ["NCCL_TIMEOUT"] = "5400"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_eval_data(dataset: str, gold_retrieval: bool, top_k: int) -> List[Dict]:
    retrieval_type = "with_pos" if gold_retrieval else "no_pos"
    data_root = os.environ.get(
        "EVALUATION_DATA_ROOT",
        os.path.join(REPO_ROOT, "evaluation_data", "end_to_end_evaluation"),
    )
    file_path = os.path.join(
        data_root, dataset, f"eval_processed_{retrieval_type}.jsonl"
    )

    processed_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            docs = item["docs"][:top_k]
            if len(docs) != top_k:
                raise ValueError(f"{dataset} sample {index} has {len(docs)} docs, expected {top_k}")
            processed_data.append(
                {
                    "original_data": item,
                    "documents": docs,
                    "question": item["question"],
                    "global_index": index,
                }
            )

    return processed_data


def cover_exact_match_score(prediction: str, ground_truth: str) -> bool:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    return all(token in pred_tokens for token in gt_tokens)


def calculate_metrics(result_list: List[Dict]) -> Dict[str, float]:
    if not result_list:
        return {"f1": 0.0, "acc": 0.0, "em": 0.0, "avg_output_length": 0.0}

    f1_total = 0.0
    acc_total = 0.0
    em_total = 0.0
    avg_output_length = 0.0

    answer_key = "golden_answers" if "golden_answers" in result_list[0] else "answer"
    for result in result_list:
        prediction = result["CoMem_normal_output"]
        ground_truth = result[answer_key][0] if answer_key == "golden_answers" else result[answer_key]

        f1_total += f1_score(prediction, ground_truth)
        acc_total += cover_exact_match_score(prediction, ground_truth)
        em_total += exact_match_score(prediction, ground_truth)
        avg_output_length += len(prediction.split())

    n = len(result_list)
    return {
        "f1": f1_total / n,
        "acc": acc_total / n,
        "em": em_total / n,
        "avg_output_length": avg_output_length / n,
    }


def generate_batch(model: CoMem, batch: List[Dict], args: argparse.Namespace) -> List[Dict]:
    tokenizer = model.decoder_tokenizer
    prompts = [
        build_raw_context_prompt(model, item["question"], item["documents"])
        for item in batch
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_ids = inputs["input_ids"].to(model.decoder.device)
    attention_mask = inputs["attention_mask"].to(model.decoder.device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
    }
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.no_grad():
        output_ids = model.decoder.generate(**generate_kwargs)

    generated_ids = output_ids[:, input_ids.shape[-1]:]
    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    batch_results = []
    for item, prediction, prompt_tokens in zip(batch, predictions, attention_mask.sum(dim=1).tolist()):
        result_item = item["original_data"].copy()
        result_item["docs"] = item["documents"]
        result_item["CoMem_normal_output"] = prediction.strip()
        result_item["mode"] = "raw_context_no_compression"
        result_item["prompt_tokens"] = int(prompt_tokens)
        result_item["adapter_mode"] = args.adapter_mode
        result_item["global_index"] = item["global_index"]
        batch_results.append(result_item)

    return batch_results


def run_dataset(dataset: str, args: argparse.Namespace) -> None:
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    processed_data = load_eval_data(dataset, args.gold_retrieval, args.generation_top_k)
    if accelerator.is_main_process:
        print(f"Loaded {len(processed_data)} samples for {dataset}")
        print("Loading CoMem decoder for raw-context evaluation...")

    model = CoMem.from_pretrained(
        args.model_path,
        training_stage=args.stage,
        generation_top_k=args.generation_top_k,
        pure_inference=True,
    )

    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size

    model = accelerator.prepare(model)
    model.eval()
    model = model.module if hasattr(model, "module") else model

    all_results = []
    with adapter_context(model, args.adapter_mode):
        with accelerator.split_between_processes(processed_data, apply_padding=False) as local_data:
            if accelerator.is_main_process:
                print(f"Process {accelerator.process_index}: processing {len(local_data)} samples")

            num_batches = (len(local_data) + args.batch_size - 1) // args.batch_size
            for batch_idx in tqdm(
                range(num_batches),
                desc=f"GPU {accelerator.process_index}",
                disable=not accelerator.is_local_main_process,
            ):
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(local_data))
                batch = local_data[start_idx:end_idx]

                batch_results = generate_batch(model, batch, args)
                all_results.extend(batch_results)

                torch.cuda.empty_cache()
                if batch_idx % 10 == 0:
                    gc.collect()

    accelerator.wait_for_everyone()
    gathered_results = accelerator.gather_for_metrics(all_results)

    if accelerator.is_main_process:
        final_results = []
        if isinstance(gathered_results, list):
            for result_batch in gathered_results:
                if isinstance(result_batch, list):
                    final_results.extend(result_batch)
                else:
                    final_results.append(result_batch)

        final_results.sort(key=lambda x: x.get("global_index", 0))
        for item in final_results:
            item.pop("global_index", None)

        output_dir = args.output_dir or args.model_path
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"{dataset}_raw_context_{args.gold_retrieval}_{args.generation_top_k}_{args.adapter_mode}.jsonl",
        )
        with open(output_path, "w", encoding="utf-8") as f:
            for item in final_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        metrics = calculate_metrics(final_results)
        metrics_path = os.path.splitext(output_path)[0] + "_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")
        print(f"Metrics saved to: {metrics_path}")
        print(f"Metrics for {dataset}: {metrics}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate raw 5-document context baseline without CoMem compression."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage1_2"])
    parser.add_argument("--dataset", type=str, default="musique", help="Comma-separated dataset names")
    parser.add_argument("--gold_retrieval", action="store_true", help="Use eval_processed_with_pos.jsonl")
    parser.add_argument("--generation_top_k", type=int, default=5, help="Number of docs to pass as raw context")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per process")
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--adapter_mode",
        type=str,
        default="none",
        choices=["none", "decoder", "all"],
        help="Use 'none' to evaluate the original base decoder without LoRA adapters.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for jsonl and metrics outputs")

    args = parser.parse_args()

    for dataset in args.dataset.split(","):
        run_dataset(dataset.strip(), args)


if __name__ == "__main__":
    main()
