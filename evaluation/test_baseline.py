import os
import json
import argparse
import gc
from datetime import timedelta
from typing import List, Dict, Tuple

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from evaluation.test import (
    DataLoader,
    ResultCalculator,
    EvaluationMetrics,
    load_results_from_jsonl,
    convert_embeddings_to_list,
)

os.environ["NCCL_TIMEOUT"] = "5400"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to extract relevant "
    "information from provided documents and to answer to questions."
)

QUESTION_ONLY_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question."


def build_prompt(tokenizer, documents: List[str], question: str) -> str:
    """Build a chat prompt that concatenates documents + question directly."""
    if documents:
        system_prompt = SYSTEM_PROMPT
        docs_text = "\n\n".join(documents)
        user_content = f"Background:\n{docs_text}\n\nQuestion:{question}"
    else:
        system_prompt = QUESTION_ONLY_SYSTEM_PROMPT
        user_content = question

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        merged = system_prompt + "\n" + user_content
        messages = [{"role": "user", "content": merged}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt


class BaselineInference:
    """Inference engine that uses the original checkpoint without LoRA or compression."""

    def __init__(self, model_path: str):
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        self.accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

        if self.accelerator.is_main_process:
            print(f"Using {self.accelerator.num_processes} GPUs")
            print(f"Loading base model from {model_path} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        if self.accelerator.state.deepspeed_plugin is not None:
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = 1
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

        if self.accelerator.is_main_process:
            print("Model loaded and ready.")

    def _get_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    @torch.no_grad()
    def generate_batch(
        self,
        batch_questions: List[str],
        batch_documents: List[List[str]],
        max_new_tokens: int = 128,
    ) -> List[str]:
        prompts = [
            build_prompt(self.tokenizer, docs, q)
            for docs, q in zip(batch_documents, batch_questions)
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        model = self._get_model()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        answers = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return answers


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation (no LoRA, no compression)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the original model checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="musique", help="Comma-separated list of datasets")
    parser.add_argument("--gold_retrieval", action="store_true")
    parser.add_argument("--generation_top_k", type=int, default=5, help="Number of top-k docs to use")
    parser.add_argument("--save_path", type=str, default=None, help="Directory to save results (default: model_path)")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_file", type=str, default=None)
    args = parser.parse_args()

    save_path = args.save_path or args.model_path

    # ── Eval-only mode ──
    if args.eval_file is not None:
        if os.path.isdir(args.eval_file):
            eval_path = os.path.join(
                args.eval_file,
                f"{args.dataset}_baseline_{args.gold_retrieval}_{args.generation_top_k}.jsonl",
            )
        else:
            eval_path = args.eval_file

        print(f"Evaluation-only mode, loading from: {eval_path}")
        final_results = load_results_from_jsonl(eval_path)
        print(f"Loaded {len(final_results)} samples")

        calculator = ResultCalculator()
        metrics = calculator.calculate_basic_metrics(final_results)
        print("Evaluation results:")
        print(json.dumps(metrics, indent=2))

        metrics_path = os.path.splitext(eval_path)[0] + "_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        return

    # ── Inference mode ──
    all_results_metrics = {}
    datasets_list = args.dataset.split(",")

    engine = BaselineInference(args.model_path)

    for dataset in datasets_list:
        if engine.accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset}")
            print(f"{'='*60}")

        processed_data = DataLoader.load_stage1_data(dataset, args.gold_retrieval)
        # load_stage1_data already takes top-5 docs
        # Trim to generation_top_k if needed
        for item in processed_data:
            item["documents"] = item["documents"][: args.generation_top_k]

        if engine.accelerator.is_main_process:
            print(f"Loaded {len(processed_data)} samples, using top-{args.generation_top_k} docs")

        engine.accelerator.wait_for_everyone()

        all_results = []
        with engine.accelerator.split_between_processes(processed_data, apply_padding=False) as local_data:
            if engine.accelerator.is_main_process:
                print(f"Process {engine.accelerator.process_index}: {len(local_data)} samples")

            batch_size = args.batch_size
            num_batches = (len(local_data) + batch_size - 1) // batch_size

            for batch_idx in tqdm(
                range(num_batches),
                desc=f"GPU {engine.accelerator.process_index}",
                disable=not engine.accelerator.is_local_main_process,
            ):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(local_data))
                batch = local_data[start_idx:end_idx]

                batch_questions = [item["question"] for item in batch]
                batch_documents = [item["documents"] for item in batch]

                try:
                    answers = engine.generate_batch(
                        batch_questions, batch_documents, max_new_tokens=args.max_new_tokens
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM at batch {batch_idx}, skipping")
                    torch.cuda.empty_cache()
                    gc.collect()
                    answers = [""] * len(batch_questions)

                for item, ans in zip(batch, answers):
                    result_item = item["original_data"].copy()
                    result_item["CLaRa_normal_output"] = ans
                    result_item["global_index"] = item["global_index"]
                    all_results.append(result_item)

                torch.cuda.empty_cache()
                if batch_idx % 10 == 0:
                    gc.collect()

        engine.accelerator.wait_for_everyone()

        all_results_gathered = engine.accelerator.gather_for_metrics(all_results)

        if engine.accelerator.is_main_process:
            final_results = []
            if isinstance(all_results_gathered, list):
                for item in all_results_gathered:
                    if isinstance(item, list):
                        final_results.extend(item)
                    else:
                        final_results.append(item)

            final_results.sort(key=lambda x: x.get("global_index", 0))

            processed_indices = set(r.get("global_index", -1) for r in final_results)
            expected_indices = set(range(len(processed_data)))
            missing = expected_indices - processed_indices
            if missing:
                print(f"Warning: missing indices {sorted(list(missing))}")
            else:
                print("Data integrity check passed")

            for item in final_results:
                item.pop("global_index", None)

            os.makedirs(save_path, exist_ok=True)
            output_path = os.path.join(
                save_path,
                f"{dataset}_baseline_{args.gold_retrieval}_{args.generation_top_k}.jsonl",
            )
            with open(output_path, "w") as f:
                for item in final_results:
                    f.write(json.dumps(item) + "\n")
            print(f"Results saved to: {output_path}")

            calculator = ResultCalculator()
            metrics = calculator.calculate_basic_metrics(final_results)
            print(f"Metrics for {dataset}: {json.dumps(metrics, indent=2)}")
            all_results_metrics[dataset] = metrics

        torch.cuda.empty_cache()
        gc.collect()

    if all_results_metrics:
        metrics_path = os.path.join(
            save_path,
            f"results_metrics_baseline_{args.gold_retrieval}_{args.generation_top_k}.json",
        )
        with open(metrics_path, "w") as f:
            json.dump(all_results_metrics, f, indent=2)
        print(f"Final metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
