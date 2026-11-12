#!/usr/bin/env python
"""Benchmark CoMem compression ratios and draw a performance-efficiency figure."""

import argparse
import gc
import json
import os
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _preparse_gpu() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default="2")
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


_preparse_gpu()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch._dynamo
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch._dynamo.config.suppress_errors = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openrlhf.models.modeling_comem import CoMem  # noqa: E402


DEFAULT_DATASETS = ("nq",)
OUTPUT_FIELD = "CoMem_normal_output"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to extract relevant "
    "information from provided documents and to answer to questions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batched CoMem inference across compression ratios and plot accuracy vs efficiency."
    )
    parser.add_argument("--gpu", default="2", help="Visible CUDA device id. Default: 2")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Benchmark source dataset. Default: nq only.")
    parser.add_argument("--models", default=None, help="Comma-separated CR=checkpoint_path entries")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "save_dir" / "performance_efficiency_analysis"))
    parser.add_argument(
        "--max_samples_per_dataset",
        type=int,
        default=None,
        help="Number of sampled examples. Defaults to batch_size for pure throughput benchmarking.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Large QA batch size used after document compression.")
    parser.add_argument("--generation_top_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend for both CoMem and baseline. Default: flash_attention_2.",
    )
    parser.add_argument("--warmup_batches", type=int, default=3)
    parser.add_argument("--repeat_runs", type=int, default=5, help="Repeat the same selected samples for stable timing.")
    parser.add_argument("--timed_runs", type=int, default=3, help="Only the last N repeats contribute to efficiency.")
    parser.add_argument("--use_general_query", action="store_true")
    parser.add_argument("--no_gold_retrieval", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--include_baseline", action="store_true", help="Benchmark and draw the direct-decoder baseline.")
    parser.add_argument("--baseline_only", action="store_true", help="Only benchmark baseline, then plot with existing CoMem summary.")
    parser.add_argument(
        "--baseline_model_path",
        default=os.environ.get("BASELINE_MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507"),
    )
    return parser.parse_args()


def parse_models(models_arg: str | None) -> Dict[int, str]:
    if not models_arg:
        return {}

    parsed = {}
    for entry in models_arg.split(","):
        if not entry.strip():
            continue
        cr_text, path = entry.split("=", 1)
        parsed[int(cr_text.strip().lower().removeprefix("cr"))] = path.strip()
    if not parsed:
        raise ValueError("No valid models were parsed from --models.")
    return dict(sorted(parsed.items()))


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        import re

        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return " ".join(remove_articles(remove_punc(str(text).lower())).split())


def cover_exact_match(prediction: str, ground_truth: str) -> bool:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    return all(token in pred_tokens for token in gt_tokens)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(len(pred_tokens), 1)
    recall = num_same / max(len(gt_tokens), 1)
    return 2 * precision * recall / (precision + recall)


def load_dataset_slice(dataset: str, gold_retrieval: bool, generation_top_k: int, limit: int) -> List[Dict]:
    retrieval_type = "with_pos" if gold_retrieval else "no_pos"
    file_path = (
        Path(__file__).resolve().parent
        / "evaluation_data"
        / "end_to_end_evaluation"
        / dataset
        / f"eval_processed_{retrieval_type}.jsonl"
    )
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            item = json.loads(line)
            rows.append(
                {
                    "original_data": item,
                    "documents": item["docs"][:generation_top_k],
                    "question": item["question"],
                    "global_index": index,
                }
            )
            if limit and limit > 0 and len(rows) >= limit:
                break
    return rows


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_cuda_call(fn):
    if not torch.cuda.is_available():
        start = time.perf_counter()
        result = fn()
        return result, time.perf_counter() - start

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = fn()
    end_event.record()
    end_event.synchronize()
    return result, start_event.elapsed_time(end_event) / 1000.0


def set_attention_backend(model, attn_implementation: str) -> None:
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = attn_implementation
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = attn_implementation


def load_model(model_path: str, generation_top_k: int, attn_implementation: str) -> CoMem:
    model = CoMem.from_pretrained(
        model_path,
        training_stage="stage1_2",
        generation_top_k=generation_top_k,
        pure_inference=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    # Benchmark-only path: use standard causal attention instead of the project's
    # FlexAttention block-mask implementation. This does not modify training code.
    set_attention_backend(model.decoder, attn_implementation)
    return model


def build_baseline_prompt(tokenizer, documents: List[str], question: str) -> str:
    docs_text = "\n\n".join(documents)
    user_content = f"Background:\n{docs_text}\n\nQuestion:{question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        merged = SYSTEM_PROMPT + "\n" + user_content
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


def load_baseline_model(model_path: str, attn_implementation: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    set_attention_backend(model, attn_implementation)
    return tokenizer, model


def causal_compress_documents(
    model: CoMem,
    enc_input_ids: torch.Tensor,
    enc_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Benchmark-only compressor: use standard causal decoder attention, no FlexAttention block mask."""
    if "encoder_adapter" in model.adapter_keys:
        model._set_active_adapters("encoder_adapter")
    else:
        raise ValueError(f"encoder_adapter not in adapter_keys: {model.adapter_keys}")

    hidden = model.decoder(
        input_ids=enc_input_ids,
        attention_mask=enc_attention_mask,
        output_hidden_states=True,
    ).hidden_states[-1]

    mem_token_ids = model.decoder_tokenizer.mem_token_ids_pt.to(enc_input_ids.device)
    mem_mask = torch.isin(enc_input_ids, mem_token_ids)
    return hidden[mem_mask].reshape(hidden.size(0), -1, hidden.size(-1))


@torch.no_grad()
def generate_baseline_batch(tokenizer, model, batch: List[Dict], max_new_tokens: int):
    prompts = [
        build_baseline_prompt(tokenizer, row["documents"], row["question"])
        for row in batch
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)
    sync_cuda()
    output_ids, generate_seconds = timed_cuda_call(
        lambda: model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    )
    del inputs, output_ids
    return generate_seconds


def benchmark_baseline(rows: List[Dict], args: argparse.Namespace) -> Dict:
    tokenizer, model = load_baseline_model(args.baseline_model_path, args.attn_implementation)
    batch_size = args.batch_size
    timed_start = max(0, args.repeat_runs - args.timed_runs)
    total_seconds = 0.0
    total_samples = 0

    progress = tqdm(total=len(rows) * args.repeat_runs, desc="Baseline", leave=False)
    try:
        for repeat_idx in range(args.repeat_runs):
            is_timed = repeat_idx >= timed_start
            index = 0
            while index < len(rows):
                batch = rows[index : index + batch_size]
                try:
                    generate_seconds = generate_baseline_batch(
                        tokenizer,
                        model,
                        batch,
                        args.max_new_tokens,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    print(f"OOM on baseline; retrying with batch_size={batch_size}")
                    continue

                if is_timed:
                    total_seconds += generate_seconds
                    total_samples += len(batch)
                index += len(batch)
                progress.update(len(batch))
    finally:
        progress.close()
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    return {
        "method": "Baseline",
        "model_path": args.baseline_model_path,
        "samples": total_samples,
        "seconds": total_seconds,
        "samples_per_second": total_samples / total_seconds if total_seconds > 0 else 0.0,
        "seconds_per_sample": total_seconds / max(total_samples, 1),
        "repeat_runs": args.repeat_runs,
        "timed_runs": args.timed_runs,
        "effective_batch_size": batch_size,
        "attn_implementation": args.attn_implementation,
    }


@torch.no_grad()
def generate_batch(
    model: CoMem,
    batch: List[Dict],
    max_new_tokens: int,
    use_general_query: bool,
) -> Tuple[List[str], Dict[str, float]]:
    questions = [row["question"] for row in batch]
    documents = [row["documents"] for row in batch]
    generation_top_k = len(documents[0])
    if any(len(docs) != generation_top_k for docs in documents):
        raise ValueError("All samples in one batch must have the same generation_top_k.")

    flat_docs = []
    flat_queries = []
    for question, docs in zip(questions, documents):
        query_text = "Extract potentially useful information from the text to help answer the question."
        if not use_general_query:
            query_text = question
        flat_docs.extend(docs)
        flat_queries.extend([query_text] * len(docs))

    prepared = model._prepare_encoder_inputs(
        flat_docs,
        max_length=model.doc_max_length,
        q_texts=flat_queries,
    )
    device = model.decoder.device
    enc_input_ids = prepared["input_ids"].to(device)
    enc_attention_mask = prepared["attention_mask"].to(device)

    sync_cuda()
    compressed_documents, compress_seconds = timed_cuda_call(
        lambda: causal_compress_documents(
            model,
            enc_input_ids=enc_input_ids,
            enc_attention_mask=enc_attention_mask,
        )
    )
    model.generation_top_k = compressed_documents.size(0) // len(questions)
    instructions = [model._blend_prompt_and_memory_tokens(query=q, stage="stage1_2") for q in questions]
    inp_dec = model.decoder_tokenizer(
        instructions,
        return_tensors="pt",
        padding="longest",
        add_special_tokens=False,
        truncation=True,
        max_length=1024,
    )
    dec_input_ids = inp_dec["input_ids"].to(model.decoder.device)
    dec_attention_mask = inp_dec["attention_mask"].to(model.decoder.device)
    inputs_embeds = model._replace_emb(compressed_documents, dec_input_ids)

    if "decoder_adapter" in model.adapter_keys:
        model._set_active_adapters("decoder_adapter")

    sync_cuda()
    output_ids, generate_seconds = timed_cuda_call(
        lambda: model.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            max_new_tokens=max_new_tokens,
        )
    )
    outputs = model.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    del enc_input_ids, enc_attention_mask, compressed_documents, inputs_embeds, output_ids

    return outputs, {
        "compress_seconds": compress_seconds,
        "generate_seconds": generate_seconds,
        "total_seconds": compress_seconds + generate_seconds,
    }


def iter_batches(rows: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def warmup_model(model: CoMem, rows: List[Dict], args: argparse.Namespace) -> None:
    if args.warmup_batches <= 0 or not rows:
        return
    warmup_rows = rows[: args.batch_size * args.warmup_batches]
    for batch in iter_batches(warmup_rows, args.batch_size):
        generate_batch(model, batch, args.max_new_tokens, args.use_general_query)
    torch.cuda.empty_cache()
    gc.collect()


def evaluate_dataset(
    model: CoMem,
    cr: int,
    dataset: str,
    rows: List[Dict],
    args: argparse.Namespace,
    prediction_dir: Path,
) -> Dict[str, float]:
    total_acc = 0.0
    total_f1 = 0.0
    total_time = 0.0
    compress_time = 0.0
    generate_time = 0.0
    total_samples = 0
    scored_samples = 0
    batch_size = args.batch_size

    pred_file = None
    if args.save_predictions:
        pred_file = (prediction_dir / f"cr{cr}_{dataset}_predictions.jsonl").open("w", encoding="utf-8")

    timed_start = max(0, args.repeat_runs - args.timed_runs)
    progress = tqdm(total=len(rows) * args.repeat_runs, desc=f"CR{cr} {dataset}", leave=False)
    try:
        for repeat_idx in range(args.repeat_runs):
            is_timed = repeat_idx >= timed_start
            index = 0
            while index < len(rows):
                batch = rows[index : index + batch_size]
                try:
                    outputs, timing = generate_batch(model, batch, args.max_new_tokens, args.use_general_query)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    print(f"OOM on CR{cr} {dataset}; retrying with batch_size={batch_size}")
                    continue

                if repeat_idx == args.repeat_runs - 1:
                    for row, output in zip(batch, outputs):
                        original = row["original_data"]
                        answer = original.get("golden_answers", original.get("answer"))
                        if isinstance(answer, list):
                            answer = answer[0]
                        total_acc += float(cover_exact_match(output, answer))
                        total_f1 += f1_score(output, answer)
                        scored_samples += 1
                        if pred_file is not None:
                            pred_file.write(
                                json.dumps(
                                    {
                                        "dataset": dataset,
                                        "compression_ratio": cr,
                                        "question": row["question"],
                                        "answer": answer,
                                        OUTPUT_FIELD: output,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                if is_timed:
                    total_time += timing["total_seconds"]
                    compress_time += timing["compress_seconds"]
                    generate_time += timing["generate_seconds"]
                    total_samples += len(batch)

                index += len(batch)
                progress.update(len(batch))
    finally:
        progress.close()
        if pred_file is not None:
            pred_file.close()

    return {
        "acc": total_acc / max(scored_samples, 1),
        "f1": total_f1 / max(scored_samples, 1),
        "samples": total_samples,
        "scored_samples": scored_samples,
        "repeat_runs": args.repeat_runs,
        "timed_runs": args.timed_runs,
        "effective_batch_size": batch_size,
        "seconds": total_time,
        "samples_per_second": total_samples / total_time if total_time > 0 else 0.0,
        "seconds_per_sample": total_time / max(total_samples, 1),
        "compress_seconds": compress_time,
        "generate_seconds": generate_time,
    }


def load_saved_acc(model_path: str, dataset: str) -> float | None:
    metrics_path = Path(model_path) / f"{dataset}_stage1_True_5_metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics.get("acc")


def plot_summary(records: List[Dict], output_dir: Path) -> None:
    records = sorted(records, key=lambda row: row["compression_ratio"])
    crs = [row["compression_ratio"] for row in records]
    accs = [row["avg_acc"] for row in records]
    speeds = [row["samples_per_second"] for row in records]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    scatter = ax.scatter(speeds, accs, c=crs, cmap="viridis", s=90, edgecolor="black", linewidth=0.8, zorder=3)
    ax.plot(speeds, accs, color="#555555", linewidth=1.4, alpha=0.7, zorder=2)

    for cr, speed, acc in zip(crs, speeds, accs):
        ax.annotate(f"CR{cr}", (speed, acc), xytext=(6, 6), textcoords="offset points", fontsize=10)

    ax.set_xlabel("Efficiency (samples / second)")
    ax.set_ylabel("Performance (average accuracy)")
    ax.set_title("CoMem Performance-Efficiency Trade-off")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Compression ratio")
    fig.tight_layout()
    fig.savefig(output_dir / "performance_efficiency.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "performance_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_only(records: List[Dict], output_dir: Path, baseline: Dict | None = None) -> None:
    comem_records = sorted(records, key=lambda row: row["compression_ratio"])
    labels = []
    speeds = []
    colors = []

    if baseline is not None:
        labels.append("Baseline")
        speeds.append(baseline["samples_per_second"])
        colors.append("#7f7f7f")

    for row in comem_records:
        labels.append(f"CR{row['compression_ratio']}")
        speeds.append(row["samples_per_second"])
        colors.append("#4C78A8")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(labels, speeds, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Efficiency (samples / second)")
    ax.set_xlabel("")
    ax.set_title("Inference Efficiency")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    for bar, speed in zip(bars, speeds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{speed:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "efficiency_only_with_baseline.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "efficiency_only_with_baseline.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    model_paths = parse_models(args.models)
    if not args.baseline_only and not model_paths:
        raise SystemExit("Provide --models CR=checkpoint_path,... for CoMem benchmarking.")
    gold_retrieval = not args.no_gold_retrieval
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    sample_limit = args.max_samples_per_dataset or args.batch_size
    print(f"Datasets: {datasets}")
    print(f"Samples per dataset: {sample_limit}")
    print(f"Compression ratios: {list(model_paths)}")

    data_by_dataset = {
        dataset: load_dataset_slice(dataset, gold_retrieval, args.generation_top_k, sample_limit)
        for dataset in datasets
    }

    if args.baseline_only:
        summary_path = output_dir / "performance_efficiency_summary.json"
        with summary_path.open("r", encoding="utf-8") as f:
            summary_records = json.load(f)
        baseline_record = benchmark_baseline(data_by_dataset[datasets[0]], args)
        baseline_path = output_dir / "baseline_efficiency_summary.json"
        with baseline_path.open("w", encoding="utf-8") as f:
            json.dump(baseline_record, f, indent=2, ensure_ascii=False)
        plot_efficiency_only(summary_records, output_dir, baseline_record)
        print(f"Saved baseline summary to {baseline_path}")
        print(f"Saved efficiency-only figure to {output_dir / 'efficiency_only_with_baseline.png'}")
        return

    summary_records = []
    for cr, model_path in model_paths.items():
        print(f"\nLoading CR{cr}: {model_path}")
        model = load_model(model_path, args.generation_top_k, args.attn_implementation)
        first_dataset = datasets[0]
        warmup_model(model, data_by_dataset[first_dataset], args)

        dataset_metrics = {}
        total_samples = 0
        total_seconds = 0.0
        for dataset in datasets:
            metrics = evaluate_dataset(model, cr, dataset, data_by_dataset[dataset], args, prediction_dir)
            saved_acc = load_saved_acc(model_path, dataset)
            if saved_acc is not None:
                metrics["saved_full_eval_acc"] = saved_acc
            dataset_metrics[dataset] = metrics
            total_samples += metrics["samples"]
            total_seconds += metrics["seconds"]
            print(
                f"CR{cr} {dataset}: acc={metrics['acc']:.4f}, "
                f"speed={metrics['samples_per_second']:.3f} samples/s"
            )

        avg_acc = sum(metric["acc"] for metric in dataset_metrics.values()) / len(dataset_metrics)
        avg_f1 = sum(metric["f1"] for metric in dataset_metrics.values()) / len(dataset_metrics)
        saved_accs = [metric["saved_full_eval_acc"] for metric in dataset_metrics.values() if "saved_full_eval_acc" in metric]
        record = {
            "compression_ratio": cr,
            "model_path": model_path,
            "avg_acc": avg_acc,
            "avg_f1": avg_f1,
            "avg_saved_full_eval_acc": sum(saved_accs) / len(saved_accs) if saved_accs else None,
            "samples": total_samples,
            "seconds": total_seconds,
            "samples_per_second": total_samples / total_seconds if total_seconds > 0 else 0.0,
            "seconds_per_sample": total_seconds / max(total_samples, 1),
            "attn_implementation": args.attn_implementation,
            "datasets": dataset_metrics,
        }
        summary_records.append(record)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    summary_path = output_dir / "performance_efficiency_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)
    plot_summary(summary_records, output_dir)
    if args.include_baseline:
        baseline_record = benchmark_baseline(data_by_dataset[datasets[0]], args)
        baseline_path = output_dir / "baseline_efficiency_summary.json"
        with baseline_path.open("w", encoding="utf-8") as f:
            json.dump(baseline_record, f, indent=2, ensure_ascii=False)
        plot_efficiency_only(summary_records, output_dir, baseline_record)
    else:
        plot_efficiency_only(summary_records, output_dir)
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved figure to {output_dir / 'performance_efficiency.png'}")


if __name__ == "__main__":
    main()
