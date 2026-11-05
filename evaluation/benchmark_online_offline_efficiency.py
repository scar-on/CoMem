#!/usr/bin/env python
"""Online/offline efficiency benchmark with FlashAttention and causal memory tokens."""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _preparse_gpu() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default="2")
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


_preparse_gpu()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openrlhf.models.modeling_clara import CLaRa  # noqa: E402


DEFAULT_MODELS = {
    4: "/user/zhangsikui/a_qcc/save_dir/stage2_clara_inst_simcse_0.01_cr4",
    8: "/user/zhangsikui/a_qcc/save_dir/stage2_clara_inst_simcse_0.01_cr8",
    16: "/user/zhangsikui/a_qcc/save_dir/stage2_clara_inst_simcse_0.01_cr16",
    32: "/user/zhangsikui/a_qcc/save_dir/stage2_clara_inst_lr1e-4",
    64: "/user/zhangsikui/a_qcc/save_dir/stage2_clara_inst_simcse_0.01_cr64",
}

SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to extract relevant "
    "information from provided documents and to answer to questions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build online/offline efficiency tables.")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--dataset", default="nq")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--offline_batch_size", type=int, default=None)
    parser.add_argument("--cache_batch_size", type=int, default=64)
    parser.add_argument("--offline_only", action="store_true")
    parser.add_argument("--repeat_runs", type=int, default=5)
    parser.add_argument("--timed_runs", type=int, default=3)
    parser.add_argument("--doc_tokens", type=int, default=256)
    parser.add_argument("--query_tokens", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--baseline_model_path", default="/user/zhangsikui/a_qcc/data/model/Qwen_Qwen3-4B-Instruct-2507")
    parser.add_argument("--output_dir", default="/user/zhangsikui/a_qcc/save_dir/online_offline_efficiency")
    return parser.parse_args()


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


def load_rows(dataset: str, top_k: int, limit: int) -> List[Dict]:
    path = (
        Path(__file__).resolve().parent
        / "evaluation_data"
        / "end_to_end_evaluation"
        / dataset
        / "eval_processed_with_pos.jsonl"
    )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            item = json.loads(line)
            rows.append(
                {
                    "question": item["question"],
                    "documents": item["docs"][:top_k],
                    "global_index": index,
                }
            )
            if len(rows) >= limit:
                break
    return rows


def parse_models() -> Dict[int, str]:
    return dict(DEFAULT_MODELS)


def set_attention_backend(model, attn_implementation: str) -> None:
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = attn_implementation
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = attn_implementation


def load_clara(model_path: str, top_k: int, attn_implementation: str) -> CLaRa:
    model = CLaRa.from_pretrained(
        model_path,
        training_stage="stage1_2",
        generation_top_k=top_k,
        pure_inference=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    set_attention_backend(model.decoder, attn_implementation)
    return model


def load_baseline(model_path: str, attn_implementation: str):
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


def causal_compress_documents(model: CLaRa, enc_input_ids: torch.Tensor, enc_attention_mask: torch.Tensor) -> torch.Tensor:
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


def build_baseline_prompt(tokenizer, documents: List[str], question: str) -> str:
    docs_text = "\n\n".join(documents)
    user_content = f"Background:\n{docs_text}\n\nQuestion:{question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": SYSTEM_PROMPT + "\n" + user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


def prepare_clara_inputs(model: CLaRa, rows: List[Dict]):
    flat_docs = []
    flat_queries = []
    for row in rows:
        flat_docs.extend(row["documents"])
        flat_queries.extend([row["question"]] * len(row["documents"]))
    prepared = model._prepare_encoder_inputs(
        flat_docs,
        max_length=model.doc_max_length,
        q_texts=flat_queries,
    )
    return (
        prepared["input_ids"].to(model.decoder.device),
        prepared["attention_mask"].to(model.decoder.device),
    )


def decode_from_compressed(model: CLaRa, rows: List[Dict], compressed_documents: torch.Tensor, max_new_tokens: int):
    questions = [row["question"] for row in rows]
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
    output_ids = model.decoder.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=dec_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    del dec_input_ids, dec_attention_mask, inputs_embeds, output_ids


@torch.no_grad()
def run_clara_once(model: CLaRa, rows: List[Dict], max_new_tokens: int, mode: str, cached=None):
    if mode == "online":
        enc_input_ids, enc_attention_mask = prepare_clara_inputs(model, rows)
        compressed, compress_s = timed_cuda_call(
            lambda: causal_compress_documents(model, enc_input_ids, enc_attention_mask)
        )
        _, decode_s = timed_cuda_call(
            lambda: decode_from_compressed(model, rows, compressed, max_new_tokens)
        )
        del enc_input_ids, enc_attention_mask, compressed
        return compress_s + decode_s, compress_s, decode_s

    compressed = cached
    _, decode_s = timed_cuda_call(lambda: decode_from_compressed(model, rows, compressed, max_new_tokens))
    return decode_s, 0.0, decode_s


@torch.no_grad()
def make_offline_cache(model: CLaRa, rows: List[Dict]) -> torch.Tensor:
    enc_input_ids, enc_attention_mask = prepare_clara_inputs(model, rows)
    compressed = causal_compress_documents(model, enc_input_ids, enc_attention_mask)
    del enc_input_ids, enc_attention_mask
    sync_cuda()
    return compressed.detach()


def benchmark_clara(model: CLaRa, rows: List[Dict], args: argparse.Namespace, mode: str) -> Dict:
    timed_start = max(0, args.repeat_runs - args.timed_runs)
    total_s = 0.0
    comp_s = 0.0
    dec_s = 0.0
    samples = 0
    cached = make_offline_cache(model, rows) if mode == "offline" else None
    for repeat_idx in tqdm(range(args.repeat_runs), desc=f"CLaRa {mode}", leave=False):
        elapsed, comp, dec = run_clara_once(model, rows, args.max_new_tokens, mode, cached)
        if repeat_idx >= timed_start:
            total_s += elapsed
            comp_s += comp
            dec_s += dec
            samples += len(rows)
    if cached is not None:
        del cached
    return {
        "samples": samples,
        "seconds": total_s,
        "compress_seconds": comp_s,
        "decode_seconds": dec_s,
        "samples_per_second": samples / total_s if total_s > 0 else 0.0,
        "seconds_per_sample": total_s / max(samples, 1),
    }


@torch.no_grad()
def run_baseline_once(tokenizer, model, rows: List[Dict], max_new_tokens: int):
    prompts = [build_baseline_prompt(tokenizer, row["documents"], row["question"]) for row in rows]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)
    _, elapsed = timed_cuda_call(
        lambda: model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    )
    del inputs
    return elapsed


def benchmark_baseline(tokenizer, model, rows: List[Dict], args: argparse.Namespace) -> Dict:
    timed_start = max(0, args.repeat_runs - args.timed_runs)
    total_s = 0.0
    samples = 0
    for repeat_idx in tqdm(range(args.repeat_runs), desc="Baseline", leave=False):
        elapsed = run_baseline_once(tokenizer, model, rows, args.max_new_tokens)
        if repeat_idx >= timed_start:
            total_s += elapsed
            samples += len(rows)
    return {
        "samples": samples,
        "seconds": total_s,
        "samples_per_second": samples / total_s if total_s > 0 else 0.0,
        "seconds_per_sample": total_s / max(samples, 1),
    }


def chunk_rows(rows: List[Dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def benchmark_baseline_chunked(tokenizer, model, rows: List[Dict], args: argparse.Namespace, batch_size: int) -> Dict:
    timed_start = max(0, args.repeat_runs - args.timed_runs)
    total_s = 0.0
    samples = 0
    effective_bs = batch_size
    for repeat_idx in tqdm(range(args.repeat_runs), desc="Baseline", leave=False):
        for batch in chunk_rows(rows, effective_bs):
            while True:
                try:
                    elapsed = run_baseline_once(tokenizer, model, batch, args.max_new_tokens)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    if effective_bs == 1:
                        raise
                    effective_bs = max(1, effective_bs // 2)
                    print(f"OOM on baseline; retrying with batch_size={effective_bs}")
                    batch = batch[:effective_bs]
            if repeat_idx >= timed_start:
                total_s += elapsed
                samples += len(batch)
    return {
        "samples": samples,
        "seconds": total_s,
        "samples_per_second": samples / total_s if total_s > 0 else 0.0,
        "seconds_per_sample": total_s / max(samples, 1),
        "effective_batch_size": effective_bs,
    }


def benchmark_clara_offline_chunked(model: CLaRa, rows: List[Dict], args: argparse.Namespace, batch_size: int) -> Dict:
    timed_start = max(0, args.repeat_runs - args.timed_runs)
    total_s = 0.0
    samples = 0
    effective_bs = batch_size
    cache_by_index = []
    for batch in chunk_rows(rows, effective_bs):
        compressed_chunks = []
        for cache_batch in chunk_rows(batch, args.cache_batch_size):
            compressed_chunks.append(make_offline_cache(model, cache_batch))
        cache_by_index.append((batch, torch.cat(compressed_chunks, dim=0)))
        del compressed_chunks
    try:
        for repeat_idx in tqdm(range(args.repeat_runs), desc="CLaRa offline", leave=False):
            for batch, compressed in cache_by_index:
                _, elapsed = timed_cuda_call(lambda b=batch, c=compressed: decode_from_compressed(model, b, c, args.max_new_tokens))
                if repeat_idx >= timed_start:
                    total_s += elapsed
                    samples += len(batch)
    finally:
        for _, compressed in cache_by_index:
            del compressed
    return {
        "samples": samples,
        "seconds": total_s,
        "compress_seconds": 0.0,
        "decode_seconds": total_s,
        "samples_per_second": samples / total_s if total_s > 0 else 0.0,
        "seconds_per_sample": total_s / max(samples, 1),
        "effective_batch_size": effective_bs,
    }


def transformer_prefill_flops(layers: int, hidden: int, intermediate: int, kv_hidden: int, seq_len: int) -> float:
    linear = 4 * seq_len * hidden * hidden + 4 * seq_len * hidden * kv_hidden
    mlp = 6 * seq_len * hidden * intermediate
    attention = 2 * seq_len * (seq_len + 1) * hidden
    return layers * (linear + mlp + attention)


def transformer_decode_flops(layers: int, hidden: int, intermediate: int, kv_hidden: int, context_len: int, new_tokens: int) -> float:
    total = 0.0
    for step in range(new_tokens):
        c = context_len + step
        linear = 4 * hidden * hidden + 4 * hidden * kv_hidden
        mlp = 6 * hidden * intermediate
        attention = 4 * (c + 1) * hidden
        total += layers * (linear + mlp + attention)
    return total


def qwen4b_flops_baseline(args: argparse.Namespace) -> float:
    layers, hidden, intermediate, kv_hidden = 36, 2560, 9728, 8 * 128
    context = args.query_tokens + args.top_k * args.doc_tokens
    return transformer_prefill_flops(layers, hidden, intermediate, kv_hidden, context) + transformer_decode_flops(
        layers, hidden, intermediate, kv_hidden, context, args.max_new_tokens
    )


def qwen4b_flops_clara(args: argparse.Namespace, cr: int, mode: str) -> float:
    layers, hidden, intermediate, kv_hidden = 36, 2560, 9728, 8 * 128
    mem = args.doc_tokens // cr
    decoder_context = args.query_tokens + args.top_k * mem
    decode = transformer_prefill_flops(layers, hidden, intermediate, kv_hidden, decoder_context) + transformer_decode_flops(
        layers, hidden, intermediate, kv_hidden, decoder_context, args.max_new_tokens
    )
    if mode == "offline":
        return decode
    compress = args.top_k * transformer_prefill_flops(
        layers, hidden, intermediate, kv_hidden, args.doc_tokens + mem
    )
    return compress + decode


def write_outputs(records: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "online_offline_efficiency.json"
    csv_path = output_dir / "online_offline_efficiency.csv"
    tex_path = output_dir / "online_offline_efficiency_table.tex"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    keys = ["setting", "model", "compression_rate", "gflops", "time_s", "samples_per_second", "batch_size"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in keys})
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llrrrr}\\toprule\n")
        f.write("Setting & Model & CR & GFlops & Time (s) & Samples/s \\\\\\midrule\n")
        for row in records:
            f.write(
                f"{row['setting']} & {row['model']} & {row['compression_rate']} & "
                f"{row['gflops']:.0f} & {row['time_s']:.3f} & {row['samples_per_second']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {tex_path}")


def main() -> None:
    args = parse_args()
    offline_batch_size = args.offline_batch_size or args.batch_size
    sample_count = offline_batch_size if args.offline_only else args.batch_size
    rows = load_rows(args.dataset, args.top_k, sample_count)
    records = []
    output_dir = Path(args.output_dir)

    tokenizer, baseline_model = load_baseline(args.baseline_model_path, args.attn_implementation)
    baseline_metrics = (
        benchmark_baseline_chunked(tokenizer, baseline_model, rows, args, offline_batch_size)
        if args.offline_only
        else benchmark_baseline(tokenizer, baseline_model, rows, args)
    )
    baseline_flops = qwen4b_flops_baseline(args) / 1e9
    records.append(
        {
            "setting": "baseline",
            "model": "Qwen3-4B",
            "compression_rate": "-",
            "gflops": baseline_flops,
            "time_s": baseline_metrics["seconds_per_sample"],
            "samples_per_second": baseline_metrics["samples_per_second"],
            "batch_size": len(rows),
            "attn_implementation": args.attn_implementation,
            **baseline_metrics,
        }
    )
    del baseline_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    for cr, model_path in parse_models().items():
        print(f"\nLoading CR{cr}: {model_path}")
        model = load_clara(model_path, args.top_k, args.attn_implementation)
        for mode in ("online", "offline"):
            if args.offline_only and mode == "online":
                continue
            metrics = (
                benchmark_clara_offline_chunked(model, rows, args, offline_batch_size)
                if args.offline_only
                else benchmark_clara(model, rows, args, mode)
            )
            records.append(
                {
                    "setting": mode,
                    "model": "CLaRa-Qwen3-4B",
                    "compression_rate": cr,
                    "gflops": qwen4b_flops_clara(args, cr, mode) / 1e9,
                    "time_s": metrics["seconds_per_sample"],
                    "samples_per_second": metrics["samples_per_second"],
                    "batch_size": len(rows),
                    "attn_implementation": args.attn_implementation,
                    **metrics,
                }
            )
        del model
        torch.cuda.empty_cache()
        gc.collect()

    write_outputs(records, output_dir)


if __name__ == "__main__":
    main()
