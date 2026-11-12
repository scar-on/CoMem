#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openrlhf.models.modeling_comem import CoMem  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "evaluation" / "evaluation_data" / "end_to_end_evaluation"


def parse_model_arg(value: str) -> Tuple[str, str]:
    if "=" not in value:
        path = value.rstrip("/")
        return os.path.basename(path), path
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def infer_training_stage(model_path: str, override: str) -> str:
    if override != "auto":
        return override
    config_path = Path(model_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("training_stage", "stage1")


def load_stage1_data(
    data_root: Path,
    dataset: str,
    generation_top_k: int,
    sample_limit: Optional[int],
    sample_seed: int,
    sample_strategy: str,
) -> List[Dict]:
    file_path = data_root / dataset / "eval_processed_with_pos.jsonl"
    data = []
    with file_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            data.append(
                {
                    "dataset": dataset,
                    "index": idx,
                    "question": item["question"],
                    "docs": item["docs"][:generation_top_k],
                    "pos_index": item.get("pos_index", []),
                    "answer": item.get("answer") or item.get("golden_answers"),
                }
            )

    if sample_limit is None or sample_limit >= len(data):
        return data

    if sample_strategy == "random":
        rng = random.Random(sample_seed)
        picked = sorted(rng.sample(range(len(data)), sample_limit))
        return [data[i] for i in picked]
    return data[:sample_limit]


def spectral_information(mem: torch.Tensor) -> Tuple[float, float, float, float]:
    """Return spectral entropy, normalized entropy, effective rank, token diversity."""
    mem = mem.float()
    n_tokens = mem.size(0)
    centered = mem - mem.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    total = energy.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0

    probs = energy / total
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    max_entropy = math.log(max(n_tokens, 2))
    normalized_entropy = entropy / max_entropy
    effective_rank = torch.exp(entropy)

    normed = F.normalize(mem, dim=-1)
    sim = torch.mm(normed, normed.t())
    off_diag = sim[~torch.eye(n_tokens, dtype=torch.bool, device=sim.device)]
    token_diversity = 1.0 - off_diag.mean()

    return (
        float(entropy.item()),
        float(normalized_entropy.item()),
        float(effective_rank.item()),
        float(token_diversity.item()),
    )


def intra_mem_statistics(mem: torch.Tensor) -> Dict:
    """Measure redundancy and information allocation across memory tokens."""
    mem = mem.float()
    n_tokens = mem.size(0)
    normed = F.normalize(mem, dim=-1)
    cosine_matrix = torch.mm(normed, normed.t())
    off_diag = cosine_matrix[~torch.eye(n_tokens, dtype=torch.bool, device=mem.device)]

    centered = mem - mem.mean(dim=0, keepdim=True)
    energy = centered.square().sum(dim=-1)
    total_energy = energy.sum()
    if total_energy <= 0:
        energy_share = torch.full_like(energy, 1.0 / max(n_tokens, 1))
    else:
        energy_share = energy / total_energy

    energy_entropy = -(energy_share * torch.log(energy_share.clamp_min(1e-12))).sum()
    energy_entropy_norm = energy_entropy / math.log(max(n_tokens, 2))
    sorted_share = torch.sort(energy_share).values
    gini_weights = torch.arange(1, n_tokens + 1, device=mem.device, dtype=energy_share.dtype)
    energy_gini = (2 * (gini_weights * sorted_share).sum() / (n_tokens * sorted_share.sum())) - (
        n_tokens + 1
    ) / n_tokens

    return {
        "mem_pairwise_cosine_mean": float(off_diag.mean().item()),
        "mem_pairwise_abs_cosine_mean": float(off_diag.abs().mean().item()),
        "mem_pairwise_cosine_std": float(off_diag.std(unbiased=False).item()),
        "mem_energy_entropy_norm": float(energy_entropy_norm.item()),
        "mem_energy_gini": float(energy_gini.item()),
        "mem_energy_max_share": float(energy_share.max().item()),
        "mem_cosine_matrix": cosine_matrix.cpu().numpy().tolist(),
        "mem_energy_share": energy_share.cpu().numpy().tolist(),
    }


def query_embedding(model: CoMem, question: str, device: torch.device) -> torch.Tensor:
    tokenizer = model.decoder_tokenizer
    encoded = tokenizer(
        question,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=256,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    embeddings = model.decoder.get_input_embeddings()(input_ids).float()
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled.squeeze(0), dim=0)


def relevance_metrics(
    mem_embeddings: torch.Tensor,
    q_emb: torch.Tensor,
    pos_index: Iterable[int],
    temperature: float,
) -> Dict[str, Optional[float]]:
    doc_vecs = F.normalize(mem_embeddings.float().mean(dim=1), dim=-1)
    token_vecs = F.normalize(mem_embeddings.float().reshape(-1, mem_embeddings.size(-1)), dim=-1)

    doc_sims = torch.mv(doc_vecs, q_emb)
    token_sims = torch.mv(token_vecs, q_emb)
    weights = F.softmax(doc_sims / temperature, dim=0)
    focus_entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum()
    max_focus_entropy = math.log(max(mem_embeddings.size(0), 2))
    focus_info_gain = 1.0 - focus_entropy / max_focus_entropy

    valid_pos = sorted({int(i) for i in pos_index if int(i) < mem_embeddings.size(0)})
    ranked = torch.argsort(doc_sims, descending=True).tolist()

    pos_gap = None
    recall_at_1 = None
    recall_at_3 = None
    if valid_pos:
        pos_mask = torch.zeros_like(doc_sims, dtype=torch.bool)
        pos_mask[valid_pos] = True
        neg_mask = ~pos_mask
        if neg_mask.any():
            pos_gap = float(doc_sims[pos_mask].mean().item() - doc_sims[neg_mask].mean().item())
        recall_at_1 = len(set(ranked[:1]) & set(valid_pos)) / len(valid_pos)
        recall_at_3 = len(set(ranked[: min(3, len(ranked))]) & set(valid_pos)) / len(valid_pos)

    return {
        "query_doc_sim_mean": float(doc_sims.mean().item()),
        "query_doc_sim_max": float(doc_sims.max().item()),
        "query_token_sim_mean": float(token_sims.mean().item()),
        "query_token_sim_max": float(token_sims.max().item()),
        "query_focus_entropy": float(focus_entropy.item()),
        "query_focus_info_gain": float(focus_info_gain.item()),
        "pos_neg_query_gap": pos_gap,
        "query_recall_at_1": recall_at_1,
        "query_recall_at_3": recall_at_3,
    }


def evaluate_model(
    model_name: str,
    model_path: str,
    training_stage: str,
    datasets: List[str],
    args: argparse.Namespace,
    completed_keys: Optional[set] = None,
) -> List[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} from {model_path} as {training_stage} on {device}")
    model = CoMem.from_pretrained(
        model_path,
        training_stage=training_stage,
        generation_top_k=args.generation_top_k,
        pure_inference=True,
    )
    model.to(device)
    model.eval()

    records = []
    completed_keys = completed_keys or set()
    try:
        for dataset in datasets:
            samples = load_stage1_data(
                data_root=Path(args.data_root),
                dataset=dataset,
                generation_top_k=args.generation_top_k,
                sample_limit=args.sample_limit,
                sample_seed=args.sample_seed,
                sample_strategy=args.sample_strategy,
            )
            pbar = tqdm(samples, desc=f"{model_name}:{dataset}")
            for sample in pbar:
                record_key = (model_name, dataset, sample["index"])
                if record_key in completed_keys:
                    continue
                question = sample["question"]
                docs = sample["docs"]
                with torch.no_grad():
                    mem_embeddings, _ = model.compress_documents(
                        documents=docs,
                        q_texts=[question] * len(docs),
                    )
                    mem_embeddings = mem_embeddings.detach().float()
                    q_emb = query_embedding(model, question, device=device)

                    doc_stats = [spectral_information(mem_embeddings[i]) for i in range(len(docs))]
                    intra_stats = [intra_mem_statistics(mem_embeddings[i]) for i in range(len(docs))]
                    rel = relevance_metrics(
                        mem_embeddings=mem_embeddings,
                        q_emb=q_emb,
                        pos_index=sample["pos_index"],
                        temperature=args.softmax_temperature,
                    )

                entropy, norm_entropy, eff_rank, diversity = map(np.asarray, zip(*doc_stats))
                mem_cosine_matrix = np.mean(
                    [item["mem_cosine_matrix"] for item in intra_stats], axis=0
                )
                mem_energy_share = np.mean(
                    [item["mem_energy_share"] for item in intra_stats], axis=0
                )
                tokenized = model.decoder_tokenizer(
                    docs,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=model.doc_max_length,
                )
                input_token_count = sum(len(ids) for ids in tokenized["input_ids"])
                mem_token_count = int(mem_embeddings.size(0) * mem_embeddings.size(1))

                record = {
                    "model": model_name,
                    "model_path": model_path,
                    "training_stage": training_stage,
                    "dataset": dataset,
                    "index": sample["index"],
                    "n_docs": len(docs),
                    "input_token_count": input_token_count,
                    "mem_token_count": mem_token_count,
                    "token_compression_ratio": mem_token_count / max(input_token_count, 1),
                    "spectral_entropy": float(entropy.mean()),
                    "spectral_entropy_norm": float(norm_entropy.mean()),
                    "effective_rank": float(eff_rank.mean()),
                    "effective_rank_norm": float(eff_rank.mean() / max(mem_embeddings.size(1), 1)),
                    "token_diversity": float(diversity.mean()),
                    "mem_pairwise_cosine_mean": float(
                        np.mean([item["mem_pairwise_cosine_mean"] for item in intra_stats])
                    ),
                    "mem_pairwise_abs_cosine_mean": float(
                        np.mean([item["mem_pairwise_abs_cosine_mean"] for item in intra_stats])
                    ),
                    "mem_pairwise_cosine_std": float(
                        np.mean([item["mem_pairwise_cosine_std"] for item in intra_stats])
                    ),
                    "mem_energy_entropy_norm": float(
                        np.mean([item["mem_energy_entropy_norm"] for item in intra_stats])
                    ),
                    "mem_energy_gini": float(np.mean([item["mem_energy_gini"] for item in intra_stats])),
                    "mem_energy_max_share": float(
                        np.mean([item["mem_energy_max_share"] for item in intra_stats])
                    ),
                    "mem_energy_share": mem_energy_share.tolist(),
                    "mem_cosine_matrix": mem_cosine_matrix.tolist(),
                    **rel,
                }
                for position, share in enumerate(mem_energy_share):
                    record[f"mem_energy_share_{position}"] = float(share)
                records.append(record)
                completed_keys.add(record_key)
                if args.save_per_sample:
                    append_jsonl(args.sample_output, record)

                del mem_embeddings, q_emb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records


def mean_optional(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def summarize(records: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["dataset"])].append(record)

    metric_names = [
        "input_token_count",
        "mem_token_count",
        "token_compression_ratio",
        "spectral_entropy",
        "spectral_entropy_norm",
        "effective_rank",
        "effective_rank_norm",
        "token_diversity",
        "mem_pairwise_cosine_mean",
        "mem_pairwise_abs_cosine_mean",
        "mem_pairwise_cosine_std",
        "mem_energy_entropy_norm",
        "mem_energy_gini",
        "mem_energy_max_share",
        "mem_energy_share_0",
        "mem_energy_share_1",
        "mem_energy_share_2",
        "mem_energy_share_3",
        "mem_energy_share_4",
        "mem_energy_share_5",
        "mem_energy_share_6",
        "mem_energy_share_7",
        "query_doc_sim_mean",
        "query_doc_sim_max",
        "query_token_sim_mean",
        "query_token_sim_max",
        "query_focus_entropy",
        "query_focus_info_gain",
        "pos_neg_query_gap",
        "query_recall_at_1",
        "query_recall_at_3",
    ]

    rows = []
    for (model, dataset), items in sorted(grouped.items()):
        row = {
            "model": model,
            "dataset": dataset,
            "sample_count": len(items),
            "training_stage": items[0].get("training_stage"),
        }
        for metric in metric_names:
            row[metric] = mean_optional([item.get(metric) for item in items])
        rows.append(row)
    return rows


def write_jsonl(path: str, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl(path: str) -> List[Dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_csv(path: str, rows: List[Dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: List[Dict], output_dir: str) -> None:
    metrics = [
        ("effective_rank_norm", "Mem effective rank (normalized)"),
        ("spectral_entropy_norm", "Mem spectral entropy (normalized)"),
        ("query_doc_sim_max", "Best doc mem-query cosine"),
        ("pos_neg_query_gap", "Positive-negative relevance gap"),
        ("query_focus_info_gain", "Query focus information gain"),
        ("token_diversity", "Mem token diversity"),
    ]

    models = sorted({row["model"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})
    x = np.arange(len(datasets))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for i, model in enumerate(models):
            values = []
            for dataset in datasets:
                match = next(
                    (row for row in rows if row["model"] == model and row["dataset"] == dataset),
                    None,
                )
                values.append(np.nan if match is None or match.get(metric) is None else match[metric])
            ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width, label=model)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20)
        ax.grid(axis="y", alpha=0.25)

    axes.flat[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mem_information_summary.png"), dpi=220)
    plt.close(fig)


def plot_scatter(records: List[Dict], output_dir: str) -> None:
    models = sorted({record["model"] for record in records})
    fig, ax = plt.subplots(figsize=(9, 7))
    for model in models:
        items = [record for record in records if record["model"] == model]
        ax.scatter(
            [item["effective_rank_norm"] for item in items],
            [item["query_doc_sim_max"] for item in items],
            s=12,
            alpha=0.25,
            label=model,
        )
    ax.set_xlabel("Mem effective rank (normalized)")
    ax.set_ylabel("Best doc mem-query cosine")
    ax.set_title("Compressed Information vs Query Relevance")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mem_information_scatter.png"), dpi=220)
    plt.close(fig)


def plot_intra_mem(records: List[Dict], rows: List[Dict], output_dir: str) -> None:
    models = sorted({record["model"] for record in records})
    datasets = sorted({record["dataset"] for record in records})
    n_models = len(models)
    n_datasets = len(datasets)

    fig, axes = plt.subplots(
        n_models,
        n_datasets,
        figsize=(4.2 * n_datasets, 3.8 * n_models),
        squeeze=False,
    )
    im = None
    for row_idx, model in enumerate(models):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            items = [
                record
                for record in records
                if record["model"] == model and record["dataset"] == dataset
            ]
            matrix = np.mean([item["mem_cosine_matrix"] for item in items], axis=0)
            im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"{model}\n{dataset}")
            ax.set_xticks(range(matrix.shape[0]))
            ax.set_yticks(range(matrix.shape[0]))
            ax.set_xticklabels([f"M{i}" for i in range(matrix.shape[0])], fontsize=8)
            ax.set_yticklabels([f"M{i}" for i in range(matrix.shape[0])], fontsize=8)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Cosine similarity")
    fig.suptitle("Average intra-mem-token cosine similarity", y=1.02)
    fig.savefig(
        os.path.join(output_dir, "intra_mem_cosine_heatmaps.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(
        n_models,
        n_datasets,
        figsize=(4.2 * n_datasets, 3.2 * n_models),
        squeeze=False,
    )
    for row_idx, model in enumerate(models):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            items = [
                record
                for record in records
                if record["model"] == model and record["dataset"] == dataset
            ]
            shares = np.mean([item["mem_energy_share"] for item in items], axis=0)
            ax.bar(np.arange(len(shares)), shares)
            ax.axhline(1.0 / len(shares), color="red", linestyle="--", linewidth=1, label="uniform")
            ax.set_ylim(0, max(0.35, float(shares.max()) * 1.2))
            ax.set_title(f"{model}\n{dataset}")
            ax.set_xticks(range(len(shares)))
            ax.set_xticklabels([f"M{i}" for i in range(len(shares))], fontsize=8)
            ax.grid(axis="y", alpha=0.25)
    axes[0][0].legend(loc="best")
    fig.suptitle("Average per-position mem-token energy share", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "intra_mem_energy_shares.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    metrics = [
        ("mem_pairwise_cosine_mean", "Mean pairwise cosine"),
        ("mem_pairwise_abs_cosine_mean", "Mean abs pairwise cosine"),
        ("mem_energy_entropy_norm", "Energy entropy"),
        ("mem_energy_max_share", "Max token energy share"),
    ]
    x = np.arange(len(datasets))
    width = 0.8 / max(n_models, 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for i, model in enumerate(models):
            values = []
            for dataset in datasets:
                match = next(
                    (row for row in rows if row["model"] == model and row["dataset"] == dataset),
                    None,
                )
                values.append(np.nan if match is None or match.get(metric) is None else match[metric])
            ax.bar(x + (i - (n_models - 1) / 2) * width, values, width, label=model)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20)
        ax.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "intra_mem_redundancy_summary.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate information carried by CoMem memory embeddings."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as name=/path/to/checkpoint. Can be repeated.",
    )
    parser.add_argument("--datasets", default="musique,2wiki,hotpotqa,nq")
    parser.add_argument("--generation_top_k", type=int, default=5)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_strategy", choices=["first", "random"], default="first")
    parser.add_argument("--softmax_temperature", type=float, default=0.1)
    parser.add_argument("--training_stage", default="auto", choices=["auto", "stage1", "stage1_2"])
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--output_dir",
        default="/user/zhangsikui/a_qcc/save_dir/mem_information_eval",
    )
    parser.add_argument("--save_per_sample", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    model_specs = [parse_model_arg(item) for item in args.model]

    all_records = []
    for model_name, model_path in model_specs:
        training_stage = infer_training_stage(model_path, args.training_stage)
        all_records.extend(evaluate_model(model_name, model_path, training_stage, datasets, args))

    summary_rows = summarize(all_records)
    summary_json = os.path.join(args.output_dir, "mem_information_summary.json")
    summary_csv = os.path.join(args.output_dir, "mem_information_summary.csv")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    write_csv(summary_csv, summary_rows)

    if args.save_per_sample:
        write_jsonl(os.path.join(args.output_dir, "mem_information_samples.jsonl"), all_records)

    plot_summary(summary_rows, args.output_dir)
    plot_scatter(all_records, args.output_dir)
    plot_intra_mem(all_records, summary_rows, args.output_dir)

    print(f"Summary saved to {summary_json}")
    print(f"CSV saved to {summary_csv}")
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
