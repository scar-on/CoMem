import argparse
import csv
import json
import os
import re
import string
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = defaultdict(int)
    for token in gt_tokens:
        common[token] += 1

    overlap = 0
    for token in pred_tokens:
        if common[token] > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def cover_acc(prediction: str, ground_truth: str) -> float:
    pred_tokens = set(normalize_answer(prediction).split())
    gt_tokens = normalize_answer(ground_truth).split()
    return float(bool(gt_tokens) and all(token in pred_tokens for token in gt_tokens))


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)

    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end
    return ranks


def corr(x_values: List[float], y_values: List[float], method: str) -> Optional[float]:
    if len(x_values) < 2:
        return None

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None

    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def finite_mean(values: List[Optional[float]]) -> Optional[float]:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    return float(np.mean(array))


def build_prediction_index(model_path: str, datasets: List[str]) -> Dict:
    predictions = {}
    for dataset in datasets:
        path = os.path.join(model_path, f"{dataset}_stage1_True_5.jsonl")
        for index, row in enumerate(read_jsonl(path)):
            predictions[(dataset, index)] = row
    return predictions


def attach_success_metrics(mem_rows: List[Dict], datasets: List[str]) -> List[Dict]:
    by_model_path = {}
    for row in mem_rows:
        model_path = row["model_path"]
        if model_path not in by_model_path:
            by_model_path[model_path] = build_prediction_index(model_path, datasets)

    merged = []
    for row in mem_rows:
        pred_row = by_model_path[row["model_path"]][(row["dataset"], row["index"])]
        answer = pred_row.get("answer") or pred_row.get("golden_answers")
        if isinstance(answer, list):
            answer = answer[0]
        prediction = pred_row.get("CoMem_normal_output", "")

        out = dict(row)
        out["prediction"] = prediction
        out["answer"] = answer
        out["qa_f1"] = f1_score(prediction, answer)
        out["qa_em"] = exact_match(prediction, answer)
        out["qa_acc"] = cover_acc(prediction, answer)
        out["qa_correct"] = out["qa_acc"]
        merged.append(out)
    return merged


def summarize_correlations(rows: List[Dict], metrics: List[str], targets: List[str]) -> List[Dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
        groups["ALL"].append(row)

    summary = []
    for group_name, items in sorted(groups.items()):
        for metric in metrics:
            for target in targets:
                x_values = [item.get(metric) for item in items]
                y_values = [item.get(target) for item in items]
                summary.append(
                    {
                        "group": group_name,
                        "metric": metric,
                        "target": target,
                        "n": len(items),
                        "pearson": corr(x_values, y_values, "pearson"),
                        "spearman": corr(x_values, y_values, "spearman"),
                        "metric_mean": finite_mean(x_values),
                        "target_mean": finite_mean(y_values),
                    }
                )
    return summary


def write_csv(path: str, rows: List[Dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_correlation_bars(summary: List[Dict], output_dir: str, target: str) -> None:
    rows = [row for row in summary if row["group"] == "ALL" and row["target"] == target]
    metrics = [row["metric"] for row in rows]
    pearson = [np.nan if row["pearson"] is None else row["pearson"] for row in rows]
    spearman = [np.nan if row["spearman"] is None else row["spearman"] for row in rows]

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 0.2, pearson, 0.4, label="Pearson")
    ax.bar(x + 0.2, spearman, 0.4, label="Spearman")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim(-1, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_title(f"Correlation between mem metrics and {target}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"correlation_bars_{target}.png"), dpi=220)
    plt.close(fig)


def plot_scatter(rows: List[Dict], metrics: List[str], output_dir: str, target: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, metric in zip(axes.flat, metrics[:6]):
        for model in sorted({row["model"] for row in rows}):
            items = [row for row in rows if row["model"] == model]
            ax.scatter(
                [item[metric] for item in items],
                [item[target] for item in items],
                s=35,
                alpha=0.8,
                label=model,
            )
        ax.set_xlabel(metric)
        ax.set_ylabel(target)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"metric_vs_{target}_scatter.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_samples", required=True)
    parser.add_argument("--datasets", default="musique,2wiki,hotpotqa,nq")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    mem_rows = read_jsonl(args.mem_samples)
    merged = attach_success_metrics(mem_rows, datasets)

    metrics = [
        "effective_rank_norm",
        "spectral_entropy_norm",
        "token_diversity",
        "mem_pairwise_abs_cosine_mean",
        "mem_energy_entropy_norm",
        "mem_energy_max_share",
        "query_doc_sim_max",
        "pos_neg_query_gap",
        "query_focus_info_gain",
    ]
    targets = ["qa_f1", "qa_acc", "qa_em"]
    summary = summarize_correlations(merged, metrics, targets)

    merged_path = os.path.join(args.output_dir, "mem_metric_success_samples.jsonl")
    summary_json = os.path.join(args.output_dir, "mem_metric_success_correlation.json")
    summary_csv = os.path.join(args.output_dir, "mem_metric_success_correlation.csv")

    write_jsonl(merged_path, merged)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_csv(summary_csv, summary)

    plot_correlation_bars(summary, args.output_dir, "qa_f1")
    plot_correlation_bars(summary, args.output_dir, "qa_acc")
    plot_scatter(merged, metrics, args.output_dir, "qa_f1")

    print(f"Merged samples saved to {merged_path}")
    print(f"Correlation summary saved to {summary_json}")
    print(f"Correlation CSV saved to {summary_csv}")
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
