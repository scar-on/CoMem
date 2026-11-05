import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path("/mnt/share/zhangsikui_workspace/comem")
OUTPUT_DIR = ROOT / "save_dir/mem_regime_summary"

SUMMARY_PATHS = {
    "causal_simcse": ROOT
    / "save_dir/mem_information_eval_simcse_160k_vs_casual_intra_200/mem_information_summary.json",
    "bidir_simcse": ROOT
    / "save_dir/mem_information_eval_simcse_160k_vs_casual_intra_200/mem_information_summary.json",
}

SAMPLES_PATH = (
    ROOT
    / "save_dir/mem_information_eval_simcse_160k_vs_casual_intra_200/mem_information_samples.jsonl"
)

MODEL_KEYS = {
    "causal_simcse": "simcse_160k_casual",
    "bidir_simcse": "simcse_160k",
}

METRIC_DIRS = {
    "causal_simcse": ROOT / "save_dir/simcse_in_batch_contrast_160k_casual",
    "bidir_simcse": ROOT / "save_dir/simcse_in_batch_contrast_160k",
}

REGIMES = {
    "causal_simcse": {
        "label": "Casual",
        "color": "#d95f02",
        "marker": "s",
    },
    "bidir_simcse": {
        "label": "Non-casual",
        "color": "#1b4e8a",
        "marker": "o",
    },
}

DATASETS = ["musique", "2wiki", "hotpotqa", "nq"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (11.0, 5.0),
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.linewidth": 1.1,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_summary_metric(regime: str, metric: str) -> float:
    rows = load_json(SUMMARY_PATHS[regime])
    model_key = MODEL_KEYS[regime]
    values = [row[metric] for row in rows if row["model"] == model_key]
    return float(np.mean(values))


def mean_eval_metric(regime: str, metric: str) -> float:
    values = []
    for dataset in DATASETS:
        path = METRIC_DIRS[regime] / f"{dataset}_stage1_True_5_metrics.json"
        values.append(load_json(path)[metric])
    return float(np.mean(values))


def collect_values():
    values = {}
    for regime in REGIMES:
        values[regime] = {
            "dispersion": mean_summary_metric(regime, "spectral_entropy_norm"),
            "shared_direction": mean_summary_metric(regime, "mem_pairwise_abs_cosine_mean"),
            "query_alignment": mean_summary_metric(regime, "query_doc_sim_max"),
            "qa_acc": mean_eval_metric(regime, "acc"),
        }
    return values


def mean_cosine_matrix(regime: str) -> np.ndarray:
    rows = load_jsonl(SAMPLES_PATH)
    model_key = MODEL_KEYS[regime]
    matrices = [row["mem_cosine_matrix"] for row in rows if row["model"] == model_key]
    return np.mean(np.asarray(matrices, dtype=float), axis=0)


def plot_panel(
    ax,
    values,
    metric_key: str,
    title: str,
    ylabel: str,
    panel_label: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    labels = list(REGIMES.keys())
    x = np.arange(len(labels))
    heights = [values[label][metric_key] for label in labels]
    colors = [REGIMES[label]["color"] for label in labels]
    markers = [REGIMES[label]["marker"] for label in labels]

    ax.bar(x, heights, color=colors, alpha=0.88, width=0.62)
    for xpos, height, color, marker in zip(x, heights, colors, markers):
        ax.plot(
            xpos,
            height,
            marker=marker,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.3,
            markeredgecolor=color,
            color=color,
        )
        ax.text(xpos, height + max(heights) * 0.025, f"{height:.3f}", ha="center", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([REGIMES[label]["label"] for label in labels], rotation=0, ha="center")
    ax.set_title(f"{panel_label} {title}")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.36)
    if ylim is None:
        ax.set_ylim(0, max(heights) * 1.18)
    else:
        ax.set_ylim(*ylim)


def plot_heatmap(ax, matrix: np.ndarray, regime: str, panel_label: str):
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title(REGIMES[regime]["label"], fontsize=13)
    ax.set_anchor("W")
    ax.set_xticks(range(matrix.shape[0]))
    ax.set_yticks(range(matrix.shape[0]))
    endpoint_labels = [f"M{i}" if i in (0, matrix.shape[0] - 1) else "" for i in range(matrix.shape[0])]
    ax.set_xticklabels(endpoint_labels, fontsize=10)
    ax.set_yticklabels(endpoint_labels, fontsize=10)
    return im


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    values = collect_values()
    with (OUTPUT_DIR / "three_mem_regimes_values.json").open("w") as f:
        json.dump(values, f, indent=2)

    fig, axes = plt.subplots(
        2,
        3,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.70]},
    )
    plot_panel(
        axes[0, 0],
        values,
        "dispersion",
        "Mem dispersion",
        "Spectral entropy",
        "(a)",
        ylim=(0, 1.0),
    )
    plot_panel(
        axes[0, 1],
        values,
        "shared_direction",
        "Shared mem direction",
        "",
        "(b)",
        ylim=(0, 0.18),
    )
    plot_panel(
        axes[1, 0],
        values,
        "query_alignment",
        "Query alignment",
        "Best mem-query cosine",
        "(c)",
        ylim=(0, 0.055),
    )
    plot_panel(
        axes[1, 1],
        values,
        "qa_acc",
        "Task effectiveness",
        "",
        "(d)",
        ylim=(0, 0.46),
    )
    plot_heatmap(axes[0, 2], mean_cosine_matrix("causal_simcse"), "causal_simcse", "(e)")
    plot_heatmap(axes[1, 2], mean_cosine_matrix("bidir_simcse"), "bidir_simcse", "(f)")

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.3,
            label=style["label"],
        )
        for style in REGIMES.values()
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.4,
        handlelength=2.0,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.10,
        top=0.86,
        wspace=0.24,
        hspace=0.58,
    )

    output_png = OUTPUT_DIR / "two_mem_regimes_with_heatmaps.png"
    output_pdf = OUTPUT_DIR / "two_mem_regimes_with_heatmaps.pdf"
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")
    print(f"Saved: {OUTPUT_DIR / 'three_mem_regimes_values.json'}")


if __name__ == "__main__":
    main()
