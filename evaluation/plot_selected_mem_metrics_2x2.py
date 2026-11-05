import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


MODEL_STYLE = {
    "simcse_160k": {
        "label": "SimCSE-160K",
        "color": "#1b4e8a",
        "marker": "o",
    },
    "simcse_160k_casual": {
        "label": "SimCSE-160K-Causal",
        "color": "#d95f02",
        "marker": "s",
    },
}

DATASET_ORDER = ["musique", "2wiki", "hotpotqa", "nq"]

METRICS = [
    ("spectral_entropy_norm", "Spectral entropy", "Entropy"),
    ("mem_pairwise_abs_cosine_mean", "Pairwise abs cosine", "Pairwise abs"),
    ("mem_energy_max_share", "Max mem-token energy share", "Max energy"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10.5, 7.2),
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.linewidth": 1.1,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def summary_lookup(summary_rows):
    return {
        (row["model"], row["dataset"]): row
        for row in summary_rows
    }


def correlation_lookup(correlation_rows):
    return {
        row["metric"]: row
        for row in correlation_rows
        if row["group"] == "ALL" and row["target"] == "qa_acc"
    }


def plot_correlation_panel(ax, corr_rows) -> None:
    metric_keys = [metric[0] for metric in METRICS]
    labels = [metric[2] for metric in METRICS]
    pearson = [corr_rows[key]["pearson"] for key in metric_keys]
    spearman = [corr_rows[key]["spearman"] for key in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.34
    ax.bar(x - width / 2, pearson, width, color="#4c78a8", label="Pearson")
    ax.bar(x + width / 2, spearman, width, color="#f58518", label="Spearman")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(-0.45, 0.45)
    ax.set_ylabel("Correlation with QA Acc")
    ax.set_title("(a) Metric-accuracy correlation")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.36)
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6)

    for offset, values in [(-width / 2, pearson), (width / 2, spearman)]:
        for i, value in enumerate(values):
            va = "bottom" if value >= 0 else "top"
            y = value + (0.015 if value >= 0 else -0.015)
            ax.text(i + offset, y, f"{value:.2f}", ha="center", va=va, fontsize=8)


def plot_metric_panel(ax, lookup, metric_key: str, title: str, panel_label: str) -> None:
    x = np.arange(len(DATASET_ORDER))
    models = list(MODEL_STYLE.keys())
    width = 0.36

    for idx, model in enumerate(models):
        style = MODEL_STYLE[model]
        values = [lookup[(model, dataset)][metric_key] for dataset in DATASET_ORDER]
        xpos = x + (idx - 0.5) * width
        ax.bar(
            xpos,
            values,
            width,
            color=style["color"],
            alpha=0.92,
            label=style["label"],
        )
        ax.plot(
            xpos,
            values,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.7,
            markersize=4.6,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER, rotation=18, ha="right")
    ax.set_title(f"{panel_label} {title}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.36)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--correlation_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--filename", default="selected_mem_metrics_2x2")
    args = parser.parse_args()

    configure_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = load_json(Path(args.summary_json))
    correlation_rows = load_json(Path(args.correlation_json))
    summary = summary_lookup(summary_rows)
    correlations = correlation_lookup(correlation_rows)

    fig, axes = plt.subplots(2, 2)
    plot_correlation_panel(axes[0, 0], correlations)
    plot_metric_panel(
        axes[0, 1],
        summary,
        "spectral_entropy_norm",
        "Spectral entropy",
        "(b)",
    )
    plot_metric_panel(
        axes[1, 0],
        summary,
        "mem_pairwise_abs_cosine_mean",
        "Pairwise abs cosine",
        "(c)",
    )
    plot_metric_panel(
        axes[1, 1],
        summary,
        "mem_energy_max_share",
        "Max mem-token energy share",
        "(d)",
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label=style["label"],
        )
        for style in MODEL_STYLE.values()
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.8,
        handlelength=2.0,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_png = output_dir / f"{args.filename}.png"
    output_pdf = output_dir / f"{args.filename}.pdf"
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
