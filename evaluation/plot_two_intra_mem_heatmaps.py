import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/share/zhangsikui_workspace/comem")
SAMPLES_PATH = (
    ROOT
    / "save_dir/mem_information_eval_simcse_160k_vs_casual_intra_200/mem_information_samples.jsonl"
)
OUTPUT_DIR = ROOT / "save_dir/mem_regime_summary"

MODELS = [
    ("simcse_160k_casual", "Casual"),
    ("simcse_160k", "Non-casual"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.6, 3.2),
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.linewidth": 1.1,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_jsonl(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_cosine_matrix(rows, model: str) -> np.ndarray:
    matrices = [row["mem_cosine_matrix"] for row in rows if row["model"] == model]
    return np.mean(np.asarray(matrices, dtype=float), axis=0)


def plot_heatmap(ax, matrix: np.ndarray, title: str) -> None:
    ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(matrix.shape[0]))
    ax.set_yticks(range(matrix.shape[0]))
    labels = [f"M{i}" if i in (0, matrix.shape[0] - 1) else "" for i in range(matrix.shape[0])]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(SAMPLES_PATH)

    fig, axes = plt.subplots(1, 2)
    for ax, (model, title) in zip(axes, MODELS):
        plot_heatmap(ax, mean_cosine_matrix(rows, model), title)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.86, wspace=0.20)

    output_png = OUTPUT_DIR / "casual_vs_noncasual_intra_mem_heatmaps.png"
    output_pdf = OUTPUT_DIR / "casual_vs_noncasual_intra_mem_heatmaps.pdf"
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
