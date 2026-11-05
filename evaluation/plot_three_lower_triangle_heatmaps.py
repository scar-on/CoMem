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

THIRD_LABELS = [f"M{i}" for i in range(1, 9)]
THIRD_MATRIX = np.array(
    [
        [1.000000, 0.546404, 0.536438, 0.611003, 0.720081, 0.611069, 0.483601, 0.399641],
        [0.546404, 1.000000, 0.687534, 0.478270, 0.571652, 0.503127, 0.552633, 0.564061],
        [0.536438, 0.687534, 1.000000, 0.557372, 0.643824, 0.599084, 0.589739, 0.561031],
        [0.611003, 0.478270, 0.557372, 1.000000, 0.677547, 0.599687, 0.506748, 0.400593],
        [0.720081, 0.571652, 0.643824, 0.677547, 1.000000, 0.556531, 0.527041, 0.447268],
        [0.611069, 0.503127, 0.599084, 0.599687, 0.556531, 1.000000, 0.469289, 0.413241],
        [0.483601, 0.552633, 0.589739, 0.506748, 0.527041, 0.469289, 1.000000, 0.619757],
        [0.399641, 0.564061, 0.561031, 0.400593, 0.447268, 0.413241, 0.619757, 1.000000],
    ],
    dtype=np.float32,
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 13,
            "figure.figsize": (9.0, 3.25),
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
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


def lower_triangle(matrix: np.ndarray) -> np.ma.MaskedArray:
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    return np.ma.array(matrix, mask=mask)


def off_diagonal_mean(matrix: np.ndarray) -> float:
    return float(matrix[~np.eye(matrix.shape[0], dtype=bool)].mean())


def plot_heatmap(ax, matrix: np.ndarray, labels, title: str):
    masked = lower_triangle(matrix)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white", alpha=1.0)
    image = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")

    ax.set_title(title, pad=8, fontsize=16)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="major", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.text(
        5.25,
        1.08,
        f"Avg. sim.\n{off_diagonal_mean(matrix):.3f}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="black",
    )
    return image


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(SAMPLES_PATH)

    matrices = [
        (THIRD_MATRIX, "(a) Query-agnostic causal", THIRD_LABELS),
        (
            mean_cosine_matrix(rows, "simcse_160k_casual"),
            "(b) Ours causal",
            [f"M{i}" for i in range(8)],
        ),
        (
            mean_cosine_matrix(rows, "simcse_160k"),
            "(c) Ours non-causal",
            [f"M{i}" for i in range(8)],
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        constrained_layout=False,
        gridspec_kw={"wspace": -0.22},
    )

    image = None
    for ax, (matrix, title, labels) in zip(axes, matrices):
        image = plot_heatmap(ax, matrix, labels, title)

    if image is not None:
        colorbar_axis = fig.add_axes([0.30, 0.075, 0.40, 0.04])
        colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
        colorbar.ax.tick_params(labelsize=13, pad=1)

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.20, top=0.82, wspace=-0.22)

    output_png = OUTPUT_DIR / "casual_noncasual_query_lower_triangle_heatmaps.png"
    output_pdf = OUTPUT_DIR / "casual_noncasual_query_lower_triangle_heatmaps.pdf"
    fig.savefig(output_png, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
