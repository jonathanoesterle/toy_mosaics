"""Generate figures for all YAML configs and save PDFs to figures/.

Usage:
    uv run python generate_figures.py [configs/foo.yaml ...]

If no config paths are given, all *.yaml files under configs/ are processed.
Each PDF is named after output.filename in the config (e.g. example.pdf) and
contains two pages: ground truth and GMM clustering.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.optimize import linear_sum_assignment

from toy_mosaics.clustering import GMMStrategy
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.simulate_dataset import dataset_from_config

_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def _relabel(labels: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Remap labels to match y_true as closely as possible via Hungarian algorithm."""
    n = int(max(labels.max(), y_true.max())) + 1
    confusion = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, labels):
        confusion[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {int(col_ind[i]): int(row_ind[i]) for i in range(len(row_ind))}
    return np.array([mapping.get(int(l), int(l)) for l in labels], dtype=int)


def _draw_mosaic(ax: plt.Axes, polygons_arr: np.ndarray, centers: np.ndarray,
                 mask: np.ndarray, color) -> None:
    patches = [Polygon(poly, closed=True) for poly in polygons_arr[mask]]
    if patches:
        ax.add_collection(PatchCollection(
            patches, facecolors=color, edgecolors="black", linewidths=1.0, alpha=0.7,
        ))
    ax.scatter(centers[mask, 0], centers[mask, 1], c="darkred", s=8, zorder=5, alpha=0.5)
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def _plot_grid(dataset: MosaicDataset, labels: np.ndarray, title: str) -> plt.Figure:
    """Figure with one subplot per cluster (spatial) plus a combined feature scatter."""
    unique = np.unique(labels)
    n = len(unique)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

    polygons_arr = np.array(dataset.polygons, dtype=object)

    for idx, label in enumerate(unique):
        ax = axes[idx]
        color = _COLORS[int(label) % 10]
        mask = labels == label
        _draw_mosaic(ax, polygons_arr, dataset.centers, mask, color)
        ax.set_title(f"Cluster {label}\n{mask.sum()} cells", fontsize=10)

    ax_feat = axes[-1]
    for label in unique:
        mask = labels == label
        color = _COLORS[int(label) % 10]
        ax_feat.scatter(
            dataset.X[mask, 0], dataset.X[mask, 1],
            color=color, label=f"Cluster {label}", s=15, alpha=0.8,
        )
    ax_feat.set_xlabel("Feature 1")
    ax_feat.set_ylabel("Feature 2")
    ax_feat.set_title("Feature space")
    ax_feat.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def _figure_path(cfg: dict, config_path: Path) -> Path:
    filename = cfg.get("output", {}).get("filename", config_path.stem + ".npz")
    return Path("figures") / Path(filename).with_suffix(".pdf")


def process_config(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = dataset_from_config(cfg)

    result = GMMStrategy(n_clusters=dataset.n_mosaics).fit(dataset)
    labels_gmm = _relabel(result.labels, dataset.y)

    out_path = _figure_path(cfg, config_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        fig_gt = _plot_grid(dataset, dataset.y, title=f"{config_path.stem} — ground truth")
        pdf.savefig(fig_gt, bbox_inches="tight")
        plt.close(fig_gt)

        fig_gmm = _plot_grid(dataset, labels_gmm, title=f"{config_path.stem} — GMM")
        pdf.savefig(fig_gmm, bbox_inches="tight")
        plt.close(fig_gmm)

    print(f"{config_path.name} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", type=Path,
                        help="YAML config files (default: all configs/*.yaml)")
    args = parser.parse_args()

    config_paths = args.configs or sorted(Path("configs").glob("*.yaml"))
    if not config_paths:
        print("No config files found.", file=sys.stderr)
        sys.exit(1)

    for config_path in config_paths:
        process_config(config_path)


if __name__ == "__main__":
    main()
