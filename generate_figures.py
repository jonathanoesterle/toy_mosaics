"""Generate figures for all YAML configs and save PDFs to figures/.

Usage:
    uv run python generate_figures.py [configs/foo.yaml ...]

If no config paths are given, all *.yaml files under configs/ are processed.
Each PDF is named after output.filename in the config (e.g. example.pdf).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from toy_mosaics.plot import plot_mosaics
from toy_mosaics.simulate_dataset import dataset_from_config


def _figure_path(cfg: dict, config_path: Path) -> Path:
    filename = cfg.get("output", {}).get("filename", config_path.stem + ".npz")
    return Path("figures") / Path(filename).with_suffix(".pdf")


def _plot_blobs(X: np.ndarray, y: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for label in np.unique(y):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], color=colors[int(label) % 10],
                   label=f"Mosaic {label}", s=15, alpha=0.7)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Simulated Feature Blobs (first 2 dims)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def process_config(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = dataset_from_config(cfg)
    out_path = _figure_path(cfg, config_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    polygons_arr = np.array(dataset.polygons, dtype=object)

    with PdfPages(out_path) as pdf:
        fig_mosaics, _ = plot_mosaics(
            groups=dataset.groups,
            polygons=polygons_arr,
            centers=dataset.centers,
            mode="basic",
        )
        fig_mosaics.suptitle(config_path.stem, fontsize=12, y=1.01)
        pdf.savefig(fig_mosaics, bbox_inches="tight")
        plt.close(fig_mosaics)

        fig_blobs = _plot_blobs(dataset.X, dataset.y)
        fig_blobs.suptitle(config_path.stem, fontsize=12, y=1.01)
        pdf.savefig(fig_blobs, bbox_inches="tight")
        plt.close(fig_blobs)

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
