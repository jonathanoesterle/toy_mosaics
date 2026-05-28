"""Generate MRF diagnostic PDFs for all YAML configs.

Each PDF contains three pages:
  1. Ground truth  — one subplot per mosaic + feature scatter
  2. MRF result    — same layout, with ARI and violation counts in the title
  3. MRF diagnostics — 6-panel figure from plot_mrf_diagnostics

Usage:
    uv run python generate_mrf_figures.py [configs/foo.yaml ...]

Output PDFs are written to figures/<stem>_mrf.pdf.
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
from scipy.spatial import KDTree
from sklearn.metrics import adjusted_rand_score

from toy_mosaics.clustering import MRFMosaicStrategy
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.plot import plot_mrf_diagnostics
from toy_mosaics.simulate_dataset import dataset_from_config

_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


# ---------------------------------------------------------------------------
# Shared drawing helpers (mirrored from generate_figures.py)
# ---------------------------------------------------------------------------

def _relabel(labels: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n = int(max(labels.max(), y_true.max())) + 1
    confusion = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, labels):
        confusion[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {int(col_ind[i]): int(row_ind[i]) for i in range(len(row_ind))}
    return np.array([mapping.get(int(l), int(l)) for l in labels], dtype=int)


def _draw_mosaic(ax, polygons_arr, centers, mask, color, error_mask=None):
    patches = [Polygon(poly, closed=True) for poly in polygons_arr[mask]]
    if patches:
        ax.add_collection(PatchCollection(
            patches, facecolors=color, edgecolors="black", linewidths=1.0, alpha=0.7,
        ))
    if error_mask is not None and error_mask.any():
        err_patches = [Polygon(poly, closed=True)
                       for poly, e in zip(polygons_arr[mask], error_mask) if e]
        if err_patches:
            ax.add_collection(PatchCollection(
                err_patches, facecolors=color, edgecolors="red",
                linewidths=2.0, alpha=0.5,
            ))
    ax.scatter(centers[mask, 0], centers[mask, 1], c="darkred", s=8, zorder=5, alpha=0.5)
    ax.set_xlim(-10, 110); ax.set_ylim(-10, 110); ax.set_aspect("equal")
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def _plot_grid(
    dataset: MosaicDataset,
    labels: np.ndarray,
    title: str,
    error_cells: np.ndarray | None = None,
) -> plt.Figure:
    """Spatial subplots (one per cluster) + feature scatter, matching generate_figures.py."""
    unique = np.unique(labels)
    n = len(unique)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    polygons_arr = np.array(dataset.polygons, dtype=object)

    for idx, label in enumerate(unique):
        ax = axes[idx]
        color = _COLORS[int(label) % 10]
        mask = labels == label
        em = None
        if error_cells is not None:
            em = error_cells[mask]
        _draw_mosaic(ax, polygons_arr, dataset.centers, mask, color, em)
        n_err = int(em.sum()) if em is not None else 0
        err_str = f"  ({n_err} err)" if n_err else ""
        ax.set_title(f"Cluster {label}\n{mask.sum()} cells{err_str}", fontsize=10)

    ax_feat = axes[-1]
    for label in unique:
        mask = labels == label
        color = _COLORS[int(label) % 10]
        ax_feat.scatter(dataset.X[mask, 0], dataset.X[mask, 1],
                        color=color, label=f"Cluster {label}", s=15, alpha=0.8)
    if error_cells is not None and error_cells.any():
        ax_feat.scatter(dataset.X[error_cells, 0], dataset.X[error_cells, 1],
                        facecolors="none", edgecolors="black", s=60, linewidths=1.5,
                        zorder=5, label="errors")
    ax_feat.set_xlabel("Feature 1"); ax_feat.set_ylabel("Feature 2")
    ax_feat.set_title("Feature space"); ax_feat.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-config processing
# ---------------------------------------------------------------------------

_SIGNED_TAU_LOW = 0.30  # intra-tile CF sits at ~0.20-0.26; this puts them in the attractive zone
_THETA_HARD = 0.85      # CF above which same-label adjacency is a near-certain error


def _run_and_relabel(dataset, spatial_radius, **kwargs):
    result = MRFMosaicStrategy(
        n_clusters=dataset.n_mosaics,
        spatial_radius=spatial_radius,
        **kwargs,
    ).fit(dataset)
    labels = _relabel(result.labels, dataset.y)
    ari = adjusted_rand_score(dataset.y, labels)
    return result, labels, ari


def process_config(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = dataset_from_config(cfg)

    # Auto-compute spatial_radius as in generate_figures.py
    nn_dists, _ = KDTree(dataset.centers).query(dataset.centers, k=2)
    spatial_radius = 3.0 * float(np.median(nn_dists[:, 1]))

    # Plain MRF
    result, labels_mrf, ari = _run_and_relabel(dataset, spatial_radius)
    model = result.model
    error_cells = labels_mrf != dataset.y

    # Signed-ramp MRF (tau_low raised to cover intra-tile CF range)
    result_s, labels_mrf_s, ari_s = _run_and_relabel(
        dataset, spatial_radius, signed_ramp=True, tau_low=_SIGNED_TAU_LOW,
    )
    model_s = result_s.model
    error_cells_s = labels_mrf_s != dataset.y

    # Signed log-ramp MRF (α=10): same attractive zone, super-linear repulsion
    result_l, labels_mrf_l, ari_l = _run_and_relabel(
        dataset, spatial_radius,
        signed_ramp=True, tau_low=_SIGNED_TAU_LOW,
        log_ramp=True, log_ramp_alpha=10.0,
    )
    model_l = result_l.model
    error_cells_l = labels_mrf_l != dataset.y

    # Signed log-ramp + H1 force-unfreeze pre-pass (theta_hard=0.85)
    result_h1, labels_mrf_h1, ari_h1 = _run_and_relabel(
        dataset, spatial_radius,
        signed_ramp=True, tau_low=_SIGNED_TAU_LOW,
        log_ramp=True, log_ramp_alpha=10.0,
        theta_hard=_THETA_HARD,
    )
    model_h1 = result_h1.model
    error_cells_h1 = labels_mrf_h1 != dataset.y

    stem = cfg.get("output", {}).get("filename", config_path.stem + ".npz")
    out_path = Path("figures") / Path(stem).with_suffix("").name
    out_path = out_path.parent / (out_path.name + "_mrf.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        # Page 1: ground truth
        fig_gt = _plot_grid(dataset, dataset.y, title=f"{config_path.stem} — ground truth")
        pdf.savefig(fig_gt, bbox_inches="tight"); plt.close(fig_gt)

        # Page 2: plain MRF result
        m = model
        plain_title = (
            f"{config_path.stem} — GMM + MRF   ARI={ari:.3f}   "
            f"violations {m['violations_before']} -> {m['violations_after']}   "
            f"changed={m['n_changed']}   iters={m['n_iters']}"
        )
        fig_mrf = _plot_grid(dataset, labels_mrf, title=plain_title, error_cells=error_cells)
        pdf.savefig(fig_mrf, bbox_inches="tight"); plt.close(fig_mrf)

        # Page 3: plain MRF diagnostics
        fig_diag, _ = plot_mrf_diagnostics(dataset, result, ground_truth=dataset.y)
        fig_diag.suptitle(f"{config_path.stem} — MRF diagnostics", fontsize=13, y=1.01)
        pdf.savefig(fig_diag, bbox_inches="tight"); plt.close(fig_diag)

        # Page 4: signed-ramp MRF result
        ms = model_s
        signed_title = (
            f"{config_path.stem} — GMM + MRF (signed ramp, tau_low={_SIGNED_TAU_LOW})   "
            f"ARI={ari_s:.3f}   "
            f"violations {ms['violations_before']} -> {ms['violations_after']}   "
            f"changed={ms['n_changed']}   iters={ms['n_iters']}"
        )
        fig_mrf_s = _plot_grid(dataset, labels_mrf_s, title=signed_title, error_cells=error_cells_s)
        pdf.savefig(fig_mrf_s, bbox_inches="tight"); plt.close(fig_mrf_s)

        # Page 5: signed-ramp MRF diagnostics
        fig_diag_s, _ = plot_mrf_diagnostics(dataset, result_s, ground_truth=dataset.y)
        fig_diag_s.suptitle(
            f"{config_path.stem} — MRF diagnostics (signed ramp)", fontsize=13, y=1.01,
        )
        pdf.savefig(fig_diag_s, bbox_inches="tight"); plt.close(fig_diag_s)

        # Page 6: signed log-ramp MRF result
        ml = model_l
        log_title = (
            f"{config_path.stem} — GMM + MRF (signed log-ramp α=10, tau_low={_SIGNED_TAU_LOW})   "
            f"ARI={ari_l:.3f}   "
            f"violations {ml['violations_before']} -> {ml['violations_after']}   "
            f"changed={ml['n_changed']}   iters={ml['n_iters']}"
        )
        fig_mrf_l = _plot_grid(dataset, labels_mrf_l, title=log_title, error_cells=error_cells_l)
        pdf.savefig(fig_mrf_l, bbox_inches="tight"); plt.close(fig_mrf_l)

        # Page 7: signed log-ramp MRF diagnostics
        fig_diag_l, _ = plot_mrf_diagnostics(dataset, result_l, ground_truth=dataset.y)
        fig_diag_l.suptitle(
            f"{config_path.stem} — MRF diagnostics (signed log-ramp α=10)", fontsize=13, y=1.01,
        )
        pdf.savefig(fig_diag_l, bbox_inches="tight"); plt.close(fig_diag_l)

        # Page 8: signed log-ramp + H1 result
        mh = model_h1
        h1_title = (
            f"{config_path.stem} — GMM + MRF (signed log-ramp a=10 + H1 th={_THETA_HARD})   "
            f"ARI={ari_h1:.3f}   "
            f"violations {mh['violations_before']} -> {mh['violations_after']}   "
            f"changed={mh['n_changed']}   iters={mh['n_iters']}   "
            f"force_unfrozen={mh['n_force_unfrozen']}"
        )
        fig_mrf_h1 = _plot_grid(dataset, labels_mrf_h1, title=h1_title, error_cells=error_cells_h1)
        pdf.savefig(fig_mrf_h1, bbox_inches="tight"); plt.close(fig_mrf_h1)

        # Page 9: signed log-ramp + H1 diagnostics
        fig_diag_h1, _ = plot_mrf_diagnostics(dataset, result_h1, ground_truth=dataset.y)
        fig_diag_h1.suptitle(
            f"{config_path.stem} — MRF diagnostics (signed log-ramp + H1)", fontsize=13, y=1.01,
        )
        pdf.savefig(fig_diag_h1, bbox_inches="tight"); plt.close(fig_diag_h1)

    n_err    = int(error_cells.sum())
    n_err_s  = int(error_cells_s.sum())
    n_err_l  = int(error_cells_l.sum())
    n_err_h1 = int(error_cells_h1.sum())

    # Empirical taus from ground-truth labels (upper bound / oracle estimate)
    raw_map = result.model["raw_map"]
    same_cfs_gt = [cf for (i, j), cf in raw_map.items() if dataset.y[i] == dataset.y[j]]
    diff_cfs_gt = [cf for (i, j), cf in raw_map.items() if dataset.y[i] != dataset.y[j]]
    emp_low  = float(np.quantile(same_cfs_gt, 0.99)) if len(same_cfs_gt) >= 10 else None
    emp_high = float(np.quantile(diff_cfs_gt, 0.25)) if len(diff_cfs_gt) >= 10 else None

    def _tau_str(model: dict) -> str:
        c = model.get("tau_calibration", {})
        src_low  = c.get("tau_low_source",  "fixed")
        src_high = c.get("tau_high_source", "fixed")
        adj = "  [tau_high adjusted]" if "tau_high_adjusted" in c else ""
        emp_low_s  = f"  GT={emp_low:.3f}"  if emp_low  is not None else ""
        emp_high_s = f"  GT={emp_high:.3f}" if emp_high is not None else ""
        return (
            f"tau_low={model['tau_low']:.3f}{emp_low_s}  ({src_low})  "
            f"tau_high={model['tau_high']:.3f}{emp_high_s}  ({src_high}){adj}"
        )

    print(
        f"{config_path.name} -> {out_path}\n"
        f"  plain:      ARI={ari:.3f}   errors={n_err}/{len(dataset)}   "
        f"violations {m['violations_before']}->{m['violations_after']}   "
        f"changed={m['n_changed']}   iters={m['n_iters']}   frozen={m['n_frozen']}   radius={spatial_radius:.1f}\n"
        f"    {_tau_str(m)}\n"
        f"  signed:     ARI={ari_s:.3f}   errors={n_err_s}/{len(dataset)}   "
        f"violations {ms['violations_before']}->{ms['violations_after']}   "
        f"changed={ms['n_changed']}   iters={ms['n_iters']}   frozen={ms['n_frozen']}\n"
        f"    {_tau_str(ms)}\n"
        f"  signed+log: ARI={ari_l:.3f}   errors={n_err_l}/{len(dataset)}   "
        f"violations {ml['violations_before']}->{ml['violations_after']}   "
        f"changed={ml['n_changed']}   iters={ml['n_iters']}   frozen={ml['n_frozen']}\n"
        f"    {_tau_str(ml)}\n"
        f"  +H1(th={_THETA_HARD}): ARI={ari_h1:.3f}   errors={n_err_h1}/{len(dataset)}   "
        f"violations {mh['violations_before']}->{mh['violations_after']}   "
        f"changed={mh['n_changed']}   iters={mh['n_iters']}   "
        f"frozen={mh['n_frozen']}   force_unfrozen={mh['n_force_unfrozen']}\n"
        f"    {_tau_str(mh)}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "configs", nargs="*", type=Path,
        help="YAML config files (default: all configs/*.yaml)",
    )
    args = parser.parse_args()

    config_paths = args.configs or sorted(Path("configs").glob("*.yaml"))
    if not config_paths:
        print("No config files found.", file=sys.stderr)
        sys.exit(1)

    for config_path in config_paths:
        process_config(config_path)


if __name__ == "__main__":
    main()
