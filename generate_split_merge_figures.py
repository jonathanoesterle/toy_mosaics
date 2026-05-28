"""Generate pipeline step figures for the split_merge MRF approach.

For each YAML config, creates a PDF with one page per pipeline step:
  0. Ground truth
  1. Step 1+2 — GMM initialization
  2. Step 3   — Post-split (K_eff sub-clusters)
  3. Step 4   — Post-ICM  (K_eff sub-clusters refined)
  4. Step 5   — Post-merge (K clusters)

Sub-clusters of original cluster k are coloured in the lighter tab20 shade
paired with k's dark shade, so the parent–child relationship is visible.

Usage:
    uv run python generate_split_merge_figures.py [configs/foo.yaml ...]

Output PDFs are written to figures/<stem>_split_merge.pdf.
"""
from __future__ import annotations

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

from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from toy_mosaics.clustering import MRFMosaicStrategy
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.simulate_dataset import dataset_from_config

# tab20: indices 2k and 2k+1 are paired (dark / light) shades of the same hue
_TAB20 = plt.cm.tab20(np.linspace(0, 1, 20, endpoint=False))

_SIGNED_TAU_LOW = 0.30
_LOG_RAMP_ALPHA = 10.0
_K_INIT_FACTOR = 2  # GMM is initialised with K * _K_INIT_FACTOR components


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relabel(labels: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Remap `labels` to best match `y_true` via linear assignment."""
    n = int(max(labels.max(), y_true.max())) + 1
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, labels):
        conf[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {int(col_ind[i]): int(row_ind[i]) for i in range(len(row_ind))}
    return np.array([mapping.get(int(lbl), int(lbl)) for lbl in labels], dtype=int)


def _label_colors(unique_labels: list[int], parent_map: dict[int, int]) -> dict[int, np.ndarray]:
    """Assign tab20 colors: original cluster k → dark shade 2k; sub-clusters → lighter shades."""
    colors: dict[int, np.ndarray] = {}
    parent_sub_count: dict[int, int] = {}
    for lbl in sorted(unique_labels):
        if lbl in parent_map:
            parent = parent_map[lbl]
            sub_idx = parent_sub_count.get(parent, 0)
            parent_sub_count[parent] = sub_idx + 1
            # light shade of parent, cycling if a parent has many sub-clusters
            colors[lbl] = _TAB20[(2 * parent + 1 + sub_idx * 2) % 20]
        else:
            colors[lbl] = _TAB20[(2 * lbl) % 20]
    return colors


def _plot_step(
    dataset: MosaicDataset,
    labels: np.ndarray,
    title: str,
    parent_map: dict[int, int] | None = None,
    gt_labels: np.ndarray | None = None,
) -> plt.Figure:
    """Row layout: feature scatter on left, one spatial panel per cluster."""
    pm = parent_map or {}
    unique = sorted(np.unique(labels).tolist())
    colors = _label_colors(unique, pm)
    polygons_arr = np.array(dataset.polygons, dtype=object)
    same_k_as_gt = gt_labels is not None and len(unique) == len(np.unique(gt_labels))

    n_panels = 1 + len(unique)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 4))
    axes = np.atleast_1d(axes)
    ax_f = axes[0]

    # Feature scatter — all clusters together
    for lbl in unique:
        mask = labels == lbl
        leg = f"{lbl} (<-{pm[lbl]})" if lbl in pm else str(lbl)
        ax_f.scatter(dataset.X[mask, 0], dataset.X[mask, 1],
                     color=colors[lbl], label=leg, s=20, alpha=0.85)
    if same_k_as_gt:
        errors = labels != gt_labels
        if errors.any():
            ax_f.scatter(dataset.X[errors, 0], dataset.X[errors, 1],
                         facecolors="none", edgecolors="black", s=70, linewidths=1.5, zorder=5)
    ax_f.set_xlabel("Feature 1"); ax_f.set_ylabel("Feature 2")
    ax_f.set_title("Feature space")
    ax_f.legend(loc="best", fontsize=7, title="cluster")

    # One spatial panel per cluster
    bg_patches = [Polygon(poly, closed=True) for poly in polygons_arr]
    for idx, lbl in enumerate(unique):
        ax = axes[1 + idx]
        mask = labels == lbl

        ax.add_collection(PatchCollection(
            bg_patches, facecolors="lightgray", edgecolors="gray",
            linewidths=0.4, alpha=0.4,
        ))
        fg = [Polygon(poly, closed=True) for poly in polygons_arr[mask]]
        if fg:
            ax.add_collection(PatchCollection(
                fg, facecolors=colors[lbl], edgecolors="black", linewidths=0.8, alpha=0.8,
            ))
        if same_k_as_gt:
            err_mask = mask & (labels != gt_labels)
            if err_mask.any():
                ax.scatter(dataset.centers[err_mask, 0], dataset.centers[err_mask, 1],
                           c="red", marker="x", s=50, zorder=6, linewidths=1.5)

        ax.set_xlim(-10, 110); ax.set_ylim(-10, 110); ax.set_aspect("equal")
        ttl = f"{lbl} (<-{pm[lbl]})" if lbl in pm else str(lbl)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Merge hierarchy
# ---------------------------------------------------------------------------

def _plot_merge_dendrogram(
    dataset: MosaicDataset,
    labels_pre_merge: np.ndarray,
    merge_history: list[tuple[int, int, float]],
    K_target: int,
    title: str,
) -> plt.Figure:
    """Dendrogram of the greedy merge sequence.

    Leaves = sub-clusters after ICM, colored by GT majority type.
    Y-axis = feature-mean L2 distance of each merge.
    A dashed red line marks the cut height separating actual merges
    from dummy padding used to complete the tree when K_target > 1.
    """
    unique = sorted(np.unique(labels_pre_merge).tolist())
    n_leaves = len(unique)
    label_to_idx = {k: i for i, k in enumerate(unique)}

    gt_majority = {
        k: int(np.bincount(dataset.y[labels_pre_merge == k],
                           minlength=dataset.n_mosaics).argmax())
        for k in unique if (labels_pre_merge == k).any()
    }
    leaf_labels = [f"{k}\n(GT:{gt_majority.get(k, '?')})" for k in unique]

    fig, ax = plt.subplots(figsize=(max(8, n_leaves * 1.1), 5))

    if not merge_history:
        ax.text(0.5, 0.5, "No merges performed",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        return fig

    # --- Build linkage matrix from merge history ---
    # node_map: label -> current scipy node index
    # node_sizes: scipy node index -> number of sub-cluster leaves beneath it
    node_map = {k: label_to_idx[k] for k in unique}
    node_sizes: dict[int, int] = {i: 1 for i in range(n_leaves)}

    Z_rows: list[list[float]] = []
    next_node = n_leaves
    max_actual_dist = 0.0

    for a, b, dist in merge_history:
        if a not in node_map or b not in node_map:
            break
        na, nb = node_map[a], node_map[b]
        merged_size = node_sizes[na] + node_sizes[nb]
        Z_rows.append([float(na), float(nb), dist, float(merged_size)])
        node_sizes[next_node] = merged_size
        node_map[a] = next_node
        del node_map[b]
        max_actual_dist = max(max_actual_dist, dist)
        next_node += 1

    n_actual = len(Z_rows)

    # Pad remaining disconnected roots with dummy high-distance merges so
    # scipy gets a complete tree.  Each pad merge is slightly taller than the
    # previous to keep distances monotone.
    remaining = sorted(node_map.values())
    pad_dist = max_actual_dist * 1.5 if max_actual_dist > 0 else 1.0
    while len(remaining) > 1:
        na, nb = remaining[0], remaining[1]
        merged_size = node_sizes[na] + node_sizes[nb]
        Z_rows.append([float(na), float(nb), pad_dist, float(merged_size)])
        node_sizes[next_node] = merged_size
        remaining = [next_node] + remaining[2:]
        next_node += 1
        pad_dist *= 1.01

    Z = np.array(Z_rows, dtype=float)

    scipy_dendrogram(
        Z, ax=ax, labels=leaf_labels, leaf_rotation=45,
        leaf_font_size=8, color_threshold=0, above_threshold_color="silver",
    )

    # Dashed cut line between real and padding merges
    if n_actual < n_leaves - 1 and max_actual_dist > 0:
        cut = max_actual_dist * 1.2
        ax.axhline(cut, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"K={K_target} cut  (above = no merge performed)")
        ax.legend(fontsize=8, loc="upper left")

    # Color leaf tick labels by GT majority type
    fig.canvas.draw()
    for lbl_obj in ax.get_xticklabels():
        text = lbl_obj.get_text()
        try:
            sub_k = int(text.split("\n")[0])
            gt = gt_majority.get(sub_k, 0)
            lbl_obj.set_color(_TAB20[(2 * gt) % 20])
        except (ValueError, IndexError):
            pass

    ax.set_xlabel("Sub-cluster  (leaf color = GT majority type)")
    ax.set_ylabel("Feature-mean L2 distance")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-config processing
# ---------------------------------------------------------------------------

def process_config(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = dataset_from_config(cfg)
    nn_dists, _ = KDTree(dataset.centers).query(dataset.centers, k=2)
    spatial_radius = 3.0 * float(np.median(nn_dists[:, 1]))

    init = "leiden"
    K = dataset.n_mosaics
    if init != "leiden":
        K_init = K * _K_INIT_FACTOR
    else:
        K_init = None  # Leiden init determines K_init internally based on resolution

    result = MRFMosaicStrategy(
        init=init,
        leiden_k_features=10,
        leiden_resolution=1.0,
        n_clusters=K,
        n_clusters_init=K_init,
        spatial_radius=spatial_radius,
        signed_ramp=True,
        tau_low=_SIGNED_TAU_LOW,
        log_ramp=True,
        log_ramp_alpha=_LOG_RAMP_ALPHA,
        split_merge=True,
    ).fit(dataset)

    model = result.model
    pm: dict[int, int] = model.get("parent_map", {})
    K_eff: int = model.get("k_after_split", K_init)

    # Intermediate label arrays
    labels_gmm    = model["labels_gmm_raw"]    # step 1+2: K_init clusters (raw)
    labels_split  = model["labels_initial"]     # step 3: post-split
    labels_icm    = model["labels_post_icm"]   # step 4: post-ICM
    labels_merged = model["labels_post_merge"] # step 5a: post-merge (before cleanup)
    labels_final  = _relabel(result.labels, dataset.y)  # step 5b: post-cleanup

    # ARI at each step (permutation-invariant, no relabeling needed)
    ari_gmm    = adjusted_rand_score(dataset.y, labels_gmm)
    ari_split  = adjusted_rand_score(dataset.y, labels_split)
    ari_icm    = adjusted_rand_score(dataset.y, labels_icm)
    ari_merged = adjusted_rand_score(dataset.y, labels_merged)
    ari_final  = adjusted_rand_score(dataset.y, result.labels)

    n_err_final = int((labels_final != dataset.y).sum())

    out_path = Path("figures") / (config_path.stem + "_split_merge.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:

        # Page 0: ground truth
        fig = _plot_step(
            dataset, dataset.y,
            f"{config_path.stem} — Ground truth   K={K}",
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 1: step 1+2 — GMM (K_init clusters)
        fig = _plot_step(
            dataset, labels_gmm,
            (f"{config_path.stem} — Step 1+2: GMM initialization ({init=})  "
             f"K_init={K_init}   ARI={ari_gmm:.3f}"),
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: step 3 — post-split
        fig = _plot_step(
            dataset, labels_split,
            (f"{config_path.stem} — Step 3: post-split   "
             f"K_eff={K_eff}   n_splits={model['n_splits']}   ARI={ari_split:.3f}"),
            parent_map=pm,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: step 4 — post-ICM
        fig = _plot_step(
            dataset, labels_icm,
            (f"{config_path.stem} — Step 4: post-ICM   "
             f"K_eff={K_eff}   iters={model['n_iters']}   ARI={ari_icm:.3f}"),
            parent_map=pm,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 4: step 5a — post-merge (greedy merge, before cleanup ICM)
        n_merged = len(np.unique(labels_merged))
        n_err_merged = int((_relabel(labels_merged, dataset.y) != dataset.y).sum())
        fig = _plot_step(
            dataset, _relabel(labels_merged, dataset.y),
            (f"{config_path.stem} — Step 5a: post-merge (before cleanup)   "
             f"K={n_merged}   n_merges={model['n_merges']}   "
             f"ARI={ari_merged:.3f}   errors={n_err_merged}/{len(dataset)}"),
            gt_labels=dataset.y,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 5: step 5b — post-cleanup (second ICM on merged labels)
        fig = _plot_step(
            dataset, labels_final,
            (f"{config_path.stem} — Step 5b: post-cleanup   "
             f"K={len(np.unique(result.labels))}   "
             f"unfrozen={model['n_unfrozen_merge']}   iters={model['n_iters_post_merge']}   "
             f"ARI={ari_final:.3f}   errors={n_err_final}/{len(dataset)}"),
            gt_labels=dataset.y,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 6: merge hierarchy dendrogram
        merge_history: list[tuple[int, int, float]] = model.get("merge_history", [])
        fig = _plot_merge_dendrogram(
            dataset, labels_icm, merge_history, K,
            (f"{config_path.stem} — Merge hierarchy   "
             f"K_eff={K_eff} -> K={len(np.unique(result.labels))}   "
             f"n_merges={model['n_merges']}"),
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(
        f"{config_path.name}  ->  {out_path}   (radius={spatial_radius:.1f})\n"
        f"  Step 1+2  GMM:        K_init={K_init}   ARI={ari_gmm:.3f}\n"
        f"  Step 3    post-split: K_eff={K_eff}     ARI={ari_split:.3f}"
        f"   n_splits={model['n_splits']}\n"
        f"  Step 4    post-ICM:   K_eff={K_eff}     ARI={ari_icm:.3f}"
        f"   iters={model['n_iters']}\n"
        f"  Step 5a   post-merge: K={n_merged}        ARI={ari_merged:.3f}"
        f"   errors={n_err_merged}   n_merges={model['n_merges']}\n"
        f"  Step 5b   cleanup:    K={len(np.unique(result.labels))}        ARI={ari_final:.3f}"
        f"   errors={n_err_final}"
        f"   unfrozen={model['n_unfrozen_merge']}   iters={model['n_iters_post_merge']}"
        f"   h1={model['n_unfrozen_h1_post']}"
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
