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
from scipy.spatial import KDTree
from sklearn.metrics import adjusted_rand_score

from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from toy_mosaics.clustering import MRFMosaicStrategy
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.simulate_dataset import dataset_from_config
from figure_utils import relabel, plot_mosaic_step

_TAB20 = plt.cm.tab20(np.linspace(0, 1, 20, endpoint=False))

_SIGNED_TAU_LOW = 0.30
_LOG_RAMP_ALPHA = 10.0
_K_INIT_FACTOR = 2  # GMM is initialised with K * _K_INIT_FACTOR components


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
    bounds = tuple(cfg.get("mosaic", {}).get("bounds", [0, 100, 0, 100]))
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
    labels_gmm     = model["labels_gmm_raw"]     # step 1+2: K_init clusters (raw)
    labels_split   = model["labels_initial"]      # step 3: post-split
    labels_icm     = model["labels_post_icm"]    # step 4: post-ICM
    labels_merged  = model["labels_post_merge"]  # step 5a: post-merge (before cleanup)
    labels_cleanup  = relabel(model["labels_post_cleanup"], dataset.y)  # step 5b
    labels_conflict = relabel(model["labels_post_conflict"], dataset.y)  # step 6
    labels_final    = relabel(result.labels, dataset.y)                  # step 7 (may have -1)

    # ARI at each step (permutation-invariant, no relabeling needed)
    ari_gmm      = adjusted_rand_score(dataset.y, labels_gmm)
    ari_split    = adjusted_rand_score(dataset.y, labels_split)
    ari_icm      = adjusted_rand_score(dataset.y, labels_icm)
    ari_merged   = adjusted_rand_score(dataset.y, labels_merged)
    ari_cleanup  = adjusted_rand_score(dataset.y, model["labels_post_cleanup"])
    ari_conflict = adjusted_rand_score(dataset.y, model["labels_post_conflict"])
    # Step 7: ARI only over labeled cells (exclude -1)
    _labeled     = result.labels >= 0
    ari_final    = adjusted_rand_score(dataset.y[_labeled], result.labels[_labeled]) if _labeled.any() else 0.0

    n_err_cleanup  = int((labels_cleanup  != dataset.y).sum())
    n_err_conflict = int((labels_conflict != dataset.y).sum())
    n_labeled      = int(_labeled.sum())
    n_err_final    = int((labels_final[_labeled] != dataset.y[_labeled]).sum())

    out_path = Path("figures") / (config_path.stem + "_split_merge.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:

        # Page 0: ground truth
        fig = plot_mosaic_step(
            dataset, dataset.y,
            f"{config_path.stem} — Ground truth   K={K}",
            bounds=bounds,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 1: step 1+2 — GMM (K_init clusters)
        fig = plot_mosaic_step(
            dataset, labels_gmm,
            (f"{config_path.stem} — Step 1+2: GMM initialization ({init=})  "
             f"K_init={K_init}   ARI={ari_gmm:.3f}"),
            bounds=bounds,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: step 3 — post-split
        fig = plot_mosaic_step(
            dataset, labels_split,
            (f"{config_path.stem} — Step 3: post-split   "
             f"K_eff={K_eff}   n_splits={model['n_splits']}   ARI={ari_split:.3f}"),
            parent_map=pm,
            bounds=bounds,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: step 4 — post-ICM
        fig = plot_mosaic_step(
            dataset, labels_icm,
            (f"{config_path.stem} — Step 4: post-ICM   "
             f"K_eff={K_eff}   iters={model['n_iters']}   ARI={ari_icm:.3f}"),
            parent_map=pm,
            bounds=bounds,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 4: step 5a — post-merge (greedy merge, before cleanup ICM)
        n_merged = len(np.unique(labels_merged))
        n_err_merged = int((relabel(labels_merged, dataset.y) != dataset.y).sum())
        fig = plot_mosaic_step(
            dataset, relabel(labels_merged, dataset.y),
            (f"{config_path.stem} — Step 5a: post-merge (before cleanup)   "
             f"K={n_merged}   n_merges={model['n_merges']}   "
             f"ARI={ari_merged:.3f}   errors={n_err_merged}/{len(dataset)}"),
            gt_labels=dataset.y,
            bounds=bounds,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 5: step 5b — post-cleanup (second ICM on merged labels)
        fig = plot_mosaic_step(
            dataset, labels_cleanup,
            (f"{config_path.stem} — Step 5b: post-cleanup   "
             f"K={len(np.unique(model['labels_post_cleanup']))}   "
             f"unfrozen={model['n_unfrozen_merge']}   iters={model['n_iters_post_merge']}   "
             f"ARI={ari_cleanup:.3f}   errors={n_err_cleanup}/{len(dataset)}"),
            gt_labels=dataset.y,
            bounds=bounds,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 6: step 6 — post-conflict-resolution
        fig = plot_mosaic_step(
            dataset, labels_conflict,
            (f"{config_path.stem} — Step 6: conflict resolution   "
             f"K={len(np.unique(model['labels_post_conflict']))}   "
             f"n_swaps={model['n_swaps']}   "
             f"ARI={ari_conflict:.3f}   errors={n_err_conflict}/{len(dataset)}"),
            gt_labels=dataset.y,
            bounds=bounds,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 7: step 7 — residual violator assignment (-1 for unfit cells)
        fig = plot_mosaic_step(
            dataset, labels_final,
            (f"{config_path.stem} — Step 7: violator assignment   "
             f"reassigned={model['n_reassigned']}   unlabeled={model['n_unlabeled']}   "
             f"ARI={ari_final:.3f}   errors={n_err_final}/{n_labeled} labeled"),
            gt_labels=dataset.y,
            bounds=bounds,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 8: merge hierarchy dendrogram
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
        f"  Step 5b   cleanup:    K={len(np.unique(model['labels_post_cleanup']))}        ARI={ari_cleanup:.3f}"
        f"   errors={n_err_cleanup}"
        f"   unfrozen={model['n_unfrozen_merge']}   iters={model['n_iters_post_merge']}\n"
        f"  Step 6    conflict:   K={len(np.unique(model['labels_post_conflict']))}        ARI={ari_conflict:.3f}"
        f"   errors={n_err_conflict}   n_swaps={model['n_swaps']}\n"
        f"  Step 7    violators:  labeled={n_labeled}/{len(dataset)}   ARI={ari_final:.3f}"
        f"   errors={n_err_final}   reassigned={model['n_reassigned']}   unlabeled={model['n_unlabeled']}"
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
