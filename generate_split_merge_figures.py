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

_POLYGON_DILATION = 0.2 # 0.15 * spatial_radius; None to disable
_SIGNED_TAU_LOW = None # 0.30; None to autocalibrate
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

def process_config(
        config_path: Path,
        init: str = "leiden",
        n_em_iters: int = 3,
        n_cleanup_steps: int = 4,
        conflict_min_posterior : float = 0.01,
        n_em_iters_post_cleanup: int = 3,
        ) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = dataset_from_config(cfg)
    bounds = tuple(cfg.get("mosaic", {}).get("bounds", [0, 100, 0, 100]))
    nn_dists, _ = KDTree(dataset.centers).query(dataset.centers, k=2)
    spatial_radius = 3.0 * float(np.median(nn_dists[:, 1]))

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
        polygon_dilation=_POLYGON_DILATION,
        log_ramp=True,
        log_ramp_alpha=_LOG_RAMP_ALPHA,
        split_merge=True,
        per_cluster_tau=True,
        n_em_iters=n_em_iters,
        n_cleanup_steps=n_cleanup_steps,
        n_em_iters_post_cleanup=n_em_iters_post_cleanup,
        conflict_min_posterior=conflict_min_posterior,
    ).fit(dataset)

    model = result.model
    pm: dict[int, int] = model.get("parent_map", {})
    K_eff: int = model.get("k_after_split", K_init)
    N = len(dataset)

    # --- Collect label arrays for each step that was actually computed ---
    labels_gmm   = model["labels_gmm_raw"]
    labels_split = model["labels_initial"]
    labels_icm   = model["labels_post_icm"]

    # 5a/5b: only present when the split_merge block ran (K_eff > K)
    labels_merged  = (
        relabel(model["labels_post_merge"], dataset.y)
        if model.get("labels_post_merge") is not None else None
    )
    labels_cleanup = (
        relabel(model["labels_post_cleanup"], dataset.y)
        if model.get("labels_post_cleanup") is not None else None
    )

    # 5c: only when n_em_iters > 0
    labels_em = (
        relabel(model["labels_post_em"], dataset.y)
        if model.get("labels_post_em") is not None else None
    )

    # After ALL cleanup passes, before post-cleanup EM (always present)
    labels_all_cleanup = relabel(model["labels_post_all_cleanup"], dataset.y)

    # Post-cleanup EM (only when n_em_iters_post_cleanup > 0)
    labels_em_post_cleanup = (
        relabel(model["labels_post_em_post_cleanup"], dataset.y)
        if model.get("labels_post_em_post_cleanup") is not None else None
    )

    # Final result
    labels_final = relabel(result.labels, dataset.y)
    _labeled = result.labels >= 0

    # --- ARI and error counts ---
    def _ari(lbl): return adjusted_rand_score(dataset.y, lbl)
    def _err(lbl): return int((lbl != dataset.y).sum())

    ari_gmm   = _ari(labels_gmm)
    ari_split = _ari(labels_split)
    ari_icm   = _ari(labels_icm)

    ari_merged   = _ari(model["labels_post_merge"])   if labels_merged  is not None else None
    n_err_merged = _err(labels_merged)                 if labels_merged  is not None else None
    n_merged     = len(np.unique(model["labels_post_merge"])) if labels_merged is not None else K

    ari_cleanup   = _ari(model["labels_post_cleanup"]) if labels_cleanup is not None else None
    n_err_cleanup = _err(labels_cleanup)                if labels_cleanup is not None else None

    ari_em      = _ari(model["labels_post_em"]) if labels_em is not None else None
    n_err_em    = _err(labels_em)                if labels_em is not None else None

    ari_all_cleanup   = _ari(model["labels_post_all_cleanup"])
    n_err_all_cleanup = _err(labels_all_cleanup)

    ari_em_post_cleanup   = _ari(model["labels_post_em_post_cleanup"]) \
                            if labels_em_post_cleanup is not None else None
    n_err_em_post_cleanup = _err(labels_em_post_cleanup) \
                            if labels_em_post_cleanup is not None else None

    ari_final   = (
        adjusted_rand_score(dataset.y[_labeled], result.labels[_labeled])
        if _labeled.any() else 0.0
    )
    n_err_final = int((labels_final[_labeled] != dataset.y[_labeled]).sum())

    out_path = Path("figures") / (config_path.stem + "_split_merge.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:

        def _save(fig):
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 0: ground truth
        _save(plot_mosaic_step(
            dataset, dataset.y,
            f"{config_path.stem} — Ground truth   K={K}",
            bounds=bounds,
        ))

        # Page 1: step 1+2 — GMM
        _save(plot_mosaic_step(
            dataset, labels_gmm,
            f"{config_path.stem} — Step 1+2: GMM ({init=})  K_init={K_init}   ARI={ari_gmm:.3f}",
            bounds=bounds,
        ))

        # Page 2: step 3 — post-split
        _save(plot_mosaic_step(
            dataset, labels_split,
            (f"{config_path.stem} — Step 3: post-split   "
             f"K_eff={K_eff}   n_splits={model['n_splits']}   ARI={ari_split:.3f}"),
            parent_map=pm, bounds=bounds,
        ))

        # Page 3: step 4 — post-ICM
        _save(plot_mosaic_step(
            dataset, labels_icm,
            (f"{config_path.stem} — Step 4: post-ICM   "
             f"K_eff={K_eff}   iters={model['n_iters']}   ARI={ari_icm:.3f}"),
            parent_map=pm, bounds=bounds,
        ))

        # Pages 5a / 5b: only when the split_merge block actually ran (K_eff > K)
        if labels_merged is not None:
            _save(plot_mosaic_step(
                dataset, labels_merged,
                (f"{config_path.stem} — Step 5a: post-merge   "
                 f"K={n_merged}   n_merges={model['n_merges']}   "
                 f"ARI={ari_merged:.3f}   errors={n_err_merged}/{N}"),
                gt_labels=dataset.y, bounds=bounds,
            ))

        if labels_cleanup is not None:
            _save(plot_mosaic_step(
                dataset, labels_cleanup,
                (f"{config_path.stem} — Step 5b: post-cleanup   "
                 f"K={len(np.unique(model['labels_post_cleanup']))}   "
                 f"unfrozen={model['n_unfrozen_merge']}   iters={model['n_iters_post_merge']}   "
                 f"ARI={ari_cleanup:.3f}   errors={n_err_cleanup}/{N}"),
                gt_labels=dataset.y, bounds=bounds,
            ))

        # Page 5c: EM re-fit (only if n_em_iters > 0)
        if labels_em is not None:
            _save(plot_mosaic_step(
                dataset, labels_em,
                (f"{config_path.stem} — Step 5c: EM re-fit × {n_em_iters}   "
                 f"ARI={ari_em:.3f}   errors={n_err_em}/{N}"),
                gt_labels=dataset.y, bounds=bounds,
            ))

        # Cleanup page: result after all n_cleanup_steps passes (before post-cleanup EM)
        if n_cleanup_steps > 0:
            _save(plot_mosaic_step(
                dataset, labels_all_cleanup,
                (f"{config_path.stem} — Cleanup ×{n_cleanup_steps}   "
                 f"changed={model['n_cleanup_changed']}/{N}   "
                 f"swaps={model['n_swaps_total']}   "
                 f"reassigned={model['n_reassigned_total']}   "
                 f"unlabeled={model['n_unlabeled_total']}   "
                 f"ARI={ari_all_cleanup:.3f}   errors={n_err_all_cleanup}/{N}"),
                gt_labels=dataset.y, bounds=bounds,
            ))

        # Post-cleanup EM page (only when n_em_iters_post_cleanup > 0)
        if labels_em_post_cleanup is not None:
            _save(plot_mosaic_step(
                dataset, labels_em_post_cleanup,
                (f"{config_path.stem} — Post-cleanup EM ×{n_em_iters_post_cleanup}   "
                 f"ARI={ari_em_post_cleanup:.3f}   errors={n_err_em_post_cleanup}/{N}"),
                gt_labels=dataset.y, bounds=bounds,
            ))

        # Final page: merge hierarchy dendrogram
        merge_history: list[tuple[int, int, float]] = model.get("merge_history", [])
        _save(_plot_merge_dendrogram(
            dataset, labels_icm, merge_history, K,
            (f"{config_path.stem} — Merge hierarchy   "
             f"K_eff={K_eff} -> K={len(np.unique(result.labels))}   "
             f"n_merges={model['n_merges']}"),
        ))

    # --- Console report: only show steps that were computed ---
    lines = [
        f"{config_path.name}  ->  {out_path}   (radius={spatial_radius:.1f})",
        f"  Step 1+2  GMM:        K_init={K_init}   ARI={ari_gmm:.3f}",
        f"  Step 3    post-split: K_eff={K_eff}   ARI={ari_split:.3f}   n_splits={model['n_splits']}",
        f"  Step 4    post-ICM:   K_eff={K_eff}   ARI={ari_icm:.3f}   iters={model['n_iters']}",
    ]
    if labels_merged is not None:
        lines.append(
            f"  Step 5a   post-merge: K={n_merged}   ARI={ari_merged:.3f}"
            f"   errors={n_err_merged}   n_merges={model['n_merges']}"
        )
    if labels_cleanup is not None:
        lines.append(
            f"  Step 5b   cleanup:    K={len(np.unique(model['labels_post_cleanup']))}   ARI={ari_cleanup:.3f}"
            f"   errors={n_err_cleanup}   unfrozen={model['n_unfrozen_merge']}   iters={model['n_iters_post_merge']}"
        )
    if labels_em is not None:
        lines.append(
            f"  Step 5c   EM ×{n_em_iters}:     ARI={ari_em:.3f}   errors={n_err_em}"
        )
    if n_cleanup_steps > 0:
        lines.append(
            f"  Cleanup   ×{n_cleanup_steps}:         "
            f"changed={model['n_cleanup_changed']}   "
            f"swaps={model['n_swaps_total']}   "
            f"reassigned={model['n_reassigned_total']}   "
            f"unlabeled={model['n_unlabeled_total']}   "
            f"ARI={ari_all_cleanup:.3f}   errors={n_err_all_cleanup}/{N}"
        )
    if labels_em_post_cleanup is not None:
        lines.append(
            f"  Post-cl EM×{n_em_iters_post_cleanup}:         "
            f"ARI={ari_em_post_cleanup:.3f}   errors={n_err_em_post_cleanup}/{N}"
        )
    if not (n_cleanup_steps > 0 or labels_em_post_cleanup is not None):
        lines.append(
            f"  Final     result:     ARI={ari_final:.3f}   errors={n_err_final}/{N}"
        )
    print("\n".join(lines))


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
