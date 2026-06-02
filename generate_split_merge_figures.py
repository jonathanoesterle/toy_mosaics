"""Generate pipeline step figures for the split_merge MRF approach.

For each YAML config, creates a PDF with one page per pipeline step:
  0. Ground truth
  1. Step 1+2 — GMM initialization
  2. Step 3   — Post-split (K_eff sub-clusters)
  3. Step 4   — Post-ICM  (K_eff sub-clusters refined)
  4. Step 5   — Post-merge (K clusters)
  ... (further steps depend on n_em_iters / n_cleanup_steps / etc.)

Sub-clusters of original cluster k are coloured in the lighter tab20 shade
paired with k's dark shade, so the parent–child relationship is visible.

Usage:
    # default toy parameters
    uv run python generate_split_merge_figures.py [configs/foo.yaml ...]

    # custom MRF parameter set
    uv run python generate_split_merge_figures.py \\
        --mrf-config mrf_configs/bcs_k15.yaml \\
        configs/bcs_train.yaml

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
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.spatial import KDTree
from sklearn.metrics import adjusted_rand_score

from toy_mosaics.clustering import MRFMosaicStrategy
from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.simulate_dataset import dataset_from_config as _sim_dataset_from_config
from toy_mosaics.preprocess_dataset import dataset_from_config as _real_dataset_from_config
from figure_utils import relabel, plot_mosaic_step

_TAB20 = plt.cm.tab20(np.linspace(0, 1, 20, endpoint=False))

_DEFAULT_MRF_CONFIG = Path("mrf_configs/toy_default.yaml")


# ---------------------------------------------------------------------------
# MRF strategy builder
# ---------------------------------------------------------------------------

def _build_mrf_strategy(K: int, spatial_radius: float, mrf_cfg: dict) -> MRFMosaicStrategy:
    """Instantiate MRFMosaicStrategy from a flat parameter dict.

    n_clusters and spatial_radius are passed in (computed from the dataset).
    k_init_factor: n_clusters_init = K * k_init_factor; ignored when init=leiden.
    leiden_resolution: null/None means auto-calibrate for target K.
    """
    cfg = dict(mrf_cfg)
    init = cfg.get("init", "kmeans")
    k_init_factor = cfg.get("k_init_factor", 2)

    if init == "leiden":
        n_clusters_init = None  # Leiden determines K_init from its resolution
    else:
        n_clusters_init = K * k_init_factor

    return MRFMosaicStrategy(
        n_clusters=K,
        spatial_radius=spatial_radius,
        n_clusters_init=n_clusters_init,
        init=init,
        leiden_k_features=cfg.get("leiden_k_features", 10),
        leiden_resolution=cfg.get("leiden_resolution", 1.0),
        covariance_type=cfg.get("covariance_type", "full"),
        kde_unary=cfg.get("kde_unary", False),
        lam=cfg.get("lam"),
        lam_alpha=cfg.get("lam_alpha", 2.0),
        signed_ramp=cfg.get("signed_ramp", True),
        log_ramp=cfg.get("log_ramp", True),
        log_ramp_alpha=cfg.get("log_ramp_alpha", 10.0),
        polygon_dilation=cfg.get("polygon_dilation", 0.05),
        per_cluster_tau=cfg.get("per_cluster_tau", True),
        cf_tau_mixture=cfg.get("cf_tau_mixture", False),
        tau_low=cfg.get("tau_low"),
        tau_high=cfg.get("tau_high"),
        tau_low_q=cfg.get("tau_low_q", 0.99),
        tau_high_q=cfg.get("tau_high_q", 0.25),
        split_merge=cfg.get("split_merge", False),
        use_global_merge=cfg.get("use_global_merge", True),
        merge_max_cells_for_dist=cfg.get("merge_max_cells_for_dist", 200),
        merge_veto_frac=cfg.get("merge_veto_frac", 0.40),
        merge_min_adj_pairs=cfg.get("merge_min_adj_pairs", 0),
        merge_min_pairs_for_veto=cfg.get("merge_min_pairs_for_veto", 10),
        remerge_after_cleanup=cfg.get("remerge_after_cleanup", False),
        max_iters=cfg.get("max_iters", 30),
        icm_jacobi=cfg.get("icm_jacobi", False),
        energy_tol=cfg.get("energy_tol", 0.0),
        conf_threshold=cfg.get("conf_threshold", 0.90),
        gmm_quality_gate=cfg.get("gmm_quality_gate", 0.0),
        theta_hard=cfg.get("theta_hard"),
        n_em_iters=cfg.get("n_em_iters", 3),
        n_cleanup_steps=cfg.get("n_cleanup_steps", 10),
        n_em_iters_post_cleanup=cfg.get("n_em_iters_post_cleanup", 10),
        n_tracked_cleanup_iters=cfg.get("n_tracked_cleanup_iters", 10),
        cleanup_sigma=cfg.get("cleanup_sigma", 1.0),
        cleanup_unfreeze_frac=cfg.get("cleanup_unfreeze_frac", 0.2),
        conflict_min_posterior=cfg.get("conflict_min_posterior", 0.01),
        lam_boost=cfg.get("lam_boost"),
        count_reg=cfg.get("count_reg", 0.0),
        n_restarts=cfg.get("n_restarts", 1),
        recalibrate=cfg.get("recalibrate", False),
        random_state=0,
        n_workers=1,
        exclude_clipped=True,
    )


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

    # Pad remaining disconnected roots with dummy high-distance merges
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

    if n_actual < n_leaves - 1 and max_actual_dist > 0:
        cut = max_actual_dist * 1.2
        ax.axhline(cut, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"K={K_target} cut  (above = no merge performed)")
        ax.legend(fontsize=8, loc="upper left")

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

def process_config(config_path: Path, mrf_cfg: dict) -> None:
    """Run MRFMosaicStrategy on one mosaic config and save a step-by-step PDF."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "input" in cfg:
        dataset = _real_dataset_from_config(cfg)
        all_pts = np.concatenate(dataset.polygons)
        bounds = (all_pts[:, 0].min(), all_pts[:, 0].max(),
                  all_pts[:, 1].min(), all_pts[:, 1].max())
    else:
        dataset = _sim_dataset_from_config(cfg)
        bounds = tuple(cfg.get("mosaic", {}).get("bounds", [0, 100, 0, 100]))

    nn_dists, _ = KDTree(dataset.centers).query(dataset.centers, k=2)
    spatial_radius = 3.0 * float(np.median(nn_dists[:, 1]))

    K = dataset.n_mosaics

    # Bind mrf_cfg values needed for figure labels / conditional logic
    init                   = mrf_cfg.get("init", "kmeans")
    k_init_factor          = mrf_cfg.get("k_init_factor", 2)
    n_em_iters             = mrf_cfg.get("n_em_iters", 3)
    n_tracked_cleanup_iters = mrf_cfg.get("n_tracked_cleanup_iters", 10)
    n_cleanup_steps        = mrf_cfg.get("n_cleanup_steps", 10)
    n_em_iters_post_cleanup = mrf_cfg.get("n_em_iters_post_cleanup", 10)

    K_init = None if init == "leiden" else K * k_init_factor

    result = _build_mrf_strategy(K, spatial_radius, mrf_cfg).fit(dataset)

    model = result.model
    pm: dict[int, int] = model.get("parent_map", {})
    K_eff: int = model.get("k_after_split", K_init)
    N = len(dataset)

    # --- Collect label arrays for each pipeline step ---
    labels_gmm   = model["labels_gmm_raw"]
    labels_split = model["labels_initial"]
    labels_icm   = model["labels_post_icm"]

    labels_merged  = (
        relabel(model["labels_post_merge"], dataset.y)
        if model.get("labels_post_merge") is not None else None
    )
    labels_cleanup = (
        relabel(model["labels_post_cleanup"], dataset.y)
        if model.get("labels_post_cleanup") is not None else None
    )
    labels_em = (
        relabel(model["labels_post_em"], dataset.y)
        if model.get("labels_post_em") is not None else None
    )

    labels_all_cleanup = relabel(model["labels_post_all_cleanup"], dataset.y)

    labels_em_post_cleanup = (
        relabel(model["labels_post_em_post_cleanup"], dataset.y)
        if model.get("labels_post_em_post_cleanup") is not None else None
    )

    correction_ran = model.get("labels_post_remerge_correction") is not None
    labels_pre_correction = (
        relabel(model["labels_pre_remerge_correction"], dataset.y)
        if correction_ran else None
    )

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

    ari_em    = _ari(model["labels_post_em"]) if labels_em is not None else None
    n_err_em  = _err(labels_em)               if labels_em is not None else None

    ari_all_cleanup   = _ari(model["labels_post_all_cleanup"])
    n_err_all_cleanup = _err(labels_all_cleanup)

    ari_em_post_cleanup   = _ari(model["labels_post_em_post_cleanup"]) \
                            if labels_em_post_cleanup is not None else None
    n_err_em_post_cleanup = _err(labels_em_post_cleanup) \
                            if labels_em_post_cleanup is not None else None

    ari_pre_correction   = _ari(model["labels_pre_remerge_correction"]) if correction_ran else None
    n_err_pre_correction = _err(labels_pre_correction)                   if correction_ran else None

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

        gmm_init    = model["gmm"]
        gmm_em_fit  = model.get("gmm_post_em")
        gmm_current = gmm_em_fit or gmm_init

        # Page 0: ground truth
        _save(plot_mosaic_step(
            dataset, dataset.y,
            f"{config_path.stem} — Ground truth   K={K}",
            bounds=bounds, gmm=None,
        ))

        # Page 1: step 1+2 — initial GMM
        _save(plot_mosaic_step(
            dataset, labels_gmm,
            f"{config_path.stem} — Step 1+2: GMM (init={init})  K_init={K_init}   ARI={ari_gmm:.3f}",
            bounds=bounds, gmm=gmm_init,
        ))

        # Page 2: step 3 — post-split
        _save(plot_mosaic_step(
            dataset, labels_split,
            (f"{config_path.stem} — Step 3: post-split   "
             f"K_eff={K_eff}   n_splits={model['n_splits']}   ARI={ari_split:.3f}"),
            parent_map=pm, bounds=bounds, gmm=gmm_init,
        ))

        # Page 3: step 4 — post-ICM
        _save(plot_mosaic_step(
            dataset, labels_icm,
            (f"{config_path.stem} — Step 4: post-ICM   "
             f"K_eff={K_eff}   iters={model['n_iters']}   ARI={ari_icm:.3f}"),
            parent_map=pm, bounds=bounds, gmm=gmm_init,
        ))

        # Pages 5a / 5b: only when split_merge actually ran
        if labels_merged is not None:
            _save(plot_mosaic_step(
                dataset, labels_merged,
                (f"{config_path.stem} — Step 5a: post-merge   "
                 f"K={n_merged}   n_merges={model['n_merges']}   "
                 f"ARI={ari_merged:.3f}   errors={n_err_merged}/{N}"),
                gt_labels=dataset.y, bounds=bounds, gmm=gmm_init,
            ))

        if labels_cleanup is not None:
            _save(plot_mosaic_step(
                dataset, labels_cleanup,
                (f"{config_path.stem} — Step 5b: post-cleanup   "
                 f"K={len(np.unique(model['labels_post_cleanup']))}   "
                 f"unfrozen={model['n_unfrozen_merge']}   iters={model['n_iters_post_merge']}   "
                 f"ARI={ari_cleanup:.3f}   errors={n_err_cleanup}/{N}"),
                gt_labels=dataset.y, bounds=bounds, gmm=gmm_init,
            ))

        # Page 5c: EM re-fit
        if labels_em is not None:
            _save(plot_mosaic_step(
                dataset, labels_em,
                (f"{config_path.stem} — Step 5c: EM re-fit ×{n_em_iters}   "
                 f"ARI={ari_em:.3f}   errors={n_err_em}/{N}"),
                gt_labels=dataset.y, bounds=bounds, gmm=gmm_current,
            ))

        # Tracked cleanup page
        if n_tracked_cleanup_iters > 0:
            energy_hist  = model.get("energy_tracked_history", [])
            best_iter    = model.get("best_tracked_iter", 0)
            actual_iters = len(energy_hist) - 1
            hist_str = "  ".join(
                f"{'*' if i == best_iter else ''}{e:.1f}"
                for i, e in enumerate(energy_hist)
            )
            _cleanup_lbl  = labels_pre_correction if (correction_ran and labels_pre_correction is not None) else labels_final
            _cleanup_ari  = ari_pre_correction    if correction_ran else ari_final
            _cleanup_err  = n_err_pre_correction  if correction_ran else n_err_final
            _cleanup_tag  = "  (before remerge correction)" if correction_ran else ""
            _save(plot_mosaic_step(
                dataset, _cleanup_lbl,
                (f"{config_path.stem} — Tracked cleanup"
                 f"  ran={actual_iters}/{n_tracked_cleanup_iters}"
                 f"  best={best_iter}{_cleanup_tag}"
                 f"   ARI={_cleanup_ari:.3f}   errors={_cleanup_err}/{N}\n"
                 f"  E: {hist_str}"),
                gt_labels=dataset.y, bounds=bounds, gmm=gmm_current,
            ))
        else:
            if n_cleanup_steps > 0:
                _cleanup_lbl = labels_pre_correction if (correction_ran and labels_pre_correction is not None) else labels_all_cleanup
                _cleanup_ari = ari_pre_correction    if correction_ran else ari_all_cleanup
                _cleanup_err = n_err_pre_correction  if correction_ran else n_err_all_cleanup
                _cleanup_tag = "  (before remerge correction)" if correction_ran else ""
                _save(plot_mosaic_step(
                    dataset, _cleanup_lbl,
                    (f"{config_path.stem} — Cleanup ×{n_cleanup_steps}{_cleanup_tag}   "
                     f"changed={model['n_cleanup_changed']}/{N}   "
                     f"swaps={model['n_swaps_total']}   "
                     f"reassigned={model['n_reassigned_total']}   "
                     f"unlabeled={model['n_unlabeled_total']}   "
                     f"ARI={_cleanup_ari:.3f}   errors={_cleanup_err}/{N}"),
                    gt_labels=dataset.y, bounds=bounds, gmm=gmm_current,
                ))
            if labels_em_post_cleanup is not None:
                _save(plot_mosaic_step(
                    dataset, labels_em_post_cleanup,
                    (f"{config_path.stem} — Post-cleanup EM ×{n_em_iters_post_cleanup}   "
                     f"ARI={ari_em_post_cleanup:.3f}   errors={n_err_em_post_cleanup}/{N}"),
                    gt_labels=dataset.y, bounds=bounds, gmm=gmm_current,
                ))

        # Remerge correction page
        if correction_ran:
            _save(plot_mosaic_step(
                dataset, labels_final,
                (f"{config_path.stem} — Remerge correction   "
                 f"ARI before={ari_pre_correction:.3f} → after={ari_final:.3f}   "
                 f"errors {n_err_pre_correction}→{n_err_final}/{N}"),
                gt_labels=dataset.y, bounds=bounds, gmm=gmm_current,
            ))

        # Final page: merge hierarchy dendrogram
        merge_history: list[tuple[int, int, float]] = model.get("merge_history", [])
        _save(_plot_merge_dendrogram(
            dataset, labels_icm, merge_history, K,
            (f"{config_path.stem} — Merge hierarchy   "
             f"K_eff={K_eff} -> K={len(np.unique(result.labels))}   "
             f"n_merges={model['n_merges']}"),
        ))

    # --- Console report ---
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
    if n_tracked_cleanup_iters > 0:
        energy_hist  = model.get("energy_tracked_history", [])
        best_iter    = model.get("best_tracked_iter", 0)
        actual_iters = len(energy_hist) - 1
        _pre_ari = ari_pre_correction if correction_ran else ari_final
        _pre_err = n_err_pre_correction if correction_ran else n_err_final
        lines.append(
            f"  Tracked   ran={actual_iters}/{n_tracked_cleanup_iters}   "
            f"best={best_iter}   "
            f"E_best={energy_hist[best_iter]:.2f}   E_final={energy_hist[-1]:.2f}   "
            f"ARI={_pre_ari:.3f}   errors={_pre_err}/{N}"
        )
    elif n_cleanup_steps > 0:
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
    else:
        lines.append(
            f"  Final     result:     ARI={ari_final:.3f}   errors={n_err_final}/{N}"
        )
    if correction_ran:
        lines.append(
            f"  Remerge   correction: ARI {ari_pre_correction:.3f} → {ari_final:.3f}   "
            f"errors {n_err_pre_correction} → {n_err_final}/{N}"
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
        help="Mosaic YAML config files (default: all configs/*.yaml)",
    )
    parser.add_argument(
        "--mrf-config", type=Path, default=_DEFAULT_MRF_CONFIG,
        metavar="PATH",
        help=(
            f"MRF parameter YAML (default: {_DEFAULT_MRF_CONFIG}).  "
            "See mrf_configs/ for examples."
        ),
    )
    args = parser.parse_args()

    if not args.mrf_config.exists():
        print(f"MRF config not found: {args.mrf_config}", file=sys.stderr)
        sys.exit(1)

    with open(args.mrf_config) as f:
        mrf_cfg = yaml.safe_load(f)

    print(f"MRF config: {args.mrf_config}")

    config_paths = args.configs or sorted(Path("configs").glob("*.yaml"))
    if not config_paths:
        print("No mosaic config files found.", file=sys.stderr)
        sys.exit(1)

    for config_path in config_paths:
        process_config(config_path, mrf_cfg)


if __name__ == "__main__":
    main()
