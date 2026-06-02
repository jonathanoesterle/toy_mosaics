"""Generate MRF diagnostic PDFs for all YAML configs.

Without --mrf-config: runs 4 hardcoded ramp variants and saves 9 pages.
With    --mrf-config: runs a single configured MRF and saves 3 pages
                      (ground truth, result, diagnostics).

Usage:
    uv run python generate_mrf_figures.py [configs/foo.yaml ...]
    uv run python generate_mrf_figures.py --mrf-config mrf_configs/bcs_k15.yaml configs/bcs_train.yaml

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
from scipy.spatial import KDTree
from sklearn.metrics import adjusted_rand_score

from toy_mosaics.clustering import MRFMosaicStrategy
from toy_mosaics.plot import plot_mrf_diagnostics
from toy_mosaics.simulate_dataset import dataset_from_config as _sim_dataset_from_config
from toy_mosaics.preprocess_dataset import dataset_from_config as _real_dataset_from_config
from figure_utils import relabel, plot_mosaic_step, build_mrf_strategy, load_mrf_cfg, get_X_2d, DEFAULT_MRF_CONFIG


_SIGNED_TAU_LOW = 0.30
_THETA_HARD = 0.75


def _run_and_relabel(dataset, spatial_radius, **kwargs):
    result = MRFMosaicStrategy(
        n_clusters=dataset.n_mosaics,
        spatial_radius=spatial_radius,
        **kwargs,
    ).fit(dataset)
    labels = relabel(result.labels, dataset.y)
    ari = adjusted_rand_score(dataset.y, labels)
    return result, labels, ari


def _tau_str(model: dict, emp_low=None, emp_high=None) -> str:
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


# ---------------------------------------------------------------------------
# Single-config run (used when --mrf-config is provided)
# ---------------------------------------------------------------------------

def _load_dataset(config_path: Path, cfg: dict):
    if "input" in cfg:
        dataset = _real_dataset_from_config(cfg)
        all_pts = np.concatenate(dataset.polygons)
        bounds = (all_pts[:, 0].min(), all_pts[:, 0].max(),
                  all_pts[:, 1].min(), all_pts[:, 1].max())
    else:
        dataset = _sim_dataset_from_config(cfg)
        bounds = tuple(cfg.get("mosaic", {}).get("bounds", [0, 100, 0, 100]))
    return dataset, bounds


def process_config_with_mrf_cfg(config_path: Path, mrf_cfg: dict) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset, bounds = _load_dataset(config_path, cfg)
    nn_dists, _ = KDTree(dataset.centers).query(dataset.centers, k=2)
    spatial_radius = 3.0 * float(np.median(nn_dists[:, 1]))
    vis_X2d, vis_labels = get_X_2d(cfg, mrf_cfg)

    result = build_mrf_strategy(dataset.n_mosaics, spatial_radius, mrf_cfg).fit(dataset)
    labels = relabel(result.labels, dataset.y)
    ari = adjusted_rand_score(dataset.y, labels)
    model = result.model

    raw_map = model["raw_map"]
    same_cfs_gt = [cf for (i, j), cf in raw_map.items() if dataset.y[i] == dataset.y[j]]
    diff_cfs_gt = [cf for (i, j), cf in raw_map.items() if dataset.y[i] != dataset.y[j]]
    emp_low  = float(np.quantile(same_cfs_gt, 0.99)) if len(same_cfs_gt) >= 10 else None
    emp_high = float(np.quantile(diff_cfs_gt, 0.25)) if len(diff_cfs_gt) >= 10 else None

    stem = cfg.get("output", {}).get("filename", config_path.stem + ".npz")
    out_path = Path("figures") / (Path(stem).stem + "_mrf.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        fig = plot_mosaic_step(dataset, dataset.y,
                               title=f"{config_path.stem} — ground truth", bounds=bounds,
                               X_2d=vis_X2d, feat_labels=vis_labels)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig = plot_mosaic_step(
            dataset, labels,
            title=(f"{config_path.stem} — MRF   ARI={ari:.3f}   "
                   f"violations {model['violations_before']} -> {model['violations_after']}   "
                   f"changed={model['n_changed']}   iters={model['n_iters']}"),
            gt_labels=dataset.y, bounds=bounds,
            X_2d=vis_X2d, feat_labels=vis_labels,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig_diag, _ = plot_mrf_diagnostics(dataset, result, ground_truth=dataset.y)
        fig_diag.suptitle(f"{config_path.stem} — MRF diagnostics", fontsize=13, y=1.01)
        pdf.savefig(fig_diag, bbox_inches="tight"); plt.close(fig_diag)

    print(
        f"{config_path.name} -> {out_path}\n"
        f"  ARI={ari:.3f}   errors={int((labels != dataset.y).sum())}/{len(dataset)}   "
        f"violations {model['violations_before']}->{model['violations_after']}   "
        f"changed={model['n_changed']}   iters={model['n_iters']}   "
        f"frozen={model['n_frozen']}   radius={spatial_radius:.1f}\n"
        f"  {_tau_str(model, emp_low, emp_high)}"
    )


# ---------------------------------------------------------------------------
# Four-variant run (legacy, used when --mrf-config is not provided)
# ---------------------------------------------------------------------------

def process_config(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset, bounds = _load_dataset(config_path, cfg)
    nn_dists, _ = KDTree(dataset.centers).query(dataset.centers, k=2)
    spatial_radius = 3.0 * float(np.median(nn_dists[:, 1]))

    result,    labels_mrf,    ari    = _run_and_relabel(dataset, spatial_radius)
    result_s,  labels_mrf_s,  ari_s  = _run_and_relabel(dataset, spatial_radius, signed_ramp=True, tau_low=_SIGNED_TAU_LOW)
    result_l,  labels_mrf_l,  ari_l  = _run_and_relabel(dataset, spatial_radius, signed_ramp=True, tau_low=_SIGNED_TAU_LOW, log_ramp=True, log_ramp_alpha=10.0)
    result_h1, labels_mrf_h1, ari_h1 = _run_and_relabel(dataset, spatial_radius, signed_ramp=True, tau_low=_SIGNED_TAU_LOW, log_ramp=True, log_ramp_alpha=10.0, theta_hard=_THETA_HARD)

    model, model_s, model_l, model_h1 = result.model, result_s.model, result_l.model, result_h1.model

    raw_map = result.model["raw_map"]
    same_cfs_gt = [cf for (i, j), cf in raw_map.items() if dataset.y[i] == dataset.y[j]]
    diff_cfs_gt = [cf for (i, j), cf in raw_map.items() if dataset.y[i] != dataset.y[j]]
    emp_low  = float(np.quantile(same_cfs_gt, 0.99)) if len(same_cfs_gt) >= 10 else None
    emp_high = float(np.quantile(diff_cfs_gt, 0.25)) if len(diff_cfs_gt) >= 10 else None

    stem = cfg.get("output", {}).get("filename", config_path.stem + ".npz")
    out_path = Path("figures") / (Path(stem).stem + "_mrf.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        def _save(fig): pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        _save(plot_mosaic_step(dataset, dataset.y, title=f"{config_path.stem} — ground truth", bounds=bounds))

        _save(plot_mosaic_step(dataset, labels_mrf, gt_labels=dataset.y, bounds=bounds, title=(
            f"{config_path.stem} — GMM + MRF   ARI={ari:.3f}   "
            f"violations {model['violations_before']} -> {model['violations_after']}   "
            f"changed={model['n_changed']}   iters={model['n_iters']}")))
        fig_diag, _ = plot_mrf_diagnostics(dataset, result, ground_truth=dataset.y)
        fig_diag.suptitle(f"{config_path.stem} — MRF diagnostics", fontsize=13, y=1.01)
        _save(fig_diag)

        _save(plot_mosaic_step(dataset, labels_mrf_s, gt_labels=dataset.y, bounds=bounds, title=(
            f"{config_path.stem} — GMM + MRF (signed ramp, tau_low={_SIGNED_TAU_LOW})   ARI={ari_s:.3f}   "
            f"violations {model_s['violations_before']} -> {model_s['violations_after']}   "
            f"changed={model_s['n_changed']}   iters={model_s['n_iters']}")))
        fig_diag_s, _ = plot_mrf_diagnostics(dataset, result_s, ground_truth=dataset.y)
        fig_diag_s.suptitle(f"{config_path.stem} — MRF diagnostics (signed ramp)", fontsize=13, y=1.01)
        _save(fig_diag_s)

        _save(plot_mosaic_step(dataset, labels_mrf_l, gt_labels=dataset.y, bounds=bounds, title=(
            f"{config_path.stem} — GMM + MRF (signed log-ramp α=10, tau_low={_SIGNED_TAU_LOW})   ARI={ari_l:.3f}   "
            f"violations {model_l['violations_before']} -> {model_l['violations_after']}   "
            f"changed={model_l['n_changed']}   iters={model_l['n_iters']}")))
        fig_diag_l, _ = plot_mrf_diagnostics(dataset, result_l, ground_truth=dataset.y)
        fig_diag_l.suptitle(f"{config_path.stem} — MRF diagnostics (signed log-ramp α=10)", fontsize=13, y=1.01)
        _save(fig_diag_l)

        _save(plot_mosaic_step(dataset, labels_mrf_h1, gt_labels=dataset.y, bounds=bounds, title=(
            f"{config_path.stem} — GMM + MRF (signed log-ramp a=10 + H1 th={_THETA_HARD})   ARI={ari_h1:.3f}   "
            f"violations {model_h1['violations_before']} -> {model_h1['violations_after']}   "
            f"changed={model_h1['n_changed']}   iters={model_h1['n_iters']}   "
            f"force_unfrozen={model_h1['n_force_unfrozen']}")))
        fig_diag_h1, _ = plot_mrf_diagnostics(dataset, result_h1, ground_truth=dataset.y)
        fig_diag_h1.suptitle(f"{config_path.stem} — MRF diagnostics (signed log-ramp + H1)", fontsize=13, y=1.01)
        _save(fig_diag_h1)

    n_err, n_err_s, n_err_l, n_err_h1 = (
        int((labels_mrf    != dataset.y).sum()),
        int((labels_mrf_s  != dataset.y).sum()),
        int((labels_mrf_l  != dataset.y).sum()),
        int((labels_mrf_h1 != dataset.y).sum()),
    )
    print(
        f"{config_path.name} -> {out_path}\n"
        f"  plain:      ARI={ari:.3f}   errors={n_err}/{len(dataset)}   "
        f"violations {model['violations_before']}->{model['violations_after']}   "
        f"changed={model['n_changed']}   iters={model['n_iters']}   frozen={model['n_frozen']}   radius={spatial_radius:.1f}\n"
        f"    {_tau_str(model, emp_low, emp_high)}\n"
        f"  signed:     ARI={ari_s:.3f}   errors={n_err_s}/{len(dataset)}   "
        f"violations {model_s['violations_before']}->{model_s['violations_after']}   "
        f"changed={model_s['n_changed']}   iters={model_s['n_iters']}   frozen={model_s['n_frozen']}\n"
        f"    {_tau_str(model_s, emp_low, emp_high)}\n"
        f"  signed+log: ARI={ari_l:.3f}   errors={n_err_l}/{len(dataset)}   "
        f"violations {model_l['violations_before']}->{model_l['violations_after']}   "
        f"changed={model_l['n_changed']}   iters={model_l['n_iters']}   frozen={model_l['n_frozen']}\n"
        f"    {_tau_str(model_l, emp_low, emp_high)}\n"
        f"  +H1(th={_THETA_HARD}): ARI={ari_h1:.3f}   errors={n_err_h1}/{len(dataset)}   "
        f"violations {model_h1['violations_before']}->{model_h1['violations_after']}   "
        f"changed={model_h1['n_changed']}   iters={model_h1['n_iters']}   "
        f"frozen={model_h1['n_frozen']}   force_unfrozen={model_h1['n_force_unfrozen']}\n"
        f"    {_tau_str(model_h1, emp_low, emp_high)}"
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
    parser.add_argument(
        "--mrf-config", type=Path, default=None, metavar="PATH",
        help=(
            "MRF parameter YAML.  When provided, runs a single configured MRF "
            "(3 pages: GT, result, diagnostics).  "
            "Without this flag, runs 4 hardcoded ramp variants (9 pages).  "
            f"See mrf_configs/ for examples."
        ),
    )
    args = parser.parse_args()

    config_paths = args.configs or sorted(Path("configs").glob("*.yaml"))
    if not config_paths:
        print("No config files found.", file=sys.stderr)
        sys.exit(1)

    if args.mrf_config is not None:
        if not args.mrf_config.exists():
            print(f"MRF config not found: {args.mrf_config}", file=sys.stderr)
            sys.exit(1)
        mrf_cfg = load_mrf_cfg(args.mrf_config)
        print(f"MRF config: {args.mrf_config}")
        for config_path in config_paths:
            process_config_with_mrf_cfg(config_path, mrf_cfg)
    else:
        for config_path in config_paths:
            process_config(config_path)


if __name__ == "__main__":
    main()
