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
from sklearn.metrics import adjusted_rand_score

from scipy.spatial import KDTree

from toy_mosaics.clustering import GMMStrategy, LeidenMosaicStrategy, LeidenProximityStrategy, WardMosaicStrategy, MRFMosaicStrategy
from toy_mosaics.simulate_dataset import dataset_from_config
from figure_utils import relabel, plot_mosaic_step


def _figure_path(cfg: dict, config_path: Path) -> Path:
    filename = cfg.get("output", {}).get("filename", config_path.stem + ".npz")
    return Path("figures") / Path(filename).with_suffix(".pdf")


def process_config(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = dataset_from_config(cfg)
    bounds = tuple(cfg.get("mosaic", {}).get("bounds", [0, 100, 0, 100]))

    result = GMMStrategy(n_clusters=dataset.n_mosaics).fit(dataset)
    labels_gmm = relabel(result.labels, dataset.y)

    # spatial_radius: 3× median nearest-neighbour distance between cell centres
    nn_dists, _ = KDTree(dataset.centers).query(dataset.centers, k=2)
    spatial_radius = 3.0 * float(np.median(nn_dists[:, 1]))
    leiden_result = LeidenMosaicStrategy(
        n_clusters=dataset.n_mosaics,
        spatial_radius=spatial_radius,
    ).fit(dataset)
    labels_leiden = relabel(leiden_result.labels, dataset.y)
    leiden_meta = leiden_result.model

    proximity_result = LeidenProximityStrategy(
        n_clusters=dataset.n_mosaics,
        spatial_radius=spatial_radius,
    ).fit(dataset)
    labels_proximity = relabel(proximity_result.labels, dataset.y)
    proximity_meta = proximity_result.model

    ward_base_result = WardMosaicStrategy(
        n_clusters=dataset.n_mosaics,
        spatial_radius=spatial_radius,
        lam=0.0,
    ).fit(dataset)
    labels_ward_base = relabel(ward_base_result.labels, dataset.y)

    ward_result = WardMosaicStrategy(
        n_clusters=dataset.n_mosaics,
        spatial_radius=spatial_radius,
        lam=1.0,
    ).fit(dataset)
    labels_ward = relabel(ward_result.labels, dataset.y)

    mrf_result = MRFMosaicStrategy(
        n_clusters=dataset.n_mosaics,
        spatial_radius=spatial_radius,
        split_merge=True,
        init='leiden',
        n_em_iters=1,
        n_cleanup_steps=3,
    ).fit(dataset)
    labels_mrf = relabel(mrf_result.labels, dataset.y)

    out_path = _figure_path(cfg, config_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _meta_str(meta: dict) -> str:
        s = "converged" if meta["converged"] else f"did not converge ({meta['n_iters_run']} iters)"
        if meta["n_merges"] > 0:
            s += f", {meta['n_merges']} merge(s)"
        return s

    ari_gmm = adjusted_rand_score(dataset.y, labels_gmm)
    ari_leiden = adjusted_rand_score(dataset.y, labels_leiden)
    ari_proximity = adjusted_rand_score(dataset.y, labels_proximity)
    ari_ward_base = adjusted_rand_score(dataset.y, labels_ward_base)
    ari_ward = adjusted_rand_score(dataset.y, labels_ward)
    ari_mrf = adjusted_rand_score(dataset.y, labels_mrf)

    with PdfPages(out_path) as pdf:
        fig_gt = plot_mosaic_step(dataset, dataset.y, title=f"{config_path.stem} — ground truth", bounds=bounds)
        pdf.savefig(fig_gt, bbox_inches="tight")
        plt.close(fig_gt)

        fig_gmm = plot_mosaic_step(dataset, labels_gmm, title=f"{config_path.stem} — GMM  ARI={ari_gmm:.3f}", gt_labels=dataset.y, bounds=bounds)
        pdf.savefig(fig_gmm, bbox_inches="tight")
        plt.close(fig_gmm)

        fig_leiden = plot_mosaic_step(
            dataset, labels_leiden,
            title=f"{config_path.stem} — Leiden/coverage  ARI={ari_leiden:.3f}  ({_meta_str(leiden_meta)})",
            gt_labels=dataset.y, bounds=bounds,
        )
        pdf.savefig(fig_leiden, bbox_inches="tight")
        plt.close(fig_leiden)

        fig_proximity = plot_mosaic_step(
            dataset, labels_proximity,
            title=f"{config_path.stem} — Leiden/proximity  ARI={ari_proximity:.3f}  ({_meta_str(proximity_meta)})",
            gt_labels=dataset.y, bounds=bounds,
        )
        pdf.savefig(fig_proximity, bbox_inches="tight")
        plt.close(fig_proximity)

        fig_ward_base = plot_mosaic_step(
            dataset, labels_ward_base,
            title=f"{config_path.stem} — Ward lam=0  ARI={ari_ward_base:.3f}",
            gt_labels=dataset.y, bounds=bounds,
        )
        pdf.savefig(fig_ward_base, bbox_inches="tight")
        plt.close(fig_ward_base)

        fig_ward = plot_mosaic_step(
            dataset, labels_ward,
            title=f"{config_path.stem} — Ward lam=1  ARI={ari_ward:.3f}",
            gt_labels=dataset.y, bounds=bounds,
        )
        pdf.savefig(fig_ward, bbox_inches="tight")
        plt.close(fig_ward)

        fig_mrf = plot_mosaic_step(
            dataset, labels_mrf,
            title=f"{config_path.stem} — GMM+MRF  ARI={ari_mrf:.3f}  (viol {mrf_result.model['violations_before']}->{mrf_result.model['violations_after']})",
            gt_labels=dataset.y, bounds=bounds,
        )
        pdf.savefig(fig_mrf, bbox_inches="tight")
        plt.close(fig_mrf)

    print(
        f"{config_path.name} -> {out_path}\n"
        f"  GMM={ari_gmm:.3f}  MRF={ari_mrf:.3f}  "
        f"Leiden/cov={ari_leiden:.3f}  Leiden/prox={ari_proximity:.3f}  "
        f"Ward(lam=0)={ari_ward_base:.3f}  Ward(lam=1)={ari_ward:.3f}"
    )


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
