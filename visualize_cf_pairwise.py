"""Visualize coverage-fraction (CF) distribution and pairwise costs.

Two-panel figure per config:

  Top — Ramp overlay scatter
    Both ramp curves (dead-zone and signed) plotted as lines.
    Every same-type cell pair within spatial_radius plotted as a dot at
    (CF, signed_ramp(CF)), coloured by ground-truth type.
    Different-type pairs shown as a rug below the axis.
    Gallery pairs labelled A–F.

  Bottom — Pair gallery
    6 representative pairs spanning the CF range (low / near-τ_low / above-τ_low
    / max, plus 2 diff-type).  Each panel shows both Voronoi polygons, the
    shaded intersection, CF value, and pairwise cost.

Usage:
    uv run python visualize_cf_pairwise.py [configs/anisotropic.yaml ...]
    uv run python visualize_cf_pairwise.py            # all configs/*.yaml
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import yaml
from scipy.spatial import KDTree
from shapely.geometry import Polygon as ShapelyPolygon

from toy_mosaics.leiden_mosaic import _build_coverage_map
from toy_mosaics.mrf_mosaic import _ramp, _signed_ramp
from toy_mosaics.simulate_dataset import dataset_from_config

_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

# Matching MRFMosaicStrategy defaults and generate_mrf_figures.py usage
TAU_LOW_PLAIN = 0.10
TAU_LOW_SIGNED = 0.30
TAU_HIGH = 0.40


# ---------------------------------------------------------------------------
# Gallery pair selection
# ---------------------------------------------------------------------------

def _pick_gallery_pairs(pairs_info: list, n: int = 6) -> list:
    """Return up to n pairs spanning CF range and both type relationships.

    Uses percentile-based selection so the gallery adapts to the actual CF
    distribution rather than assuming pairs straddle τ_low.
    Targets: 4 same-type across p5/p35/p70/p100, 2 diff-type at p5/p95.
    """
    same = sorted((cf, i, j, True)  for cf, i, j, s in pairs_info if s)
    diff = sorted((cf, i, j, False) for cf, i, j, s in pairs_info if not s)

    def _at_percentile(pool, pct):
        if not pool:
            return None
        idx = int(round(pct / 100 * (len(pool) - 1)))
        return pool[max(0, min(idx, len(pool) - 1))]

    candidates = [
        p for p in [
            _at_percentile(same, 5),
            _at_percentile(same, 35),
            _at_percentile(same, 70),
            _at_percentile(same, 100),
            _at_percentile(diff,  5),
            _at_percentile(diff, 95),
        ] if p is not None
    ]

    seen: set = set()
    result = []
    for item in candidates:
        key = (item[1], item[2])
        if key not in seen and len(result) < n:
            seen.add(key)
            result.append(item)

    return sorted(result, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_pair(ax, poly_i, poly_j, ci, cj, cf: float, cost: float, same: bool) -> None:
    """Draw two polygons with intersection shaded, zoomed to the pair bounding box."""
    for poly, color, lw in [(poly_i, ci, 1.2), (poly_j, cj, 1.2)]:
        ax.add_patch(MplPolygon(
            poly, closed=True, facecolor=color, edgecolor="black",
            linewidth=lw, alpha=0.45, zorder=2,
        ))

    # Intersection via shapely
    try:
        sp_i = ShapelyPolygon(poly_i)
        sp_j = ShapelyPolygon(poly_j)
        isect = sp_i.intersection(sp_j)
        if not isect.is_empty and isect.area > 0:
            geoms = [isect] if isect.geom_type == "Polygon" else list(isect.geoms)
            for g in geoms:
                coords = np.array(g.exterior.coords)
                ax.add_patch(MplPolygon(
                    coords, closed=True, facecolor="#555", edgecolor="#c00",
                    linewidth=1.5, alpha=0.65, zorder=4,
                ))
    except Exception:
        pass

    all_pts = np.vstack([poly_i, poly_j])
    pad = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.12 + 1
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    cost_plain = _ramp(cf, TAU_LOW_PLAIN, TAU_HIGH)
    rel = "same-type" if same else "diff-type"
    sign = "+" if cost >= 0 else ""
    ax.set_title(
        f"CF={cf:.3f}\nsigned={sign}{cost:.2f}  plain={cost_plain:.2f}\n{rel}",
        fontsize=8, pad=3,
    )


# ---------------------------------------------------------------------------
# Main per-config routine
# ---------------------------------------------------------------------------

def process_config(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = dataset_from_config(cfg)

    nn_dists, _ = KDTree(dataset.centers).query(dataset.centers, k=2)
    spatial_radius = 3.0 * float(np.median(nn_dists[:, 1]))

    raw_map = _build_coverage_map(
        dataset.polygons, dataset.centers, spatial_radius,
        dataset.clipped, exclude_clipped=True,
    )

    pairs_info = [
        (cf, i, j, bool(dataset.y[i] == dataset.y[j]))
        for (i, j), cf in raw_map.items()
    ]

    gallery = _pick_gallery_pairs(pairs_info)
    labels = list("ABCDEF")[: len(gallery)]

    n_same = sum(s for _, _, _, s in pairs_info)
    n_diff = len(pairs_info) - n_same

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.3, 1.0], hspace=0.55, wspace=0.30)
    ax_ramp = fig.add_subplot(gs[0, :])
    ax_gal = [fig.add_subplot(gs[1, k]) for k in range(6)]

    # --- Ramp curves ---
    cf_grid = np.linspace(0, 1, 600)
    cost_signed = np.array([_signed_ramp(c, TAU_LOW_SIGNED, TAU_HIGH) for c in cf_grid])
    cost_plain  = np.array([_ramp(c, TAU_LOW_PLAIN,  TAU_HIGH) for c in cf_grid])

    # Shade attractive and repulsive zones relative to the signed-ramp curve
    mask_attr = cf_grid <= TAU_LOW_SIGNED
    mask_rep  = cf_grid >= TAU_LOW_SIGNED
    ax_ramp.fill_between(cf_grid[mask_attr], cost_signed[mask_attr], 0,
                         color="#e8f5e9", alpha=0.75, zorder=0, label="_nolegend_")
    ax_ramp.fill_between(cf_grid[mask_rep],  cost_signed[mask_rep],  0,
                         color="#ffebee", alpha=0.55, zorder=0, label="_nolegend_")

    ax_ramp.plot(cf_grid, cost_plain,  "--", color="#aaa", lw=1.8, zorder=2,
                 label=f"dead-zone ramp  (τ_low={TAU_LOW_PLAIN}, τ_high={TAU_HIGH})")
    ax_ramp.plot(cf_grid, cost_signed, "-",  color="#2a6496", lw=2.2, zorder=3,
                 label=f"signed-ramp  (τ_low={TAU_LOW_SIGNED}, τ_high={TAU_HIGH})")

    # Scatter same-type pairs
    for cf, i, j, same in pairs_info:
        if same:
            c = _COLORS[int(dataset.y[i]) % 10]
            ax_ramp.scatter(cf, _signed_ramp(cf, TAU_LOW_SIGNED, TAU_HIGH),
                            color=c, s=18, alpha=0.45, linewidths=0, zorder=4)

    # Rug for diff-type pairs
    diff_cfs = [cf for cf, _, _, s in pairs_info if not s]
    if diff_cfs:
        ax_ramp.scatter(diff_cfs, np.full(len(diff_cfs), -1.18),
                        marker="|", color="#888", s=25, alpha=0.35, linewidths=0.8,
                        zorder=3, label=f"diff-type pairs (rug, n={n_diff})")

    # Gallery markers
    for lbl, (cf, i, j, same) in zip(labels, gallery):
        y = _signed_ramp(cf, TAU_LOW_SIGNED, TAU_HIGH)
        ax_ramp.scatter(cf, y, s=90, color="white", edgecolors="black", linewidths=1.5, zorder=6)
        ax_ramp.text(cf, y + 0.1, lbl, ha="center", va="bottom",
                     fontsize=9, fontweight="bold", zorder=7)

    # Reference lines
    ax_ramp.axvline(TAU_LOW_SIGNED, color="#2a6496", lw=0.9, ls=":", alpha=0.7)
    ax_ramp.axvline(TAU_HIGH,       color="#888",    lw=0.9, ls=":", alpha=0.7)
    ax_ramp.axhline(0,              color="#444",    lw=0.6, alpha=0.5)

    ax_ramp.set_xlim(-0.02, 1.02)
    ax_ramp.set_ylim(-1.32, 1.2)
    ax_ramp.set_xlabel("Coverage fraction CF", fontsize=11)
    ax_ramp.set_ylabel("Pairwise cost φ · [same-type]", fontsize=11)
    ax_ramp.set_title(
        f"{config_path.stem} — {len(pairs_info)} pairs within radius {spatial_radius:.1f}"
        f"  ({n_same} same-type  ·  {n_diff} diff-type)",
        fontsize=10,
    )
    ax_ramp.legend(fontsize=9, loc="upper left")

    ax_ramp.text(TAU_LOW_SIGNED, -1.29, f"τ_low\n{TAU_LOW_SIGNED}",
                 ha="center", va="bottom", fontsize=8, color="#2a6496")
    ax_ramp.text(TAU_HIGH, -1.29, f"τ_high\n{TAU_HIGH}",
                 ha="center", va="bottom", fontsize=8, color="#888")

    # Zone annotations
    ax_ramp.text(TAU_LOW_SIGNED * 0.5, -0.55, "attractive\n(same-tile)",
                 ha="center", va="center", fontsize=8, color="#2e7d32", style="italic")
    ax_ramp.text((TAU_LOW_SIGNED + TAU_HIGH) * 0.5, 0.5, "repulsive\n(territorial overlap)",
                 ha="center", va="center", fontsize=8, color="#c62828", style="italic")

    # --- Gallery ---
    for k, (lbl, (cf, i, j, same)) in enumerate(zip(labels, gallery)):
        cost = _signed_ramp(cf, TAU_LOW_SIGNED, TAU_HIGH)
        ci = _COLORS[int(dataset.y[i]) % 10]
        cj = _COLORS[int(dataset.y[j]) % 10]
        _draw_pair(ax_gal[k], dataset.polygons[i], dataset.polygons[j],
                   ci, cj, cf, cost, same)
        ax_gal[k].set_xlabel(f"[{lbl}]  cells {i} & {j}", fontsize=8)

    for k in range(len(gallery), 6):
        ax_gal[k].set_visible(False)

    stem = cfg.get("output", {}).get("filename", config_path.stem + ".npz")
    out_path = Path("figures") / (Path(stem).with_suffix("").name + "_cf_pairwise.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    same_cfs = [cf for cf, _, _, s in pairs_info if s]
    n_same_above_tau = sum(c > TAU_LOW_SIGNED for c in same_cfs)
    note = (
        f"  NOTE: all same-type pairs have CF < tau_low_signed={TAU_LOW_SIGNED} "
        "(signed-ramp is fully attractive for same-type)"
        if n_same_above_tau == 0 and same_cfs
        else f"  {n_same_above_tau}/{n_same} same-type pairs above tau_low_signed={TAU_LOW_SIGNED}"
    )
    print(f"{config_path.name} -> {out_path}  ({n_same} same-type, {n_diff} diff-type pairs)\n{note}")


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
