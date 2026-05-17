"""Simulate a mosaic dataset from a YAML config and save to data/.

Usage:
    uv run python -m toy_mosaics.simulate_dataset configs/example.yaml

Loading saved data:
    data = np.load("data/example.npz", allow_pickle=False)
    groups, centers, X, y = data["groups"], data["centers"], data["X"], data["y"]
    clipped = data["clipped"]
    # Reconstruct ragged polygon list:
    verts, offs = data["polygon_vertices"], data["polygon_offsets"]
    polygons = [verts[offs[i]:offs[i+1]] for i in range(len(offs) - 1)]
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.simulate import simulate_rgc_mosaics


def _broadcast(value, n, name):
    """Accept a scalar or a list; always return a list of length n."""
    if isinstance(value, (int, float, str)):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"'{name}' has {len(value)} entries but n_mosaics={n}")
    return value


def _make_covariance(n_dims, std, aspect_ratio, rotation_deg):
    """Build a covariance matrix for one cluster.

    std controls the overall scale; aspect_ratio > 1 elongates along the first
    axis; rotation_deg rotates the ellipse (only applied in 2D).
    """
    std_major = float(std)
    std_minor = std_major / float(aspect_ratio)
    variances = [std_major**2, std_minor**2] + [std_major**2] * max(0, n_dims - 2)
    cov = np.diag(variances[:n_dims])
    if n_dims == 2 and rotation_deg:
        angle = float(rotation_deg) * np.pi / 180.0
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        cov = R @ cov @ R.T
    return cov


def _generate_features(groups, fc):
    n_dims = fc.get("n_dims", 2)
    spread = float(fc["spread"])
    center_jitter = fc.get("center_jitter", 0.0)

    u_groups, n_per_group = np.unique(groups, return_counts=True)
    n_groups = int(len(u_groups))

    # Broadcast per-cluster covariance params (scalar or list of length n_groups)
    stds = _broadcast(fc["std"], n_groups, "std")
    aspect_ratios = _broadcast(fc.get("aspect_ratio", 1.0), n_groups, "aspect_ratio")
    rotation = fc.get("rotation", None)
    if rotation == "random":
        rot_angles = list(np.random.uniform(0, 360, n_groups))
    else:
        rot_angles = _broadcast(rotation if rotation is not None else 0.0, n_groups, "rotation")

    # Place centers equidistantly on a circle (first 2 dims); rest are zero
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
    centers = np.zeros((n_groups, n_dims))
    centers[:, 0] = spread * np.cos(angles)
    centers[:, 1] = spread * np.sin(angles)
    if center_jitter:
        centers += np.random.normal(0, float(center_jitter), centers.shape)

    X_parts, y_parts = [], []
    for i, (n_i, center) in enumerate(zip(n_per_group, centers)):
        cov = _make_covariance(n_dims, stds[i], aspect_ratios[i], rot_angles[i])
        X_parts.append(np.random.multivariate_normal(center, cov, size=n_i))
        y_parts.append(np.full(n_i, i))
    return np.concatenate(X_parts), np.concatenate(y_parts)



def dataset_from_config(cfg: dict) -> MosaicDataset:
    """Build a :class:`MosaicDataset` from a parsed YAML config dict."""
    if seed := cfg.get("seed"):
        np.random.seed(seed)

    mc = cfg["mosaic"]
    n = mc["n_mosaics"]
    groups, polygons, centers, clipped = simulate_rgc_mosaics(
        n_mosaics=n,
        mean_diameters=_broadcast(mc["mean_diameters"], n, "mean_diameters"),
        center_noise=mc["center_noise"],
        diameter_noise=mc["diameter_noise"],
        n_missing_list=_broadcast(mc.get("n_missing_list", 0), n, "n_missing_list"),
        overlap_factors=_broadcast(mc.get("overlap_factors", 1.0), n, "overlap_factors"),
    )
    X, y = _generate_features(groups, cfg.get("features", {}))
    return MosaicDataset(groups=groups, centers=centers, clipped=clipped, polygons=polygons, X=X, y=y)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = dataset_from_config(cfg)

    filename = cfg.get("output", {}).get("filename", args.config.stem + ".npz")
    out_path = Path("data") / filename
    dataset.save(out_path)
    print(f"Saved {len(dataset)} cells to {out_path}")


if __name__ == "__main__":
    main()
