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

from toy_mosaics.simulate import simulate_rgc_mosaics


def _generate_features(groups, feat_configs):
    u_groups, n_per_group = np.unique(groups, return_counts=True)
    if len(feat_configs) != len(u_groups):
        raise ValueError(
            f"Config has {len(feat_configs)} feature entries but "
            f"simulation produced {len(u_groups)} groups"
        )
    X_parts, y_parts = [], []
    for i, (n_i, fc) in enumerate(zip(n_per_group, feat_configs)):
        center = np.array(fc["center"], dtype=float)
        std = np.array(fc["std"], dtype=float)
        X_parts.append(np.random.normal(center, std, size=(n_i, len(center))))
        y_parts.append(np.full(n_i, i))
    return np.concatenate(X_parts), np.concatenate(y_parts)


def _pack_polygons(polygons):
    """Serialize ragged polygon list to flat array + offsets for .npz storage."""
    vertices = np.concatenate([p for p in polygons], axis=0)
    offsets = np.zeros(len(polygons) + 1, dtype=np.int64)
    for i, p in enumerate(polygons):
        offsets[i + 1] = offsets[i] + len(p)
    return vertices, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if seed := cfg.get("seed"):
        np.random.seed(seed)

    mc = cfg["mosaic"]
    groups, polygons, centers, clipped = simulate_rgc_mosaics(
        n_mosaics=mc["n_mosaics"],
        mean_diameters=mc["mean_diameters"],
        center_noise=mc["center_noise"],
        diameter_noise=mc["diameter_noise"],
        n_missing_list=mc["n_missing_list"],
        overlap_factors=mc.get("overlap_factors"),
    )

    X, y = _generate_features(groups, cfg["features"])
    polygon_vertices, polygon_offsets = _pack_polygons(polygons)

    filename = cfg.get("output", {}).get("filename", args.config.stem + ".npz")
    out_path = Path("data") / filename
    out_path.parent.mkdir(exist_ok=True)

    np.savez_compressed(
        out_path,
        groups=groups,
        centers=centers,
        clipped=clipped,
        polygon_vertices=polygon_vertices,
        polygon_offsets=polygon_offsets,
        X=X,
        y=y,
    )
    print(f"Saved {len(groups)} cells to {out_path}")


if __name__ == "__main__":
    main()
