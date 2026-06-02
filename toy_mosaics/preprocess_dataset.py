"""Build a MosaicDataset from a pickled DataFrame and save to data/.

Usage:
    uv run python -m toy_mosaics.preprocess_dataset configs/bcs_train.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from toy_mosaics.dataset import MosaicDataset
from toy_mosaics.simulate import HULL_FEATURE_NAMES, compute_hull_features


def dataset_from_dataframe(
    path: str | Path,
    feature_cols: list[str],
    polygon_col: str = "hull_points",
    group_col: str = "celltype_final",
    zero_center: bool = True,
    normalize_features: bool = False,
) -> MosaicDataset:
    """Build a :class:`MosaicDataset` from a pickled DataFrame of real cells.

    Parameters
    ----------
    path:
        Path to a pickle file produced by ``df[cols].to_pickle(path)``.
    feature_cols:
        DataFrame columns to use as the feature matrix X. Must have at least 2.
    polygon_col:
        Column holding per-cell convex hull vertices, each an (N, 2) array.
    group_col:
        Column holding the cell-type label (string or int); encoded as integers.
    """
    if len(feature_cols) < 2:
        raise ValueError(f"feature_cols must have at least 2 columns, got {feature_cols!r}")

    df = pd.read_pickle(path)

    n_before = len(df)
    df = df.dropna(subset=[group_col]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with missing '{group_col}'")

    polygons = [np.asarray(p, dtype=float) for p in df[polygon_col]]

    _, groups = np.unique(df[group_col].values, return_inverse=True)
    groups = groups.astype(np.int_)

    centers = np.array([p.mean(axis=0) for p in polygons], dtype=float)

    clipped = df["clipped"].values.astype(bool) if "clipped" in df.columns else np.zeros(len(df), dtype=bool)

    X = df[feature_cols].to_numpy(dtype=float)

    if zero_center:
        X = X - X.mean(axis=0)
    if normalize_features:
        X = X / X.std(axis=0)

    hull_features = compute_hull_features(polygons)

    return MosaicDataset(
        groups=groups,
        centers=centers,
        clipped=clipped,
        polygons=polygons,
        X=X,
        y=groups.copy(),
        hull_features=hull_features,
        hull_feature_names=list(HULL_FEATURE_NAMES),
    )


def dataset_from_config(cfg: dict) -> MosaicDataset:
    """Build a :class:`MosaicDataset` from a parsed YAML config dict."""
    inp = cfg["input"]
    return dataset_from_dataframe(
        path=inp["path"],
        feature_cols=list(inp["feature_cols"]),
        polygon_col=inp.get("polygon_col", "hull_points"),
        group_col=inp.get("group_col", "celltype_final"),
    )


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
