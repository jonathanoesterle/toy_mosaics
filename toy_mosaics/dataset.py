"""MosaicDataset: a thin dataclass for mosaic + feature datasets saved as .npz."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class MosaicDataset:
    """One simulated dataset: spatial mosaics plus per-cell feature vectors.

    Attributes
    ----------
    groups:
        Integer mosaic label per cell, shape (n_cells,).
    centers:
        Voronoi centre coordinates per cell, shape (n_cells, 2).
    clipped:
        Boolean mask – True when a polygon was clipped by the bounding box,
        shape (n_cells,).
    polygons:
        Ragged list of (n_vertices, 2) float arrays, one per cell.
    X:
        Feature matrix, shape (n_cells, n_dims).
    y:
        Feature cluster label per cell, shape (n_cells,).
    hull_features:
        Optional geometric features derived from Voronoi polygons,
        shape (n_cells, n_hull_feats). None when not computed.
    hull_feature_names:
        Column names for hull_features (e.g. ["area", "perimeter", "circularity"]).
    """

    groups: NDArray[np.int_]
    centers: NDArray[np.floating]
    clipped: NDArray[np.bool_]
    polygons: List[NDArray[np.floating]]
    X: NDArray[np.floating]
    y: NDArray[np.int_]
    hull_features: Optional[NDArray[np.floating]] = None
    hull_feature_names: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a compressed .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vertices = np.concatenate(self.polygons, axis=0)
        offsets = np.zeros(len(self.polygons) + 1, dtype=np.int64)
        for i, p in enumerate(self.polygons):
            offsets[i + 1] = offsets[i] + len(p)

        arrays = dict(
            groups=self.groups,
            centers=self.centers,
            clipped=self.clipped,
            polygon_vertices=vertices,
            polygon_offsets=offsets,
            X=self.X,
            y=self.y,
        )
        if self.hull_features is not None:
            arrays["hull_features"] = self.hull_features
            arrays["hull_feature_names"] = np.array(self.hull_feature_names)

        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MosaicDataset":
        """Load from a .npz file produced by :meth:`save` or simulate_dataset."""
        data = np.load(path, allow_pickle=False)
        verts, offs = data["polygon_vertices"], data["polygon_offsets"]
        polygons = [verts[offs[i] : offs[i + 1]] for i in range(len(offs) - 1)]

        hull_features = data["hull_features"] if "hull_features" in data else None
        hull_feature_names = (
            data["hull_feature_names"].tolist() if "hull_feature_names" in data else []
        )

        return cls(
            groups=data["groups"],
            centers=data["centers"],
            clipped=data["clipped"].astype(bool),
            polygons=polygons,
            X=data["X"],
            y=data["y"],
            hull_features=hull_features,
            hull_feature_names=hull_feature_names,
        )

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.groups)

    @property
    def n_mosaics(self) -> int:
        return int(self.groups.max()) + 1

    @property
    def n_feature_dims(self) -> int:
        return self.X.shape[1]
