# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test function
uv run pytest tests/test_dataset.py::test_save_load_roundtrip

# Run tests for a specific config
uv run pytest tests/test_dataset.py -k "high_overlap"

# Generate a dataset from a YAML config (output goes to data/)
uv run python -m toy_mosaics.simulate_dataset configs/example.yaml
```

No linter or formatter is configured. The project uses `uv` for environment management.

## Architecture

The library simulates **retinal ganglion cell (RGC) mosaics**: multiple independent tilings of 2D space, each representing one cell type, paired with high-dimensional feature vectors so that clustering algorithms can be tested against ground-truth spatial structure.

### Simulation pipeline

`simulate_dataset.py` → `dataset_from_config(cfg)` drives the full pipeline:

1. **`simulate.py`** — `simulate_rgc_mosaics()` generates `n_mosaics` independent spatial tilings:
   - Poisson disk sampling produces jittered cell centres
   - `compute_voronoi_polygons()` builds the Voronoi diagram using mirror points to bound edge regions
   - `polygons.py` clips each polygon to the bounding box (Sutherland-Hodgman) and optionally removes random cells (`n_missing_list`)
   - Returns `groups`, `polygons`, `centers`, `clipped` — all indexed per cell

2. **`simulate_dataset.py`** — `_generate_features()` creates Gaussian blob features:
   - One cluster per mosaic, centres arranged on a circle of radius `spread`
   - Per-cluster covariance controlled by `std`, `aspect_ratio`, `rotation`
   - Returns `X` (n_cells × n_dims) and `y` (cluster label = mosaic index)

3. **`dataset.py`** — `MosaicDataset` wraps all arrays. Polygons are stored as a ragged Python list; serialized to `.npz` as a flat vertex array + int64 offsets.

### Two parallel index spaces

Every per-cell array (`groups`, `centers`, `clipped`, `polygons`, `X`, `y`) has length `n_cells`. The connection between spaces:
- `groups[i]` — which mosaic cell `i` belongs to (integer 0…n_mosaics-1)
- `y[i]` — which feature cluster cell `i` belongs to (always equals `groups[i]` in simulated data)

### Analysis modules

- **`overlap.py`** — pairwise IoU between convex hulls of different mosaics; measures spatial interleaving
- **`coverage.py`** — `CoverageDensityMapper`: kernel density estimation over the spatial domain per mosaic
- **`spatial_losses.py`** — `spatial_separation_loss()`: a GMM regulariser that penalises spatially adjacent cells being assigned to the same cluster; expects raw GMM internals (means, covariances, weights, responsibilities)

### Visualization

`plot/` has three entry points (all exported from `toy_mosaics.plot`):
- `plot_mosaics(dataset)` — polygon patches; modes: `"basic"`, `"violations"`, `"iou"`
- `plot_blobs(dataset)` — 2D scatter of feature vectors
- `plot_coverage(dataset)` — spatial density maps

### Config schema

YAML configs in `configs/` drive dataset generation. All mosaic-level parameters (`mean_diameters`, `n_missing_list`, `overlap_factors`) accept either a scalar (broadcast to all mosaics) or a list of length `n_mosaics`. Feature parameters `std`, `aspect_ratio`, and `rotation` accept scalar, list, or `"random"`.
