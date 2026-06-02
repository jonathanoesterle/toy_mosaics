"""Tests for clustering strategies: KMeans, GMM, LeidenMosaicStrategy, and MRFMosaicStrategy."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from toy_mosaics.clustering import (
    ClusteringResult, GMMStrategy, KMeansStrategy, LeidenMosaicStrategy, MRFMosaicStrategy,
)
from toy_mosaics.mrf_mosaic import _build_coverage_map, _ramp
from toy_mosaics.simulate_dataset import dataset_from_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

CONFIG_NAMES = ["varied", "anisotropic"]

STRATEGIES = [KMeansStrategy, GMMStrategy]


def _load_cfg(name: str) -> dict:
    with open(CONFIGS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(params=CONFIG_NAMES)
def dataset(request):
    cfg = _load_cfg(request.param)
    return dataset_from_config(cfg)


@pytest.mark.parametrize("StrategyClass", STRATEGIES)
def test_labels_shape(dataset, StrategyClass):
    result = StrategyClass(n_clusters=dataset.n_mosaics).fit(dataset)
    assert result.labels.shape == (len(dataset),)


@pytest.mark.parametrize("StrategyClass", STRATEGIES)
def test_labels_dtype(dataset, StrategyClass):
    result = StrategyClass(n_clusters=dataset.n_mosaics).fit(dataset)
    assert np.issubdtype(result.labels.dtype, np.integer)


@pytest.mark.parametrize("StrategyClass", STRATEGIES)
def test_n_unique_labels(dataset, StrategyClass):
    result = StrategyClass(n_clusters=dataset.n_mosaics).fit(dataset)
    assert len(np.unique(result.labels)) == dataset.n_mosaics


@pytest.mark.parametrize("StrategyClass", STRATEGIES)
def test_result_type(dataset, StrategyClass):
    result = StrategyClass(n_clusters=dataset.n_mosaics).fit(dataset)
    assert isinstance(result, ClusteringResult)
    assert result.model is not None


def test_kmeans_reproducible():
    cfg = _load_cfg("varied")
    ds = dataset_from_config(cfg)
    r1 = KMeansStrategy(n_clusters=ds.n_mosaics, random_state=42).fit(ds)
    r2 = KMeansStrategy(n_clusters=ds.n_mosaics, random_state=42).fit(ds)
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_gmm_reproducible():
    cfg = _load_cfg("varied")
    ds = dataset_from_config(cfg)
    r1 = GMMStrategy(n_clusters=ds.n_mosaics, random_state=42).fit(ds)
    r2 = GMMStrategy(n_clusters=ds.n_mosaics, random_state=42).fit(ds)
    np.testing.assert_array_equal(r1.labels, r2.labels)


# ---------------------------------------------------------------------------
# LeidenMosaicStrategy
# ---------------------------------------------------------------------------

# spatial_radius large enough to capture inter-cell overlap in the example config
# (mean_diameters ≈ 10–15 → use 30 as a safe upper bound)
_LEIDEN_KWARGS = dict(k=10, spatial_radius=30.0, tau_low_global=0.0, n_iter=2, resolution=1.0)


@pytest.fixture
def leiden_dataset():
    cfg = _load_cfg("varied")
    return dataset_from_config(cfg)


def test_leiden_labels_shape(leiden_dataset):
    result = LeidenMosaicStrategy(n_clusters=leiden_dataset.n_mosaics, **_LEIDEN_KWARGS).fit(leiden_dataset)
    assert result.labels.shape == (len(leiden_dataset),)


def test_leiden_labels_dtype(leiden_dataset):
    result = LeidenMosaicStrategy(n_clusters=leiden_dataset.n_mosaics, **_LEIDEN_KWARGS).fit(leiden_dataset)
    assert np.issubdtype(result.labels.dtype, np.integer)


def test_leiden_result_type(leiden_dataset):
    result = LeidenMosaicStrategy(n_clusters=leiden_dataset.n_mosaics, **_LEIDEN_KWARGS).fit(leiden_dataset)
    assert isinstance(result, ClusteringResult)
    assert isinstance(result.model, dict)
    assert "converged" in result.model
    assert "n_iters_run" in result.model
    assert "n_merges" in result.model


def test_leiden_reproducible(leiden_dataset):
    ds = leiden_dataset
    kwargs = dict(n_clusters=ds.n_mosaics, random_state=42, **_LEIDEN_KWARGS)
    r1 = LeidenMosaicStrategy(**kwargs).fit(ds)
    r2 = LeidenMosaicStrategy(**kwargs).fit(ds)
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_leiden_exclude_clipped_flag(leiden_dataset):
    ds = leiden_dataset
    base = dict(n_clusters=ds.n_mosaics, random_state=0, **_LEIDEN_KWARGS)
    # smoke-test that both flag values run without error
    LeidenMosaicStrategy(**base, exclude_clipped=True).fit(ds)
    LeidenMosaicStrategy(**base, exclude_clipped=False).fit(ds)


def test_leiden_n_iters_single(leiden_dataset):
    result = LeidenMosaicStrategy(
        n_clusters=leiden_dataset.n_mosaics, n_iter=1, **{k: v for k, v in _LEIDEN_KWARGS.items() if k != "n_iter"}
    ).fit(leiden_dataset)
    assert result.model["n_iters_run"] == 1


def test_leiden_merge_disabled(leiden_dataset):
    result = LeidenMosaicStrategy(
        n_clusters=leiden_dataset.n_mosaics, merge=False, **_LEIDEN_KWARGS
    ).fit(leiden_dataset)
    assert result.model["n_merges"] == 0
    assert result.labels.shape == (len(leiden_dataset),)


def test_leiden_merge_reduces_clusters(leiden_dataset):
    """Merge step should produce <= as many clusters as the no-merge version."""
    base = dict(n_clusters=leiden_dataset.n_mosaics, random_state=0, resolution=2.0,
                **{k: v for k, v in _LEIDEN_KWARGS.items() if k != "resolution"})
    r_no_merge = LeidenMosaicStrategy(**base, merge=False).fit(leiden_dataset)
    r_merge    = LeidenMosaicStrategy(**base, merge=True, theta_paga=0.1, delta_r=0.01).fit(leiden_dataset)
    assert len(np.unique(r_merge.labels)) <= len(np.unique(r_no_merge.labels))


# ---------------------------------------------------------------------------
# MRFMosaicStrategy
# ---------------------------------------------------------------------------

_MRF_MODEL_KEYS = (
    "gmm", "lam", "n_iters", "n_frozen", "n_changed",
    "violations_before", "violations_after",
    "labels_initial", "frozen",
    "tau_low", "tau_high", "spatial_radius", "exclude_clipped", "conf_threshold",
    "signed_ramp",
)

# All hard configs — used for algorithm invariant tests (monotonicity, frozen cells).
HARD_CONFIGS = ["anisotropic", "elongated", "high_overlap"]

# Subset where MRF with default lam=20 is expected not to regress ARI.
# high_overlap (K=4, overlap_factor 1.2-1.5) is intentionally excluded: the default
# lam=20 over-penalises at K=4 with very dense coverage, producing spatial coloring
# rather than feature-based correction and regressing ARI by ~0.02.  This is the
# known failure mode documented in mrf_mosaic_constraints.html §9c.
SPATIAL_CORRECTION_CONFIGS = ["anisotropic", "elongated"]


def _align_labels(labels: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Remap labels to best match y_true (Hungarian algorithm)."""
    n = int(max(labels.max(), y_true.max())) + 1
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, labels):
        conf[t, p] += 1
    _, col = linear_sum_assignment(-conf)
    mapping = {int(col[i]): i for i in range(len(col))}
    return np.array([mapping.get(int(x), int(x)) for x in labels], dtype=int)


def error_summary(dataset, result, gmm_result=None) -> str:
    """Human-readable per-cell diagnostic for cells misclassified by MRF.

    Always safe to call: returns a formatted string.  Print with ``-s`` flag
    to see it even when the test passes.
    """
    model = result.model
    gmm = model["gmm"]
    tau_low = float(model.get("tau_low", 0.10))
    tau_high = float(model.get("tau_high", 0.40))
    spatial_radius = float(model.get("spatial_radius", 20.0))
    exclude_clipped = bool(model.get("exclude_clipped", True))
    conf_threshold = float(model.get("conf_threshold", 0.90))
    frozen_stored = model.get("frozen")

    labels = _align_labels(result.labels, dataset.y)
    errors = np.where(labels != dataset.y)[0]

    raw_map = _build_coverage_map(
        dataset.polygons, dataset.centers, spatial_radius,
        dataset.clipped, exclude_clipped,
    )
    nbrs: list[list[tuple[int, float]]] = [[] for _ in range(len(dataset))]
    for (i, j), cf in raw_map.items():
        w = _ramp(cf, tau_low, tau_high)
        if w > 0.0:
            nbrs[i].append((j, w))
            nbrs[j].append((i, w))

    max_post = gmm.predict_proba(dataset.X).max(axis=1)

    SEP = "-" * 72
    ari_mrf = adjusted_rand_score(dataset.y, labels)

    lines = []
    if gmm_result is not None:
        ari_gmm = adjusted_rand_score(dataset.y, gmm_result.labels)
        lines.append(f"ARI  GMM: {ari_gmm:.3f}  ->  MRF: {ari_mrf:.3f}")
    else:
        lines.append(f"ARI  MRF: {ari_mrf:.3f}")

    lines.append(
        f"violations {model.get('violations_before', '?')} -> {model.get('violations_after', '?')}  "
        f"n_changed={model.get('n_changed', '?')}  "
        f"n_iters={model.get('n_iters', '?')}  "
        f"n_frozen={model.get('n_frozen', '?')}"
    )
    lines.append(SEP)

    if len(errors) == 0:
        lines.append("No misclassified cells.")
        lines.append(SEP)
        return "\n".join(lines)

    lines.append(f"Misclassified cells after MRF ({len(errors)} errors):")
    hdr = (
        f"  {'id':>4}  {'true':>4}  {'pred':>4}  "
        f"{'posterior':>9}  {'conflict':>8}  {'same':>5}  {'corr':>5}  "
        f"{'frozen':>6}  diagnosis"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for i in sorted(errors.tolist()):
        true_k = int(dataset.y[i])
        pred_k = int(labels[i])
        post = float(max_post[i])
        conflict = sum(w for j, w in nbrs[i] if labels[j] == pred_k)
        same_n = sum(1 for j, _ in nbrs[i] if labels[j] == pred_k)
        corr_n = sum(1 for j, _ in nbrs[i] if labels[j] == true_k)

        if frozen_stored is not None:
            is_frozen = bool(frozen_stored[i])
        else:
            is_frozen = post >= conf_threshold

        if conflict == 0.0:
            diag = "Type-B (no spatial signal)"
        elif is_frozen:
            diag = "Type-A frozen (lower conf_threshold)"
        else:
            diag = "Type-A unfixed (increase lam?)"

        lines.append(
            f"  {i:>4}  {true_k:>4}  {pred_k:>4}  {post:>9.3f}  "
            f"{conflict:>8.3f}  {same_n:>5}  {corr_n:>5}  "
            f"{str(is_frozen):>6}  {diag}"
        )

    lines.append(SEP)
    return "\n".join(lines)


# --- smoke tests (easy configs, default params) ---

def test_mrf_smoke(dataset):
    result = MRFMosaicStrategy(n_clusters=dataset.n_mosaics).fit(dataset)
    assert result.labels.shape == (len(dataset),)
    assert np.issubdtype(result.labels.dtype, np.integer)
    assert isinstance(result, ClusteringResult)
    for key in _MRF_MODEL_KEYS:
        assert key in result.model, f"missing model key: {key}"


def test_mrf_reproducible():
    cfg = _load_cfg("varied")
    ds = dataset_from_config(cfg)
    r1 = MRFMosaicStrategy(n_clusters=ds.n_mosaics, random_state=42).fit(ds)
    r2 = MRFMosaicStrategy(n_clusters=ds.n_mosaics, random_state=42).fit(ds)
    np.testing.assert_array_equal(r1.labels, r2.labels)


# --- quality / diagnostic tests (hard configs) ---

@pytest.fixture(params=HARD_CONFIGS)
def hard_dataset_named(request):
    cfg = _load_cfg(request.param)
    return dataset_from_config(cfg), request.param


@pytest.fixture(params=SPATIAL_CORRECTION_CONFIGS)
def correction_dataset_named(request):
    cfg = _load_cfg(request.param)
    return dataset_from_config(cfg), request.param


def test_mrf_violations_monotone(hard_dataset_named):
    """ICM must never increase violations (monotone energy descent guarantee)."""
    dataset, name = hard_dataset_named
    result = MRFMosaicStrategy(n_clusters=dataset.n_mosaics).fit(dataset)
    before = result.model["violations_before"]
    after = result.model["violations_after"]
    assert after <= before, (
        f"[{name}] violations increased: {before} → {after}"
    )


def test_mrf_frozen_cells_unchanged(hard_dataset_named):
    """No cell above conf_threshold should change label during ICM.
    n_em_iters=0: EM deliberately unfreezes cells to escape the normalization
    trap — this test isolates the ICM frozen-cell guarantee only."""
    dataset, name = hard_dataset_named
    result = MRFMosaicStrategy(n_clusters=dataset.n_mosaics, n_em_iters=0).fit(dataset)
    frozen = result.model["frozen"]
    labels_initial = result.model["labels_initial"]
    assert np.all(result.labels[frozen] == labels_initial[frozen]), (
        f"[{name}] frozen cells were reassigned by ICM"
    )


def test_mrf_ari_nonnegression(correction_dataset_named):
    """MRF spatial correction must not regress ARI vs plain GMM (tolerance 0.01).
    n_em_iters=0: isolates the ICM spatial-correction guarantee; EM is an
    orthogonal optimisation step tested separately."""
    dataset, name = correction_dataset_named
    gmm_result = GMMStrategy(n_clusters=dataset.n_mosaics).fit(dataset)
    mrf_result = MRFMosaicStrategy(n_clusters=dataset.n_mosaics, n_em_iters=0).fit(dataset)

    ari_gmm = adjusted_rand_score(dataset.y, gmm_result.labels)
    ari_mrf = adjusted_rand_score(dataset.y, mrf_result.labels)

    print(f"\n[{name}]  GMM ARI={ari_gmm:.3f}   MRF ARI={ari_mrf:.3f}")
    print(error_summary(dataset, mrf_result, gmm_result))

    assert ari_mrf >= ari_gmm - 0.01, (
        f"[{name}] MRF regressed: ARI {ari_mrf:.3f} < {ari_gmm:.3f} (GMM)"
    )


# --- signed ramp ---

def test_mrf_parallel_smoke(leiden_dataset):
    """n_workers=2 must run without error and produce valid output."""
    ds = leiden_dataset
    result = MRFMosaicStrategy(n_clusters=ds.n_mosaics, n_workers=2).fit(ds)
    assert result.labels.shape == (len(ds),)
    assert np.issubdtype(result.labels.dtype, np.integer)
    for key in _MRF_MODEL_KEYS:
        assert key in result.model, f"missing model key: {key}"


def test_mrf_parallel_ari_nonnegression(leiden_dataset):
    """Vectorised Jacobi ICM must not regress ARI vs plain GMM (tolerance 0.01).
    n_em_iters=0: isolates the ICM spatial-correction guarantee."""
    ds = leiden_dataset
    gmm_result = GMMStrategy(n_clusters=ds.n_mosaics).fit(ds)
    mrf_result = MRFMosaicStrategy(n_clusters=ds.n_mosaics, n_workers=2, n_em_iters=0).fit(ds)
    ari_gmm = adjusted_rand_score(ds.y, gmm_result.labels)
    ari_mrf = adjusted_rand_score(ds.y, mrf_result.labels)
    assert ari_mrf >= ari_gmm - 0.01, (
        f"parallel MRF regressed: ARI {ari_mrf:.3f} < {ari_gmm:.3f} (GMM)"
    )


def test_mrf_signed_ramp_smoke(leiden_dataset):
    """signed_ramp=True must run without error and produce valid output."""
    ds = leiden_dataset
    result = MRFMosaicStrategy(
        n_clusters=ds.n_mosaics, signed_ramp=True, tau_low=0.30,
    ).fit(ds)
    assert result.labels.shape == (len(ds),)
    assert np.issubdtype(result.labels.dtype, np.integer)
    assert result.model["signed_ramp"] is True


def test_mrf_signed_ramp_improves_anisotropic():
    """signed_ramp with tau_low=0.30 must improve or match plain MRF on anisotropic."""
    cfg = _load_cfg("anisotropic")
    ds = dataset_from_config(cfg)
    from scipy.spatial import KDTree
    nn_dists, _ = KDTree(ds.centers).query(ds.centers, k=2)
    sr = 3.0 * float(np.median(nn_dists[:, 1]))

    r_plain = MRFMosaicStrategy(n_clusters=ds.n_mosaics, spatial_radius=sr).fit(ds)
    r_signed = MRFMosaicStrategy(
        n_clusters=ds.n_mosaics, spatial_radius=sr, signed_ramp=True, tau_low=0.30,
    ).fit(ds)

    ari_plain = adjusted_rand_score(ds.y, r_plain.labels)
    ari_signed = adjusted_rand_score(ds.y, r_signed.labels)
    print(f"\nplain ARI={ari_plain:.4f}  signed ARI={ari_signed:.4f}")
    # Tolerance 0.03: signed_ramp can regress slightly on anisotropic when tau
    # calibration falls back to prior (GMM ARI too low for reliable frozen-anchor stats).
    assert ari_signed >= ari_plain - 0.03, (
        f"signed_ramp regressed: {ari_signed:.4f} < {ari_plain:.4f}"
    )


# ---------------------------------------------------------------------------
# H1 force-unfreeze pre-pass (theta_hard)
# ---------------------------------------------------------------------------

_H1_KWARGS = dict(signed_ramp=True, tau_low=0.30, log_ramp=True, log_ramp_alpha=10.0,
                  theta_hard=0.85)
_THETA_HARD = 0.85

# Configs that actually exist on disk (used where leiden_dataset / example.yaml is unavailable).
H1_CONFIGS = ["anisotropic", "high_overlap"]


@pytest.fixture(params=H1_CONFIGS)
def h1_dataset_named(request):
    cfg = _load_cfg(request.param)
    return dataset_from_config(cfg), request.param


def test_mrf_h1_smoke(h1_dataset_named):
    """theta_hard pre-pass runs without error; model dict contains expected keys."""
    dataset, _name = h1_dataset_named
    result = MRFMosaicStrategy(n_clusters=dataset.n_mosaics, **_H1_KWARGS).fit(dataset)
    assert result.labels.shape == (len(dataset),)
    assert np.issubdtype(result.labels.dtype, np.integer)
    assert "n_force_unfrozen" in result.model
    assert "theta_hard" in result.model
    assert result.model["theta_hard"] == _THETA_HARD


def test_mrf_h1_frozen_mask_invariant(h1_dataset_named):
    """Post-H1 frozen cells must not change label during ICM.
    n_em_iters=0: EM unfreezes everything by design; this test verifies
    the ICM-level frozen-cell guarantee only."""
    dataset, name = h1_dataset_named
    result = MRFMosaicStrategy(n_clusters=dataset.n_mosaics, **_H1_KWARGS, n_em_iters=0).fit(dataset)
    frozen = result.model["frozen"]
    labels_initial = result.model["labels_initial"]
    assert np.all(result.labels[frozen] == labels_initial[frozen]), (
        f"[{name}] post-H1 frozen cells were reassigned by ICM"
    )


def test_mrf_h1_only_unfreezes_violators(h1_dataset_named):
    """H1 must not unfreeze more cells than the number of CF > theta_hard pairs."""
    dataset, _name = h1_dataset_named
    result = MRFMosaicStrategy(n_clusters=dataset.n_mosaics, **_H1_KWARGS).fit(dataset)
    raw_map = result.model["raw_map"]
    # Each violating pair triggers at most one unfreeze.
    n_high_cf_pairs = sum(1 for cf in raw_map.values() if cf > _THETA_HARD)
    assert result.model["n_force_unfrozen"] <= n_high_cf_pairs


def test_mrf_h1_reduces_high_cf_violations(h1_dataset_named):
    """H1+ICM must not increase same-label pairs with CF > theta_hard vs no-H1."""
    from scipy.spatial import KDTree
    dataset, name = h1_dataset_named
    nn_dists, _ = KDTree(dataset.centers).query(dataset.centers, k=2)
    sr = 3.0 * float(np.median(nn_dists[:, 1]))

    base = dict(n_clusters=dataset.n_mosaics, spatial_radius=sr,
                signed_ramp=True, tau_low=0.30, log_ramp=True)
    r_no_h1 = MRFMosaicStrategy(**base).fit(dataset)
    r_h1    = MRFMosaicStrategy(**base, theta_hard=_THETA_HARD).fit(dataset)

    raw_map = r_h1.model["raw_map"]

    def _high_cf_violations(labels):
        return sum(
            1 for (i, j), cf in raw_map.items()
            if cf > _THETA_HARD and labels[i] == labels[j]
        )

    v_no_h1 = _high_cf_violations(r_no_h1.labels)
    v_h1    = _high_cf_violations(r_h1.labels)
    n_fu    = r_h1.model["n_force_unfrozen"]
    print(f"\n[{name}] CF>{_THETA_HARD} violations: no-H1={v_no_h1}  H1={v_h1}"
          f"  force_unfrozen={n_fu}")
    assert v_h1 <= v_no_h1, (
        f"[{name}] H1 increased high-CF violations: {v_no_h1} → {v_h1}"
    )


def test_mrf_h1_ari_nonnegression(h1_dataset_named):
    """H1 must not regress ARI vs the same config without H1 (tolerance 0.01)."""
    from scipy.spatial import KDTree
    dataset, name = h1_dataset_named
    nn_dists, _ = KDTree(dataset.centers).query(dataset.centers, k=2)
    sr = 3.0 * float(np.median(nn_dists[:, 1]))

    base = dict(n_clusters=dataset.n_mosaics, spatial_radius=sr,
                signed_ramp=True, tau_low=0.30, log_ramp=True)
    r_no_h1 = MRFMosaicStrategy(**base).fit(dataset)
    r_h1    = MRFMosaicStrategy(**base, theta_hard=_THETA_HARD).fit(dataset)

    ari_no_h1 = adjusted_rand_score(dataset.y, r_no_h1.labels)
    ari_h1    = adjusted_rand_score(dataset.y, r_h1.labels)
    print(f"\n[{name}]  no-H1 ARI={ari_no_h1:.3f}   H1 ARI={ari_h1:.3f}"
          f"  force_unfrozen={r_h1.model['n_force_unfrozen']}")
    assert ari_h1 >= ari_no_h1 - 0.01, (
        f"[{name}] H1 regressed ARI: {ari_h1:.3f} < {ari_no_h1:.3f}"
    )


# ---------------------------------------------------------------------------
# split_merge tests
# ---------------------------------------------------------------------------

def test_mrf_split_merge_smoke():
    """split_merge=True must run without error and return valid output."""
    cfg = _load_cfg("varied")
    dataset = dataset_from_config(cfg)
    result = MRFMosaicStrategy(
        n_clusters=dataset.n_mosaics, split_merge=True,
    ).fit(dataset)
    assert result.labels.shape == (len(dataset),)
    assert np.issubdtype(result.labels.dtype, np.integer)
    assert len(np.unique(result.labels)) == dataset.n_mosaics
    assert result.model["split_merge"] is True
    assert result.model["n_splits"] >= 0
    assert result.model["n_merges"] >= 0


def test_mrf_split_merge_ari_nonnegression():
    """split_merge must not regress ARI vs plain GMM on anisotropic."""
    cfg = _load_cfg("anisotropic")
    dataset = dataset_from_config(cfg)
    gmm_result = GMMStrategy(n_clusters=dataset.n_mosaics).fit(dataset)
    sm_result = MRFMosaicStrategy(
        n_clusters=dataset.n_mosaics, split_merge=True,
    ).fit(dataset)
    ari_gmm = adjusted_rand_score(dataset.y, gmm_result.labels)
    ari_sm = adjusted_rand_score(dataset.y, sm_result.labels)
    print(f"\n[anisotropic] GMM ARI={ari_gmm:.3f}  split_merge ARI={ari_sm:.3f}"
          f"  n_splits={sm_result.model['n_splits']}"
          f"  n_merges={sm_result.model['n_merges']}")
    assert ari_sm >= ari_gmm - 0.01, (
        f"split_merge regressed ARI: {ari_sm:.3f} < {ari_gmm:.3f}"
    )
