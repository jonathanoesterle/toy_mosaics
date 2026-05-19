"""Shared result type for all clustering strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ClusteringResult:
    """Output of a clustering strategy.

    Attributes
    ----------
    labels:
        Integer cluster assignment per cell, shape (n_cells,).
    model:
        Fitted estimator or a dict of algorithm-specific metadata.
    """

    labels: NDArray[np.int_]
    model: Any
