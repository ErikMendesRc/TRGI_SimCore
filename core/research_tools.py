"""Extra analysis tools for TRGI research experiments."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy import ndimage
from scipy.stats import linregress

from .manifold import Manifold

logger = logging.getLogger(__name__)


def save_results(data: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", path)


def linear_regression(x: Iterable[float], y: Iterable[float]):
    """Return slope, intercept and p-value for a linear regression."""
    res = linregress(list(x), list(y))
    return res.slope, res.intercept, res.pvalue


def detect_domains(manifold: Manifold, threshold: float = 0.8) -> np.ndarray:
    """Label connected domains where P(|0>) > threshold."""
    if manifold.infon_type == "qubit":
        field = np.vectorize(lambda q: q.p0)(manifold.grid)
    else:
        field = manifold.grid
    mask = field > threshold
    labeled, _ = ndimage.label(mask)
    return labeled


def track_perturbation(
    manifold: Manifold,
    dynamics,
    start: Tuple[int, int],
    steps: int = 10,
    flip_axis: str = "x",
) -> list[int]:
    """Apply a single qubit flip and track spread distance over time."""
    from .infon_qubit import Qubit

    original = manifold.get_infon_state(start)
    if flip_axis == "x":
        U = np.array([[0, 1], [1, 0]], dtype=complex)
    else:
        U = np.array([[1, 0], [0, -1]], dtype=complex)
    flipped = Qubit(*(U @ original.state))
    manifold.set_infon_state(start, flipped)
    distances = []
    base = np.vectorize(lambda q: q.state.copy())(manifold.grid)
    for t in range(steps):
        dynamics.step()
        diff = np.vectorize(lambda s, q: np.linalg.norm(q.state - s))(base, manifold.grid)
        coords = np.argwhere(diff > 1e-3)
        if len(coords) == 0:
            distances.append(0)
        else:
            dmax = max(np.hypot(r - start[0], c - start[1]) for r, c in coords)
            distances.append(int(round(dmax)))
    return distances