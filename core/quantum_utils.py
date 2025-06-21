"""Utility functions for quantum information analysis.

This module provides helpers for computing entanglement entropy
and correlation observables on the TRGI lattice.
"""
from __future__ import annotations
import itertools
import numpy as np
from typing import Iterable, Tuple

from .infon_qubit import Qubit


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Return the von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ)."""
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-12]
    return float(-np.sum(vals * np.log2(vals)))


def reduced_density_matrix(pair_state: np.ndarray, keep: int = 0) -> np.ndarray:
    """Return reduced density matrix of one qubit from a two-qubit state."""
    pair_state = pair_state.reshape(2, 2)
    rho = pair_state @ pair_state.conj().T
    if keep == 0:
        # trace out second qubit
        return np.array([
            [rho[0, 0] + rho[1, 1], rho[0, 2] + rho[1, 3]],
            [rho[2, 0] + rho[3, 1], rho[2, 2] + rho[3, 3]],
        ])
    else:
        # trace out first qubit
        return np.array([
            [rho[0, 0] + rho[2, 2], rho[0, 1] + rho[2, 3]],
            [rho[1, 0] + rho[3, 2], rho[1, 1] + rho[3, 3]],
        ])


def pair_entanglement_entropy(pair_state: np.ndarray) -> float:
    """Entanglement entropy of a two-qubit state."""
    rho_a = reduced_density_matrix(pair_state, keep=0)
    return von_neumann_entropy(rho_a)


def region_entropy(manifold, positions: Iterable[Tuple[int, int]]) -> float:
    """Approximate entanglement entropy of a region.

    The current TRGI dynamics stores qubits in product states, so the
    entropy reduces to a sum of single-qubit entropies. This helper is
    provided for completeness.
    """
    H = 0.0
    for pos in positions:
        q: Qubit = manifold.get_infon_state(pos)
        rho = np.outer(q.state, q.state.conj())
        H += von_neumann_entropy(rho)
    return H


def x_expectation(q: Qubit) -> float:
    """Return expectation value of σ_x for a qubit."""
    a, b = q.state
    return float(2 * np.real(np.conj(a) * b))


def spatial_correlation(manifold, max_distance: int = 5) -> np.ndarray:
    """Compute ⟨σ_x(i) σ_x(j)⟩ correlations as a function of distance."""
    rows, cols = manifold.rows, manifold.cols
    corr = np.zeros(max_distance + 1)
    counts = np.zeros(max_distance + 1)
    exp_x = np.vectorize(lambda q: x_expectation(q))(manifold.grid)
    for (r1, c1), (r2, c2) in itertools.combinations(
        itertools.product(range(rows), range(cols)), 2
    ):
        d = int(round(np.hypot(r1 - r2, c1 - c2)))
        if d <= max_distance:
            corr[d] += exp_x[r1, c1] * exp_x[r2, c2]
            counts[d] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(counts > 0, corr / counts, 0)
    return corr


def autocorrelation(data: np.ndarray, lag: int) -> float:
    """Temporal autocorrelation of a 1-D array."""
    if lag >= len(data):
        return 0.0
    data = np.asarray(data)
    mean = data.mean()
    var = data.var()
    if var == 0:
        return 0.0
    return float(np.correlate(data - mean, data - mean, mode="full")[len(data)-1+lag] / (var * (len(data)-lag)))