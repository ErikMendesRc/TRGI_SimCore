"""
core/metrics.py
---------------
Métricas globais de complexidade / informação.
"""

import numpy as np
from .infon_qubit import Qubit


def shannon_entropy(grid) -> float:
    """
    Entropia média H = −∑ p log₂ p calculada sobre
    a projeção computacional |0⟩ / |1⟩ de cada qubit.
    """
    probs = []
    for row in grid:
        for q in row:
            p0 = abs(q.state[0]) ** 2
            probs.append([p0, 1 - p0])
    probs = np.array(probs)
    eps = 1e-12
    H = -np.sum(probs * np.log2(probs + eps)) / len(probs)
    return float(H)