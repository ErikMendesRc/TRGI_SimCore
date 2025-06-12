import numpy as np
from .infon_qubit import Qubit


def shannon_entropy(grid) -> float:
    """
    Calcula a entropia média de Shannon da grade de qubits.

    A entropia é baseada nas probabilidades de medição dos estados |0⟩ e |1⟩
    para cada qubit, usando a fórmula:

        H = −∑ p * log₂(p)

    Onde p são as probabilidades de medição em cada célula.

    Args:
        grid: Grade 2D contendo objetos do tipo Qubit.

    Returns:
        float: Valor da entropia média da grade.
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