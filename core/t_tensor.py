# core/t_tensor.py
from __future__ import annotations
import numpy as np
from typing import Tuple

from .manifold import Manifold
from .dynamics import HamiltonianDynamics  # Depende da dinâmica para o Hamiltoniano

class TTensor:
    """
    Calcula componentes do Tensor de Informação T_ab(I).
    Focaremos em T_00, a densidade de energia informacional.
    """
    def __init__(self, manifold: Manifold, dynamics: HamiltonianDynamics):
        if not isinstance(dynamics, HamiltonianDynamics):
            raise TypeError("TTensor requer HamiltonianDynamics para calcular energia.")
        self.manifold = manifold
        self.dynamics = dynamics
        # Cache para a matriz T00, para não recalcular a cada frame
        self.T00_matrix = np.zeros(manifold.grid.shape)
        print("TTensor (real) initialized.")

    def compute_T00_local(self, position: Tuple[int, int]) -> float:
        """
        Calcula T00 = ⟨H_local⟩, o valor esperado do Hamiltoniano local.
        Isto é a "densidade de energia informacional".
        """
        m = self.manifold
        r, c = position

        # Pega o estado do par (horizontal)
        pos1, pos2 = (r, c), m._wrap(r, c + 1)
        q1, q2 = m.get_infon_state(pos1), m.get_infon_state(pos2)
        pair_state = np.kron(q1.state, q2.state)

        # Pega o Hamiltoniano local para este par
        H_local = self.dynamics.get_local_hamiltonian(pos1)

        # Calcula o valor esperado: ⟨ψ|H|ψ⟩
        # Para vetores complexos, é conj(ψ) @ H @ ψ
        expected_energy = np.real(np.conj(pair_state).T @ H_local @ pair_state)
        return float(expected_energy)

    def compute_T_matrix_global(self) -> np.ndarray:
        """
        Calcula a matriz T00 (densidade de energia) para toda a variedade.
        """
        t_matrix = np.zeros(self.manifold.grid.shape)
        for r in range(self.manifold.rows):
            for c in range(self.manifold.cols):
                t_matrix[r, c] = self.compute_T00_local((r, c))
        self.T00_matrix = t_matrix
        return self.T00_matrix