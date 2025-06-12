from __future__ import annotations
import numpy as np
from typing import Tuple

from .manifold import Manifold
from .dynamics import HamiltonianDynamics


class TTensor:
    """
    Calcula componentes do Tensor de Informação T_ab(I),
    com foco em T_00, a densidade de energia informacional local.

    A densidade T_00 é definida como o valor esperado do Hamiltoniano local
    aplicado a pares de qubits vizinhos.
    """

    def __init__(self, manifold: Manifold, dynamics: HamiltonianDynamics):
        """
        Inicializa o TTensor com a variedade e a dinâmica hamiltoniana.

        Args:
            manifold (Manifold): A grade contendo os infons quânticos.
            dynamics (HamiltonianDynamics): A dinâmica que fornece os Hamiltonianos locais.

        Raises:
            TypeError: Se dynamics não for uma instância de HamiltonianDynamics.
        """
        if not isinstance(dynamics, HamiltonianDynamics):
            raise TypeError("TTensor requer HamiltonianDynamics para calcular energia.")
        self.manifold = manifold
        self.dynamics = dynamics
        self.T00_matrix = np.zeros(manifold.grid.shape)

    def compute_T00_local(self, position: Tuple[int, int]) -> float:
        """
        Calcula o valor esperado do Hamiltoniano local (T_00) em uma posição.

        T_00 ≡ ⟨ψ|H_local|ψ⟩, onde ψ é o estado do par de qubits e H_local é o
        Hamiltoniano entre eles.

        Args:
            position (Tuple[int, int]): Coordenada (linha, coluna) na grade.

        Returns:
            float: Valor escalar representando a densidade de energia T_00.
        """
        m = self.manifold
        r, c = position

        pos1, pos2 = (r, c), m._wrap(r, c + 1)
        q1, q2 = m.get_infon_state(pos1), m.get_infon_state(pos2)
        pair_state = np.kron(q1.state, q2.state)

        H_local = self.dynamics.get_local_hamiltonian(pos1)
        expected_energy = np.real(np.conj(pair_state).T @ H_local @ pair_state)

        return float(expected_energy)

    def compute_T_matrix_global(self) -> np.ndarray:
        """
        Calcula a matriz T_00 para todos os pontos da grade.

        Returns:
            np.ndarray: Matriz 2D com os valores locais de densidade de energia.
        """
        t_matrix = np.zeros(self.manifold.grid.shape)
        for r in range(self.manifold.rows):
            for c in range(self.manifold.cols):
                t_matrix[r, c] = self.compute_T00_local((r, c))
        self.T00_matrix = t_matrix
        return self.T00_matrix