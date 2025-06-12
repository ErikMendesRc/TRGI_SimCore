from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .manifold import Manifold
    from .geometry import EmergentGeometry
    from .infon_qubit import Qubit

# Operadores de Pauli e Identidade (modelo de Ising)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
ID = np.identity(2, dtype=complex)


class GameOfLifeTRGI:
    """
    Implementa a dinâmica clássica de autômatos celulares baseada nas regras de Conway.
    Usado para simulações com infons escalares.
    """

    def __init__(self, manifold: Manifold, rule_b: List[int] | None = None, rule_s: List[int] | None = None):
        if manifold.infon_type != "scalar":
            raise ValueError("GameOfLifeTRGI requer infons escalares.")
        self.manifold = manifold
        self.rule_b = rule_b or [3]
        self.rule_s = rule_s or [2, 3]

    def step(self):
        """Executa um passo de atualização das células vivas/mortas conforme as regras B/S."""
        m = self.manifold
        new_grid = m.grid.copy()
        for r in range(m.rows):
            for c in range(m.cols):
                alive = m.grid[r, c] == 1.0
                n = int(m.neighbors_sum((r, c)))
                if alive and n not in self.rule_s:
                    new_grid[r, c] = 0.0
                elif not alive and n in self.rule_b:
                    new_grid[r, c] = 1.0
        m.grid = new_grid


class HamiltonianDynamics:
    """
    Dinâmica quântica baseada no Modelo de Ising Transverso com acoplamento geométrico.
    Simula evolução unitária de qubits em uma variedade discreta.
    """

    def __init__(
        self,
        manifold: Manifold,
        geometry: EmergentGeometry,
        J: float = 1.0,
        h: float = 0.5,
        dt: float = 0.1,
        use_geometric_coupling: bool = True
    ):
        if manifold.infon_type != "qubit":
            raise ValueError("HamiltonianDynamics requer infons 'qubit'.")
        self.manifold = manifold
        self.geometry = geometry
        self.J_base = J
        self.h = h
        self.dt = dt
        self.use_geometric_coupling = use_geometric_coupling

    def get_local_hamiltonian(self, pos: Tuple[int, int]) -> np.ndarray:
        """
        Constrói o Hamiltoniano local 4x4 entre um qubit e seu vizinho à direita.
        O acoplamento pode ser modulado pela distância informacional entre os qubits.
        """
        J_eff = self.J_base
        if self.use_geometric_coupling:
            neighbor_pos = self.manifold._wrap(pos[0], pos[1] + 1)
            dist = self.geometry.compute_local_metric_analogue(pos, neighbor_pos)
            coupling_factor = 1.0 - (dist / np.pi)
            J_eff *= coupling_factor

        H_interaction = -J_eff * np.kron(SIGMA_Z, SIGMA_Z)
        H_field = -self.h * (np.kron(SIGMA_X, ID) + np.kron(ID, SIGMA_X))
        return H_interaction + H_field

    def step(self):
        """
        Executa um passo temporal usando decomposição de Trotter-Suzuki (ordem 1).
        Aplica evolução horizontal e vertical sobre pares de qubits vizinhos.
        """
        m = self.manifold

        for r in range(m.rows):
            for c in range(m.cols):
                self._evolve_pair((r, c), (r, c + 1))

        for r in range(m.rows):
            for c in range(m.cols):
                self._evolve_pair((r, c), (r + 1, c))

    def _evolve_pair(self, pos1_raw: Tuple[int, int], pos2_raw: Tuple[int, int]):
        """
        Evolui o estado conjunto de um par de qubits usando a matriz U = exp(-iH dt).
        Aproxima os estados individuais com projeção via SVD.
        """
        m = self.manifold
        pos1 = m._wrap(*pos1_raw)
        pos2 = m._wrap(*pos2_raw)

        q1 = m.get_infon_state(pos1)
        q2 = m.get_infon_state(pos2)

        H_pair = self.get_local_hamiltonian(pos1)
        U_pair = expm(-1j * H_pair * self.dt)

        pair_state = np.kron(q1.state, q2.state)
        evolved_pair_state = U_pair @ pair_state

        q1_new_state, q2_new_state = self._unentangle_and_update(evolved_pair_state)

        from .infon_qubit import Qubit
        m.set_infon_state(pos1, Qubit(a=q1_new_state[0], b=q1_new_state[1]))
        m.set_infon_state(pos2, Qubit(a=q2_new_state[0], b=q2_new_state[1]))

    def _unentangle_and_update(self, pair_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aproxima o par emaranhado por dois estados de qubit usando SVD.
        Retorna os dois vetores aproximados.
        """
        try:
            U, S, Vh = np.linalg.svd(pair_state.reshape(2, 2))
            q1_state = U[:, 0] * np.sqrt(S[0])
            q2_state = Vh[0, :] * np.sqrt(S[0])
            return q1_state, q2_state
        except np.linalg.LinAlgError:
            return np.array([1, 0], dtype=complex), np.array([1, 0], dtype=complex)