# core/dynamics.py
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from typing import List, Tuple, TYPE_CHECKING

# Evita importação circular, mas permite anotações de tipo
if TYPE_CHECKING:
    from .manifold import Manifold
    from .geometry import EmergentGeometry
    from .infon_qubit import Qubit

# --- Constantes Físicas ---
# Matrizes de Pauli, a base para operadores de qubit
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
ID = np.identity(2, dtype=complex)


# =========================================================================
# 1. Dinâmica Clássica (Escalar) - Game of Life
# =========================================================================
class GameOfLifeTRGI:
    """Dinâmica de autômato celular clássico (regras de Conway)."""

    def __init__(
            self,
            manifold: 'Manifold',
            rule_b: List[int] | None = None,
            rule_s: List[int] | None = None,
    ):
        if manifold.infon_type != "scalar":
            raise ValueError("GameOfLifeTRGI requer infons escalares.")
        self.manifold = manifold
        self.rule_b = rule_b or [3]
        self.rule_s = rule_s or [2, 3]
        print(f"GameOfLifeTRGI B{''.join(map(str, self.rule_b))}/S{''.join(map(str, self.rule_s))}")

    def step(self):
        m = self.manifold
        new_grid = m.grid.copy()
        for r in range(m.rows):
            for c in range(m.cols):
                alive = m.grid[r, c] == 1.0
                n = int(m.neighbors_sum((r, c)))
                if alive:
                    if n not in self.rule_s:
                        new_grid[r, c] = 0.0
                else:
                    if n in self.rule_b:
                        new_grid[r, c] = 1.0
        m.grid = new_grid


# =========================================================================
# 2. Dinâmica Quântica (Qubit) - Baseada em Hamiltoniano
# =========================================================================
class HamiltonianDynamics:
    """
    Evolui o sistema quântico usando um Hamiltoniano local (Modelo de Ising Transverso),
    com um acoplamento opcional à geometria emergente, simulando a TRGI.

    Hamiltoniano para um par de vizinhos ⟨i,j⟩:
        H_ij = -J_eff * Z_i ⊗ Z_j   - h * (X_i ⊗ I + I ⊗ X_j)
    Onde:
    - J_eff: Força de interação (ferromagnética para J > 0), pode depender da geometria.
    - h: Força do campo transverso, induz flutuações quânticas (superposição).
    - Z, X, I: Matrizes de Pauli e Identidade.
    """

    def __init__(
            self,
            manifold: 'Manifold',
            geometry: 'EmergentGeometry',
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
        print(f"HamiltonianDynamics: J={J}, h={h}, dt={dt}, GeoCoupling={use_geometric_coupling}")

    def get_local_hamiltonian(self, pos: Tuple[int, int]) -> np.ndarray:
        """
        Constrói o Hamiltoniano 4x4 para a interação entre o qubit em `pos`
        e seu vizinho da direita.
        """
        # --- O NÚCLEO DA TRGI: ACOPLAMENTO GEOMETRIA-DINÂMICA ---
        J_eff = self.J_base
        if self.use_geometric_coupling:
            neighbor_pos = self.manifold._wrap(pos[0], pos[1] + 1)
            # A distância informacional é o ângulo entre os vetores de Bloch [0, π]
            dist = self.geometry.compute_local_metric_analogue(pos, neighbor_pos)
            # Mapeia a distância para um fator de acoplamento:
            # - dist = 0 (alinhados) -> fator = 1 (acoplamento máximo)
            # - dist = π (anti-alinhados) -> fator = 0 (acoplamento nulo)
            coupling_factor = 1.0 - (dist / np.pi)
            J_eff *= coupling_factor

        # Termo de interação (energia menor se os spins Z estiverem alinhados)
        H_interaction = -J_eff * np.kron(SIGMA_Z, SIGMA_Z)

        # Termo de campo transverso (age em cada qubit individualmente)
        H_field = -self.h * (np.kron(SIGMA_X, ID) + np.kron(ID, SIGMA_X))

        return H_interaction + H_field

    def step(self):
        """
        Executa um passo de evolução temporal.

        Usa a decomposição de Trotter-Suzuki de primeira ordem:
        U(dt) = exp(-i*H*dt) ≈ exp(-i*H_hor*dt) * exp(-i*H_ver*dt)
        Onde H_hor é a soma das interações horizontais e H_ver das verticais.
        Isso nos permite evoluir pares de qubits de forma independente em cada sub-passo.
        """
        m = self.manifold

        # Sub-passo 1: Evoluir todos os pares HORIZONTAIS
        for r in range(m.rows):
            for c in range(m.cols):  # Itera sobre todos para garantir periodicidade
                self._evolve_pair((r, c), (r, c + 1))

        # Sub-passo 2: Evoluir todos os pares VERTICAIS
        for r in range(m.rows):  # Itera sobre todos
            for c in range(m.cols):
                self._evolve_pair((r, c), (r + 1, c))

    def _evolve_pair(self, pos1_raw: Tuple[int, int], pos2_raw: Tuple[int, int]):
        """Função auxiliar para evoluir um par de qubits."""
        m = self.manifold
        pos1 = m._wrap(*pos1_raw)
        pos2 = m._wrap(*pos2_raw)

        q1 = m.get_infon_state(pos1)
        q2 = m.get_infon_state(pos2)

        # 1. Obter o Hamiltoniano para este par
        H_pair = self.get_local_hamiltonian(pos1)

        # 2. Calcular o operador de evolução temporal U = exp(-iHΔt)
        U_pair = expm(-1j * H_pair * self.dt)

        # 3. Aplicar a evolução ao estado combinado do par
        pair_state = np.kron(q1.state, q2.state)
        evolved_pair_state = U_pair @ pair_state

        # 4. Atualizar os qubits individuais (com aproximação)
        q1_new_state, q2_new_state = self._unentangle_and_update(evolved_pair_state)

        # Importante: Criar NOVAS instâncias de Qubit para evitar modificar
        # o estado que outros cálculos no mesmo passo possam estar usando.
        # Isto é uma simplificação; um modelo mais complexo usaria dois grids (atual e próximo).
        from .infon_qubit import Qubit
        m.set_infon_state(pos1, Qubit(a=q1_new_state[0], b=q1_new_state[1]))
        m.set_infon_state(pos2, Qubit(a=q2_new_state[0], b=q2_new_state[1]))

    def _unentangle_and_update(self, pair_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aproxima os estados de qubit individuais a partir de um estado de par emaranhado.

        !! AVISO: Esta é uma APROXIMAÇÃO DE CAMPO MÉDIO e a maior simplificação do modelo !!
        Um estado de dois qubits emaranhado, em geral, NÃO PODE ser escrito como um produto
        tensorial de dois estados de qubit únicos. A maneira fisicamente rigorosa de
        lidar com isso seria usar matrizes de densidade e o traço parcial.

        Para manter a simulação computacionalmente tratável com vetores de estado,
        projetamos o estado emaranhado de volta para o espaço de produto mais próximo.
        Isso captura parte da dinâmica, mas perde a informação de emaranhamento entre passos.
        """
        # Decomposição em valores singulares (SVD) é o método mais robusto
        # para encontrar a melhor aproximação de produto tensorial.
        # U, S, Vh = svd(matrix) -> A ≈ S[0] * U[:,0] ⊗ Vh[0,:]
        try:
            U, S, Vh = np.linalg.svd(pair_state.reshape(2, 2))

            # Os novos estados são a primeira coluna de U e a primeira linha de Vh
            q1_state = U[:, 0] * np.sqrt(S[0])
            q2_state = Vh[0, :] * np.sqrt(S[0])

            return q1_state, q2_state
        except np.linalg.LinAlgError:
            # Em caso de erro numérico, retorna um estado padrão.
            return np.array([1, 0], dtype=complex), np.array([1, 0], dtype=complex)