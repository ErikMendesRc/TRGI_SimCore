# core/geometry.py
# (Versão Corrigida)

from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .manifold import Manifold
    from .infon_qubit import Qubit

# --- Helper fora da classe para não depender de 'self' ---
def bloch_vector(q: 'Qubit') -> np.ndarray:
    a, b = q.state
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(b) * a)
    z = abs(a)**2 - abs(b)**2
    return np.array([x, y, z])

# --- Classe Principal ---
class EmergentGeometry:
    def __init__(self, manifold: 'Manifold'):
        self.manifold = manifold
        
        # --- CORREÇÃO APLICADA AQUI ---
        # Inicializa os campos como arrays de zeros com o shape correto.
        # Isso garante que o atributo exista desde a criação do objeto.
        grid_shape = self.manifold.grid.shape
        self.curvature_field = np.zeros(grid_shape)
        # É uma boa ideia fazer o mesmo para outros campos que você possa calcular
        self.metric_analogue_field = np.zeros(grid_shape)

    def compute_local_metric_analogue(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calcula a 'distância' informacional entre dois pontos."""
        if self.manifold.infon_type == "scalar":
            r1, c1, r2, c2 = *p1, *p2
            return float(np.hypot(r1 - r2, c1 - c2))

        # Para qubits, a distância é o ângulo entre seus vetores de Bloch.
        v1 = bloch_vector(self.manifold.get_infon_state(p1))
        v2 = bloch_vector(self.manifold.get_infon_state(p2))
        
        # O produto escalar entre vetores normalizados é o cosseno do ângulo.
        dot_product = np.dot(v1, v2)
        # Adicionar 1e-9 previne divisão por zero se um vetor for nulo.
        norm_product = norm(v1) * norm(v2) + 1e-9
        cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
        
        return float(np.arccos(cos_angle))

    def compute_curvature_analogue(self, pos: Tuple[int, int]) -> float:
        """
        Calcula um proxy de curvatura local: o desvio padrão das distâncias
        informacionais aos 8 vizinhos. Alta variação = alta curvatura.
        """
        dists = [
            self.compute_local_metric_analogue(pos, n_pos)
            for n_pos in self.manifold.neighbor_indices(pos)
        ]
        return float(np.std(dists))

    def compute_curvature_field(self) -> np.ndarray:
        """
        Calcula o campo de curvatura para toda a variedade e o armazena
        no atributo self.curvature_field.
        """
        for r in range(self.manifold.rows):
            for c in range(self.manifold.cols):
                self.curvature_field[r, c] = self.compute_curvature_analogue((r, c))
        return self.curvature_field