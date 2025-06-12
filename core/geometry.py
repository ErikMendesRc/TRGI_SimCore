from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .manifold import Manifold
    from .infon_qubit import Qubit


def bloch_vector(q: Qubit) -> np.ndarray:
    """
    Converte o estado de um qubit em seu vetor de Bloch (x, y, z) na esfera de Bloch.

    Args:
        q (Qubit): Estado quântico do infon.

    Returns:
        np.ndarray: Vetor de Bloch tridimensional correspondente ao estado.
    """
    a, b = q.state
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(b) * a)
    z = abs(a) ** 2 - abs(b) ** 2
    return np.array([x, y, z])


class EmergentGeometry:
    """
    Classe responsável por calcular métricas geométricas emergentes
    a partir da distribuição de qubits em uma variedade discreta.
    """

    def __init__(self, manifold: Manifold):
        """
        Inicializa a geometria emergente com base na variedade fornecida.

        Args:
            manifold (Manifold): Objeto representando a estrutura espacial da simulação.
        """
        self.manifold = manifold
        grid_shape = self.manifold.grid.shape
        self.curvature_field = np.zeros(grid_shape)
        self.metric_analogue_field = np.zeros(grid_shape)

    def compute_local_metric_analogue(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """
        Calcula a distância informacional entre dois pontos, baseada na diferença
        entre os vetores de Bloch (ângulo entre os estados quânticos).

        Args:
            p1 (Tuple[int, int]): Coordenadas do primeiro ponto.
            p2 (Tuple[int, int]): Coordenadas do segundo ponto.

        Returns:
            float: Distância em radianos (0 a π).
        """
        if self.manifold.infon_type == "scalar":
            r1, c1, r2, c2 = *p1, *p2
            return float(np.hypot(r1 - r2, c1 - c2))

        v1 = bloch_vector(self.manifold.get_infon_state(p1))
        v2 = bloch_vector(self.manifold.get_infon_state(p2))
        dot_product = np.dot(v1, v2)
        norm_product = norm(v1) * norm(v2) + 1e-9
        cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def compute_curvature_analogue(self, pos: Tuple[int, int]) -> float:
        """
        Calcula um análogo de curvatura baseado no desvio padrão das distâncias
        informacionais entre um ponto e seus vizinhos.

        Args:
            pos (Tuple[int, int]): Posição (linha, coluna) na variedade.

        Returns:
            float: Valor de curvatura local.
        """
        dists = [
            self.compute_local_metric_analogue(pos, n_pos)
            for n_pos in self.manifold.neighbor_indices(pos)
        ]
        return float(np.std(dists))

    def compute_curvature_field(self) -> np.ndarray:
        """
        Calcula e armazena o campo de curvatura para toda a variedade.

        Returns:
            np.ndarray: Matriz 2D com valores de curvatura em cada célula.
        """
        for r in range(self.manifold.rows):
            for c in range(self.manifold.cols):
                self.curvature_field[r, c] = self.compute_curvature_analogue((r, c))
        return self.curvature_field