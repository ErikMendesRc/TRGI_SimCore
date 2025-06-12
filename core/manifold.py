from __future__ import annotations

import numpy as np
from typing import Tuple, Iterable, Iterator

from .infon_qubit import Qubit


class Manifold:
    """
    Representa uma variedade discreta 2D contendo infons.

    Pode operar em dois modos:
    - 'scalar': usa valores float (0.0 ou 1.0) em cada célula.
    - 'qubit': usa instâncias da classe Qubit em cada célula.
    """

    def __init__(
        self,
        dimensions: Tuple[int, int],
        infon_type: str = "scalar",
        boundary_conditions: str = "periodic",
    ):
        """
        Inicializa a grade 2D da variedade.

        Args:
            dimensions (Tuple[int, int]): Tamanho da grade (linhas, colunas).
            infon_type (str): Tipo de infon ('scalar' ou 'qubit').
            boundary_conditions (str): Condição de contorno ('periodic' ou 'fixed').
        """
        if len(dimensions) != 2:
            raise ValueError("Atualmente só suportamos grids 2-D.")

        self.rows, self.cols = dimensions
        self.infon_type = infon_type.lower()
        self.boundary_conditions = boundary_conditions

        if self.infon_type == "scalar":
            self.grid: np.ndarray = np.zeros(dimensions, dtype=float)

        elif self.infon_type == "qubit":
            self.grid = np.empty(dimensions, dtype=object)
            for r in range(self.rows):
                for c in range(self.cols):
                    self.grid[r, c] = Qubit()

        else:
            raise NotImplementedError(f"Infon type '{infon_type}' não suportado.")

    def initialize_infons(self, method: str = "random", **kwargs):
        """
        Inicializa ou reinicializa o grid escalar com valores padrão, aleatórios ou padrões definidos.

        Args:
            method (str): Método de inicialização ('random', 'pattern', 'clear').
            **kwargs: Parâmetros específicos para o método selecionado.
        """
        if self.infon_type != "scalar":
            return

        if method == "random":
            density = kwargs.get("density", 0.5)
            seed = kwargs.get("seed_value")
            if seed is not None:
                np.random.seed(seed)
            self.grid = (np.random.rand(self.rows, self.cols) < density).astype(float)

        elif method == "pattern":
            pattern_array = kwargs.get("pattern_array")
            if pattern_array is None or pattern_array.shape != self.grid.shape:
                raise ValueError("pattern_array ausente ou com shape incompatível.")
            self.grid = pattern_array.astype(float)

        elif method == "clear":
            value = kwargs.get("value", 0.0)
            self.grid.fill(value)

        else:
            raise ValueError(f"Método de init desconhecido: {method}")

    def _wrap(self, r: int, c: int) -> Tuple[int, int]:
        """
        Aplica condição de contorno à coordenada fornecida.

        Args:
            r (int): Linha.
            c (int): Coluna.

        Returns:
            Tuple[int, int]: Coordenada ajustada conforme as condições de contorno.
        """
        if self.boundary_conditions == "periodic":
            return r % self.rows, c % self.cols
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        raise IndexError("Posição fora dos limites para BC 'fixed'.")

    def get_infon_state(self, pos: Tuple[int, int]):
        """
        Retorna o estado do infon na posição indicada.

        Args:
            pos (Tuple[int, int]): Posição (linha, coluna).

        Returns:
            Valor escalar ou qubit, dependendo do tipo de infon.
        """
        r, c = self._wrap(*pos)
        return self.grid[r, c]

    def set_infon_state(self, pos: Tuple[int, int], value):
        """
        Define o valor do infon na posição indicada.

        Args:
            pos (Tuple[int, int]): Posição (linha, coluna).
            value: Valor escalar (float) ou Qubit.
        """
        r, c = self._wrap(*pos)
        self.grid[r, c] = value

    def neighbor_indices(self, pos: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
        """
        Gera as posições dos 8 vizinhos ao redor de uma célula.

        Args:
            pos (Tuple[int, int]): Posição central.

        Yields:
            Tuple[int, int]: Posições vizinhas (com wrapping se necessário).
        """
        r0, c0 = pos
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                yield self._wrap(r0 + dr, c0 + dc)

    def neighbors_sum(self, pos: Tuple[int, int]) -> float:
        """
        Soma os valores dos 8 vizinhos de uma célula escalar.

        Args:
            pos (Tuple[int, int]): Posição da célula.

        Returns:
            float: Soma dos valores dos vizinhos.

        Raises:
            TypeError: Se o tipo de infon não for escalar.
        """
        if self.infon_type != "scalar":
            raise TypeError("neighbors_sum só faz sentido para escalares.")
        return sum(self.grid[r, c] for r, c in self.neighbor_indices(pos))