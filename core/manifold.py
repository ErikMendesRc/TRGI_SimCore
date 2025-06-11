"""
core/manifold.py
----------------
Grade 2-D que armazena infons:
- Escalares 0/1 (Game of Life)
- Qubits (novo modo quântico)
Mantém compatibilidade total com funções antigas.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Iterable, Iterator, List

from .infon_qubit import Qubit   # NEW


class Manifold:
    """Variedade discreta 2-D contendo infons."""

    # ----------------------------- INIT -------------------------------- #
    def __init__(
        self,
        dimensions: Tuple[int, int],
        infon_type: str = "scalar",
        boundary_conditions: str = "periodic",
    ):
        if len(dimensions) != 2:
            raise ValueError("Atualmente só suportamos grids 2-D.")

        self.rows, self.cols = dimensions
        self.infon_type = infon_type.lower()
        self.boundary_conditions = boundary_conditions

        # Alocação do grid
        if self.infon_type == "scalar":
            self.grid: np.ndarray = np.zeros(dimensions, dtype=float)

        elif self.infon_type == "qubit":
            self.grid = np.empty(dimensions, dtype=object)
            for r in range(self.rows):
                for c in range(self.cols):
                    self.grid[r, c] = Qubit()  # estado aleatório

        else:
            raise NotImplementedError(f"Infon type '{infon_type}' não suportado.")

        print(
            f"Manifold inicializado — {self.rows}×{self.cols}, "
            f"infon={self.infon_type}, BC={self.boundary_conditions}"
        )

    # ---------------------- Inicialização ESCALAR ---------------------- #
    def initialize_infons(self, method: str = "random", **kwargs):
        """
        Re-inicializa SOMENTE grids escalares (para qubits isso não se aplica).
        Métodos suportados: random, pattern, clear.
        """
        if self.infon_type != "scalar":
            print("Aviso: initialize_infons ignorado (modo qubit).")
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

        print(f"Infons escalares inicializados ({method}).")

    # ------------------------- Boundary helper ------------------------- #
    def _wrap(self, r: int, c: int) -> Tuple[int, int]:
        if self.boundary_conditions == "periodic":
            return r % self.rows, c % self.cols
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        raise IndexError("Posição fora dos limites para BC 'fixed'.")

    # --------------------------- Acesso -------------------------------- #
    def get_infon_state(self, pos: Tuple[int, int]):
        r, c = self._wrap(*pos)
        return self.grid[r, c]

    def set_infon_state(self, pos: Tuple[int, int], value):
        r, c = self._wrap(*pos)
        self.grid[r, c] = value

    # ------------------------- Vizinhanças ----------------------------- #
    def neighbor_indices(self, pos: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
        r0, c0 = pos
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                yield self._wrap(r0 + dr, c0 + dc)

    def neighbors_sum(self, pos: Tuple[int, int]) -> float:
        """Somente para grids escalares."""
        if self.infon_type != "scalar":
            raise TypeError("neighbors_sum só faz sentido para escalares.")
        return sum(self.grid[r, c] for r, c in self.neighbor_indices(pos))
