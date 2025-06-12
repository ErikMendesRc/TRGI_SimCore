from __future__ import annotations
import numpy as np


class Qubit:
    """
    Representa um infon quântico (qubit), com estado vetorial normalizado |ψ⟩ = a|0⟩ + b|1⟩.

    Se nenhum valor for fornecido na criação, o estado é inicializado aleatoriamente
    na esfera de Bloch.
    """
    __slots__ = ("state",)

    def __init__(self, a: complex | None = None, b: complex | None = None):
        """
        Inicializa um qubit com os coeficientes complexos a e b.

        Args:
            a (complex | None): Amplitude para o estado |0⟩.
            b (complex | None): Amplitude para o estado |1⟩.
        """
        if a is None and b is None:
            theta = np.arccos(2 * np.random.rand() - 1)
            phi = 2 * np.pi * np.random.rand()
            a = np.cos(theta / 2)
            b = np.exp(1j * phi) * np.sin(theta / 2)
        elif a is None:
            a = np.sqrt(1 - abs(b)**2)
        elif b is None:
            b = np.sqrt(1 - abs(a)**2)

        norm = np.hypot(abs(a), abs(b))
        if norm < 1e-9:
            self.state = np.array([1.0, 0.0], dtype=np.complex128)
        else:
            self.state = np.array([a, b], dtype=np.complex128) / norm

    def apply_unitary(self, U: np.ndarray):
        """
        Aplica uma matriz unitária 2x2 ao estado do qubit (in-place).

        Args:
            U (np.ndarray): Matriz unitária 2x2.
        """
        self.state = U @ self.state

    def measure(self) -> int:
        """
        Mede o qubit na base computacional {|0⟩, |1⟩} e colapsa o estado.

        Returns:
            int: 0 ou 1, resultado da medição.
        """
        p0 = abs(self.state[0])**2
        result = 0 if np.random.rand() < p0 else 1
        self.state = np.array([1.0, 0.0] if result == 0 else [0.0, 1.0], dtype=np.complex128)
        return result

    @property
    def p0(self) -> float:
        """
        Retorna a probabilidade de medir o estado |0⟩.

        Returns:
            float: Valor de probabilidade entre 0 e 1.
        """
        return abs(self.state[0])**2