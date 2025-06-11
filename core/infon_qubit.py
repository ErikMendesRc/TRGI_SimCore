# core/infon_qubit.py
from __future__ import annotations
import numpy as np

class Qubit:
    """
    Infon quântico elementar. Estado |ψ⟩ = a|0⟩ + b|1⟩.
    """
    __slots__ = ("state",)

    def __init__(self, a: complex | None = None, b: complex | None = None):
        if a is None and b is None:
            # Estado aleatório uniforme na esfera de Bloch
            theta, phi = np.arccos(2 * np.random.rand() - 1), np.random.rand() * 2 * np.pi
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
        """Aplica matriz 2×2 unitária U sobre o qubit in-place."""
        self.state = U @ self.state

    def measure(self) -> int:
        """Mede o qubit na base computacional e colapsa o estado."""
        p0 = abs(self.state[0])**2
        result = 0 if np.random.rand() < p0 else 1
        self.state = np.array([1.0, 0.0] if result == 0 else [0.0, 1.0], dtype=np.complex128)
        return result

    @property
    def p0(self) -> float:
        """Probabilidade de medir |0⟩."""
        return abs(self.state[0])**2