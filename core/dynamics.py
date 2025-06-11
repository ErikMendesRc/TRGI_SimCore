import numpy as np
from .manifold import Manifold
from typing import List

class InformationalDynamics:
    """
    Classe base abstrata para as regras de evolução dos infons.
    """
    def __init__(self, manifold: Manifold, **params):
        """
        Inicializa a dinâmica.

        Args:
            manifold (Manifold): A instância da variedade onde a dinâmica opera.
            **params: Parâmetros específicos para a dinâmica.
        """
        self.manifold = manifold
        self.params = params

    def step(self):
        """
        Aplica um passo das regras de evolução na variedade.
        Este método deve ser implementado por subclasses.
        """
        raise NotImplementedError("Subclasses must implement the 'step' method.")

class GameOfLifeTRGI(InformationalDynamics):
    """
    Implementa a dinâmica do Jogo da Vida de Conway como um exemplo.
    Infons são 'vivos' (1.0) ou 'mortos' (0.0).
    """
    def __init__(self, manifold: Manifold, rule_b: List[int] = [3], rule_s: List[int] = [2, 3]):
        """
        Inicializa a dinâmica do Jogo da Vida.

        Args:
            manifold (Manifold): A instância da variedade.
            rule_b (List[int]): Lista de contagens de vizinhos para um infon 'nascer'.
            rule_s (List[int]): Lista de contagens de vizinhos para um infon 'sobreviver'.
        """
        super().__init__(manifold)
        self.rule_b = rule_b
        self.rule_s = rule_s
        print(f"GameOfLifeTRGI dynamics initialized with B{'/'.join(map(str,rule_b))}/S{'/'.join(map(str,rule_s))}")

    def step(self):
        """
        Aplica um passo da lógica do Jogo da Vida.
        """
        new_grid = self.manifold.grid.copy() # Trabalha em uma cópia para atualizações simultâneas

        for r in range(self.manifold.rows):
            for c in range(self.manifold.cols):
                current_state = self.manifold.grid[r, c]
                # Usamos get_neighbors_sum pois os estados são 0 ou 1, a soma é a contagem de vizinhos vivos.
                num_alive_neighbors = int(self.manifold.get_neighbors_sum((r, c), neighborhood_type='moore'))

                if current_state == 1.0:  # Infon 'vivo'
                    if num_alive_neighbors not in self.rule_s:
                        new_grid[r, c] = 0.0  # Morre por solidão ou superpopulação
                else:  # Infon 'morto'
                    if num_alive_neighbors in self.rule_b:
                        new_grid[r, c] = 1.0  # Nasce

        self.manifold.grid = new_grid