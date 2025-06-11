import numpy as np
from typing import Tuple, List, Union

class Manifold:
    """
    Representa a variedade discreta (grade n-dimensional) e os infons.
    """
    def __init__(self, dimensions: Tuple[int, ...], infon_type: str = 'scalar', boundary_conditions: str = 'periodic'):
        """
        Inicializa a variedade.

        Args:
            dimensions (Tuple[int, ...]): Dimensões da grade (e.g., (rows, cols) para 2D).
            infon_type (str): Tipo de infon (atualmente, apenas 'scalar' suportado).
            boundary_conditions (str): Condições de contorno ('periodic' ou 'fixed').
        """
        if not dimensions or len(dimensions) != 2: # Atualmente forçando 2D
            raise ValueError("Dimensions must be a 2-tuple (rows, cols).")
        
        self.rows, self.cols = dimensions
        self.infon_type = infon_type
        self.boundary_conditions = boundary_conditions
        
        if self.infon_type == 'scalar':
            self.grid: np.ndarray = np.zeros(dimensions, dtype=float) # Usar float para potencializar futuros tipos
        else:
            raise NotImplementedError(f"Infon type '{infon_type}' not yet supported.")

        print(f"Manifold initialized: {self.rows}x{self.cols}, Infon: {self.infon_type}, Boundary: {self.boundary_conditions}")

    def initialize_infons(self, method: str = 'random', **kwargs):
        """
        Inicializa o estado dos infons na grade.

        Args:
            method (str): Metodo de inicialização ('random', 'pattern', 'clear').
            **kwargs: Argumentos adicionais dependendo do metodo.
                      Para 'random': density (float, 0-1), seed_value (int, opcional).
                      Para 'pattern': pattern_array (np.ndarray).
                      Para 'clear': value (float, default 0.0).
        """
        if method == 'random':
            density = kwargs.get('density', 0.5)
            seed = kwargs.get('seed_value', None)
            if seed is not None:
                np.random.seed(seed)
            self.grid = (np.random.rand(self.rows, self.cols) < density).astype(float)
        elif method == 'pattern':
            pattern_array = kwargs.get('pattern_array')
            if pattern_array is None or pattern_array.shape != self.grid.shape:
                raise ValueError("Pattern array must be provided and match grid dimensions.")
            self.grid = pattern_array.astype(float)
        elif method == 'clear':
            clear_value = kwargs.get('value', 0.0)
            self.grid.fill(clear_value)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        print(f"Infons initialized using method: {method}")

    def _apply_boundary_conditions(self, r: int, c: int) -> Tuple[int, int]:
        """Aplica condições de contorno para uma dada coordenada."""
        if self.boundary_conditions == 'periodic':
            r = r % self.rows
            c = c % self.cols
        elif self.boundary_conditions == 'fixed':
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                # Para 'fixed', fora dos limites pode retornar um valor especial ou levantar erro
                # Aqui, vamos tratar como se tivesse um valor que não contribui (e.g. 0 para contagem de vizinhos)
                # mas para get_infon_state, levantaremos erro.
                raise IndexError("Position out of bounds for 'fixed' boundary condition.")
        return r, c

    def get_infon_state(self, position: Tuple[int, int]) -> float:
        """
        Retorna o estado do infon na posição especificada.

        Args:
            position (Tuple[int, int]): Coordenadas (row, col).

        Returns:
            float: Estado do infon.
        """
        r, c = position
        if self.boundary_conditions == 'periodic':
            r_eff, c_eff = self._apply_boundary_conditions(r, c)
            return self.grid[r_eff, c_eff]
        elif self.boundary_conditions == 'fixed':
            if 0 <= r < self.rows and 0 <= c < self.cols:
                return self.grid[r, c]
            else:
                # Ou retornar um valor padrão para vizinhos fora, como 0.0,
                # mas para acesso direto, melhor levantar erro.
                raise IndexError("Position out of bounds for 'fixed' boundary condition during get_infon_state.")
        return 0.0 # Caso padrão, não deveria acontecer com as lógicas acima

    def set_infon_state(self, position: Tuple[int, int], value: float):
        """
        Define o estado do infon na posição especificada.

        Args:
            position (Tuple[int, int]): Coordenadas (row, col).
            value (float): Novo estado do infon.
        """
        r, c = position
        # Não aplicamos condições de contorno aqui, pois o set deve ser dentro dos limites reais.
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[r, c] = value
        else:
            print(f"Warning: Attempted to set infon state out of bounds: {position}")


    def get_neighbors_sum(self, position: Tuple[int, int], neighborhood_type: str = 'moore') -> float:
        """
        Retorna a SOMA dos estados dos infons vizinhos.

        Args:
            position (Tuple[int, int]): Coordenadas (row, col) do infon central.
            neighborhood_type (str): Tipo de vizinhança ('moore' para 8 vizinhos).

        Returns:
            float: Soma dos estados dos vizinhos.
        """
        r_center, c_center = position
        neighbor_sum = 0.0

        if neighborhood_type == 'moore':
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Não conta o próprio infon

                    r, c = r_center + dr, c_center + dc
                    
                    try:
                        if self.boundary_conditions == 'periodic':
                             r_eff, c_eff = self._apply_boundary_conditions(r,c)
                             neighbor_sum += self.grid[r_eff, c_eff]
                        elif self.boundary_conditions == 'fixed':
                            if 0 <= r < self.rows and 0 <= c < self.cols: # Verifica se o vizinho está dentro
                                neighbor_sum += self.grid[r,c]
                            # Se 'fixed' e fora do limite, o vizinho é efetivamente 0, então não adicionamos.
                    except IndexError:
                        # Para 'fixed' e vizinho fora dos limites, efetivamente contribui com 0.
                        pass 
        else:
            raise NotImplementedError(f"Neighborhood type '{neighborhood_type}' not supported.")
        
        return neighbor_sum