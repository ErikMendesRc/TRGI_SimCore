# gui/interactive_sim_mpl.py
# (Versão Completa e Corrigida)

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from core.infon_qubit import Qubit
from core.manifold import Manifold
from core.dynamics import GameOfLifeTRGI, HamiltonianDynamics
from core.geometry import EmergentGeometry
from core.t_tensor import TTensor
from core.metrics import shannon_entropy
from core.analysis import plot_history, plot_correlation


class TRGIInteractiveSim:
    def __init__(self, config_path: str = "config/default_params.json"):
        # 1. Carregar configuração
        try:
            with open(config_path, "r") as f:
                self.params = json.load(f)
            print(f"Config loaded from {config_path}")
        except FileNotFoundError:
            print(f"⚠ Config file '{config_path}' not found. Using default parameters.")
            self.params = {"manifold_dims": [40, 40], "infon_type": "qubit", "boundary_conditions": "periodic",
                           "J": 1.0, "h": 0.8, "dt": 0.1, "use_geometric_coupling": True, "animation_interval_ms": 50}

        # 2. Inicializar os componentes da simulação
        self._initialize_simulation()

        # 3. Configurar a GUI
        self.running = False
        self.animation = None

        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(bottom=0.25, top=0.9)

        self._setup_plots()
        self._setup_widgets()
        self._connect_events()

        # 4. Desenhar o estado inicial completo
        self._refresh_plot()

    def _initialize_simulation(self):
        """Cria ou recria todas as instâncias necessárias para a simulação."""
        self.history = {'step': [], 'entropy': [], 'avg_curvature': [], 'avg_energy': []}
        self.step_count = 0

        dims = tuple(self.params["manifold_dims"])
        infon_type = self.params["infon_type"]
        bc = self.params.get("boundary_conditions", "periodic")

        print("\n--- Initializing Simulation Core ---")
        self.manifold = Manifold(dims, infon_type=infon_type, boundary_conditions=bc)
        self.geometry = EmergentGeometry(self.manifold)

        if infon_type == "scalar":
            self.manifold.initialize_infons()
            self.dynamics = GameOfLifeTRGI(self.manifold)
            self.tensor = None
        else:
            # *** CORREÇÃO APLICADA AQUI ***
            # 1. Definimos as chaves que o construtor de HamiltonianDynamics aceita.
            allowed_keys = ['J', 'h', 'dt', 'use_geometric_coupling']

            # 2. Criamos um dicionário `dynamics_params` contendo apenas essas chaves.
            #    Usamos .get() para segurança, caso uma chave não esteja no JSON.
            dynamics_params = {key: self.params.get(key) for key in allowed_keys if key in self.params}

            # 3. Passamos o dicionário filtrado para o construtor.
            self.dynamics = HamiltonianDynamics(self.manifold, self.geometry, **dynamics_params)
            self.tensor = TTensor(self.manifold, self.dynamics)

        # Pré-calcula e coleta métricas para o passo 0
        if self.manifold.infon_type == 'qubit':
            self._collect_metrics()

        print("--- Initialization Complete ---\n")

    def _setup_plots(self):
        self.ax_state, self.ax_curvature, self.ax_energy = self.axes
        grid_shape = self.manifold.grid.shape
        cmap_state = "binary" if self.manifold.infon_type == "scalar" else "RdBu"
        self.im_state = self.ax_state.imshow(np.zeros(grid_shape), cmap=cmap_state, vmin=0, vmax=1,
                                             interpolation='nearest')
        self.ax_state.set_title("Infon State (P(|0⟩))")
        self.ax_state.set_xticks([]);
        self.ax_state.set_yticks([])
        self.im_curvature = self.ax_curvature.imshow(np.zeros(grid_shape), cmap='viridis', vmin=0, vmax=1.2,
                                                     interpolation='nearest')
        self.ax_curvature.set_title("Emergent Curvature")
        self.ax_curvature.set_xticks([]);
        self.ax_curvature.set_yticks([])
        self.im_energy = self.ax_energy.imshow(np.zeros(grid_shape), cmap='inferno', vmin=-2, vmax=2,
                                               interpolation='nearest')
        self.ax_energy.set_title("Info. Energy Density (T₀₀)")
        self.ax_energy.set_xticks([]);
        self.ax_energy.set_yticks([])
        self.txt_info = self.fig.text(0.5, 0.95, "Status: Paused", ha='center', fontsize=12)

    def _setup_widgets(self):
        self.btn_play = Button(plt.axes([0.25, 0.05, 0.1, 0.075]), "Play/Pause")
        self.btn_step = Button(plt.axes([0.37, 0.05, 0.1, 0.075]), "Step")
        self.btn_reset = Button(plt.axes([0.49, 0.05, 0.1, 0.075]), "Reset")
        self.btn_clear = Button(plt.axes([0.61, 0.05, 0.1, 0.075]), "Clear")
        self.btn_analyze = Button(plt.axes([0.73, 0.05, 0.1, 0.075]), "Analyze")

    def _connect_events(self):
        self.btn_play.on_clicked(self._on_play_pause)
        self.btn_step.on_clicked(self._on_step)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_clear.on_clicked(self._on_clear)
        self.btn_analyze.on_clicked(self._on_analyze)

    def _on_play_pause(self, _):
        self.running = not self.running
        if self.running:
            if self.animation is None:
                self.animation = FuncAnimation(self.fig, self._update_frame,
                                               interval=self.params["animation_interval_ms"], blit=False)
            self.animation.resume()
        else:
            if self.animation: self.animation.pause()
        self.fig.canvas.draw_idle()

    def _on_step(self, _):
        if not self.running: self._simulation_step()

    def _on_reset(self, _):
        if self.running: self._on_play_pause(None)
        self._initialize_simulation()
        self._refresh_plot()

    def _on_clear(self, _):
        if self.running: self._on_play_pause(None)
        if self.manifold.infon_type == "scalar":
            self.manifold.initialize_infons("clear")
        else:
            for r in range(self.manifold.rows):
                for c in range(self.manifold.cols):
                    self.manifold.set_infon_state((r, c), Qubit(a=1.0, b=0.0))
        # Recalcula e redesenha após limpar
        self._collect_metrics()
        self._refresh_plot()

    def _on_analyze(self, _):
        print("\n--- Generating Analysis Plots ---")
        plot_history(self.history)
        plot_correlation(self.geometry, self.tensor)

    def _collect_metrics(self):
        self.geometry.compute_curvature_field()
        if self.tensor: self.tensor.compute_T_matrix_global()

        self.history['step'].append(self.step_count)
        self.history['entropy'].append(shannon_entropy(self.manifold.grid))
        self.history['avg_curvature'].append(np.mean(self.geometry.curvature_field))
        if self.tensor: self.history['avg_energy'].append(np.mean(self.tensor.T00_matrix))

    def _simulation_step(self):
        self.dynamics.step()
        self.step_count += 1

        if self.manifold.infon_type == 'qubit':
            self._collect_metrics()

        self._refresh_plot()

    def _grid_to_plot(self):
        if self.manifold.infon_type == "scalar": return self.manifold.grid
        return np.vectorize(lambda q: q.p0)(self.manifold.grid)

    def _update_frame(self, _):
        self._simulation_step()
        return self.axes

    def _refresh_plot(self):
        self.im_state.set_data(self._grid_to_plot())

        status_text = f"Status: {'Running' if self.running else 'Paused'} (Step: {self.step_count})"

        if self.manifold.infon_type == 'qubit' and self.history['entropy']:
            H = self.history['entropy'][-1]
            self.txt_info.set_text(f"{status_text} | Avg. Entropy: {H:.3f}")
            self.im_curvature.set_data(self.geometry.curvature_field)
            if self.tensor:
                energy_data = self.tensor.T00_matrix
                self.im_energy.set_data(energy_data)
                emin, emax = np.min(energy_data), np.max(energy_data)
                self.im_energy.set_clim(vmin=emin - 1e-9, vmax=emax + 1e-9)
        else:
            self.txt_info.set_text(status_text)

        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()