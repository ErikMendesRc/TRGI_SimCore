import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, TextBox, Slider
import numpy as np
import json
from core.manifold import Manifold
from core.dynamics import GameOfLifeTRGI # Ou sua classe de dinâmica

class TRGIInteractiveSim:
    """
    Interface gráfica interativa para a simulação TRGI usando Matplotlib.
    """
    def __init__(self, config_path: str = 'config/default_params.json'):
        """
        Inicializa a simulação interativa.

        Args:
            config_path (str): Caminho para o arquivo de configuração JSON.
        """
        try:
            with open(config_path, 'r') as f:
                self.params = json.load(f)
            print(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default parameters.")
            # Definir alguns padrões se o arquivo não for encontrado
            self.params = {
                "manifold_dims": [50, 50], "infon_type": "scalar", "boundary_conditions": "periodic",
                "initialization_method": "random", "initialization_density": 0.2,
                "animation_interval_ms": 100, "gol_rule_b": [3], "gol_rule_s": [2, 3]
            }

        self.manifold_dims = tuple(self.params.get("manifold_dims", [50, 50]))

        self.manifold = Manifold(
            dimensions=self.manifold_dims,
            infon_type=self.params.get("infon_type", "scalar"),
            boundary_conditions=self.params.get("boundary_conditions", "periodic")
        )
        self.manifold.initialize_infons(
            method=self.params.get("initialization_method", "random"),
            density=self.params.get("initialization_density", 0.2)
        )

        # Usar GameOfLifeTRGI como exemplo de dinâmica
        self.dynamics = GameOfLifeTRGI(
            self.manifold,
            rule_b=self.params.get("gol_rule_b", [3]),
            rule_s=self.params.get("gol_rule_s", [2, 3])
        )

        self.running = False
        self.animation_interval = self.params.get("animation_interval_ms", 100)
        self.fig, self.ax_grid = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25, top=0.95, left=0.05, right=0.95) # Ajustar para dar espaço aos botões

        self.grid_image = self.ax_grid.imshow(self.manifold.grid, cmap='binary', interpolation='nearest', vmin=0, vmax=1)
        self.ax_grid.set_title("TRGI Simulation - Paused")
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])

        self._setup_widgets()
        self._connect_events()

        self.animation = None # Será definido em run()

    def _setup_widgets(self):
        """Configura os widgets da GUI."""
        ax_play_pause = plt.axes([0.1, 0.05, 0.15, 0.075])
        self.btn_play_pause = Button(ax_play_pause, 'Play/Pause')

        ax_step = plt.axes([0.3, 0.05, 0.1, 0.075])
        self.btn_step = Button(ax_step, 'Step')

        ax_reset = plt.axes([0.45, 0.05, 0.1, 0.075])
        self.btn_reset = Button(ax_reset, 'Reset')

        ax_clear = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.btn_clear = Button(ax_clear, 'Clear')

        # Slider de Densidade para Reset
        ax_density_slider = plt.axes([0.1, 0.15, 0.65, 0.03])
        self.density_slider = Slider(
            ax=ax_density_slider,
            label='Reset Density',
            valmin=0.0,
            valmax=1.0,
            valinit=self.params.get("initialization_density", 0.2),
            valstep=0.01
        )


    def _connect_events(self):
        """Conecta os eventos dos widgets e do mouse."""
        self.btn_play_pause.on_clicked(self._on_play_pause)
        self.btn_step.on_clicked(self._on_step)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_clear.on_clicked(self._on_clear)
        self.fig.canvas.mpl_connect('button_press_event', self._on_grid_click)

    def _on_play_pause(self, event):
        self.running = not self.running
        if self.running:
            self.ax_grid.set_title("TRGI Simulation - Running")
            if self.animation is None: # Iniciar pela primeira vez ou se foi parado
                 self.animation = FuncAnimation(self.fig, self._update_simulation_frame,
                                           interval=self.animation_interval, blit=False) # blit=False pode ser mais estável
            else:
                self.animation.resume()
        else:
            self.ax_grid.set_title("TRGI Simulation - Paused")
            if self.animation:
                self.animation.pause()
        self.fig.canvas.draw_idle()

    def _on_step(self, event):
        if not self.running:
            self.dynamics.step()
            self.grid_image.set_data(self.manifold.grid)
            self.fig.canvas.draw_idle()

    def _on_reset(self, event):
        self.running = False
        if self.animation:
            self.animation.pause()
        self.ax_grid.set_title("TRGI Simulation - Paused (Reset)")
        current_density = self.density_slider.val
        self.manifold.initialize_infons(method='random', density=current_density)
        self.grid_image.set_data(self.manifold.grid)
        self.fig.canvas.draw_idle()

    def _on_clear(self, event):
        self.running = False
        if self.animation:
            self.animation.pause()
        self.ax_grid.set_title("TRGI Simulation - Paused (Cleared)")
        self.manifold.initialize_infons(method='clear')
        self.grid_image.set_data(self.manifold.grid)
        self.fig.canvas.draw_idle()

    def _on_grid_click(self, event):
        if event.inaxes == self.ax_grid and event.button == 1: # Botão esquerdo
            # Converter coordenadas do evento para índices da grade
            # event.xdata e event.ydata são floats, precisamos arredondar para o inteiro mais próximo
            col = int(round(event.xdata))
            row = int(round(event.ydata))

            if 0 <= row < self.manifold.rows and 0 <= col < self.manifold.cols:
                current_state = self.manifold.get_infon_state((row, col))
                new_state = 1.0 - current_state # Alterna 0 -> 1, 1 -> 0
                self.manifold.set_infon_state((row, col), new_state)
                self.grid_image.set_data(self.manifold.grid)
                self.fig.canvas.draw_idle()

    def _update_simulation_frame(self, frame_num):
        """Chamado por FuncAnimation."""
        if self.running:
            self.dynamics.step()
            self.grid_image.set_data(self.manifold.grid)
        return [self.grid_image] # Retorna uma lista de artistas modificados

    def run(self):
        """Inicia a GUI e o loop de animação."""
        # A animação é iniciada/retomada no _on_play_pause
        # self.animation = FuncAnimation(self.fig, self._update_simulation_frame,
        #                                interval=self.animation_interval, blit=False)
        plt.show()