"""
analysis.py

Módulo responsável pela análise e visualização dos resultados da simulação TRGI.
Inclui funções para plotar métricas globais e correlação local entre curvatura e energia.

Autores: Colaboração Humano + IA
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_history(history: dict):
    """
    Plota a evolução das métricas globais ao longo do tempo.

    Parâmetros:
        history (dict): Dicionário contendo listas de valores por passo de simulação.
            Espera-se as chaves: 'step', 'entropy', 'avg_curvature', 'avg_energy'.
    """
    if not history or not history.get('step'):
        print("Análise de histórico: Nenhum dado coletado. Execute a simulação primeiro.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Evolução Temporal das Métricas Globais da TRGI", fontsize=16)

    steps = history['step']

    axes[0].plot(steps, history['entropy'], color='blue', label='Entropia de Shannon')
    axes[0].set_ylabel("Entropia (bits)")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(steps, history['avg_curvature'], color='green', label='Curvatura Média')
    axes[1].set_ylabel("Curvatura")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(steps, history['avg_energy'], color='red', label='Energia Média (T₀₀)')
    axes[2].set_ylabel("Energia")
    axes[2].set_xlabel("Passo da Simulação")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_correlation(geometry, tensor):
    """
    Plota um gráfico de dispersão entre curvatura local e densidade de energia (T₀₀),
    incluindo uma linha de regressão linear.

    Parâmetros:
        geometry (EmergentGeometry): Objeto contendo o campo de curvatura.
        tensor (TTensor): Objeto contendo a matriz de energia informacional T₀₀.
    """
    if geometry is None or tensor is None:
        print("Análise de correlação: Disponível apenas para o modo qubit.")
        return

    curvature_data = geometry.curvature_field.flatten()
    energy_data = tensor.T00_matrix.flatten()

    plt.figure(figsize=(8, 8))

    sample_indices = np.random.choice(
        len(curvature_data),
        min(len(curvature_data), 5000),
        replace=False
    )

    plt.scatter(
        curvature_data[sample_indices],
        energy_data[sample_indices],
        alpha=0.2,
        s=15,
        label='Amostra de Células'
    )

    try:
        m, b = np.polyfit(curvature_data, energy_data, 1)
        x_trend = np.array([min(curvature_data), max(curvature_data)])
        y_trend = m * x_trend + b
        plt.plot(
            x_trend, y_trend,
            color='red',
            linewidth=2,
            label=f'Tendência Linear (y={m:.2f}x + {b:.2f})'
        )
        plt.legend()
    except Exception as e:
        print(f"Não foi possível calcular a linha de tendência: {e}")

    plt.title("Correlação: Curvatura vs. Densidade de Energia (T₀₀)")
    plt.xlabel("Curvatura Local")
    plt.ylabel("Densidade de Energia Local (T₀₀)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()