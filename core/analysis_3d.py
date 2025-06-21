"""
core/analysis_3d.py
Rotinas de visualização 3-D para a simulação TRGI.

Requisitos:
- Matplotlib >= 3.4  (mplot3d embutido)
"""

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (ativa proj='3d')
import matplotlib.pyplot as plt
import numpy as np


def _surface(Z, title="", cmap="viridis", elev=40, azim=-60):
    """
    Renderiza uma matriz 2-D como superfície 3-D.
    """
    rows, cols = Z.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_zlabel("valor")
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.5, aspect=12)
    plt.show()


# ---------- superfícies ---------------------------------------------------- #

def plot_curvature_surface(geometry):
    """Superfície 3-D da curvatura emergente."""
    _surface(
        geometry.curvature_field,
        title="Curvatura emergente (superfície 3-D)",
        cmap="viridis",
    )


def plot_energy_surface(tensor):
    """Superfície 3-D da densidade de energia informacional T₀₀."""
    _surface(
        tensor.T00_matrix,
        title="Densidade de energia T₀₀ (superfície 3-D)",
        cmap="inferno",
    )


# ---------- nuvem de vetores de Bloch ------------------------------------- #

def plot_bloch_quiver(manifold, sample_step=3):
    """
    Nuvem de vetores de Bloch (quiver 3-D).
    """
    # import leve aqui para evitar dependência circular
    from core.geometry import bloch_vector

    rows, cols = manifold.rows, manifold.cols
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for r in range(0, rows, sample_step):
        for c in range(0, cols, sample_step):
            v = bloch_vector(manifold.get_infon_state((r, c)))
            ax.quiver(
                c, r, 0,
                v[0], v[1], v[2],
                length=0.9,
                normalize=True,
                linewidth=0.8,
                alpha=0.8,
            )

    ax.set_title("Vetores de Bloch (quiver 3-D)")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_zlabel("Bloch z")
    plt.show()
