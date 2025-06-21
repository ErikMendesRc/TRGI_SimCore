"""Run automated TRGI simulations varying a set of parameters."""
from __future__ import annotations
import json
from itertools import product
from pathlib import Path

import numpy as np

from core.manifold import Manifold
from core.dynamics import HamiltonianDynamics
from core.geometry import EmergentGeometry
from core.t_tensor import TTensor
from core.metrics import shannon_entropy
from core.research_tools import save_results


DEF_PARAMS = {
    "manifold_dims": [20, 20],
    "J": [1.0],
    "h": [0.2, 0.5, 0.8],
    "steps": 20,
}


def run_once(rows: int, cols: int, J: float, h: float, steps: int):
    manifold = Manifold((rows, cols), infon_type="qubit")
    geom = EmergentGeometry(manifold)
    dyn = HamiltonianDynamics(manifold, geom, J=J, h=h)
    tensor = TTensor(manifold, dyn)

    history = {"entropy": [], "avg_curvature": [], "avg_energy": []}
    for _ in range(steps):
        geom.compute_curvature_field()
        tensor.compute_T_matrix_global()
        history["entropy"].append(shannon_entropy(manifold.grid))
        history["avg_curvature"].append(np.mean(geom.curvature_field))
        history["avg_energy"].append(np.mean(tensor.T00_matrix))
        dyn.step()
    return history


def main(cfg: str | None = None):
    if cfg:
        with open(cfg) as f:
            params = json.load(f)
    else:
        params = DEF_PARAMS

    dims = params.get("manifold_dims", [20, 20])
    J_list = params.get("J", [1.0])
    h_list = params.get("h", [0.2])
    steps = params.get("steps", 10)

    results = {}
    for J, h in product(J_list, h_list):
        hist = run_once(dims[0], dims[1], J, h, steps)
        key = f"J{J}_h{h}"
        results[key] = hist

    save_results(results, Path("results") / "sweep_results.json")


if __name__ == "__main__":
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)