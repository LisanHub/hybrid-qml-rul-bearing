"""
Variational quantum circuit (VQC) for hybrid quantum–classical models.

This module uses PennyLane: angle embedding maps classical inputs to qubit rotations,
:class:`~pennylane.StronglyEntanglingLayers` implements a generic hardware-efficient
ansatz, and single-qubit :math:`\\langle Z \\rangle` readouts feed a classical head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Defaults matching the thesis VQC specification
# -----------------------------------------------------------------------------
DEFAULT_N_QUBITS: int = 6
DEFAULT_N_LAYERS: int = 3

_project_root = Path(__file__).resolve().parent.parent
_DEFAULT_DIAGRAM_PATH = _project_root / "results" / "figures" / "vqc_diagram.png"


def create_quantum_device(n_qubits: int = 6) -> qml.Device:
    """
    Create a noiseless statevector simulator on ``n_qubits`` wires.

    ``default.qubit`` tracks the full quantum state |ψ⟩ and applies gates exactly.
    This is the usual starting point before moving to noisy / smaller QPUs.
    """
    return qml.device("default.qubit", wires=n_qubits)


def make_vqc_qnode(
    dev: qml.Device,
    n_layers: int,
    **qnode_kwargs: Any,
):
    """
    Factory: build a QNode with angle embedding + StronglyEntanglingLayers + ⟨Z⟩.

    Parameters
    ----------
    dev
        PennyLane device (wires implied by ``dev.num_wires``).
    n_layers
        Number of stacked strongly-entangling layers (depth of the ansatz).
    **qnode_kwargs
        Passed to ``@qml.qnode`` (e.g. ``interface="torch"``, ``diff_method="backprop"``).

    Returns
    -------
    Callable
        ``circuit(inputs, weights)`` — see module-level :func:`quantum_circuit`.
    """
    # PennyLane v0.4x ``default.qubit`` exposes wire labels via ``dev.wires``.
    wires = list(dev.wires)

    @qml.qnode(dev, **qnode_kwargs)
    def circuit(
        inputs: Union[torch.Tensor, np.ndarray],
        weights: Union[torch.Tensor, np.ndarray],
    ) -> List[Any]:
        # --- State preparation (feature map) ---------------------------------
        # AngleEmbedding writes each classical component x_j into a *rotation* on qubit j.
        # Here we use RX: |0⟩ → cos(x/2)|0⟩ − i sin(x/2)|1⟩ (up to global phase).
        # Intuition: higher-dimensional classical vectors need more qubits; we use
        # n_qubits features per forward pass (pad or project upstream if needed).
        qml.AngleEmbedding(inputs, wires=wires, rotation="X")

        # --- Trainable ansatz -------------------------------------------------
        # StronglyEntanglingLayers stacks, per layer, single-qubit rotations plus a
        # ring (cyclic) pattern of CNOTs.  The ``weights`` tensor has shape
        # (n_layers, n_qubits, 3): three Euler-like angles per wire per layer.
        # This is a standard *hardware-inspired* template: expressive but structured.
        qml.StronglyEntanglingLayers(weights, wires=wires)

        # --- Measurement (information extraction) ----------------------------
        # Pauli-Z expectation ⟨Z_k⟩ ∈ [-1, 1] on each qubit k.  Together they form a
        # bounded 6-D quantum feature vector downstream classical layers can regress on.
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return circuit


# Simulator used by the module-level ``quantum_circuit`` (Torch interface for training).
_default_dev: qml.Device = create_quantum_device(DEFAULT_N_QUBITS)
quantum_circuit = make_vqc_qnode(
    _default_dev,
    DEFAULT_N_LAYERS,
    interface="torch",
    diff_method="backprop",
)


def strongly_entangling_weights_shape(n_layers: int, n_qubits: int) -> tuple[int, int, int]:
    """Return the ``weights`` shape expected by :class:`~pennylane.StronglyEntanglingLayers`."""
    return (n_layers, n_qubits, 3)


class QuantumLayer(nn.Module):
    """
    PyTorch module wrapping the VQC via :class:`pennylane.qnn.TorchLayer`.

    ``forward`` expects ``x`` shaped ``(batch, n_qubits)`` — one classical scalar per
    qubit after any classical preprocessing.  Trainable parameters live in the
    StronglyEntanglingLayers block only; embedding uses the *input* data.
    """

    def __init__(
        self,
        n_qubits: int = DEFAULT_N_QUBITS,
        n_layers: int = DEFAULT_N_LAYERS,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = create_quantum_device(n_qubits)
        qnode = make_vqc_qnode(
            dev,
            n_layers,
            interface="torch",
            diff_method="backprop",
        )
        wshape = strongly_entangling_weights_shape(n_layers, n_qubits)
        self.qnn: qml.qnn.TorchLayer = qml.qnn.TorchLayer(
            qnode,
            weight_shapes={"weights": wshape},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(x)


def save_circuit_diagram(
    filepath: Optional[Union[str, Path]] = None,
    *,
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
    rng_seed: int = 0,
) -> Path:
    """
    Render the VQC and save an image (diagram) for reports / thesis figures.

    Uses :func:`pennylane.draw_mpl` with dummy NumPy inputs and random weights.
    Falls back to a text diagram in ``.txt`` if Matplotlib drawing fails.

    Parameters
    ----------
    filepath
        Output path (``.png`` recommended).  Default: ``results/figures/vqc_diagram.png``.
    n_qubits, n_layers
        Topology to draw (should match training if used in documentation).
    rng_seed
        Seed for reproducible random weights in the picture (shape only matters).

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """
    out = Path(filepath) if filepath is not None else _DEFAULT_DIAGRAM_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    dev = create_quantum_device(n_qubits)
    circ = make_vqc_qnode(dev, n_layers)  # NumPy / default interface — sufficient for drawing.

    inputs = np.zeros(n_qubits, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    weights = rng.random(strongly_entangling_weights_shape(n_layers, n_qubits))

    try:
        fig, _ = qml.draw_mpl(circ, decimals=2)(inputs, weights)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        txt_path = out.with_suffix(".txt")
        txt_path.write_text(qml.draw(circ)(inputs, weights), encoding="utf-8")
        out = txt_path

    return out
