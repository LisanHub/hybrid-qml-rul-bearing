"""
Hybrid quantum–classical RUL regressor: classical encoder → VQC → classical decoder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .quantum_circuit import (
        DEFAULT_N_LAYERS,
        DEFAULT_N_QUBITS,
        QuantumLayer,
        strongly_entangling_weights_shape,
    )
except ImportError:  # running with ``sys.path`` pointing at ``src/`` (notebooks)
    from quantum_circuit import (
        DEFAULT_N_LAYERS,
        DEFAULT_N_QUBITS,
        QuantumLayer,
        strongly_entangling_weights_shape,
    )

_project_root = Path(__file__).resolve().parent.parent
_METRICS_PATH = _project_root / "results" / "metrics.json"


def _create_mixed_device(n_qubits: int) -> qml.Device:
    """Density-matrix simulator (required for depolarizing / CPTP channels)."""
    return qml.device("default.mixed", wires=n_qubits)


def make_noisy_vqc_qnode(
    dev: qml.Device,
    n_layers: int,
    depolarize_prob: float,
    **qnode_kwargs: Any,
):
    """
    Same layout as the clean VQC, with :class:`~pennylane.DepolarizingChannel`
    on every qubit after the embedding and after the strongly-entangling block.

    ``depolarize_prob`` ∈ [0, 1] is the single-qubit depolarization strength
    (probability mass moved off the Bloch sphere pole in the standard channel model).
    """
    wires = list(dev.wires)
    p = float(depolarize_prob)

    @qml.qnode(dev, **qnode_kwargs)
    def circuit(
        inputs: Union[torch.Tensor, np.ndarray],
        weights: Union[torch.Tensor, np.ndarray],
    ) -> list:
        qml.AngleEmbedding(inputs, wires=wires, rotation="X")
        if p > 0.0:
            for w in wires:
                qml.DepolarizingChannel(p, wires=w)
        qml.StronglyEntanglingLayers(weights, wires=wires)
        if p > 0.0:
            for w in wires:
                qml.DepolarizingChannel(p, wires=w)
        return [qml.expval(qml.PauliZ(ww)) for ww in wires]

    return circuit


class NoisyQuantumLayer(nn.Module):
    """
    Trainable VQC on a mixed-state device with depolarizing noise (evaluation/study).
    """

    def __init__(
        self,
        n_qubits: int = DEFAULT_N_QUBITS,
        n_layers: int = DEFAULT_N_LAYERS,
        noise_level: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_level = float(noise_level)

        dev = _create_mixed_device(n_qubits)
        qnode = make_noisy_vqc_qnode(
            dev,
            n_layers,
            depolarize_prob=self.noise_level,
            interface="torch",
            diff_method="parameter-shift",
        )
        wshape = strongly_entangling_weights_shape(n_layers, n_qubits)
        self.qnn: qml.qnn.TorchLayer = qml.qnn.TorchLayer(
            qnode,
            weight_shapes={"weights": wshape},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(x)


class HybridQMLModel(nn.Module):
    """
    Classical encoder (features → 6), quantum residual processor (6 → 6), decoder (6 → 1 RUL).

    Encoder: ``Linear(in → 16) → ReLU → Linear(16 → 6)`` to match ``n_qubits``.
    Decoder: ``Linear(6 → 32) → ReLU → Linear(32 → 1)``.

    If input has shape ``(batch, seq, feat)`` (e.g. LSTM-style windows), only the **last**
    timestep is encoded—consistent with predicting RUL from the most recent window.
    """

    def __init__(self, input_features: int) -> None:
        super().__init__()
        if input_features < 1:
            raise ValueError("input_features must be >= 1")

        self.input_features = input_features
        self.latent_dim = DEFAULT_N_QUBITS

        self.encoder = nn.Sequential(
            nn.Linear(input_features, 16),
            nn.ReLU(),
            nn.Linear(16, self.latent_dim),
        )
        self.quantum = QuantumLayer(DEFAULT_N_QUBITS, DEFAULT_N_LAYERS)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._noisy_modules: Dict[float, NoisyQuantumLayer] = {}

    @staticmethod
    def _collapse_time(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x[:, -1, :]
        if x.dim() == 2:
            return x
        raise ValueError(f"Expected x of shape (B, F) or (B, T, F); got {tuple(x.shape)}")

    def sync_noisy_weights(self, noise_level: float) -> NoisyQuantumLayer:
        """Return a cached noisy layer at ``noise_level`` with weights copied from ``self.quantum``."""
        key = round(float(noise_level), 8)
        if key not in self._noisy_modules or self._noisy_modules[key].noise_level != noise_level:
            self._noisy_modules[key] = NoisyQuantumLayer(
                DEFAULT_N_QUBITS,
                DEFAULT_N_LAYERS,
                noise_level=noise_level,
            )
        noisy = self._noisy_modules[key]
        noisy.qnn.load_state_dict(self.quantum.qnn.state_dict(), strict=True)
        return noisy

    def forward(self, x: torch.Tensor, noise_level: float = 0.0) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            ``(batch, features)`` or ``(batch, seq, features)``.
        noise_level
            If > 0 and the module is in ``eval`` mode, uses :meth:`sync_noisy_weights`
            so weights match ``self.quantum`` (call that once before batched inference).
            Training always uses the noiseless ``QuantumLayer``.
        """
        x_in = self._collapse_time(x)
        z = self.encoder(x_in)

        if float(noise_level) > 0.0 and not self.training:
            key = round(float(noise_level), 8)
            if key not in self._noisy_modules:
                raise RuntimeError(
                    "Eval with noise requires `model.sync_noisy_weights(noise_level)` "
                    "before `forward` (``evaluate_hybrid`` does this automatically)."
                )
            q = self._noisy_modules[key](z)
        else:
            q = self.quantum(z)

        return self.decoder(q).squeeze(-1)


def train_hybrid(
    model: HybridQMLModel,
    train_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    weights_path: Optional[Union[str, Path]] = None,
    show_progress: bool = False,
) -> HybridQMLModel:
    """
    Optimize with MSE on normalized RUL; quantum block is the noiseless ``QuantumLayer``.

    Optionally saves ``HybridQMLModel`` ``state_dict`` to ``results/hybrid_model.pt``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    epoch_iter = range(1, epochs + 1)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Hybrid QML train", unit="epoch")

    for epoch in epoch_iter:
        running = 0.0
        n_batches = 0
        batch_it = train_loader
        if show_progress:
            batch_it = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, unit="batch")
        for xb, yb in batch_it:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb, noise_level=0.0)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            running += float(loss.item())
            n_batches += 1

        if verbose and not show_progress and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"epoch {epoch:4d}/{epochs}  train_mse: {running / max(1, n_batches):.6f}")
        elif show_progress and verbose and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(mse=f"{running / max(1, n_batches):.6f}")

    if weights_path is not None:
        wp = Path(weights_path)
        wp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), wp)
        if verbose:
            print(f"Saved hybrid state_dict to {wp}")

    return model


@torch.no_grad()
def evaluate_hybrid(
    model: HybridQMLModel,
    test_loader: DataLoader,
    noise_level: float = 0.0,
    device: Optional[torch.device] = None,
    bearing_name: str = "test",
    save_results: bool = True,
) -> Tuple[float, float, np.ndarray]:
    """
    MAE / RMSE on normalized RUL.  When ``noise_level > 0``, weights are copied into a
    noisy ``TorchLayer`` for the quantum subroutine (mixed-state depolarizing simulation).

    Persists arrays to ``results/hybrid_predictions_<bearing>.npz`` and merges scalars
    into ``results/metrics.json`` when ``save_results`` is True.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    if float(noise_level) > 0.0:
        model.sync_noisy_weights(float(noise_level))

    preds: list[np.ndarray] = []
    targs: list[np.ndarray] = []

    for xb, yb in test_loader:
        xb = xb.to(device)
        pb = model(xb, noise_level=float(noise_level)).detach().cpu().numpy()
        preds.append(pb)
        targs.append(yb.numpy())

    pred = np.concatenate(preds, axis=0).ravel()
    true = np.concatenate(targs, axis=0).ravel()

    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))

    if save_results:
        _persist_hybrid_results(
            mae=mae,
            rmse=rmse,
            noise_level=float(noise_level),
            bearing_name=bearing_name,
            true_rul=true,
            predicted_rul=pred,
        )

    return mae, rmse, pred


def _persist_hybrid_results(
    *,
    mae: float,
    rmse: float,
    noise_level: float,
    bearing_name: str,
    true_rul: np.ndarray,
    predicted_rul: np.ndarray,
) -> None:
    """Write ``.npz`` arrays and merge a summary entry into ``results/metrics.json``."""
    (_project_root / "results").mkdir(parents=True, exist_ok=True)

    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", bearing_name).strip("_") or "bearing"
    nz_path = _project_root / "results" / f"hybrid_predictions_{safe}.npz"
    np.savez(
        nz_path,
        true_rul=true_rul,
        predicted_rul=predicted_rul,
        bearing_name=np.array(bearing_name),
        noise_level=np.array(noise_level),
    )

    entry = {
        "mae": mae,
        "rmse": rmse,
        "noise_level": noise_level,
        "bearing_name": bearing_name,
        "n_samples": int(len(predicted_rul)),
        "predictions_npz": str(nz_path.relative_to(_project_root)).replace("\\", "/"),
    }
    # Optional inline for small runs (keeps metrics.json self-contained for quick inspection)
    if len(predicted_rul) <= 5000:
        entry["true_rul"] = true_rul.astype(float).tolist()
        entry["predicted_rul"] = predicted_rul.astype(float).tolist()

    blob: Dict[str, Any] = {}
    if _METRICS_PATH.exists():
        try:
            blob = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blob = {}
    if not isinstance(blob, dict):
        blob = {}

    blob.setdefault("hybrid_model", {})
    run_key = f"{safe}_noise_{noise_level}"
    blob["hybrid_model"][run_key] = entry
    _METRICS_PATH.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def plot_degradation_curve(
    true: Union[np.ndarray, list],
    predicted: Union[np.ndarray, list],
    bearing_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """Plot normalized RUL vs window index; save under ``results/figures/`` by default."""
    y = np.asarray(true, dtype=float).ravel()
    y_hat = np.asarray(predicted, dtype=float).ravel()
    if len(y) != len(y_hat):
        raise ValueError("true and predicted must have the same length")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(len(y))
    ax.plot(idx, y, label="True RUL", color="#2ca02c", linewidth=1.2)
    ax.plot(idx, y_hat, label="Predicted RUL", color="#d62728", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("test window index")
    ax.set_ylabel("normalized RUL")
    ax.set_title(f"Hybrid QML — degradation curve ({bearing_name})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", bearing_name).strip("_") or "bearing"
        save_path = _project_root / "results" / "figures" / f"hybrid_degradation_{safe}.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
