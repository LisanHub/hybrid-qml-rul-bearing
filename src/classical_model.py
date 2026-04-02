"""
Classical LSTM baseline for bearing RUL regression on windowed feature sequences.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_project_root = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _project_root / "results" / "classical_lstm.pt"


class BearingDataset(Dataset):
    """
    PyTorch dataset of sliding-window feature rows (and optional short history) with RUL.

    Each item has shape ``(seq_len, n_features)``: ``seq_len`` consecutive windows,
    predicting **normalized RUL** at the **last** window in that span.
    """

    def __init__(
        self,
        features: Union[np.ndarray, torch.Tensor],
        rul: Union[np.ndarray, torch.Tensor],
        seq_len: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        features
            ``(n_windows, n_features)`` — one row per sliding window.
        rul
            Normalized RUL aligned with each window (same length as ``n_windows``).
        seq_len
            Number of **consecutive** windows stacked as LSTM timesteps. Use ``1`` for
            a single window per sample; use e.g. ``10`` to give the LSTM short history.
        """
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(rul, dtype=np.float32).ravel()

        if x.ndim != 2:
            raise ValueError(f"features must be 2-D (n_windows, n_features); got {x.shape}")
        if len(y) != x.shape[0]:
            raise ValueError("rul length must match number of feature rows")

        n, n_feat = x.shape
        if n < seq_len:
            raise ValueError(
                f"Need at least seq_len={seq_len} windows; got n_windows={n}"
            )

        n_samples = n - seq_len + 1
        self._x = np.empty((n_samples, seq_len, n_feat), dtype=np.float32)
        self._y = np.empty((n_samples,), dtype=np.float32)

        for i in range(n_samples):
            self._x[i] = x[i : i + seq_len]
            self._y[i] = y[i + seq_len - 1]

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self._x[idx]), torch.tensor(self._y[idx], dtype=torch.float32)


class LSTMRegressor(nn.Module):
    """
    Two-layer LSTM followed by a linear head (single normalized RUL value per sequence).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: Optional[torch.device] = None,
    weights_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    show_progress: bool = False,
) -> nn.Module:
    """
    Train with MSE loss and Adam ( ``lr`` default 0.001 ).

    Saves state dict to ``results/classical_lstm.pt`` under the project root
    unless ``weights_path`` is provided (parent directories are created).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    epoch_iter = range(1, epochs + 1)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="LSTM train", unit="epoch")

    for epoch in epoch_iter:
        total_loss = 0.0
        n_batches = 0
        batch_it = train_loader
        if show_progress:
            batch_it = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, unit="batch")
        for xb, yb in batch_it:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        if verbose and not show_progress and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            avg = total_loss / max(1, n_batches)
            print(f"epoch {epoch:4d}/{epochs}  train_mse: {avg:.6f}")
        elif show_progress and verbose and hasattr(epoch_iter, "set_postfix"):
            avg = total_loss / max(1, n_batches)
            epoch_iter.set_postfix(mse=f"{avg:.6f}")

    out_path = Path(weights_path) if weights_path is not None else _DEFAULT_WEIGHTS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    if verbose:
        print(f"Saved weights to {out_path}")

    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Return MAE, RMSE (both in normalized RUL units), and stacked predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    preds: list[np.ndarray] = []
    targs: list[np.ndarray] = []

    for xb, yb in test_loader:
        xb = xb.to(device)
        pb = model(xb).detach().cpu().numpy()
        preds.append(pb)
        targs.append(yb.numpy())

    pred = np.concatenate(preds, axis=0).ravel()
    true = np.concatenate(targs, axis=0).ravel()

    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    return mae, rmse, pred


def plot_predictions(
    true_rul: Union[np.ndarray, list],
    predicted_rul: Union[np.ndarray, list],
    bearing_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot true vs predicted RUL along the sequence index (test windows).

    Saves under ``results/figures/`` by default if ``save_path`` is omitted
    (filename derived from ``bearing_name``).
    """
    y = np.asarray(true_rul, dtype=float).ravel()
    y_hat = np.asarray(predicted_rul, dtype=float).ravel()

    if len(y) != len(y_hat):
        raise ValueError("true_rul and predicted_rul must have the same length")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(len(y))
    ax.plot(idx, y, label="True RUL", color="#1f77b4", linewidth=1.2, alpha=0.9)
    ax.plot(idx, y_hat, label="Predicted RUL", color="#ff7f0e", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("window index (test set order)")
    ax.set_ylabel("normalized RUL")
    ax.set_title(f"RUL prediction — {bearing_name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", bearing_name).strip("_") or "bearing"
        save_path = _project_root / "results" / "figures" / f"predictions_{safe}.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
