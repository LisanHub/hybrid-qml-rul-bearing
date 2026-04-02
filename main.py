#!/usr/bin/env python3
"""
End-to-end pipeline: preprocessing, feature extraction, LSTM + hybrid QML training, evaluation.

Run from repository root, e.g.::

    python main.py --data_dir data/raw --mode all --epochs 50

``--data_dir`` should contain (or be a parent of) ``Bearing1_1`` … ``Bearing1_5`` (XJTU-SY layout).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Paths & imports from src/
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from classical_model import (  # noqa: E402
    BearingDataset,
    LSTMRegressor,
    evaluate_model,
    plot_predictions,
    train_model,
)
from features import extract_all_features, normalize_features  # noqa: E402
from hybrid_model import (  # noqa: E402
    HybridQMLModel,
    evaluate_hybrid,
    plot_degradation_curve,
    train_hybrid,
)
from preprocessing import compute_rul, split_bearings  # noqa: E402

WINDOW = 1024
STRIDE = 512
SEQ_LEN = 10
BATCH_SIZE = 64
TEST_BEARING = "Bearing1_5"
TRAIN_NAMES = [f"Bearing1_{i}" for i in range(1, 5)]

RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"
CLASSICAL_CKPT = RESULTS / "classical_lstm.pt"
HYBRID_CKPT = RESULTS / "hybrid_model.pt"
METRICS_JSON = RESULTS / "metrics.json"
SCALER_PATH = PROCESSED / "feature_scaler.joblib"

PipelinePack = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], Any]


def _feature_columns(feat_df: pd.DataFrame) -> list[str]:
    return [c for c in feat_df.columns if c not in ("window_start", "window_end", "bearing")]


def rul_labels_for_windows(bearing_df: pd.DataFrame, feat_df: pd.DataFrame) -> np.ndarray:
    if feat_df.empty:
        return np.array([], dtype=np.float32)
    rul = compute_rul(bearing_df).to_numpy(dtype=np.float64)
    ends = feat_df["window_end"].to_numpy(dtype=np.int64) - 1
    return rul[ends].astype(np.float32)


def apply_scaler(feat_df: pd.DataFrame, scaler, cols: list[str]) -> pd.DataFrame:
    out = feat_df.copy()
    x = feat_df[cols].to_numpy(dtype=np.float64, copy=True)
    out[cols] = scaler.transform(x)
    return out


def save_classical_metrics(
    mae: float,
    rmse: float,
    bearing_name: str,
    npz_relpath: str,
    training_time_sec: Optional[float] = None,
) -> None:
    entry: dict[str, Any] = {
        "mae": mae,
        "rmse": rmse,
        "bearing_name": bearing_name,
        "noise_level": 0.0,
        "predictions_npz": npz_relpath.replace("\\", "/"),
    }
    if training_time_sec is not None:
        entry["training_time_sec"] = float(training_time_sec)

    blob: dict[str, Any] = {}
    if METRICS_JSON.is_file():
        try:
            blob = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blob = {}
    if not isinstance(blob, dict):
        blob = {}
    blob.setdefault("classical_model", {})
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", bearing_name).strip("_") or "test"
    blob["classical_model"][safe] = entry
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def load_bearings(data_dir: Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    train_list, test_list = split_bearings(data_dir)
    return train_list, test_list


def maybe_subsample(
    x: np.ndarray,
    y: np.ndarray,
    max_windows: Optional[int],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if max_windows is None or len(x) <= max_windows:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_windows, replace=False)
    return x[idx], y[idx]


def prepare_scaled_windows(
    data_dir: Path,
    max_train_windows: Optional[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    Any,
]:
    """Train/test arrays (scaled features), labels, feature column names, scaler."""
    train_list, test_list = load_bearings(data_dir)
    train_raw: list[pd.DataFrame] = []
    for df, name in tqdm(list(zip(train_list, TRAIN_NAMES)), desc="Extract features (train)", unit="bearing"):
        fdf = extract_all_features(df, window_size=WINDOW, stride=STRIDE)
        fdf.insert(0, "bearing", name)
        train_raw.append(fdf)

    test_df = test_list[0]
    test_feat = extract_all_features(test_df, window_size=WINDOW, stride=STRIDE)
    tqdm.write("Concatenating training windows & fitting scaler…")
    train_concat = pd.concat(train_raw, axis=0, ignore_index=True)
    train_for_scale = train_concat.drop(columns=["bearing"], errors="ignore")
    train_norm, scaler = normalize_features(train_for_scale, scaler_path=SCALER_PATH)

    feat_cols = _feature_columns(train_norm)
    nx = train_norm[feat_cols].to_numpy(dtype=np.float32, copy=True)
    y_parts = []
    for raw, fdf in zip(train_list, train_raw):
        ff = fdf.drop(columns=["bearing"], errors="ignore").reset_index(drop=True)
        y_parts.append(rul_labels_for_windows(raw, ff))
    ny = np.concatenate(y_parts, axis=0).astype(np.float32)
    assert len(nx) == len(ny)

    nx, ny = maybe_subsample(nx, ny, max_train_windows)
    tx = apply_scaler(test_feat, scaler, feat_cols)[feat_cols].to_numpy(dtype=np.float32, copy=True)
    ty = rul_labels_for_windows(test_df, test_feat)
    return nx, ny, tx, ty, feat_cols, scaler


def run_preprocess(data_dir: Path) -> None:
    train_list, test_list = load_bearings(data_dir)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"bearings": {}}
    for df, name in tqdm(
        list(zip(train_list + test_list, TRAIN_NAMES + [TEST_BEARING])),
        desc="Preprocess",
        unit="bearing",
    ):
        manifest["bearings"][name] = {"n_samples": int(len(df)), "columns": list(df.columns)}
        fdf = extract_all_features(df, window_size=WINDOW, stride=STRIDE)
        fdf.to_pickle(PROCESSED / f"{name}_features.pkl")
    (PROCESSED / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tqdm.write(f"Wrote feature pickles and manifest under {PROCESSED}")


def run_train_classical(
    data_dir: Path,
    epochs: int,
    max_train_windows: Optional[int],
    show_progress: bool,
    pack: Optional[PipelinePack] = None,
) -> dict[str, float]:
    if pack is None:
        nx, ny, _, _, feat_cols, _ = prepare_scaled_windows(data_dir, max_train_windows)
    else:
        nx, ny, _, _, feat_cols, _ = pack
    n_feat = len(feat_cols)
    tr_ds = BearingDataset(nx, ny, seq_len=SEQ_LEN)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    model = LSTMRegressor(input_size=n_feat, hidden_size=64, num_layers=2, dropout=0.2)
    t0 = time.perf_counter()
    train_model(
        model,
        tr_ld,
        epochs=epochs,
        weights_path=CLASSICAL_CKPT,
        verbose=not show_progress,
        show_progress=show_progress,
    )
    t_classical = time.perf_counter() - t0
    return {"training_time_sec_classical": t_classical, "n_train_windows": float(len(tr_ds))}


def run_train_hybrid(
    data_dir: Path,
    epochs: int,
    max_train_windows: Optional[int],
    show_progress: bool,
    pack: Optional[PipelinePack] = None,
) -> dict[str, float]:
    if pack is None:
        nx, ny, _, _, feat_cols, _ = prepare_scaled_windows(data_dir, max_train_windows)
    else:
        nx, ny, _, _, feat_cols, _ = pack
    n_feat = len(feat_cols)
    tr_ds = BearingDataset(nx, ny, seq_len=SEQ_LEN)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    model = HybridQMLModel(input_features=n_feat)
    t0 = time.perf_counter()
    train_hybrid(
        model,
        tr_ld,
        epochs=epochs,
        weights_path=HYBRID_CKPT,
        verbose=not show_progress,
        show_progress=show_progress,
    )
    t_hybrid = time.perf_counter() - t0
    return {"training_time_sec_hybrid": t_hybrid, "n_train_windows": float(len(tr_ds))}


def run_evaluate(
    data_dir: Path,
    noise_level: float,
    max_train_windows: Optional[int],
    pack: Optional[PipelinePack] = None,
) -> dict[str, Any]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    if pack is None:
        _, _, tx, ty, feat_cols, _ = prepare_scaled_windows(data_dir, max_train_windows)
    else:
        _, _, tx, ty, feat_cols, _ = pack
    n_feat = len(feat_cols)
    te_ds = BearingDataset(tx, ty, seq_len=SEQ_LEN)
    y_true = np.array([te_ds[i][1].float().numpy() for i in range(len(te_ds))], dtype=np.float32)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics: dict[str, Any] = {}

    lstm = LSTMRegressor(input_size=n_feat, hidden_size=64, num_layers=2, dropout=0.2)
    lstm.load_state_dict(torch.load(CLASSICAL_CKPT, map_location=device))
    mae_c, rmse_c, pred_c = evaluate_model(lstm, te_ld, device=device)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", TEST_BEARING).strip("_")
    npz_c = RESULTS / f"classical_predictions_{slug}.npz"
    np.savez(npz_c, true_rul=y_true, predicted_rul=pred_c)
    rel_c = f"results/classical_predictions_{slug}.npz"

    plot_predictions(
        y_true,
        pred_c,
        f"{TEST_BEARING} (classical)",
        save_path=FIGURES / f"pipeline_classical_{slug}.png",
        show=False,
    )
    plt.close("all")

    save_classical_metrics(mae_c, rmse_c, TEST_BEARING, rel_c, training_time_sec=None)
    metrics["classical_mae"] = mae_c
    metrics["classical_rmse"] = rmse_c

    hybrid = HybridQMLModel(input_features=n_feat)
    hybrid.load_state_dict(torch.load(HYBRID_CKPT, map_location=device))
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)
    mae_h, rmse_h, pred_h = evaluate_hybrid(
        hybrid,
        te_ld,
        noise_level=noise_level,
        bearing_name=TEST_BEARING,
        save_results=True,
    )
    metrics["hybrid_mae"] = mae_h
    metrics["hybrid_rmse"] = rmse_h
    metrics["noise_level"] = noise_level

    plot_degradation_curve(
        y_true,
        pred_h,
        f"{TEST_BEARING} (hybrid)",
        save_path=FIGURES / f"pipeline_hybrid_{slug}.png",
        show=False,
    )
    plt.close("all")

    return metrics


def run_all(
    data_dir: Path,
    epochs: int,
    noise_level: float,
    max_train_windows: Optional[int],
    show_progress: bool,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")

    summary_metrics: dict[str, Any] = {}

    tqdm.write("\n--- preprocess (export pickles) ---")
    for _ in tqdm([0], desc="Preprocess", disable=not show_progress):
        run_preprocess(data_dir)

    tqdm.write("\n--- prepare windows (single pass) ---")
    pack = prepare_scaled_windows(data_dir, max_train_windows)

    tqdm.write("\n--- train classical LSTM ---")
    for _ in tqdm([0], desc="Train classical", disable=not show_progress):
        summary_metrics.update(
            run_train_classical(data_dir, epochs, max_train_windows, show_progress, pack=pack)
        )

    tqdm.write("\n--- train hybrid QML ---")
    for _ in tqdm([0], desc="Train hybrid", disable=not show_progress):
        summary_metrics.update(
            run_train_hybrid(data_dir, epochs, max_train_windows, show_progress, pack=pack)
        )

    tqdm.write("\n--- evaluate ---")
    for _ in tqdm([0], desc="Evaluate", disable=not show_progress):
        summary_metrics.update(run_evaluate(data_dir, noise_level, max_train_windows, pack=pack))

    print_summary(
        classical_mae=summary_metrics.get("classical_mae"),
        classical_rmse=summary_metrics.get("classical_rmse"),
        hybrid_mae=summary_metrics.get("hybrid_mae"),
        hybrid_rmse=summary_metrics.get("hybrid_rmse"),
        noise_level=noise_level,
    )


def print_summary(
    classical_mae: Optional[float] = None,
    classical_rmse: Optional[float] = None,
    hybrid_mae: Optional[float] = None,
    hybrid_rmse: Optional[float] = None,
    noise_level: float = 0.0,
) -> None:
    if METRICS_JSON.is_file():
        try:
            blob = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
            ch = blob.get("classical_model", {})
            hh = blob.get("hybrid_model", {})
            for _, ent in ch.items():
                if isinstance(ent, dict) and str(ent.get("bearing_name", "")) == TEST_BEARING:
                    classical_mae = classical_mae if classical_mae is not None else ent.get("mae")
                    classical_rmse = classical_rmse if classical_rmse is not None else ent.get("rmse")
                    break
            for ent in hh.values():
                if (
                    isinstance(ent, dict)
                    and str(ent.get("bearing_name", "")) == TEST_BEARING
                    and abs(float(ent.get("noise_level", -1)) - noise_level) < 1e-9
                ):
                    hybrid_mae = hybrid_mae if hybrid_mae is not None else ent.get("mae")
                    hybrid_rmse = hybrid_rmse if hybrid_rmse is not None else ent.get("rmse")
                    break
        except (json.JSONDecodeError, TypeError):
            pass

    print("\n" + "=" * 56)
    print("FINAL METRICS (normalized RUL on Bearing1_5)")
    print("=" * 56)
    print(f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'Notes':>12}")
    print("-" * 56)
    def fmt(x: Any) -> str:
        if x is None:
            return f"{'—':>10}"
        return f"{float(x):10.4f}"

    print(f"{'Classical LSTM':<22} {fmt(classical_mae)} {fmt(classical_rmse)} {'':>12}")
    note = f"noise={noise_level}"
    print(f"{'Hybrid QML':<22} {fmt(hybrid_mae)} {fmt(hybrid_rmse)} {note:>12}")
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bearing RUL pipeline (classical + hybrid QML).")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing or under Bearing1_1 … Bearing1_5",
    )
    parser.add_argument(
        "--mode",
        choices=("preprocess", "train_classical", "train_hybrid", "evaluate", "all"),
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for LSTM and hybrid")
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.0,
        help="Depolarizing noise probability for hybrid evaluation (0 = noiseless)",
    )
    parser.add_argument(
        "--max_train_windows",
        type=int,
        default=None,
        help="Optional cap on training windows (subsampled, reproducible) for faster dry runs",
    )
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm for training inner loops")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    show = not args.no_progress

    if args.mode == "preprocess":
        run_preprocess(data_dir)
        return
    if args.mode == "train_classical":
        run_train_classical(data_dir, args.epochs, args.max_train_windows, show)
        print_summary(noise_level=args.noise_level)
        return
    if args.mode == "train_hybrid":
        run_train_hybrid(data_dir, args.epochs, args.max_train_windows, show)
        print_summary(noise_level=args.noise_level)
        return
    if args.mode == "evaluate":
        run_evaluate(data_dir, args.noise_level, args.max_train_windows, pack=None)
        print_summary(noise_level=args.noise_level)
        return
    if args.mode == "all":
        run_all(data_dir, args.epochs, args.noise_level, args.max_train_windows, show_progress=show)
        return


if __name__ == "__main__":
    main()
