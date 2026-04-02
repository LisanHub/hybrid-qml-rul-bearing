"""
Slide-window feature extraction for bearing vibration (time + frequency domain).

Expects :mod:`preprocessing.load_bearing_data` column names ``horizontal_acc`` and
``vertical_acc``. Each row of the output corresponds to one window; channels are
prefixed with ``h_`` and ``v_``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler

# Default path for persisting the feature normalizer (project root / data/processed)
_DEFAULT_SCALER_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "feature_scaler.joblib"
)

_TIME_KEYS = (
    "rms",
    "mean",
    "std",
    "kurtosis",
    "skewness",
    "crest_factor",
    "peak_to_peak",
)
_FREQ_KEYS = (
    "dominant_frequency",
    "spectral_entropy",
    "spectral_kurtosis",
    "band_energy_ratio_1",
    "band_energy_ratio_2",
    "band_energy_ratio_3",
    "band_energy_ratio_4",
)


def _finite_or_zero(x: float) -> float:
    v = float(x)
    if not np.isfinite(v):
        return 0.0
    return v


def extract_window_features(signal_window: np.ndarray) -> Dict[str, float]:
    """
    Compute time-domain and FFT-based features for a single 1-D window.

    The FFT uses the mean-centered window. ``dominant_frequency`` is in cycles per
    sample (Nyquist interval ``[0, 0.5]``); multiply by the sampling rate (Hz) to
    convert to Hz.

    Parameters
    ----------
    signal_window
        1-D array of samples (any shape is flattened).

    Returns
    -------
    dict
        Keys: RMS, mean, std, kurtosis (excess Fisher), skewness, crest factor,
        peak-to-peak, dominant_frequency, spectral_entropy (bits, base-2 log),
        spectral_kurtosis (excess kurtosis of FFT magnitudes, DC excluded),
        ``band_energy_ratio_1`` … ``_4`` (non-negative, sum to 1 across bands).
    """
    x = np.asarray(signal_window, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        raise ValueError("signal_window must contain at least 2 samples")

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    ptp = float(np.ptp(x))

    crest = peak / rms if rms > 1e-12 else 0.0

    k_t = kurtosis(x, fisher=True, bias=False)
    s_t = skew(x, bias=False)

    x_zm = x - mean
    spectrum = np.fft.rfft(x_zm)
    mag = np.abs(spectrum)
    power = mag**2
    freqs = np.fft.rfftfreq(n, d=1.0)

    if power.size > 1:
        idx_dom = int(1 + np.argmax(power[1:]))
    else:
        idx_dom = 0
    dom_f = float(freqs[idx_dom])

    p_sum = float(np.sum(power)) + 1e-12
    p_norm = power / p_sum
    p_pos = p_norm[p_norm > 0]
    spec_entropy = float(-np.sum(p_pos * np.log2(p_pos))) if p_pos.size else 0.0

    mag_no_dc = mag[1:] if mag.size > 1 else mag
    if mag_no_dc.size > 4:
        spec_k = float(kurtosis(mag_no_dc, fisher=True, bias=False))
    else:
        spec_k = 0.0

    nb = power.size
    edges = np.linspace(0, nb, 5, dtype=int)
    ratios: list[float] = []
    for b in range(4):
        lo, hi = edges[b], edges[b + 1]
        ratios.append(float(np.sum(power[lo:hi])) if hi > lo else 0.0)
    rsum = sum(ratios) + 1e-12
    ratios = [r / rsum for r in ratios]

    out: Dict[str, float] = {
        "rms": _finite_or_zero(rms),
        "mean": _finite_or_zero(mean),
        "std": _finite_or_zero(std),
        "kurtosis": _finite_or_zero(k_t),
        "skewness": _finite_or_zero(s_t),
        "crest_factor": _finite_or_zero(crest),
        "peak_to_peak": _finite_or_zero(ptp),
        "dominant_frequency": _finite_or_zero(dom_f),
        "spectral_entropy": _finite_or_zero(spec_entropy),
        "spectral_kurtosis": _finite_or_zero(spec_k),
        "band_energy_ratio_1": ratios[0],
        "band_energy_ratio_2": ratios[1],
        "band_energy_ratio_3": ratios[2],
        "band_energy_ratio_4": ratios[3],
    }
    return out


def extract_all_features(
    bearing_df: pd.DataFrame,
    window_size: int = 1024,
    stride: int = 512,
) -> pd.DataFrame:
    """
    Slide a window over horizontal and vertical acceleration and stack features.

    For each window start ``s``, features are computed on
    ``horizontal_acc[s:s+window_size]`` and ``vertical_acc[s:s+window_size]``, with
    column prefixes ``h_`` and ``v_``.

    Parameters
    ----------
    bearing_df
        Must contain ``horizontal_acc`` and ``vertical_acc``.
    window_size
        Number of samples per window (default 1024).
    stride
        Advance in samples between consecutive windows (default 512).

    Returns
    -------
    pd.DataFrame
        One row per window. Includes ``window_start`` and ``window_end`` (exclusive)
        for alignment with raw indices / RUL labeling.
    """
    required = ("horizontal_acc", "vertical_acc")
    for col in required:
        if col not in bearing_df.columns:
            raise KeyError(f"bearing_df must include column '{col}'")

    n = len(bearing_df)
    if n < window_size:
        cols = ["window_start", "window_end"] + [
            f"{pfx}{k}" for pfx in ("h_", "v_") for k in _TIME_KEYS + _FREQ_KEYS
        ]
        return pd.DataFrame(columns=cols)

    h_all = bearing_df["horizontal_acc"].to_numpy(dtype=np.float64, copy=False)
    v_all = bearing_df["vertical_acc"].to_numpy(dtype=np.float64, copy=False)

    rows: list[Dict[str, Any]] = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        h_win = h_all[start:end]
        v_win = v_all[start:end]

        fh = extract_window_features(h_win)
        fv = extract_window_features(v_win)

        row: Dict[str, Any] = {
            "window_start": start,
            "window_end": end,
        }
        for k, val in fh.items():
            row[f"h_{k}"] = val
        for k, val in fv.items():
            row[f"v_{k}"] = val
        rows.append(row)

    return pd.DataFrame(rows)


def normalize_features(
    features_df: pd.DataFrame,
    scaler_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Z-score each numeric feature column (excluding window indices).

    Fits a :class:`~sklearn.preprocessing.StandardScaler` on ``features_df``,
    transforms the data, and persists the scaler with :mod:`joblib` for inference.

    Parameters
    ----------
    features_df
        Output of :func:`extract_all_features` (or compatible numeric columns).
    scaler_path
        Where to save the fitted scaler. If ``None``, uses
        ``data/processed/feature_scaler.joblib`` under the project root (next to
        ``src/``).

    Returns
    -------
    normalized_df
        Same row count as ``features_df``; ``window_start`` / ``window_end`` unchanged.
    scaler
        Fitted ``StandardScaler`` (also written to disk).
    """
    if features_df.empty:
        scaler = StandardScaler()
        return features_df.copy(), scaler

    meta_cols = {"window_start", "window_end", "bearing"}
    meta = features_df[[c for c in features_df.columns if c in meta_cols]].copy()
    feat_cols = [c for c in features_df.columns if c not in meta_cols]

    if not feat_cols:
        raise ValueError("features_df has no feature columns to normalize")

    X = features_df[feat_cols].to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    feat_part = pd.DataFrame(Xn, columns=feat_cols, index=features_df.index)
    normalized_df = pd.concat([meta, feat_part], axis=1)

    out_path = Path(scaler_path) if scaler_path is not None else _DEFAULT_SCALER_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_path)

    return normalized_df, scaler
