"""
Bearing vibration loading, RUL labeling, and train/test splits (XJTU-SY style layout).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Column names on disk (XJTU-SY CSV) vs. canonical names used in this project
_COLUMN_ALIASES = {
    "Horizontal_vibration_signals": "horizontal_acc",
    "horizontal_vibration_signals": "horizontal_acc",
    "Vertical_vibration_signals": "vertical_acc",
    "vertical_vibration_signals": "vertical_acc",
}


def _sort_csv_paths(paths: List[Path]) -> List[Path]:
    """Sort CSV paths by numeric stem (1.csv, 2.csv, …, 10.csv)."""

    def key(p: Path) -> Tuple[int, str]:
        try:
            return (int(p.stem), p.name)
        except ValueError:
            return (10**9, p.name)

    return sorted(paths, key=key)


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known XJTU headers to horizontal_acc / vertical_acc."""
    out = df.copy()
    rename = {c: _COLUMN_ALIASES[c] for c in out.columns if c in _COLUMN_ALIASES}
    out = out.rename(columns=rename)

    expected = ("horizontal_acc", "vertical_acc")
    if tuple(out.columns) != expected:
        if len(out.columns) == 2:
            out.columns = list(expected)
        else:
            raise ValueError(
                f"Expected two columns (acc channels), got {list(out.columns)}"
            )
    return out


def load_bearing_data(bearing_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and concatenate all CSV segments for one bearing, in segment order.

    Each CSV is expected to contain two acceleration columns (horizontal, vertical).
    Files are read in numeric order of the filename stem (e.g. ``1.csv``, ``2.csv``, …).

    Parameters
    ----------
    bearing_path
        Directory containing the bearing's ``*.csv`` files.

    Returns
    -------
    pd.DataFrame
        Concatenated data with columns ``horizontal_acc`` and ``vertical_acc``,
        and a ``RangeIndex`` named ``time`` (sample index from run start; multiply
        by ``1 / f_s`` for seconds if needed).
    """
    root = Path(bearing_path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    csv_paths = [p for p in root.glob("*.csv") if p.is_file()]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {root}")

    frames: List[pd.DataFrame] = []
    for path in _sort_csv_paths(csv_paths):
        seg = pd.read_csv(path)
        seg = _canonicalize_columns(seg)
        frames.append(seg)

    data = pd.concat(frames, axis=0, ignore_index=True)
    data.index = pd.RangeIndex(stop=len(data), name="time")
    return data


def compute_rul(
    bearing_data: pd.DataFrame,
    total_life: Optional[int] = None,
) -> pd.Series:
    """
    Assign normalized RUL labels under a linear degradation assumption.

    For sample index ``i`` in ``0 .. N-1``, raw remaining life is::

        RUL_raw(i) = T - i

    where ``T`` is ``total_life`` if given, otherwise ``N = len(bearing_data)``.
    Values are then scaled linearly to the closed interval ``[0, 1]`` (end of life → 0).

    Parameters
    ----------
    bearing_data
        Table with one row per time sample (e.g. output of :func:`load_bearing_data`).
    total_life
        Optional horizon ``T`` in samples. If ``None``, ``T = len(bearing_data)``.

    Returns
    -------
    pd.Series
        Normalized RUL, same index as ``bearing_data``, name ``rul``.
    """
    n = len(bearing_data)
    if n == 0:
        return pd.Series(dtype=float, name="rul")

    if total_life is not None:
        if total_life < 1:
            raise ValueError("total_life must be >= 1 when provided")
        t_horizon = int(total_life)
    else:
        t_horizon = n

    idx = np.arange(n, dtype=np.float64)
    rul_raw = t_horizon - idx

    r_min = float(np.min(rul_raw))
    r_max = float(np.max(rul_raw))
    if np.isclose(r_max, r_min):
        # Degenerate: constant raw RUL (e.g. N=1); convention = fully remaining
        rul_norm = np.ones(n, dtype=np.float64)
    else:
        rul_norm = (rul_raw - r_min) / (r_max - r_min)

    return pd.Series(rul_norm, index=bearing_data.index, name="rul", dtype="float64")


def _find_bearing_root(data_dir: Path) -> Path:
    """
    Resolve the directory that directly contains ``Bearing1_1`` … ``Bearing1_5``.
    """
    root = data_dir.expanduser().resolve()
    if (root / "Bearing1_1").is_dir():
        return root

    matches = [p for p in root.rglob("Bearing1_1") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find folder 'Bearing1_1' under {root}. "
            "Pass the directory that contains Bearing1_1 … Bearing1_5, "
            "or a parent (e.g. data/raw) to search."
        )

    return matches[0].parent


def split_bearings(
    data_dir: Union[str, Path],
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Load train bearings (``Bearing1_1`` … ``Bearing1_4``) and test (``Bearing1_5``).

    ``data_dir`` may be the folder that directly contains the ``Bearing1_*``
    directories, or any ancestor; the first ``Bearing1_1`` found under ``data_dir``
    defines the sibling layout used for all IDs.

    Parameters
    ----------
    data_dir
        Path to search for ``Bearing1_1`` … ``Bearing1_5``.

    Returns
    -------
    train_list
        Four DataFrames, ordered as bearing 1 … 4.
    test_list
        Single DataFrame for ``Bearing1_5`` (list of one element for API symmetry).
    """
    parent = _find_bearing_root(Path(data_dir))

    train_ids = ["Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4"]
    test_ids = ["Bearing1_5"]

    train_list: List[pd.DataFrame] = []
    for bid in train_ids:
        bp = parent / bid
        if not bp.is_dir():
            raise FileNotFoundError(f"Missing training bearing folder: {bp}")
        train_list.append(load_bearing_data(bp))

    test_list: List[pd.DataFrame] = []
    for bid in test_ids:
        bp = parent / bid
        if not bp.is_dir():
            raise FileNotFoundError(f"Missing test bearing folder: {bp}")
        test_list.append(load_bearing_data(bp))

    return train_list, test_list
