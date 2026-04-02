"""Generate notebooks/05_results_comparison.ipynb"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "05_results_comparison.ipynb"


def lines(s: str) -> list[str]:
    return [ln + "\n" for ln in s.strip("\n").split("\n")]

cells: list[dict] = []


def md(t: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(t)})


def code(t: str) -> None:
    cells.append(
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(t)}
    )


md(
    """# 05 — Results comparison (Classical LSTM vs hybrid QML)

Publication-style summary of test performance on **Bearing1_5** (or your configured test bearing).

**Inputs (produced by training notebooks or scripts):**
- `results/metrics.json` — optional keys `classical_model`, `hybrid_model` with MAE/RMSE and optional `training_time_sec`, `predictions_npz`
- `results/classical_predictions_Bearing1_5.npz` — arrays `true_rul`, `predicted_rul`
- `results/hybrid_predictions_Bearing1_5.npz` — same layout

If NPZ files are missing, run your `03_classical_baseline` / `04_hybrid_qml_model` pipelines to export them, or set **`USE_SYNTHETIC_DEMO = True`** below to plot **illustrative** curves only."""
)

code(
    """from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Headless runs only (keep default in Jupyter for inline figures)
try:
    get_ipython()  # type: ignore[name-defined]
except NameError:
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ——— Notebook configuration ———
TEST_BEARING = "Bearing1_5"
USE_SYNTHETIC_DEMO = False  # set True only to preview figure layout without real NPZ files

NOTEBOOK_DIR = Path.cwd().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR
if not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = NOTEBOOK_DIR

RESULTS = PROJECT_ROOT / "results"
METRICS_JSON = RESULTS / "metrics.json"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication defaults (matplotlib)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

try:
    from IPython.display import display
except ImportError:

    def display(obj):
        print(obj)


def _show_fig() -> None:
    try:
        get_ipython()  # type: ignore[name-defined]
        plt.show()
    except NameError:
        plt.close("all")


print(f"PROJECT_ROOT = {PROJECT_ROOT}")"""
)

code(
    """from typing import Optional

def safe_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "bearing"


def load_metrics_blob() -> dict:
    if not METRICS_JSON.is_file():
        return {}
    try:
        return json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def find_run_entry(section: dict, bearing: str, prefer_noise: Optional[float] = 0.0) -> Optional[dict]:
    \"\"\"Pick metrics row for a bearing; for hybrid, optionally match noise_level.\"\"\"
    if not section:
        return None
    candidates = []
    for key, entry in section.items():
        if not isinstance(entry, dict):
            continue
        bn = str(entry.get("bearing_name", ""))
        if bn != bearing:
            continue
        nl = float(entry.get("noise_level", -1.0))
        candidates.append((nl, key, entry))
    if not candidates:
        return None
    if prefer_noise is not None:
        for nl, _, ent in sorted(candidates, key=lambda t: abs(t[0] - prefer_noise)):
            if abs(nl - prefer_noise) < 1e-6:
                return ent
    return sorted(candidates, key=lambda t: t[0])[0][2]


def load_npz_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return np.asarray(data["true_rul"]).ravel(), np.asarray(data["predicted_rul"]).ravel()


_metrics = load_metrics_blob()
_classical_sec = _metrics.get("classical_model", {}) if isinstance(_metrics, dict) else {}
_hybrid_sec = _metrics.get("hybrid_model", {}) if isinstance(_metrics, dict) else {}

row_classical = find_run_entry(_classical_sec, TEST_BEARING, None) or (
    _classical_sec.get(TEST_BEARING) if isinstance(_classical_sec.get(TEST_BEARING), dict) else None
)
row_hybrid_clean = find_run_entry(_hybrid_sec, TEST_BEARING, 0.0)

print("Classical metrics row:", "found" if row_classical else "not in metrics.json")
print("Hybrid (noise=0) row:", "found" if row_hybrid_clean else "not in metrics.json")"""
)

code(
    """slug = safe_slug(TEST_BEARING)
path_classical_npz = RESULTS / f"classical_predictions_{slug}.npz"
path_hybrid_npz = RESULTS / f"hybrid_predictions_{slug}.npz"

if USE_SYNTHETIC_DEMO:
    rng = np.random.default_rng(0)
    n = 200
    t = np.linspace(0, 1, n)
    true_rul = 1.0 - t
    pred_lstm = true_rul + 0.06 * np.cumsum(rng.normal(0, 0.02, n)) / np.sqrt(n)
    pred_lstm = np.clip(pred_lstm, 0, 1)
    pred_hybrid = true_rul + 0.04 * np.cumsum(rng.normal(0, 0.015, n)) / np.sqrt(n)
    pred_hybrid = np.clip(pred_hybrid, 0, 1)
    mae_lstm = float(np.mean(np.abs(true_rul - pred_lstm)))
    rmse_lstm = float(np.sqrt(np.mean((true_rul - pred_lstm) ** 2)))
    mae_hybrid = float(np.mean(np.abs(true_rul - pred_hybrid)))
    rmse_hybrid = float(np.sqrt(np.mean((true_rul - pred_hybrid) ** 2)))
    t_lstm = t_hybrid = np.nan
    print("Using SYNTHETIC demo arrays.")
else:
    if not path_classical_npz.is_file() or not path_hybrid_npz.is_file():
        raise FileNotFoundError(
            f"Expected:\\n  {path_classical_npz}\\n  {path_hybrid_npz}\\n"
            "Export these from your training notebooks or set USE_SYNTHETIC_DEMO = True."
        )
    true_c, pred_lstm = load_npz_arrays(path_classical_npz)
    true_h, pred_hybrid = load_npz_arrays(path_hybrid_npz)
    if len(true_c) != len(pred_lstm) or len(true_h) != len(pred_hybrid):
        raise ValueError("true_rul and predicted_rul length mismatch inside an NPZ file.")
    m = min(len(true_c), len(true_h))
    true_c, pred_lstm = true_c[:m], pred_lstm[:m]
    true_h, pred_hybrid = true_h[:m], pred_hybrid[:m]
    if not np.allclose(true_c, true_h, rtol=1e-4, atol=1e-4):
        print("Warning: `true_rul` differs between classical and hybrid NPZ; using classical labels.")
    true_rul = true_c
    mae_lstm = float(row_classical["mae"]) if row_classical and "mae" in row_classical else float(np.mean(np.abs(true_rul - pred_lstm)))
    rmse_lstm = float(row_classical["rmse"]) if row_classical and "rmse" in row_classical else float(np.sqrt(np.mean((true_rul - pred_lstm) ** 2)))
    mae_hybrid = float(row_hybrid_clean["mae"]) if row_hybrid_clean and "mae" in row_hybrid_clean else float(np.mean(np.abs(true_rul - pred_hybrid)))
    rmse_hybrid = float(row_hybrid_clean["rmse"]) if row_hybrid_clean and "rmse" in row_hybrid_clean else float(np.sqrt(np.mean((true_rul - pred_hybrid) ** 2)))
    t_lstm = row_classical.get("training_time_sec") if row_classical else np.nan
    t_hybrid = row_hybrid_clean.get("training_time_sec") if row_hybrid_clean else np.nan

def fmt_time(sec) -> str:
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return "—"
    return f"{float(sec):.1f} s"

table = pd.DataFrame({
    "Model": ["Classical LSTM", "Hybrid QML"],
    "MAE": [mae_lstm, mae_hybrid],
    "RMSE": [rmse_lstm, rmse_hybrid],
    "Training Time": [fmt_time(t_lstm), fmt_time(t_hybrid)],
})
table"""
)

md(
    """## 1. Side-by-side metrics

**Interpretation:** MAE / RMSE are on **normalized RUL** `[0, 1]` (same units as training). Training time is read from `training_time_sec` in `metrics.json` if your training scripts record it; otherwise shown as an em dash."""
)

code(
    """display(table.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}"}).set_caption("Test performance summary"))

# Optional LaTeX-style export
fig, ax = plt.subplots(figsize=(5.5, 1.6))
ax.axis("off")
tbl = ax.table(
    cellText=table.values,
    colLabels=table.columns,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.15, 1.35)
fig.savefig(FIG_DIR / "table_metrics_comparison.png", bbox_inches="tight", facecolor="white")
_show_fig()"""
)

md("""## 2. Predicted vs actual RUL (test bearing, overlaid)""")

code(
    """fig, ax = plt.subplots(figsize=(7.0, 4.2))
idx = np.arange(len(true_rul))
ax.plot(idx, true_rul, color="0.15", linewidth=1.6, label="Actual RUL", zorder=3)
ax.plot(idx, pred_lstm, color="#1f77b4", linewidth=1.2, alpha=0.9, label="Classical LSTM")
ax.plot(idx, pred_hybrid, color="#d62728", linewidth=1.2, alpha=0.9, label="Hybrid QML")
ax.set_xlabel("Test window index")
ax.set_ylabel("Normalized RUL")
ax.set_title(f"RUL trajectories — {TEST_BEARING}")
ax.legend(frameon=True, loc="upper right")
fig.tight_layout()
fig.savefig(FIG_DIR / "rul_overlay_Bearing1_5.png", bbox_inches="tight", facecolor="white")
_show_fig()"""
)

md("""## 3. Bar chart — MAE & RMSE""")

code(
    """fig, ax = plt.subplots(figsize=(6.0, 4.0))
models = ["Classical\\nLSTM", "Hybrid\\nQML"]
x = np.arange(len(models))
w = 0.36
ax.bar(x - w / 2, [mae_lstm, mae_hybrid], width=w, label="MAE", color="#4c72b0", edgecolor="0.2", linewidth=0.6)
ax.bar(x + w / 2, [rmse_lstm, rmse_hybrid], width=w, label="RMSE", color="#dd8452", edgecolor="0.2", linewidth=0.6)
ax.set_xticks(x, models)
ax.set_ylabel("Error (normalized RUL)")
ax.set_title("Error comparison on test bearing")
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(FIG_DIR / "bar_mae_rmse_comparison.png", bbox_inches="tight", facecolor="white")
_show_fig()"""
)

md(
    """## 4. Noise robustness (hybrid QML)

We plot **MAE vs depolarizing noise probability** for the hybrid model on the same test bearing.  
Points are **loaded from** `metrics.json` (`hybrid_model` entries with matching `bearing_name`).  

To populate the curve, evaluate the hybrid model at noise levels in `[0, 0.1]` (e.g. 0, 0.02, …, 0.1) and ensure each run writes metrics (see `evaluate_hybrid(..., save_results=True)`)."""
)

code(
    """def collect_hybrid_noise_curve(section: dict, bearing: str) -> tuple[list[float], list[float]]:
    noises, maes = [], []
    if not section:
        return noises, maes
    for entry in section.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("bearing_name", "")) != bearing:
            continue
        try:
            noises.append(float(entry["noise_level"]))
            maes.append(float(entry["mae"]))
        except (KeyError, TypeError, ValueError):
            continue
    order = np.argsort(noises)
    return list(np.asarray(noises)[order]), list(np.asarray(maes)[order])


n_vals, mae_vals = collect_hybrid_noise_curve(_hybrid_sec, TEST_BEARING)

fig, ax = plt.subplots(figsize=(6.5, 4.0))
if len(n_vals) >= 2:
    ax.plot(n_vals, mae_vals, "o-", color="#2ca02c", linewidth=1.8, markersize=7, markeredgecolor="0.2", label="Hybrid MAE")
elif len(n_vals) == 1:
    ax.scatter(n_vals, mae_vals, s=80, color="#2ca02c", edgecolors="0.2", label="Hybrid MAE", zorder=3)
    ax.text(0.02, 0.95, "Single noise point — add more `evaluate_hybrid` runs.", transform=ax.transAxes, va="top")
else:
    if USE_SYNTHETIC_DEMO:
        demo_x = np.linspace(0, 0.1, 11)
        demo_y = mae_hybrid * (1.0 + 4.0 * demo_x)
        ax.plot(demo_x, demo_y, "--", color="#9467bd", linewidth=1.5, label="Illustrative (demo mode)")
    ax.text(
        0.02, 0.95,
        "No hybrid noise sweep in `metrics.json`. Run `evaluate_hybrid` at p in [0, 0.1] or set USE_SYNTHETIC_DEMO = True.",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

ax.set_xlabel("Depolarizing noise level (p)")
ax.set_ylabel("MAE (normalized RUL)")
ax.set_title("Hybrid model — sensitivity to gate noise (evaluation)")
ax.set_xlim(0, 0.1)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(FIG_DIR / "hybrid_noise_robustness.png", bbox_inches="tight", facecolor="white")
_show_fig()"""
)

md(
    """## 5. Conclusion

*This cell is filled by the next code block from current metrics so the take-away stays consistent with the table above. Edit manually for your thesis narrative if needed.*"""
)

code(
    '''better = "Hybrid QML" if mae_hybrid < mae_lstm else "Classical LSTM"
delta = abs(mae_lstm - mae_hybrid)
noise_note = (
    f"The hybrid MAE noise sweep has {len(n_vals)} saved point(s)."
    if len(n_vals)
    else "Noise robustness data were not yet logged; add multi-noise evaluations to `metrics.json`."
)

conclusion = f"""
### Summary ({TEST_BEARING}, normalized RUL)

- **Lower MAE:** **{better}** (absolute gap ≈ **{delta:.4f}**).
- **RMSE:** Classical **{rmse_lstm:.4f}** vs Hybrid **{rmse_hybrid:.4f}**.
- **Training cost:** Compare wall times once `training_time_sec` is written by your training scripts; optimizers and quantum simulation overhead usually make hybrid **wall-clock** training slower at equal epochs.
- **Noise:** {noise_note} Under increasing depolarizing noise, hybrid MAE typically **rises**; the classical baseline is unaffected by that knob (shown for context only).

**Practical takeaway:** Prefer the model with lower MAE/RMSE on the **held-out** bearing; report hybrid **noise curves** when claiming NISQ-era robustness.
""".strip()

try:
    from IPython.display import Markdown

    display(Markdown(conclusion))
except ImportError:
    print(conclusion)'''
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", NB)
