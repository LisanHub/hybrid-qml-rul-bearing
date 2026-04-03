<div align="center">

<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PennyLane-QML-6929C4?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/B.Sc.%20Thesis-DIU%202026-0369A1?style=for-the-badge"/>

<br/><br/>

# Hybrid Quantum–Classical RUL Prediction<br/>for Rolling Bearings

**Zadid Al Lisan** · Daffodil International University · Supervised by **Dr. Md Alamgir Kabir**

*Reproducibility code for the B.Sc. thesis:*  
**"Hybrid Quantum-Classical Regression Model for Remaining Useful-Life Prediction of Rolling Bearings"**

[![CI](https://github.com/LisanHub/hybrid-qml-rul-bearing/actions/workflows/ci.yml/badge.svg)](https://github.com/LisanHub/hybrid-qml-rul-bearing/actions/workflows/ci.yml)
[![XJTU-SY Dataset](https://img.shields.io/badge/Dataset-XJTU--SY-orange?style=flat-square)](https://biaowang.tech/xjtu-sy-bearing-datasets/)
[![GitHub](https://img.shields.io/badge/GitHub-LisanHub-181717?style=flat-square&logo=github)](https://github.com/LisanHub)

</div>

---

## Overview

Accurate **remaining useful life (RUL)** prediction for rolling-element bearings is critical to condition-based maintenance — reducing unplanned downtime before it begins. This repository implements and benchmarks a **hybrid quantum-classical regressor** against a classical LSTM baseline on the [XJTU-SY](https://biaowang.tech/xjtu-sy-bearing-datasets/) run-to-failure dataset.

The hybrid model routes windowed vibration features through a **fully-connected encoder** → a **PennyLane variational quantum circuit (VQC)** → a **classical decoder**, all trained end-to-end. A depolarizing noise sweep probes circuit robustness under realistic quantum hardware conditions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION & FEATURES                        │
│                                                                         │
│   Raw CSVs (H / V acceleration)  ──►  Sliding window  ──►  Standardise │
│   Window: 1024 samples │ Stride: 512 │ Features: Time-domain + FFT     │
└──────────────────────┬──────────────────────────────┬───────────────────┘
                       │                              │
          ╔════════════▼═════════════╗     ╔══════════▼══════════╗
          ║     HYBRID QML PATH      ║     ║  CLASSICAL BASELINE  ║
          ╠══════════════════════════╣     ╠═════════════════════╣
          ║                          ║     ║                      ║
          ║  ┌──────────────────┐    ║     ║  ┌───────────────┐  ║
          ║  │   FC Encoder     │    ║     ║  │ 2-layer LSTM  │  ║
          ║  │  Linear(n → 16)  │    ║     ║  │  hidden = 64  │  ║
          ║  │  ReLU            │    ║     ║  └──────┬────────┘  ║
          ║  │  Linear(16 → 6)  │    ║     ║         │           ║
          ║  └────────┬─────────┘    ║     ║  ┌──────▼────────┐  ║
          ║           │              ║     ║  │ Linear(64→1)  │  ║
          ║  ┌────────▼─────────┐    ║     ║  └──────┬────────┘  ║
          ║  │  VQC — 6 qubits  │    ║     ╚═════════╪═══════════╝
          ║  │  AngleEmbedding  │    ║               │
          ║  │  StronglyEntangl │    ║               │
          ║  │  3 layers        │    ║               │
          ║  │  ⟨Z⟩ readouts    │    ║               │
          ║  └────────┬─────────┘    ║               │
          ║           │              ║               │
          ║  ┌────────▼─────────┐    ║               │
          ║  │   FC Decoder     │    ║               │
          ║  │  Linear(6 → 32)  │    ║               │
          ║  │  ReLU            │    ║               │
          ║  │  Linear(32 → 1)  │    ║               │
          ║  └────────┬─────────┘    ║               │
          ╚═══════════╪══════════════╝               │
                      │                              │
                      └──────────────┬───────────────┘
                                     ▼
                          ┌────────────────────┐
                          │    RUL ∈ [0, 1]    │
                          │  (normalised,      │
                          │   linear decay)    │
                          └────────────────────┘
```

> **VQC details:** 6 qubits · AngleEmbedding · 3× StronglyEntanglingLayers · Pauli-Z readouts.  
> Depolarizing noise can be injected at inference time for hardware-robustness studies.

---

## Results

Evaluated on **Bearing1\_5** (held-out test bearing), normalised RUL ∈ [0, 1]:

| Model | MAE ↓ | RMSE ↓ | Δ MAE vs. LSTM |
|:------|:-----:|:------:|:--------------:|
| Classical LSTM (2-layer, hidden=64) | 0.2219 | 0.2892 | — |
| **Hybrid VQC + FC (ours)** | **0.1986** | **0.2548** | **−10.5 %** |

The hybrid model achieves a **10.5% reduction in MAE** and **11.9% reduction in RMSE** over the classical baseline, demonstrating that variational quantum circuits can serve as competitive regression heads on real-world time-series health data.

> Exact numbers may vary slightly with hardware and library builds. Re-run `main.py` to reproduce.

---

## Repository Structure

```
hybrid-qml-rul-bearing/
├── .github/
│   └── workflows/ci.yml               # Import & compile checks
│
├── data/
│   ├── raw/                           # XJTU-SY CSVs  (gitignored)
│   └── processed/                     # Cached features & scalers (gitignored)
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_preprocessing.ipynb        # Feature engineering walkthrough
│   ├── 03_classical_baseline.ipynb
│   ├── 04_hybrid_qml_model.ipynb
│   └── 05_results_comparison.ipynb   # Publication figures
│
├── scripts/
│   └── build_05_notebook.py          # Auto-generate notebook 05 source
│
├── src/
│   ├── preprocessing.py              # CSV loading, windowing
│   ├── features.py                   # Time-domain + FFT feature extraction
│   ├── classical_model.py            # LSTM baseline
│   ├── quantum_circuit.py            # PennyLane VQC definition
│   └── hybrid_model.py               # End-to-end hybrid regressor
│
├── results/
│   ├── figures/                      # Plots (gitignored; .gitkeep only)
│   └── metrics.json                  # Example merged metrics (tracked)
│
├── main.py                           # End-to-end pipeline entry point
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

> Large artefacts (`*.pt`, `*.pkl`, raw CSVs, most figures) are **gitignored** — run `main.py` to regenerate locally.

---

## Installation

**Requirements:** Python 3.10 · `git` · ~4 GB RAM minimum for full bearing runs

```bash
# 1. Clone
git clone https://github.com/LisanHub/hybrid-qml-rul-bearing.git
cd hybrid-qml-rul-bearing

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU (optional):** Reinstall `torch` from [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA build; remaining pins are kept for reproducibility.

### Dataset Layout

Download the **XJTU-SY** bearing CSVs and place them so `Bearing1_1` … `Bearing1_5` appear under your chosen root:

```
data/
└── raw/
    └── XJTU-SY_Bearing_Datasets/
        └── 35Hz12kN/
            ├── Bearing1_1/
            ├── Bearing1_2/
            ├── Bearing1_3/
            ├── Bearing1_4/
            └── Bearing1_5/        ← test bearing
```

The loader recursively searches for `Bearing1_*` under `--data_dir`.  
**Default split:** `Bearing1_1` – `Bearing1_4` → train · `Bearing1_5` → test

---

## Usage

### Full Pipeline

```bash
python main.py \
  --data_dir  data/raw \
  --mode      all \
  --epochs    50 \
  --noise_level 0.0
```

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--data_dir` | — | Root folder containing `Bearing1_*` subdirectories |
| `--mode` | `all` | `preprocess` · `train_classical` · `train_hybrid` · `evaluate` · `all` |
| `--epochs` | `50` | Training epochs (applies to both models) |
| `--noise_level` | `0.0` | Depolarizing probability injected during hybrid evaluation |
| `--max_train_windows` | — | Optional subsample cap for quick dry-runs |
| `--no_progress` | — | Suppress `tqdm` progress bars |

**Estimated runtime (desktop CPU, full windows, 50 epochs):** ~50–60 min  

Quick smoke-test:

```bash
python main.py --data_dir data/raw --mode all --epochs 5 --max_train_windows 500
```

### Exploratory Analysis

```bash
python -m jupyter lab notebooks/01_eda.ipynb
```

### Reproducing Publication Figures

After `main.py` has written `results/metrics.json` and the `*_predictions_Bearing1_5.npz` files:

```bash
# Option A — execute notebook directly
jupyter nbconvert --to notebook --execute notebooks/05_results_comparison.ipynb

# Option B — regenerate notebook source first, then execute
python scripts/build_05_notebook.py
jupyter nbconvert --to notebook --execute notebooks/05_results_comparison.ipynb
```

### Noise Robustness Sweep

To reproduce the full multi-point noise curve:

```bash
for NOISE in 0.0 0.02 0.04 0.06 0.08 0.10; do
  python main.py --data_dir data/raw --mode evaluate --noise_level $NOISE
done
```

Then re-execute notebook **05** — `hybrid_noise_robustness.png` renders with the complete sweep.

---

## Output Artefacts

| File | Produced by | Description |
|:-----|:-----------:|:------------|
| `data/processed/*.pkl` | `main.py` | Feature pickles, scaler, manifest |
| `results/classical_lstm.pt` | `main.py` | Trained LSTM weights |
| `results/hybrid_model.pt` | `main.py` | Trained hybrid model weights |
| `results/metrics.json` | `main.py` | Merged evaluation metrics |
| `results/figures/pipeline_classical_Bearing1_5.png` | `main.py` | True vs. predicted RUL — LSTM |
| `results/figures/pipeline_hybrid_Bearing1_5.png` | `main.py` | Degradation plot — hybrid |
| `results/figures/table_metrics_comparison.png` | Notebook 05 | Formatted metrics table |
| `results/figures/rul_overlay_Bearing1_5.png` | Notebook 05 | Both models overlaid |
| `results/figures/bar_mae_rmse_comparison.png` | Notebook 05 | MAE / RMSE bar chart |
| `results/figures/hybrid_noise_robustness.png` | Notebook 05 | Noise sweep curve |
| `results/figures/vqc_diagram.png` | `src/quantum_circuit.py` | Circuit diagram (optional) |

---

## Citation

If you use this code or build on this thesis work, please cite:

```bibtex
@misc{lisan2026hybridqmlrulbearing,
  title        = {Hybrid Quantum-Classical Regression for Remaining Useful Life
                  Prediction of Rolling Bearings},
  author       = {Lisan, Zadid Al},
  year         = {2026},
  howpublished = {B.Sc.\ thesis, Daffodil International University,
                  supervised by Dr.\ Md Alamgir Kabir},
  url          = {https://github.com/LisanHub/hybrid-qml-rul-bearing}
}
```

GitHub also surfaces [`CITATION.cff`](CITATION.cff) in the repository sidebar for one-click export.

---

## Acknowledgements

- **Dataset:** [XJTU-SY Bearing Datasets](https://biaowang.tech/xjtu-sy-bearing-datasets/) and related publications by Biao Wang et al.
- **Supervisor:** Dr. Md Alamgir Kabir, Daffodil International University
- **Libraries:** [PyTorch](https://pytorch.org) · [PennyLane](https://pennylane.ai) · [NumPy](https://numpy.org) · [pandas](https://pandas.pydata.org) · [scikit-learn](https://scikit-learn.org)
- **Contact:** [lisan15-5426@diu.edu.bd](mailto:lisan15-5426@diu.edu.bd) · [@LisanHub](https://github.com/LisanHub)

---

<div align="center">

Made with care at **Daffodil International University**, Dhaka, Bangladesh · 2026

</div>
