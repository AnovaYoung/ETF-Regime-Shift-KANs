# Regime Detection in ETF Flows with Kolmogorov–Arnold Networks (KANs)

This project investigates whether **Kolmogorov–Arnold Networks (KANs)** can detect latent **regime shifts** in cross-asset ETF flows and related macro features — and whether they can do so **more compactly** and **interpretably** than standard deep nets.

> **Demo status:** A GPU-ready training stub is included. It:
>
> * Loads your parquet features
> * Coerces numeric columns and **z-scores X**
> * Detects if the target is **binary** (e.g., `regime_high_vol`) and uses **BCEWithLogitsLoss**; otherwise uses MSE
> * Trains with mini-batches + train/val split on **CUDA**

---

## Motivation

Classic MLPs/LSTMs can model regimes but are often black-box and parameter-heavy. **KANs**, inspired by the Kolmogorov–Arnold representation theorem, use learnable edge functions instead of fixed activations, aiming for:

* **Parameter efficiency**
* **Interpretability** (critical in finance)
* **Flexible nonlinearity** for noisy, regime-shifting data

This repo tests KANs on real, noisy ETF data and compares them to strong baselines.

---

## What are KANs?

KANs replace fixed activations with **learnable edge functions**, enabling:

* Fewer parameters for similar expressivity

* Inspectable learned functions (for intuition)

* Strong nonlinear modeling capacity

* Original paper: [https://arxiv.org/abs/2404.19756](https://arxiv.org/abs/2404.19756)

---

## Tech Stack

* **Python 3.11**
* **PyTorch** (GPU builds; RTX 50-series supported via nightly cu129)
* **PyKAN** (KAN implementation; optional — wrapper falls back to MLP if KAN constructor differs)
* **Pandas / NumPy** (+ light scikit-learn usage)
* **Matplotlib / Plotly** (optional)
* **Dask** (planned, for larger merges)
* **Dash** (planned, for a demo UI)

---

## Data Sources (current + planned)

* **ETF Flow data** (PDF → CSV by ticker)
* **ETF OHLCV daily data**
* **Macro & Rates** (VIX, Treasuries, spreads, etc.)
* **Google Trends** (20+ regime keywords)
* **Synthetic datasets** (benchmarking)
* **(Planned)** News/event features via LLM or APIs

---

## Project Layout

```
src/etf_kan/              # Python package
  data/                   # data loaders / readers
  models/                 # KAN wrapper (falls back to MLP if KAN unavailable)
  training/               # training loops, evaluation
configs/                  # YAML configs for experiments
data/                     # local data (gitignored)
  raw/  interim/  processed/
notebooks/                # EDA & experiments
scripts/                  # setup / run helpers
tests/                    # unit tests
```

---

## Quickstart

### 1) Clone

```bash
git clone https://github.com/<your-user>/ETF-Regime-Shift-KANs.git
cd ETF-Regime-Shift-KANs
```

### 2) One virtual environment (Python 3.11)

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### GPU builds (choose ONE)

* **RTX 50-series (Blackwell)** – use **nightly cu129** (supports sm_120):

```bash
pip install --pre --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
```

* **CUDA 12.4** (older GPUs):

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

* **CPU-only**:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

Verify:

```bash
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU?', torch.cuda.is_available())"
```

### 4) First run

```bash
# ensure PYTHONPATH includes src (e.g., VS Code .env: PYTHONPATH=${workspaceFolder}/src)
python -m etf_kan.training.train --config configs/default.yaml
```

The stub:

* z-scores features
* auto-detects binary vs regression targets
* uses BCEWithLogitsLoss for binary (e.g., `regime_high_vol`)
* prints train/val loss and best value

---

## Roadmap

**1. Literature & Theory Review**

* [x] Survey KAN literature/code
* [x] Review regime detection SOTA
* [x] Document theoretical motivation

**2. Data Acquisition & Preprocessing**

* [x] Gather ETF flows, prices, macro, Trends
* [x] Convert flow PDFs → per-ticker CSVs
* [x] Merge into a unified parquet (5k+ rows × 200+ cols)
* [x] Local HPC setup (RTX 5080 Laptop GPU)
* [ ] (Planned) News/event features

**3. KAN Prototyping & Benchmarks**

* [x] GPU training stub (binary/regression switch)
* [x] KAN wrapper with MLP fallback
* [ ] Wire real KAN (`pykan`) and align constructor signature
* [ ] Baselines (LogReg, MLP, LSTM)

**4. Experimentation & Tuning**

* [ ] Stratified splits, **early stopping**, **AUROC** metric
* [ ] Hyperparams (KAN width/depth/sparsity)
* [ ] Compare to baselines

**5. Interpretability & Analysis**

* [ ] Visualize learned KAN edge functions
* [ ] When/why KAN > baseline (or not)

**6. Results & Dissemination**

* [ ] Draft paper (preprint)
* [ ] Open-source demo notebooks
* [ ] Present results

---

## Current Status

* **GPU verified:** nightly cu129 (CUDA 12.9) on RTX 5080; `CAP (12,0), GPU? True`
* **Training stub live:** z-scored X, BCE for `regime_high_vol`, mini-batches, val split
* **Overfitting observed:** val loss bottoms early → next step: early stopping + weight decay (see README.dev if added)

---

## Developer Notes (short)

* Use **one** venv at repo root (`.venv`).
* Keep **data/** out of git (already in `.gitignore`).
* VS Code: **Python: Select Interpreter** → `.\.venv\Scripts\python.exe`, add `.env`:

  ```
  PYTHONPATH=${workspaceFolder}\src
  ```
* If KAN import fails, the wrapper falls back to MLP so experiments still run.
  Try a real KAN:

  ```bash
  pip install -U pykan
  ```

  If the constructor differs, adjust the wrapper and re-run.

---

## References

* Kolmogorov–Arnold Networks (arXiv): [https://arxiv.org/abs/2404.19756](https://arxiv.org/abs/2404.19756)
* PyKAN repo: [https://github.com/KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)
* Survey on KANs: [https://arxiv.org/abs/2411.06078](https://arxiv.org/abs/2411.06078)

