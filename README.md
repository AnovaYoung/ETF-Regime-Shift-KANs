# ETF Regime Shift KANs

Kolmogorov–Arnold Networks (KANs) for modeling ETF regime shifts using fund flows and market features.

## Quickstart

### 1) Clone
```bash
git clone https://github.com/<your-user>/ETF-Regime-Shift-KANs.git
cd ETF-Regime-Shift-KANs
```

### 2) Create ONE virtual environment (Python 3.11 recommended)
On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
> **Choose ONE**: GPU (CUDA 12.4) or CPU-only.

**GPU (CUDA 12.4, recommended if you have an NVIDIA GPU):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

**CPU only:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify:
```bash
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU?', torch.cuda.is_available())"
```

### 4) Project layout
```
src/etf_kan/              # Python package
  data/                   # data loaders / readers
  models/                 # KAN models and wrappers
  training/               # training loops, evaluation
configs/                  # yaml configs for experiments
data/                     # local data (ignored by git)
notebooks/                # exploratory work
scripts/                  # setup / run helpers
tests/                    # unit tests
```

### 5) First run (example)
```bash
python -m etf_kan.training.train --config configs/default.yaml
```

---

## Why KANs here?
KANs provide compact, interpretable basis functions well-suited for regime segmentation where relationships are non-linear and time-varying.

## License
MIT (see `LICENSE` if/when added).
