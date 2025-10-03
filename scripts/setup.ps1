\
# Create venv and install deps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
# Pick ONE torch install:
# pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio  # GPU (CUDA 12.4)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu   # CPU only
python -m ipykernel install --user --name etf-kan --display-name "Python (.venv) etf-kan"
