#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Python and venv"
PYTHON_BIN=${PYTHON_BIN:-python3}
$PYTHON_BIN -V

if [ ! -d "venv" ]; then
  $PYTHON_BIN -m venv venv
fi
source venv/bin/activate
python -V

echo "[2/4] Upgrading pip and core tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing packages (PyTorch with MPS, Transformers, etc.)"
# PyTorch stable wheels on macOS arm64 include MPS support by default
pip install "torch>=2.3.0" "torchvision>=0.18.0" "torchaudio>=2.3.0"
pip install transformers datasets accelerate protobuf scikit-learn imbalanced-learn matplotlib seaborn

echo "[4/4] Verifying MPS availability"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("MPS built:", hasattr(torch.backends, 'mps'))
print("MPS available:", torch.backends.mps.is_available())

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
x = torch.ones((2, 2), device=device)
y = x * 3
print("device:", device, "result:", y)
PY

echo "Done. Activate with: source venv/bin/activate"

