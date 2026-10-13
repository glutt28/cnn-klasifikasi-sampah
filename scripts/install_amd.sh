#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    printf 'Error: uv was not found. Install uv first.\n' >&2
    exit 127
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    uv venv --python 3.12 "$ROOT/.venv"
fi

# The unsafe strategy is intentional: uv must compare the PyTorch ROCm index
# with PyPI. Exact version pins in requirements-amd.txt prevent selecting CUDA wheels.
uv pip install \
    --python "$ROOT/.venv/bin/python" \
    --index-url https://download.pytorch.org/whl/rocm7.2 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    -r "$ROOT/requirements-amd.txt"

"$ROOT/.venv/bin/python" - <<'PY'
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if torch.version.hip is None:
    raise SystemExit("ERROR: Installed torch is not a ROCm build.")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: ROCm is installed, but no GPU is available to PyTorch.")
PY
