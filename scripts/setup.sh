#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_DIR}/venv"

echo "=========================================="
echo "GPU Utilization Manager Setup"
echo "=========================================="

GPU_ID=${1:-0}

get_cuda_version() {
  local cuda_version=""

  if command -v nvcc &>/dev/null; then
    cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
  elif command -v nvidia-smi &>/dev/null; then
    local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    case "$driver_version" in
    550.*) cuda_version="12.4" ;;
    545.*) cuda_version="12.3" ;;
    535.*) cuda_version="12.2" ;;
    525.*) cuda_version="12.0" ;;
    520.*) cuda_version="11.8" ;;
    515.*) cuda_version="11.7" ;;
    510.*) cuda_version="11.6" ;;
    470.*) cuda_version="11.4" ;;
    460.*) cuda_version="11.2" ;;
    450.*) cuda_version="11.0" ;;
    *) cuda_version="12.1" ;;
    esac
  fi

  echo "$cuda_version"
}

get_torch_index_url() {
  local cuda_version="$1"
  local index_url="https://download.pytorch.org/whl/cpu"

  local major_minor=$(echo "$cuda_version" | sed 's/^\([0-9]\+\.[0-9]\+\).*/\1/')

  case "$major_minor" in
  11.0|11.1|11.2|11.3|11.4|11.5|11.6|11.7|11.8)
    index_url="https://download.pytorch.org/whl/cu118"
    ;;
  12.0|12.1|12.2|12.3|12.4)
    index_url="https://download.pytorch.org/whl/cu124"
    ;;
  12.5|12.6)
    index_url="https://download.pytorch.org/whl/cu126"
    ;;
  12.7|12.8)
    index_url="https://download.pytorch.org/whl/cu128"
    ;;
  12.9|12.10|12.*)
    echo "Warning: CUDA $cuda_version is newer than tested, using cu128" >&2
    index_url="https://download.pytorch.org/whl/cu128"
    ;;
  13.*)
    echo "Warning: CUDA $cuda_version is very new, trying cu128" >&2
    index_url="https://download.pytorch.org/whl/cu128"
    ;;
  *)
    echo "Warning: Unknown CUDA version '$cuda_version', using CPU version" >&2
    index_url="https://download.pytorch.org/whl/cpu"
    ;;
  esac

  echo "$index_url"
}

echo "Step 1: Checking CUDA version..."
CUDA_VERSION=$(get_cuda_version)
if [ -z "$CUDA_VERSION" ]; then
  echo "Warning: Could not detect CUDA version"
  echo "Will install CPU-only PyTorch"
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  echo "Detected CUDA version: $CUDA_VERSION"
  TORCH_INDEX=$(get_torch_index_url "$CUDA_VERSION")
  echo "PyTorch index: $TORCH_INDEX"
fi

echo ""
echo "Step 2: Creating Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
  echo "Virtual environment already exists at $VENV_DIR"
  read -p "Recreate? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    echo "Created new virtual environment"
  else
    echo "Using existing virtual environment"
  fi
else
  python3 -m venv "$VENV_DIR"
  echo "Created virtual environment at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Step 4: Installing PyTorch with CUDA support..."
echo "Installing from: $TORCH_INDEX"
pip install torch --index-url "$TORCH_INDEX"

echo ""
echo "Step 5: Installing other dependencies..."
pip uninstall -y pynvml >/dev/null 2>&1 || true
pip install pyyaml nvidia-ml-py

echo ""
echo "Step 6: Installing system dependencies..."
if command -v apt-get &>/dev/null; then
  apt-get update
  apt-get install -y datacenter-gpu-manager dcgm || echo "Warning: Could not install DCGM"
fi

echo ""
echo "Step 7: Creating directories..."
mkdir -p /tmp/nvidia-mps
mkdir -p /tmp/nvidia-mps-log
mkdir -p /var/log/gpu_manager
mkdir -p "$PROJECT_DIR/logs"

echo ""
echo "Step 8: Configuring GPU..."
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -i $GPU_ID -c EXCLUSIVE_PROCESS || echo "Warning: Could not set exclusive mode"
else
  echo "Warning: nvidia-smi not found"
fi

echo ""
echo "Step 9: Testing MPS availability..."
if command -v nvidia-cuda-mps-control &>/dev/null; then
  echo "✓ MPS control available"
else
  echo "⚠ Warning: nvidia-cuda-mps-control not found"
fi

echo ""
echo "Step 10: Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || echo "Warning: PyTorch verification failed"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment: $VENV_DIR"
echo "Python: $(which python3)"
echo "PyTorch: $(pip show torch | grep Version | cut -d' ' -f2)"
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To start the manager:"
echo "  source $VENV_DIR/bin/activate"
echo "  python $PROJECT_DIR/src/daemon.py"
echo ""
echo "Or use the run script:"
echo "  ./scripts/run.sh"
