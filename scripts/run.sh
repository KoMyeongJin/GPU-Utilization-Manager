#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/config.yaml"
VENV_DIR="${PROJECT_DIR}/venv"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Warning: Virtual environment not found at $VENV_DIR"
    echo "Run ./scripts/setup.sh first"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

cd "$PROJECT_DIR"

if [ "$1" == "stop" ]; then
    echo "Stopping GPU Manager..."
    pkill -f "python.*daemon.py" || true
    pkill -f "python.*filler_worker.py" || true
    echo "Stopped"
    exit 0
fi

if [ "$1" == "status" ]; then
    echo "GPU Manager Status:"
    if pgrep -f "python.*daemon.py" > /dev/null; then
        echo "  Manager: Running"
        nvidia-smi pmon -c 1 2>/dev/null || nvidia-smi
    else
        echo "  Manager: Not running"
    fi
    exit 0
fi

echo "Starting GPU Utilization Manager..."
echo "Config: $CONFIG_FILE"
echo "Python: $(which python3)"

if [ -f "$CONFIG_FILE" ]; then
    python3 "$PROJECT_DIR/src/daemon.py" "$CONFIG_FILE" &
else
    python3 "$PROJECT_DIR/src/daemon.py" &
fi

MANAGER_PID=$!
echo "Manager started (PID: $MANAGER_PID)"
echo ""
echo "Logs will be written to /var/log/gpu_manager/"
echo ""
echo "To stop: $0 stop"
echo "To check status: $0 status"
