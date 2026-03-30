# GPU Utilization Manager for B200

Maintains target GPU utilization on a single selected GPU by running low-priority filler workloads when that GPU is underused.

## Current Scope

- Single-GPU control per manager instance
- Target GPU selected by `gpu_id` in `config.yaml`
- Other GPUs are ignored by the current instance
- Filler intensity levels now span `0..8`

## Behavior

- If the selected GPU is under the target, filler level increases gradually
- If the selected GPU is in the healthy range, the current level is held
- If the selected GPU is too busy, filler level is reduced gradually
- When level is `0`, workers are paused
- When level is `1..8`, workers are resumed and compute on the selected GPU

The controller uses smoothed utilization rather than a single instantaneous sample, and it moves one level at a time to reduce oscillation.

## Architecture

```text
GPU Manager Daemon
  ├─ DCGM / nvidia-smi monitor
  ├─ metrics aggregator
  ├─ scaling engine
  ├─ state machine
  ├─ experiment registry
  ├─ MPS adapter
  └─ filler controller
       └─ filler workers (PyTorch GEMM)
```

## Quick Start

### 1. Setup

```bash
./scripts/setup.sh
```

What setup does now:

- creates `venv/`
- installs PyTorch inside the virtualenv
- chooses the PyTorch wheel index from the detected CUDA version
- installs `pyyaml` and `pynvml`
- tries to install DCGM packages
- prepares MPS directories

### 2. Start

```bash
./scripts/run.sh
```

`run.sh` activates `venv/` automatically if it exists.

### 3. Stop / Status

```bash
./scripts/run.sh stop
./scripts/run.sh status
```

## Experiment Detection

Two modes exist:

### Auto-detect

```bash
python train.py
```

### Explicit registration

```python
from src.experiment_registry import ExperimentClient

client = ExperimentClient()
client.start(name="my_experiment")

# run experiment

client.stop()
```

## Configuration

Main settings live in `config.yaml`.

```yaml
gpu_id: 0
poll_interval_sec: 2.0
target_util_pct: 70.0

thresholds:
  low_boost_pct: 65.0
  target_floor_pct: 70.0
  healthy_high_pct: 88.0
  reduce_pct: 92.0
  emergency_reduce_pct: 95.0
  critical_pause_pct: 98.0
```

## Filler Levels

The current default filler levels are:

```yaml
0: workers=0  batch_size=0   streams=0  sleep_ms=100
1: workers=1  batch_size=32  streams=1  sleep_ms=20
2: workers=1  batch_size=64  streams=2  sleep_ms=10
3: workers=2  batch_size=96  streams=2  sleep_ms=5
4: workers=2  batch_size=128 streams=4  sleep_ms=0
5: workers=3  batch_size=160 streams=4  sleep_ms=0
6: workers=3  batch_size=192 streams=5  sleep_ms=0
7: workers=4  batch_size=224 streams=6  sleep_ms=0
8: workers=4  batch_size=256 streams=8  sleep_ms=0
```

Important: these are relative intensity levels, not guaranteed utilization percentages. On a larger GPU, even level 8 may still be below 70%.

## MPS Caps

Default caps:

```yaml
mps_caps_no_experiment:      [0, 40, 60, 80, 90, 92, 94, 96, 98]
mps_caps_experiment_active:  [0, 5, 10, 20, 30, 35, 40, 45, 50]
```

## Monitoring

```bash
./scripts/run.sh status
watch -n 1 nvidia-smi
dcgmi dmon -e 1001,1002,1003
```

## Current Limitations

- single-GPU only per manager instance
- level definitions are static defaults
- level 8 is not guaranteed to hit 70% on every GPU
- terminal output from the daemon can still appear in the launching shell unless redirected

## Project Layout

```text
gpu_util_manager/
├── src/
├── scripts/
├── tests/
├── config.yaml
├── venv/
└── README.md
```
