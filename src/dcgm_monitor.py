"""DCGM-based GPU monitoring."""

import subprocess
import json
import time
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class GpuMetrics:
    """GPU metrics snapshot."""
    timestamp: float
    gpu_id: int
    gpu_util: float
    memory_util: float
    memory_used_mb: float
    memory_total_mb: float
    temperature: Optional[float] = None
    power_draw: Optional[float] = None
    sm_active: Optional[float] = None


class DCGMMonitor:
    """DCGM-based GPU monitor with NVML fallback."""

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.use_dcgm = self._check_dcgm()
        self._last_metrics: Optional[GpuMetrics] = None

    def _check_dcgm(self) -> bool:
        try:
            subprocess.run(
                ["dcgmi", "discovery", "-l"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _get_nvidia_smi_metrics(self) -> Optional[GpuMetrics]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_id}",
                    "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return None

            parts = result.stdout.strip().split(", ")
            return GpuMetrics(
                timestamp=time.time(),
                gpu_id=self.gpu_id,
                gpu_util=float(parts[1]) if len(parts) > 1 else 0.0,
                memory_util=float(parts[2]) if len(parts) > 2 else 0.0,
                memory_used_mb=float(parts[3]) if len(parts) > 3 else 0.0,
                memory_total_mb=float(parts[4]) if len(parts) > 4 else 0.0,
                temperature=float(parts[5]) if len(parts) > 5 else None,
                power_draw=float(parts[6]) if len(parts) > 6 else None,
                sm_active=None
            )
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return None

    def _get_dcgm_metrics(self) -> Optional[GpuMetrics]:
        try:
            result = subprocess.run(
                ["dcgmi", "dmon", "-e", "1001,1002,1003,1004,1005", "-c", "1"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return self._get_nvidia_smi_metrics()

            lines = result.stdout.strip().split("\n")
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    gpu_id = int(parts[1])
                    if gpu_id == self.gpu_id:
                        return GpuMetrics(
                            timestamp=time.time(),
                            gpu_id=gpu_id,
                            gpu_util=float(parts[2]) if len(parts) > 2 else 0.0,
                            memory_util=float(parts[3]) if len(parts) > 3 else 0.0,
                            memory_used_mb=float(parts[4]) if len(parts) > 4 else 0.0,
                            memory_total_mb=float(parts[5]) if len(parts) > 5 else 0.0,
                        )

            return self._get_nvidia_smi_metrics()
        except subprocess.TimeoutExpired:
            return self._get_nvidia_smi_metrics()

    def read_sample(self) -> GpuMetrics:
        """Read GPU metrics sample."""
        if self.use_dcgm:
            metrics = self._get_dcgm_metrics()
        else:
            metrics = self._get_nvidia_smi_metrics()

        if metrics is None:
            if self._last_metrics:
                return self._last_metrics
            return GpuMetrics(
                timestamp=time.time(),
                gpu_id=self.gpu_id,
                gpu_util=0.0,
                memory_util=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0
            )

        self._last_metrics = metrics
        return metrics


class MetricsAggregator:
    """Aggregate metrics over time windows."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.samples: List[GpuMetrics] = []

    def add(self, sample: GpuMetrics) -> None:
        self.samples.append(sample)
        if len(self.samples) > self.window_size:
            self.samples = self.samples[-self.window_size:]

    def ema_util(self, alpha: float = 0.3) -> float:
        if not self.samples:
            return 0.0

        ema = self.samples[0].gpu_util
        for sample in self.samples[1:]:
            ema = alpha * sample.gpu_util + (1 - alpha) * ema
        return ema

    def trend(self, window: int = 3) -> float:
        if len(self.samples) < window + 1:
            return 0.0

        recent = sum(s.gpu_util for s in self.samples[-window:]) / window
        previous = sum(s.gpu_util for s in self.samples[-(window*2):-window]) / window
        return recent - previous

    def avg_util(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.gpu_util for s in self.samples) / len(self.samples)

    def is_stable(self, threshold: float = 5.0) -> bool:
        if len(self.samples) < 3:
            return True

        recent = [s.gpu_util for s in self.samples[-3:]]
        variance = max(recent) - min(recent)
        return variance < threshold
