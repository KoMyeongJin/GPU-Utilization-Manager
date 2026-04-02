"""DCGM-based GPU monitoring with diagnostic enhancements."""

import importlib
import subprocess
import json
import time
import os
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
    # NEW: Diagnostic fields
    kernel_launches_per_sec: Optional[float] = None
    gpu_stall_reason: Optional[str] = None
    nsight_timestamp: Optional[float] = None


class NsightProfiler:
    """Optional background Nsight Systems profiler for kernel-level diagnostics."""

    def __init__(self, gpu_id: int, enable: bool = False):
        self.gpu_id = gpu_id
        self.enable = enable and self._check_nsight_installed()
        self.process = None
        self.output_file = None

    def _check_nsight_installed(self) -> bool:
        """Check if Nsight Systems is installed and available."""
        try:
            subprocess.run(["nsys", "--version"], capture_output=True, timeout=2)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def start(self, experiment_name: str, duration_sec: int = 60):
        """Start background profiling."""
        if not self.enable:
            return

        self.output_file = f"/tmp/nsight_{experiment_name}_{self.gpu_id}.nsys-rep"
        cmd = [
            "nsys",
            "profile",
            "-o",
            self.output_file,
            "-d",
            str(duration_sec),
            f"--gpu-metrics-device={self.gpu_id}",
            "--stats=full",
            "--capture=cuda,cudaMemcpy,osrt",
            "-f",
            "true",
        ]

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            # Graceful degradation: disable Nsight on error
            print(f"[WARN] Nsight profiling failed: {e}")
            self.enable = False

    def stop(self):
        """Stop profiling gracefully."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None

    def report_path(self) -> str:
        """Return path to profiling output file."""
        return self.output_file if self.output_file else ""


class DCGMMonitor:
    """DCGM-based GPU monitor with NVML fallback."""

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self._pynvml = None
        self._nvml_handle = None
        self.use_nvml = self._init_nvml()
        self.use_dcgm = self._check_dcgm()
        self._last_metrics: Optional[GpuMetrics] = None

    def _init_nvml(self) -> bool:
        try:
            pynvml = importlib.import_module("pynvml")
        except ImportError:
            return False

        try:
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            return True
        except Exception:
            self._pynvml = None
            self._nvml_handle = None
            return False

    def _check_dcgm(self) -> bool:
        try:
            subprocess.run(["dcgmi", "discovery", "-l"], capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _normalize_percent_metric(self, raw_value: str) -> float:
        value = float(raw_value)
        if 0.0 <= value <= 1.0:
            return value * 100.0
        return value

    def _get_nvidia_smi_metrics(self) -> Optional[GpuMetrics]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_id}",
                    "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            parts = result.stdout.strip().split(", ")
            return GpuMetrics(
                timestamp=time.time(),
                gpu_id=self.gpu_id,
                gpu_util=self._normalize_percent_metric(parts[1])
                if len(parts) > 1
                else 0.0,
                memory_util=self._normalize_percent_metric(parts[2])
                if len(parts) > 2
                else 0.0,
                memory_used_mb=float(parts[3]) if len(parts) > 3 else 0.0,
                memory_total_mb=float(parts[4]) if len(parts) > 4 else 0.0,
                temperature=float(parts[5]) if len(parts) > 5 else None,
                power_draw=float(parts[6]) if len(parts) > 6 else None,
                sm_active=None,
            )
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return None

    def _get_nvml_metrics(self) -> Optional[GpuMetrics]:
        if not self.use_nvml or self._pynvml is None or self._nvml_handle is None:
            return None

        try:
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)

            temperature = None
            try:
                temperature = float(
                    self._pynvml.nvmlDeviceGetTemperature(
                        self._nvml_handle, self._pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass

            power_draw = None
            try:
                power_draw = (
                    float(self._pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle))
                    / 1000.0
                )
            except Exception:
                pass

            memory_total_mb = float(memory.total) / (1024 * 1024)
            memory_used_mb = float(memory.used) / (1024 * 1024)
            memory_util = 0.0
            if memory.total > 0:
                memory_util = (float(memory.used) / float(memory.total)) * 100.0

            return GpuMetrics(
                timestamp=time.time(),
                gpu_id=self.gpu_id,
                gpu_util=float(utilization.gpu),
                memory_util=memory_util,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                temperature=temperature,
                power_draw=power_draw,
                sm_active=None,
            )
        except Exception:
            return None

    def _get_dcgm_metrics(self) -> Optional[GpuMetrics]:
        try:
            result = subprocess.run(
                ["dcgmi", "dmon", "-e", "1001,1002,1003,1004,1005", "-c", "1"],
                capture_output=True,
                text=True,
                timeout=5,
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
                            gpu_util=self._normalize_percent_metric(parts[2])
                            if len(parts) > 2
                            else 0.0,
                            memory_util=self._normalize_percent_metric(parts[3])
                            if len(parts) > 3
                            else 0.0,
                            memory_used_mb=float(parts[4]) if len(parts) > 4 else 0.0,
                            memory_total_mb=float(parts[5]) if len(parts) > 5 else 0.0,
                        )

            return self._get_nvidia_smi_metrics()
        except subprocess.TimeoutExpired:
            return self._get_nvidia_smi_metrics()

    def read_sample(self) -> GpuMetrics:
        """Read GPU metrics sample."""
        if self.use_nvml:
            metrics = self._get_nvml_metrics()
            if metrics is None and self.use_dcgm:
                metrics = self._get_dcgm_metrics()
            elif metrics is None:
                metrics = self._get_nvidia_smi_metrics()
        elif self.use_dcgm:
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
                memory_total_mb=0.0,
            )

        self._last_metrics = metrics
        return metrics


class MetricsAggregator:
    """Aggregate metrics over time windows with diagnostic pattern detection."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.samples: List[GpuMetrics] = []
        self._util_variance_threshold = 5.0  # % points for noise detection

    def add(self, sample: GpuMetrics) -> None:
        self.samples.append(sample)
        if len(self.samples) > self.window_size:
            self.samples = self.samples[-self.window_size :]

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
        previous = (
            sum(s.gpu_util for s in self.samples[-(window * 2) : -window]) / window
        )
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

    def detect_measurement_noise(self) -> bool:
        """Detect if variance is high (potential measurement artifact)."""
        if len(self.samples) < 3:
            return False

        utils = [s.gpu_util for s in self.samples[-10:]]
        if not utils:
            return False

        mean_util = sum(utils) / len(utils)
        variance = sum((u - mean_util) ** 2 for u in utils) / len(utils)
        std_dev = variance**0.5
        return std_dev > self._util_variance_threshold

    def get_kv_cache_pressure_estimate(self, memory_util_pct: float) -> str:
        """Estimate KV cache memory pressure based on utilization."""
        if memory_util_pct > 90:
            return "CRITICAL"
        elif memory_util_pct > 75:
            return "HIGH"
        elif memory_util_pct > 50:
            return "MODERATE"
        else:
            return "LOW"

    def get_smoothed_utilization(self) -> float:
        """Return EMA-smoothed GPU utilization."""
        if not self.samples:
            return 0.0
        recent = self.samples[-3:]
        return sum(s.gpu_util for s in recent) / len(recent)
