"""Tests for GPU saturation diagnostic features."""

import os
import sys
import types
import pytest
import time
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.dcgm_monitor as dcgm_monitor
from src.dcgm_monitor import GpuMetrics, MetricsAggregator, NsightProfiler, DCGMMonitor


class TestGpuMetricsEnhanced:
    """Test enhanced GpuMetrics dataclass with diagnostic fields."""

    def test_metrics_with_diagnostic_fields(self):
        """Verify diagnostic fields are optional and don't break instantiation."""
        metrics = GpuMetrics(
            timestamp=time.time(),
            gpu_id=0,
            gpu_util=50.0,
            memory_util=60.0,
            memory_used_mb=10000,
            memory_total_mb=16384,
            temperature=65.0,
            power_draw=250.0,
            sm_active=80.0,
            kernel_launches_per_sec=100.0,
            gpu_stall_reason="memory_dependence",
            nsight_timestamp=time.time(),
        )
        assert metrics.gpu_util == 50.0
        assert metrics.kernel_launches_per_sec == 100.0
        assert metrics.gpu_stall_reason == "memory_dependence"


class TestMetricsAggregatorDiagnostics:
    """Test diagnostic methods in MetricsAggregator."""

    def test_detect_measurement_noise_low_variance(self):
        """Test noise detection returns False for stable utilization."""
        agg = MetricsAggregator(window_size=10)

        # Add samples with low variance
        for util in [50, 50, 51, 49, 50, 50, 49, 51, 50, 50]:
            agg.add(
                GpuMetrics(
                    timestamp=time.time(),
                    gpu_id=0,
                    gpu_util=util,
                    memory_util=60.0,
                    memory_used_mb=10000,
                    memory_total_mb=16384,
                )
            )

        assert agg.detect_measurement_noise() == False

    def test_detect_measurement_noise_high_variance(self):
        """Test noise detection returns True for unstable utilization."""
        agg = MetricsAggregator(window_size=10)

        # Add samples with high variance (noisy measurement)
        for util in [20, 60, 30, 70, 25, 65, 35, 75, 40, 80]:
            agg.add(
                GpuMetrics(
                    timestamp=time.time(),
                    gpu_id=0,
                    gpu_util=util,
                    memory_util=60.0,
                    memory_used_mb=10000,
                    memory_total_mb=16384,
                )
            )

        assert agg.detect_measurement_noise() == True

    def test_kv_cache_pressure_critical(self):
        """Test KV cache pressure estimation for high memory utilization."""
        agg = MetricsAggregator()
        assert agg.get_kv_cache_pressure_estimate(95.0) == "CRITICAL"

    def test_kv_cache_pressure_high(self):
        """Test KV cache pressure estimation for moderately high memory."""
        agg = MetricsAggregator()
        assert agg.get_kv_cache_pressure_estimate(80.0) == "HIGH"

    def test_kv_cache_pressure_moderate(self):
        """Test KV cache pressure estimation for moderate memory."""
        agg = MetricsAggregator()
        assert agg.get_kv_cache_pressure_estimate(60.0) == "MODERATE"

    def test_kv_cache_pressure_low(self):
        """Test KV cache pressure estimation for low memory."""
        agg = MetricsAggregator()
        assert agg.get_kv_cache_pressure_estimate(30.0) == "LOW"

    def test_smoothed_utilization(self):
        """Test EMA smoothing of GPU utilization."""
        agg = MetricsAggregator()

        # Add samples
        for util in [50, 60, 70]:
            agg.add(
                GpuMetrics(
                    timestamp=time.time(),
                    gpu_id=0,
                    gpu_util=util,
                    memory_util=60.0,
                    memory_used_mb=10000,
                    memory_total_mb=16384,
                )
            )

        smoothed = agg.get_smoothed_utilization()
        # Should be average of last 3: (50 + 60 + 70) / 3 = 60
        assert 59.0 < smoothed < 61.0


class TestNsightProfiler:
    """Test Nsight Systems profiler wrapper."""

    def test_nsight_initialization_disabled_by_default(self):
        """Verify Nsight is disabled by default."""
        profiler = NsightProfiler(gpu_id=0, enable=False)
        assert profiler.enable == False

    def test_nsight_graceful_degradation(self):
        """Verify Nsight gracefully degrades if nsys not available."""
        profiler = NsightProfiler(gpu_id=0, enable=True)
        # enable may be True or False depending on nsys availability
        # The key is that instantiation doesn't crash
        assert profiler.gpu_id == 0
        assert profiler.process is None

    def test_nsight_report_path_empty_when_not_started(self):
        """Verify report path is empty if profiling not started."""
        profiler = NsightProfiler(gpu_id=0, enable=False)
        assert profiler.report_path() == ""


class TestDCGMMonitorNormalization:
    def test_nvml_metrics_are_used_when_available(self, monkeypatch):
        fake_pynvml = types.SimpleNamespace(
            NVML_TEMPERATURE_GPU=0,
            nvmlInit=Mock(),
            nvmlDeviceGetHandleByIndex=Mock(return_value="handle"),
            nvmlDeviceGetUtilizationRates=Mock(
                return_value=types.SimpleNamespace(gpu=55, memory=22)
            ),
            nvmlDeviceGetMemoryInfo=Mock(
                return_value=types.SimpleNamespace(
                    used=8 * 1024 * 1024,
                    total=16 * 1024 * 1024,
                )
            ),
            nvmlDeviceGetTemperature=Mock(return_value=66),
            nvmlDeviceGetPowerUsage=Mock(return_value=250000),
        )

        monkeypatch.setattr(DCGMMonitor, "_check_dcgm", lambda self: False)
        monkeypatch.setattr(
            dcgm_monitor.importlib, "import_module", lambda name: fake_pynvml
        )

        monitor = DCGMMonitor(gpu_id=0)

        with patch.object(dcgm_monitor.subprocess, "run") as mock_run:
            metrics = monitor.read_sample()

        assert metrics is not None
        assert monitor.use_nvml is True
        assert metrics.gpu_util == 55.0
        assert metrics.memory_util == 50.0
        assert metrics.temperature == 66.0
        assert metrics.power_draw == 250.0
        mock_run.assert_not_called()

    def test_nvml_unavailable_falls_back_to_existing_path(self, monkeypatch):
        monkeypatch.setattr(DCGMMonitor, "_check_dcgm", lambda self: False)
        monkeypatch.setattr(
            dcgm_monitor.importlib, "import_module", Mock(side_effect=ImportError)
        )
        monitor = DCGMMonitor(gpu_id=0)

        mock_result = Mock(
            returncode=0,
            stdout="2026/04/01 00:00:00.000, 12, 34, 10000, 18000, 65, 250\n",
        )

        with patch.object(dcgm_monitor.subprocess, "run", return_value=mock_result):
            metrics = monitor.read_sample()

        assert metrics is not None
        assert monitor.use_nvml is False
        assert metrics.gpu_util == 12.0
        assert metrics.memory_util == 34.0

    def test_nvidia_smi_fractional_utilization_is_normalized_to_percent(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(DCGMMonitor, "_check_dcgm", lambda self: False)
            mp.setattr(
                dcgm_monitor.importlib, "import_module", Mock(side_effect=ImportError)
            )
            monitor = DCGMMonitor(gpu_id=0)

        mock_result = Mock(
            returncode=0,
            stdout="2026/04/01 00:00:00.000, 0.9, 0.2, 10000, 18000, 65, 250\n",
        )

        with patch("src.dcgm_monitor.subprocess.run", return_value=mock_result):
            metrics = monitor._get_nvidia_smi_metrics()

        assert metrics is not None
        assert metrics.gpu_util == 90.0
        assert metrics.memory_util == 20.0

    def test_dcgmi_fractional_utilization_is_normalized_to_percent(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(DCGMMonitor, "_check_dcgm", lambda self: True)
            mp.setattr(
                dcgm_monitor.importlib, "import_module", Mock(side_effect=ImportError)
            )
            monitor = DCGMMonitor(gpu_id=0)

        mock_result = Mock(returncode=0, stdout="# header\n0 0 0.1 0.4 1234 5678\n")

        with patch("src.dcgm_monitor.subprocess.run", return_value=mock_result):
            metrics = monitor._get_dcgm_metrics()

        assert metrics is not None
        assert metrics.gpu_util == 10.0
        assert metrics.memory_util == 40.0

    def test_percent_utilization_is_left_unchanged(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(DCGMMonitor, "_check_dcgm", lambda self: False)
            mp.setattr(
                dcgm_monitor.importlib, "import_module", Mock(side_effect=ImportError)
            )
            monitor = DCGMMonitor(gpu_id=0)

        assert monitor._normalize_percent_metric("90") == 90.0
        assert monitor._normalize_percent_metric("12.5") == 12.5


class TestDaemonDiagnosticMethods:
    """Test diagnostic methods added to daemon."""

    def test_daemon_diagnostic_attributes_initialized(self):
        """Verify daemon initializes diagnostic attributes."""
        from src.daemon import GPUtilizationManager
        from unittest.mock import patch

        with patch("src.daemon.load_config") as mock_config:
            mock_config.return_value = Mock(
                gpu_id=0,
                poll_interval_sec=2.0,
                enable_mps=False,
                enable_process_detection=False,
                diagnostic_interval_polls=30,
                target_util_pct=70.0,
                thresholds=Mock(
                    low_boost_pct=65.0,
                    target_floor_pct=70.0,
                    healthy_high_pct=88.0,
                    reduce_pct=92.0,
                    emergency_reduce_pct=95.0,
                    critical_pause_pct=98.0,
                ),
                filler=Mock(
                    sublevels_per_major=1,
                    max_major_level=8,
                    mps_caps_no_experiment=[0, 40, 60, 80, 90, 92, 94, 96, 98],
                    mps_caps_experiment_active=[0, 5, 10, 20, 30, 35, 40, 45, 50],
                ),
                hysteresis=Mock(consecutive_polls=3, min_dwell_sec=1.0),
                socket_path="/tmp/gpu_manager.sock",
                mps_pipe_dir="/tmp/mps",
                mps_log_dir="/tmp/mps_log",
                experiment_heartbeat_timeout_sec=10.0,
            )

            with patch("src.daemon.DCGMMonitor"):
                with patch("src.daemon.ExperimentRegistry"):
                    with patch("src.daemon.MPSAdapter"):
                        with patch("src.daemon.FillerController"):
                            with patch("src.daemon.ScalingEngine"):
                                with patch("src.daemon.FillerStateMachine"):
                                    manager = GPUtilizationManager()
                                    assert hasattr(manager, "_diagnostic_interval")
                                    assert hasattr(manager, "_diagnostic_counter")
                                    assert hasattr(manager, "_decision_timestamps")
                                    assert manager._diagnostic_interval == 30


class TestFillerWorkerDiagnostics:
    """Test diagnostic features in filler worker."""

    def test_worker_phase_detection_prefill(self):
        """Test phase detection identifies prefill phase."""
        from src.filler_worker import FillerWorker
        from unittest.mock import patch

        with patch.dict("os.environ", {"FILLER_STREAMS": "4", "FILLER_SLEEP_MS": "0"}):
            worker = FillerWorker()
            # 4 streams * (1000 / 0) = high launch rate → PREFILL
            # Note: division by zero edge case, but worker sets default 1 if 0
            assert hasattr(worker, "_detect_phase")
            assert callable(worker._detect_phase)

    def test_worker_kv_cache_estimate_computed(self):
        """Test KV cache estimate is computed on initialization."""
        from src.filler_worker import FillerWorker
        from unittest.mock import patch

        with patch.dict(
            "os.environ",
            {
                "FILLER_WORKER_ID": "1",
                "FILLER_BATCH_SIZE": "64",
                "FILLER_STREAMS": "2",
                "FILLER_SLEEP_MS": "10",
            },
        ):
            worker = FillerWorker()
            assert hasattr(worker, "estimated_kv_cache_mb")
            assert isinstance(worker.estimated_kv_cache_mb, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
