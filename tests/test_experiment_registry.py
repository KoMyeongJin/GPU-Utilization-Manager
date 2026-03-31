import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiment_registry import ExperimentRegistry


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid


class FakeUtilSample:
    def __init__(self, pid, sm=0, mem=0, enc=0, dec=0):
        self.pid = pid
        self.smUtil = sm
        self.memUtil = mem
        self.encUtil = enc
        self.decUtil = dec


class TestExperimentRegistry:
    def test_nvml_gpu_process_detection(self, monkeypatch):
        fake_pynvml = types.SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda idx: f"gpu-{idx}",
            nvmlDeviceGetComputeRunningProcesses_v3=lambda handle: [FakeProcess(111), FakeProcess(222)],
            nvmlDeviceGetGraphicsRunningProcesses_v3=lambda handle: [FakeProcess(333)],
            nvmlDeviceGetProcessUtilization=lambda handle, lastSeenTimeStamp=0: [
                FakeUtilSample(111, sm=10),
                FakeUtilSample(222, sm=0),
                FakeUtilSample(333, mem=5),
            ],
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

        registry = ExperimentRegistry(enable_process_detection=True)
        registry.set_gpu_id(2)

        assert registry._get_gpu_processes() == {111, 333}

    def test_nvml_process_detection_falls_back_without_util_api(self, monkeypatch):
        fake_pynvml = types.SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda idx: f"gpu-{idx}",
            nvmlDeviceGetComputeRunningProcesses_v3=lambda handle: [FakeProcess(111), FakeProcess(222)],
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

        registry = ExperimentRegistry(enable_process_detection=True)
        registry.set_gpu_id(0)

        assert registry._get_gpu_processes() == {111, 222}

    def test_status_excludes_filler_pids(self, monkeypatch):
        registry = ExperimentRegistry(enable_process_detection=True)
        monkeypatch.setattr(registry, "cleanup_stale", lambda: 0)
        monkeypatch.setattr(registry, "_get_gpu_processes", lambda: {1001, 2002})

        status = registry.get_status(current_gpu_util_pct=50.0, filler_pids={2002})

        assert status.active is True
        assert status.active_pids == [1001]

    def test_status_ignores_allocated_but_idle_processes(self, monkeypatch):
        registry = ExperimentRegistry(enable_process_detection=True)
        monkeypatch.setattr(registry, "cleanup_stale", lambda: 0)
        monkeypatch.setattr(registry, "_get_gpu_processes", lambda: {1001})

        status = registry.get_status(current_gpu_util_pct=0.0, filler_pids=set())

        assert status.active is False
        assert status.active_pids == []

    def test_status_requires_busy_selected_gpu_for_auto_detect(self, monkeypatch):
        registry = ExperimentRegistry(enable_process_detection=True)
        monkeypatch.setattr(registry, "cleanup_stale", lambda: 0)
        monkeypatch.setattr(registry, "_get_gpu_processes", lambda: {1001})

        status = registry.get_status(current_gpu_util_pct=5.0, filler_pids=set())

        assert status.active is False
        assert status.active_pids == []

    def test_status_excludes_namespace_pid_aliases(self, monkeypatch):
        registry = ExperimentRegistry(enable_process_detection=True)
        monkeypatch.setattr(registry, "cleanup_stale", lambda: 0)
        monkeypatch.setattr(registry, "_get_gpu_processes", lambda: {9001, 7777})
        monkeypatch.setattr(registry, "_expand_pid_aliases", lambda pid: {pid, 9001} if pid == 42 else {pid})

        status = registry.get_status(current_gpu_util_pct=50.0, filler_pids={42})

        assert status.active is True
        assert status.active_pids == [7777]

    def test_status_excludes_worker_parent_family(self, monkeypatch):
        registry = ExperimentRegistry(enable_process_detection=True)
        monkeypatch.setattr(registry, "cleanup_stale", lambda: 0)
        monkeypatch.setattr(registry, "_get_gpu_processes", lambda: {9001, 7000, 7777})
        monkeypatch.setattr(registry, "_expand_pid_aliases", lambda pid: {pid, 9001} if pid == 42 else ({pid, 7000} if pid == 500 else {pid}))
        monkeypatch.setattr(registry, "_get_parent_pid", lambda pid: 500 if pid in {42, 9001} else None)

        status = registry.get_status(current_gpu_util_pct=50.0, filler_pids={42})

        assert status.active is True
        assert status.active_pids == [7777]
