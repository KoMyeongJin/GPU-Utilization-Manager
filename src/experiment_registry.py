"""Experiment registry for detecting active experiments."""

import subprocess
import importlib
import time
from dataclasses import dataclass, field
from typing import Any, List, Dict, Set, Optional
from threading import Lock


@dataclass
class ExperimentInfo:
    """Information about a registered experiment."""
    pid: int
    name: str
    start_time: float
    last_heartbeat: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentStatus:
    """Current experiment status."""
    active: bool
    active_pids: List[int]
    estimated_load_pct: float
    experiments: List[ExperimentInfo]


class ExperimentRegistry:
    """Registry for tracking active experiments."""

    def __init__(
        self,
        heartbeat_timeout_sec: float = 30.0,
        enable_process_detection: bool = True,
        auto_detect_min_gpu_util_pct: float = 10.0,
    ):
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.enable_process_detection = enable_process_detection
        self.auto_detect_min_gpu_util_pct = auto_detect_min_gpu_util_pct
        self._experiments: Dict[int, ExperimentInfo] = {}
        self._lock = Lock()
        self._gpu_id: int = 0

    def set_gpu_id(self, gpu_id: int) -> None:
        self._gpu_id = gpu_id

    def register(self, pid: int, name: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()
        with self._lock:
            self._experiments[pid] = ExperimentInfo(
                pid=pid,
                name=name or f"experiment_{pid}",
                start_time=now,
                last_heartbeat=now,
                metadata=metadata or {}
            )

    def heartbeat(self, pid: int) -> bool:
        with self._lock:
            if pid in self._experiments:
                self._experiments[pid].last_heartbeat = time.time()
                return True
            return False

    def unregister(self, pid: int) -> bool:
        with self._lock:
            if pid in self._experiments:
                del self._experiments[pid]
                return True
            return False

    def _is_alive(self, pid: int) -> bool:
        try:
            subprocess.run(
                ["kill", "-0", str(pid)],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def _get_gpu_processes(self) -> Set[int]:
        try:
            pids = set()
            try:
                pynvml = importlib.import_module("pynvml")
            except ImportError:
                return set()

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_id)
                process_lists = []

                for attr in (
                    "nvmlDeviceGetComputeRunningProcesses_v3",
                    "nvmlDeviceGetComputeRunningProcesses_v2",
                    "nvmlDeviceGetComputeRunningProcesses",
                    "nvmlDeviceGetGraphicsRunningProcesses_v3",
                    "nvmlDeviceGetGraphicsRunningProcesses_v2",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                ):
                    getter = getattr(pynvml, attr, None)
                    if getter is None:
                        continue
                    try:
                        process_lists.extend(getter(handle) or [])
                    except Exception:
                        continue

                for process in process_lists:
                    pid = getattr(process, "pid", None)
                    if isinstance(pid, int) and pid > 0:
                        pids.add(pid)

                active_pids = self._get_active_gpu_processes_by_utilization(pynvml, handle, pids)
                if active_pids is not None:
                    return active_pids
            finally:
                pynvml.nvmlShutdown()

            return pids
        except Exception:
            return set()

    def _get_active_gpu_processes_by_utilization(self, pynvml: Any, handle: Any, candidate_pids: Set[int]) -> Optional[Set[int]]:
        getter = getattr(pynvml, "nvmlDeviceGetProcessUtilization", None)
        if getter is None:
            return None

        try:
            samples = getter(handle, 0)
        except TypeError:
            try:
                samples = getter(handle, lastSeenTimeStamp=0)
            except Exception:
                return None
        except Exception:
            return None

        active_pids: Set[int] = set()
        for sample in samples or []:
            pid = getattr(sample, "pid", None)
            if not isinstance(pid, int) or pid not in candidate_pids:
                continue

            sm_util = int(getattr(sample, "smUtil", 0) or 0)
            mem_util = int(getattr(sample, "memUtil", 0) or 0)
            enc_util = int(getattr(sample, "encUtil", 0) or 0)
            dec_util = int(getattr(sample, "decUtil", 0) or 0)

            if sm_util > 0 or mem_util > 0 or enc_util > 0 or dec_util > 0:
                active_pids.add(pid)

        return active_pids

    def _get_filler_pids(self) -> Set[int]:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "filler_worker.py"],
                capture_output=True,
                text=True,
                timeout=2
            )

            pids = set()
            for line in result.stdout.split("\n"):
                if line.strip():
                    try:
                        pids.add(int(line.strip()))
                    except ValueError:
                        continue
            return pids
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return set()

    def _expand_pid_aliases(self, pid: int) -> Set[int]:
        aliases = {pid}
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("NSpid:"):
                        parts = line.split()[1:]
                        for value in parts:
                            try:
                                aliases.add(int(value))
                            except ValueError:
                                continue
                        break
        except OSError:
            pass
        return aliases

    def _normalize_pid_set(self, pids: Set[int]) -> Set[int]:
        normalized: Set[int] = set()
        for pid in pids:
            normalized.update(self._expand_pid_aliases(pid))
        return normalized

    def _get_parent_pid(self, pid: int) -> Optional[int]:
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        value = line.split()[1]
                        parent_pid = int(value)
                        return parent_pid if parent_pid > 0 else None
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _expand_process_family(self, pids: Set[int]) -> Set[int]:
        family = self._normalize_pid_set(pids)
        queue = list(family)
        seen = set(family)

        while queue:
            current_pid = queue.pop()
            parent_pid = self._get_parent_pid(current_pid)
            if parent_pid is None or parent_pid in seen:
                continue

            parent_aliases = self._expand_pid_aliases(parent_pid)
            new_aliases = parent_aliases - seen
            if new_aliases:
                seen.update(new_aliases)
                queue.extend(new_aliases)

        return seen

    def cleanup_stale(self) -> int:
        removed = 0
        now = time.time()

        with self._lock:
            stale_pids = []
            for pid, info in self._experiments.items():
                if now - info.last_heartbeat > self.heartbeat_timeout_sec:
                    stale_pids.append(pid)
                elif not self._is_alive(pid):
                    stale_pids.append(pid)

            for pid in stale_pids:
                del self._experiments[pid]
                removed += 1

        return removed

    def get_status(
        self,
        current_gpu_util_pct: Optional[float] = None,
        filler_pids: Optional[Set[int]] = None,
    ) -> ExperimentStatus:
        self.cleanup_stale()

        active_pids = []
        experiments = []

        if self._experiments:
            with self._lock:
                active_pids = list(self._experiments.keys())
                experiments = list(self._experiments.values())

        if self.enable_process_detection and not active_pids:
            if current_gpu_util_pct is not None and current_gpu_util_pct < self.auto_detect_min_gpu_util_pct:
                return ExperimentStatus(
                    active=False,
                    active_pids=[],
                    estimated_load_pct=0.0,
                    experiments=experiments,
                )

            gpu_pids = self._get_gpu_processes()
            effective_filler_pids = filler_pids if filler_pids is not None else self._get_filler_pids()
            normalized_filler_pids = self._expand_process_family(effective_filler_pids)
            non_filler_pids = gpu_pids - normalized_filler_pids

            if non_filler_pids:
                active_pids = list(non_filler_pids)

        estimated_load = 0.0
        if active_pids:
            estimated_load = 50.0

        return ExperimentStatus(
            active=len(active_pids) > 0,
            active_pids=active_pids,
            estimated_load_pct=estimated_load,
            experiments=experiments
        )

    def is_active(self) -> bool:
        return self.get_status().active

    def get_active_pids(self) -> List[int]:
        return self.get_status().active_pids

    def get_experiment_count(self) -> int:
        with self._lock:
            return len(self._experiments)


class ExperimentClient:
    """Client for experiment registration."""

    def __init__(self, manager_socket: str = "/tmp/gpu_manager.sock"):
        self.manager_socket = manager_socket
        self._pid = None

    def start(self, name: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        import socket
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(self.manager_socket)

            import json
            msg = json.dumps({
                "action": "start",
                "pid": None,
                "name": name,
                "metadata": metadata or {}
            })
            sock.sendall(msg.encode())
            sock.close()

            self._pid = None
            return True
        except (socket.error, ConnectionRefusedError, FileNotFoundError):
            return False

    def heartbeat(self) -> bool:
        import socket
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(self.manager_socket)

            import json
            msg = json.dumps({"action": "heartbeat", "pid": None})
            sock.sendall(msg.encode())
            sock.close()
            return True
        except (socket.error, ConnectionRefusedError, FileNotFoundError):
            return False

    def stop(self) -> bool:
        import socket
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(self.manager_socket)

            import json
            msg = json.dumps({"action": "stop", "pid": None})
            sock.sendall(msg.encode())
            sock.close()
            return True
        except (socket.error, ConnectionRefusedError, FileNotFoundError):
            return False
