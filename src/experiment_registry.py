"""Experiment registry for detecting active experiments."""

import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from threading import Lock


@dataclass
class ExperimentInfo:
    """Information about a registered experiment."""
    pid: int
    name: str
    start_time: float
    last_heartbeat: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExperimentStatus:
    """Current experiment status."""
    active: bool
    active_pids: List[int]
    estimated_load_pct: float
    experiments: List[ExperimentInfo]


class ExperimentRegistry:
    """Registry for tracking active experiments."""

    def __init__(self, heartbeat_timeout_sec: float = 30.0, enable_process_detection: bool = True):
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.enable_process_detection = enable_process_detection
        self._experiments: Dict[int, ExperimentInfo] = {}
        self._lock = Lock()
        self._gpu_id: int = 0

    def set_gpu_id(self, gpu_id: int) -> None:
        self._gpu_id = gpu_id

    def register(self, pid: int, name: str = "", metadata: Optional[Dict] = None) -> None:
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
            result = subprocess.run(
                ["nvidia-smi", "pmon", "-c", "1"],
                capture_output=True,
                text=True,
                timeout=5
            )

            pids = set()
            for line in result.stdout.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        pid = int(parts[3])
                        if pid > 0:
                            pids.add(pid)
                    except ValueError:
                        continue
            return pids
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return set()

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

    def get_status(self) -> ExperimentStatus:
        self.cleanup_stale()

        active_pids = []
        experiments = []

        if self._experiments:
            with self._lock:
                active_pids = list(self._experiments.keys())
                experiments = list(self._experiments.values())

        if self.enable_process_detection and not active_pids:
            gpu_pids = self._get_gpu_processes()
            filler_pids = self._get_filler_pids()
            non_filler_pids = gpu_pids - filler_pids

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

    def start(self, name: str = "", metadata: Optional[Dict] = None) -> bool:
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
