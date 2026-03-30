"""MPS adapter for thread percentage control."""

import subprocess
import os
from typing import Optional


class MPSAdapter:
    """Adapter for CUDA MPS control."""

    def __init__(self, pipe_dir: str = "/tmp/nvidia-mps", log_dir: str = "/tmp/nvidia-mps-log"):
        self.pipe_dir = pipe_dir
        self.log_dir = log_dir
        self._server_pid: Optional[int] = None

    def is_running(self) -> bool:
        try:
            result = subprocess.run(
                ["pgrep", "-x", "nvidia-cuda-mps-control"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def start(self) -> bool:
        if self.is_running():
            return True

        os.makedirs(self.pipe_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
        env["CUDA_MPS_LOG_DIRECTORY"] = self.log_dir

        try:
            subprocess.Popen(
                ["nvidia-cuda-mps-control", "-d"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            import time
            time.sleep(1)
            return self.is_running()
        except FileNotFoundError:
            return False

    def stop(self) -> bool:
        if not self.is_running():
            return True

        try:
            subprocess.run(
                ["echo", "quit"],
                stdout=subprocess.PIPE,
                timeout=2
            )
            subprocess.run(
                ["nvidia-cuda-mps-control"],
                input=b"quit\n",
                capture_output=True,
                timeout=5
            )
            return not self.is_running()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def set_active_thread_percentage(self, percentage: int) -> bool:
        """Set default active thread percentage for MPS clients."""
        if not self.is_running():
            return False

        percentage = max(1, min(100, percentage))

        try:
            result = subprocess.run(
                ["nvidia-cuda-mps-control"],
                input=f"set_default_active_thread_percentage {percentage}\n".encode(),
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def get_server_status(self) -> Optional[str]:
        if not self.is_running():
            return None

        try:
            result = subprocess.run(
                ["nvidia-cuda-mps-control"],
                input=b"get_server_status\n",
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def list_clients(self) -> list:
        if not self.is_running():
            return []

        try:
            result = subprocess.run(
                ["nvidia-cuda-mps-control"],
                input=b"ps\n",
                capture_output=True,
                text=True,
                timeout=5
            )

            clients = []
            for line in result.stdout.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    clients.append({
                        "pid": int(parts[0]),
                        "id": int(parts[1]),
                        "server": int(parts[2]),
                        "device": parts[3],
                        "namespace": parts[4],
                        "command": " ".join(parts[5:])
                    })
            return clients
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    def terminate_client(self, server_pid: int, client_pid: int) -> bool:
        if not self.is_running():
            return False

        try:
            result = subprocess.run(
                ["nvidia-cuda-mps-control"],
                input=f"terminate_client {server_pid} {client_pid}\n".encode(),
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def set_environment(self, percentage: Optional[int] = None) -> dict:
        """Get environment variables for MPS client."""
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
        env["CUDA_MPS_LOG_DIRECTORY"] = self.log_dir

        if percentage is not None:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)

        return env
