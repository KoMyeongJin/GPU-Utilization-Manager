import subprocess
import os
import signal
from typing import List, Dict, Optional
from config import ManagerConfig, FillerLevel
from shared_memory import SharedMemoryManager


class FillerController:
    def __init__(self, config: ManagerConfig):
        self.config = config
        self.shm = SharedMemoryManager(config.shm_name)
        self._workers: Dict[int, subprocess.Popen[bytes]] = {}
        self._current_level = 0

    def initialize(self) -> bool:
        try:
            self.shm.create()
            return True
        except Exception:
            return False

    def cleanup(self) -> None:
        self.stop_all()
        self.shm.close()

    def ensure_workers(self, target_count: int, level_config: FillerLevel) -> int:
        current = len(self._workers)

        if current < target_count:
            for i in range(current, target_count):
                self._start_worker(i, level_config)
        elif current > target_count:
            to_stop = list(self._workers.keys())[target_count:]
            for pid in to_stop:
                self._stop_worker(pid)

        return len(self._workers)

    def _start_worker(self, worker_id: int, level_config: FillerLevel) -> Optional[int]:
        env = os.environ.copy()
        env.update({
            "GPU_MANAGER_SHM": self.config.shm_name,
            "FILLER_WORKER_ID": str(worker_id),
            "FILLER_BATCH_SIZE": str(level_config.batch_size),
            "FILLER_STREAMS": str(level_config.streams),
            "FILLER_SLEEP_MS": str(level_config.sleep_ms),
            "CUDA_VISIBLE_DEVICES": str(self.config.gpu_id),
        })

        script_path = os.path.join(os.path.dirname(__file__), "filler_worker.py")

        try:
            proc = subprocess.Popen(
                ["python", script_path],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            self._workers[proc.pid] = proc
            return proc.pid
        except Exception:
            return None

    def _stop_worker(self, pid: int) -> bool:
        if pid not in self._workers:
            return False

        proc = self._workers[pid]
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        del self._workers[pid]
        return True

    def stop_all(self) -> None:
        for pid in list(self._workers.keys()):
            self._stop_worker(pid)

    def apply_level(self, level: int) -> bool:
        if level not in self.config.filler.levels:
            return False

        level_config = self.config.filler.levels[level]
        self.shm.set_level(level)
        self.ensure_workers(level_config.workers, level_config)
        if level == 0:
            self.shm.pause()
        else:
            self.shm.resume()
        self._current_level = level

        return True

    def get_worker_count(self) -> int:
        return len(self._workers)

    def get_active_pids(self) -> List[int]:
        return list(self._workers.keys())

    def pause_all(self) -> None:
        self.shm.pause()

    def resume_all(self) -> None:
        self.shm.resume()
