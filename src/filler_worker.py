import os
import sys
import time
import torch
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class FillerWorker:

    def __init__(self):
        self.worker_id = int(os.environ.get("FILLER_WORKER_ID", "0"))
        self.batch_size = int(os.environ.get("FILLER_BATCH_SIZE", "64"))
        self.num_streams = int(os.environ.get("FILLER_STREAMS", "2"))
        self.sleep_ms = float(os.environ.get("FILLER_SLEEP_MS", "10"))
        self.shm_name = os.environ.get("GPU_MANAGER_SHM", "/gpu_manager_shm")
        self.sublevels_per_major = int(os.environ.get("FILLER_SUBLEVELS_PER_MAJOR", "1"))

        self.device = torch.device("cuda:0")
        self._running = True
        self._paused = False
        self.shm_client = None

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def handle_signal(signum, frame):
            self._running = False

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

    def _init_shared_memory(self):
        try:
            try:
                from .shared_memory import SharedMemoryClient
            except ImportError:
                from src.shared_memory import SharedMemoryClient
            self.shm_client = SharedMemoryClient(self.shm_name)
            if not self.shm_client.attach():
                self.shm_client = None
        except Exception:
            self.shm_client = None

    def _check_command(self):
        if self.shm_client is None:
            return

        try:
            cmd, param = self.shm_client.get_command()

            if cmd.name == "PAUSE":
                self._paused = True
            elif cmd.name == "RESUME":
                self._paused = False
            elif cmd.name == "SHUTDOWN":
                self._running = False

            self.shm_client.clear_command()
        except Exception:
            pass

    def _get_target_step(self) -> int:
        if self.shm_client is None:
            return 2 * self.sublevels_per_major

        try:
            return self.shm_client.get_target_step()
        except Exception:
            return 2 * self.sublevels_per_major

    def _matrix_size_for_step(self, step: int) -> int:
        size_map = {
            0: 0,
            1: 1024,
            2: 2048,
            3: 4096,
            4: 8192,
            5: 10240,
            6: 12288,
            7: 14336,
            8: 16384,
        }
        major_level = min(step // self.sublevels_per_major, 8)
        sublevel = step % self.sublevels_per_major
        if self.sublevels_per_major == 1 or major_level >= 8:
            return size_map[major_level]
        next_level = min(major_level + 1, 8)
        ratio = sublevel / self.sublevels_per_major
        interpolated = size_map[major_level] + ratio * (size_map[next_level] - size_map[major_level])
        return int(interpolated)

    def _compute_gemm(self, size: int) -> torch.Tensor:
        a = torch.randn(size, size, device=self.device, dtype=torch.float16)
        b = torch.randn(size, size, device=self.device, dtype=torch.float16)
        return torch.matmul(a, b)

    def run(self):
        self._init_shared_memory()

        torch.cuda.set_device(self.device)

        iteration = 0

        while self._running:
            self._check_command()

            if self._paused:
                time.sleep(0.1)
                continue

            target_step = self._get_target_step()
            matrix_size = self._matrix_size_for_step(target_step)

            if matrix_size == 0:
                time.sleep(0.1)
                continue

            for i in range(self.num_streams):
                _ = i
                self._compute_gemm(matrix_size)

            torch.cuda.synchronize()

            if self.sleep_ms > 0:
                time.sleep(self.sleep_ms / 1000.0)

            iteration += 1

        if self.shm_client:
            self.shm_client.close()


def main():
    worker = FillerWorker()
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
