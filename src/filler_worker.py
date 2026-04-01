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
        self.sublevels_per_major = int(
            os.environ.get("FILLER_SUBLEVELS_PER_MAJOR", "1")
        )

        self.device = torch.device("cuda:0")
        self._running = True
        self._paused = False
        self.shm_client = None
        self._buffer_size = None
        self._compute_slots = []
        self._stream_pool = []

        # NEW: Diagnostic fields
        self.estimated_kv_cache_mb = 0.0
        self._last_phase = "PREFILL"
        self._poll_count = 0
        self._compute_kv_cache_estimate()

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
        interpolated = size_map[major_level] + ratio * (
            size_map[next_level] - size_map[major_level]
        )
        return int(interpolated)

    def _compute_gemm(self, size: int) -> torch.Tensor:
        a = torch.randn(size, size, device=self.device, dtype=torch.float16)
        b = torch.randn(size, size, device=self.device, dtype=torch.float16)
        return torch.matmul(a, b)

    def _active_stream_count(self) -> int:
        return max(1, self.num_streams)

    def _use_real_cuda_streams(self) -> bool:
        return torch.cuda.is_available() and self._active_stream_count() > 1

    def _build_stream_pool(self):
        if not self._use_real_cuda_streams():
            return []
        return [
            torch.cuda.Stream(device=self.device)
            for _ in range(self._active_stream_count())
        ]

    def _allocate_work_item(self, size: int, stream):
        dtype = torch.float16
        a = torch.randn(size, size, device=self.device, dtype=dtype)
        b = torch.randn(size, size, device=self.device, dtype=dtype)
        out = torch.empty(size, size, device=self.device, dtype=dtype)
        return {
            "stream": stream,
            "a": a,
            "b": b,
            "out": out,
        }

    def _ensure_compute_resources(self, size: int):
        needs_rebuild = (
            self._buffer_size != size
            or len(self._compute_slots) != self._active_stream_count()
        )
        if not needs_rebuild:
            return self._compute_slots

        self._stream_pool = self._build_stream_pool()
        streams = self._stream_pool if self._stream_pool else [None]
        self._compute_slots = [
            self._allocate_work_item(size, stream) for stream in streams
        ]
        self._buffer_size = size
        return self._compute_slots

    def _launch_gemm(self, work_item):
        stream = work_item["stream"]
        if stream is None:
            torch.mm(work_item["a"], work_item["b"], out=work_item["out"])
            return

        with torch.cuda.stream(stream):
            torch.mm(work_item["a"], work_item["b"], out=work_item["out"])

    def _synchronize_device(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _dispatch_compute(self, size: int):
        work_items = self._ensure_compute_resources(size)
        launch_count = max(1, self.batch_size)
        for launch_idx in range(launch_count):
            work_item = work_items[launch_idx % len(work_items)]
            self._launch_gemm(work_item)
        self._synchronize_device()

    def _compute_kv_cache_estimate(self):
        """Estimate KV cache memory pressure for GEMM operation.

        Formula from research: KV_bytes = 2 × L × H_kv × D × S × B × bytes_per_element
        For GEMM: approximate as matrix multiply with sequence length proxy.
        """
        max_step = 8 * self.sublevels_per_major
        matrix_size = self._matrix_size_for_step(max_step)

        # Assume FP16 KV cache (2 bytes per element)
        # Matrix represents approximation of KV cache size
        # Estimate: KV_bytes ≈ matrix_size^2 × 16 bytes (FP16 for K and V)
        self.estimated_kv_cache_mb = (matrix_size**2 * 16) / (1024 * 1024)

    def _detect_phase(self) -> str:
        """Detect GPU phase: 'prefill' (compute-bound) or 'decode' (memory-bound).

        Heuristic: kernel launch rate
        - High launch rate → prefill (many tokens per batch)
        - Low launch rate → decode (few tokens, high latency)
        """
        # Estimate kernel launches per second from configuration
        kernel_launches_per_second = self.num_streams * (1000 / max(self.sleep_ms, 1))

        # If <10 launches/sec, likely decode phase
        if kernel_launches_per_second < 10:
            return "DECODE"
        else:
            return "PREFILL"

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

            self._dispatch_compute(matrix_size)

            if self.sleep_ms > 0:
                time.sleep(self.sleep_ms / 1000.0)

            iteration += 1

            # NEW: Diagnostic logging
            self._poll_count += 1
            phase = self._detect_phase()
            if phase != self._last_phase:
                print(
                    f"[Worker {self.worker_id}] Phase change: {self._last_phase} → {phase}"
                )
                self._last_phase = phase

            # Log estimated memory usage periodically
            if self._poll_count % 1000 == 0:
                try:
                    with open(f"/tmp/filler_worker_{self.worker_id}.log", "a") as f:
                        f.write(
                            f"{time.time():.3f} "
                            f"Step={target_step} "
                            f"Phase={phase} "
                            f"EstimatedKV={self.estimated_kv_cache_mb:.1f}MB "
                            f"BatchSize={self.batch_size} "
                            f"Streams={self.num_streams}\n"
                        )
                except Exception:
                    pass

        if self.shm_client:
            self.shm_client.close()


def main():
    worker = FillerWorker()
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
