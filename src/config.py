"""Configuration management for GPU Utilization Manager."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional
import os
import yaml


MATRIX_SIZES_BY_MAJOR_LEVEL: Dict[int, int] = {
    0: 0,
    1: 1024,
    2: 2048,
    3: 4096,
    4: 6144,
    5: 8192,
    6: 10240,
    7: 13312,
    8: 16384,
}


def matrix_size_for_step(
    step: int,
    sublevels_per_major: int,
    max_major_level: int = 8,
    size_map: Optional[Dict[int, int]] = None,
) -> int:
    size_map = size_map or MATRIX_SIZES_BY_MAJOR_LEVEL
    step_size = max(1, sublevels_per_major)
    capped_major_level = min(max_major_level, max(size_map))
    clamped_step = max(0, min(step, capped_major_level * step_size))
    if clamped_step == 0:
        return size_map[0]

    active_step = clamped_step - 1
    major_level = min(capped_major_level, 1 + (active_step // step_size))
    sublevel = active_step % step_size

    if step_size == 1 or major_level >= capped_major_level or sublevel == 0:
        return size_map[major_level]

    next_level = min(major_level + 1, capped_major_level)
    ratio = sublevel / step_size
    start = size_map[major_level]
    end = size_map[next_level]
    interpolated_size = ((1 - ratio) * start) + (ratio * end)
    return int(round(interpolated_size))


@dataclass
class Thresholds:
    """GPU utilization thresholds for scaling decisions."""

    low_boost_pct: float = 65.0
    target_floor_pct: float = 70.0
    healthy_high_pct: float = 88.0
    reduce_pct: float = 92.0
    emergency_reduce_pct: float = 95.0
    critical_pause_pct: float = 98.0


@dataclass
class HysteresisConfig:
    """Hysteresis configuration to prevent flapping."""

    consecutive_polls: int = 2
    min_dwell_sec: float = 8.0


@dataclass
class FillerLevel:
    """Configuration for a single filler intensity level."""

    workers: int = 0
    batch_size: int = 0
    streams: int = 0
    sleep_ms: float = 0.0


def level_work_proxy(level: FillerLevel, matrix_size: int) -> float:
    if (
        level.workers <= 0
        or level.batch_size <= 0
        or level.streams <= 0
        or matrix_size <= 0
    ):
        return 0.0

    sleep_divisor = 1.0 + max(0.0, level.sleep_ms)
    return (
        level.workers
        * level.batch_size
        * level.streams
        * (matrix_size**3)
        / sleep_divisor
    )


@dataclass
class FillerConfig:
    """Filler workload configuration."""

    sublevels_per_major: int = 1
    levels: Dict[int, FillerLevel] = field(
        default_factory=lambda: {
            0: FillerLevel(workers=0, batch_size=0, streams=0, sleep_ms=100),
            1: FillerLevel(workers=1, batch_size=6, streams=1, sleep_ms=40),
            2: FillerLevel(workers=1, batch_size=10, streams=1, sleep_ms=32),
            3: FillerLevel(workers=1, batch_size=16, streams=1, sleep_ms=24),
            4: FillerLevel(workers=1, batch_size=20, streams=1, sleep_ms=16),
            5: FillerLevel(workers=1, batch_size=20, streams=2, sleep_ms=18),
            6: FillerLevel(workers=1, batch_size=24, streams=2, sleep_ms=15),
            7: FillerLevel(workers=1, batch_size=26, streams=2, sleep_ms=12),
            8: FillerLevel(workers=1, batch_size=32, streams=2, sleep_ms=8),
        }
    )
    mps_caps_no_experiment: List[int] = field(
        default_factory=lambda: [0, 40, 60, 80, 90, 92, 94, 96, 98]
    )
    mps_caps_experiment_active: List[int] = field(
        default_factory=lambda: [0, 5, 10, 20, 30, 35, 40, 45, 50]
    )

    @property
    def max_major_level(self) -> int:
        return max(self.levels)

    @property
    def max_step(self) -> int:
        return self.max_major_level * self.sublevels_per_major

    def step_for_major_level(self, major_level: int) -> int:
        if major_level <= 0:
            return 0
        return 1 + ((major_level - 1) * self.sublevels_per_major)

    def split_step(self, step: int) -> tuple[int, int]:
        clamped_step = max(0, min(step, self.max_step))
        if clamped_step == 0:
            return (0, 0)
        active_step = clamped_step - 1
        major_level = 1 + (active_step // self.sublevels_per_major)
        sublevel = active_step % self.sublevels_per_major
        return (min(major_level, self.max_major_level), sublevel)

    def interpolate_mps_cap(self, step: int, experiment_active: bool) -> int:
        caps = (
            self.mps_caps_experiment_active
            if experiment_active
            else self.mps_caps_no_experiment
        )
        major_level, sublevel = self.split_step(step)
        if self.sublevels_per_major == 1 or major_level >= self.max_major_level:
            return caps[major_level]
        next_level = min(major_level + 1, self.max_major_level)
        ratio = sublevel / self.sublevels_per_major
        interpolated = caps[major_level] + ratio * (
            caps[next_level] - caps[major_level]
        )
        return int(round(interpolated))

    def matrix_size_for_step(self, step: int) -> int:
        return matrix_size_for_step(
            step, self.sublevels_per_major, self.max_major_level
        )

    def major_level_work_proxy(self, major_level: int) -> float:
        level = self.levels[major_level]
        step = self.step_for_major_level(major_level)
        return level_work_proxy(level, self.matrix_size_for_step(step))

    @property
    def first_active_major_level(self) -> int:
        active_major_levels = [
            major_level
            for major_level in sorted(self.levels)
            if self.major_level_work_proxy(major_level) > 0
        ]
        if not active_major_levels:
            return self.max_major_level
        return active_major_levels[0]

    @property
    def first_active_step(self) -> int:
        return self.step_for_major_level(self.first_active_major_level)

    def target_work_proxy(self, step: int) -> float:
        clamped_step = max(0, min(step, self.max_step))
        if clamped_step < self.first_active_step:
            return 0.0

        start_work = self.major_level_work_proxy(self.first_active_major_level)
        end_work = self.major_level_work_proxy(self.max_major_level)
        if (
            clamped_step >= self.max_step
            or self.first_active_step >= self.max_step
            or start_work <= 0
            or end_work <= 0
        ):
            return end_work

        progress = (clamped_step - self.first_active_step) / (
            self.max_step - self.first_active_step
        )
        return start_work + (progress * (end_work - start_work))

    def interpolate_level_config(self, step: int) -> FillerLevel:
        clamped_step = max(0, min(step, self.max_step))
        if clamped_step < self.first_active_step:
            base = self.levels[0]
            return FillerLevel(
                workers=base.workers,
                batch_size=base.batch_size,
                streams=base.streams,
                sleep_ms=base.sleep_ms,
            )

        major_level, sublevel = self.split_step(clamped_step)
        base = self.levels[major_level]
        next_level = min(major_level + 1, self.max_major_level)
        nxt = self.levels[next_level]
        ratio = 0.0
        if self.sublevels_per_major > 1 and major_level < self.max_major_level:
            ratio = sublevel / self.sublevels_per_major
        return self._blend_level_configs(clamped_step, base, nxt, ratio)

    def _blend_level_configs(
        self, step: int, start: FillerLevel, end: FillerLevel, ratio: float
    ) -> FillerLevel:
        workers = start.workers
        streams = start.streams
        batch_size = int(
            round(start.batch_size + ratio * (end.batch_size - start.batch_size))
        )
        sleep_ms = start.sleep_ms + ratio * (end.sleep_ms - start.sleep_ms)

        return FillerLevel(
            workers=workers,
            batch_size=batch_size,
            streams=streams,
            sleep_ms=sleep_ms,
        )


@dataclass
class ManagerConfig:
    """Main GPU Utilization Manager configuration."""

    # GPU settings
    gpu_id: int = 0
    poll_interval_sec: float = 2.0
    target_util_pct: float = 70.0
    enable_mps: bool = True

    # Thresholds
    thresholds: Thresholds = field(default_factory=Thresholds)

    # Hysteresis
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)

    # Filler configuration
    filler: FillerConfig = field(default_factory=FillerConfig)

    # Shared memory
    shm_name: str = "/gpu_manager_shm"

    # Socket
    socket_path: str = "/tmp/gpu_manager.sock"

    # MPS
    mps_pipe_dir: str = "/tmp/nvidia-mps"
    mps_log_dir: str = "/tmp/nvidia-mps-log"

    # Experiment detection
    experiment_heartbeat_timeout_sec: float = 30.0
    enable_process_detection: bool = True

    # Safety
    max_filler_memory_gb: float = 20.0

    @classmethod
    def from_yaml(cls, path: str) -> "ManagerConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManagerConfig":
        """Create config from dictionary."""
        config = cls()

        if "gpu_id" in data:
            config.gpu_id = data["gpu_id"]
        if "poll_interval_sec" in data:
            config.poll_interval_sec = data["poll_interval_sec"]
        if "target_util_pct" in data:
            config.target_util_pct = data["target_util_pct"]
        if "enable_mps" in data:
            config.enable_mps = bool(data["enable_mps"])

        if "thresholds" in data:
            config.thresholds = Thresholds(**data["thresholds"])

        if "hysteresis" in data:
            config.hysteresis = HysteresisConfig(**data["hysteresis"])

        if "filler" in data:
            filler_data = data["filler"]
            if "sublevels_per_major" in filler_data:
                config.filler.sublevels_per_major = int(
                    filler_data["sublevels_per_major"]
                )
            if "levels" in filler_data:
                levels = {}
                for level_id, level_data in filler_data["levels"].items():
                    levels[int(level_id)] = FillerLevel(**level_data)
                config.filler.levels = levels
            if "mps_caps_no_experiment" in filler_data:
                config.filler.mps_caps_no_experiment = filler_data[
                    "mps_caps_no_experiment"
                ]
            if "mps_caps_experiment_active" in filler_data:
                config.filler.mps_caps_experiment_active = filler_data[
                    "mps_caps_experiment_active"
                ]

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "gpu_id": self.gpu_id,
            "poll_interval_sec": self.poll_interval_sec,
            "target_util_pct": self.target_util_pct,
            "enable_mps": self.enable_mps,
            "thresholds": {
                "low_boost_pct": self.thresholds.low_boost_pct,
                "target_floor_pct": self.thresholds.target_floor_pct,
                "healthy_high_pct": self.thresholds.healthy_high_pct,
                "reduce_pct": self.thresholds.reduce_pct,
                "emergency_reduce_pct": self.thresholds.emergency_reduce_pct,
                "critical_pause_pct": self.thresholds.critical_pause_pct,
            },
            "hysteresis": {
                "consecutive_polls": self.hysteresis.consecutive_polls,
                "min_dwell_sec": self.hysteresis.min_dwell_sec,
            },
            "filler": {
                "sublevels_per_major": self.filler.sublevels_per_major,
                "levels": {
                    k: {
                        "workers": v.workers,
                        "batch_size": v.batch_size,
                        "streams": v.streams,
                        "sleep_ms": v.sleep_ms,
                    }
                    for k, v in self.filler.levels.items()
                },
                "mps_caps_no_experiment": self.filler.mps_caps_no_experiment,
                "mps_caps_experiment_active": self.filler.mps_caps_experiment_active,
            },
        }


def get_default_config() -> ManagerConfig:
    """Get default configuration."""
    return ManagerConfig()


def load_config(path: Optional[str] = None) -> ManagerConfig:
    """Load configuration from file or environment."""
    if path and os.path.exists(path):
        return ManagerConfig.from_yaml(path)

    # Check environment variables
    config = ManagerConfig()

    if "GPU_MANAGER_GPU_ID" in os.environ:
        config.gpu_id = int(os.environ["GPU_MANAGER_GPU_ID"])
    if "GPU_MANAGER_POLL_INTERVAL" in os.environ:
        config.poll_interval_sec = float(os.environ["GPU_MANAGER_POLL_INTERVAL"])
    if "GPU_MANAGER_TARGET_UTIL" in os.environ:
        config.target_util_pct = float(os.environ["GPU_MANAGER_TARGET_UTIL"])

    return config
