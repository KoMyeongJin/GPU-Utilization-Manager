"""Configuration management for GPU Utilization Manager."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import yaml


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


@dataclass
class FillerConfig:
    """Filler workload configuration."""
    sublevels_per_major: int = 1
    levels: Dict[int, FillerLevel] = field(default_factory=lambda: {
        0: FillerLevel(workers=0, batch_size=0, streams=0, sleep_ms=100),
        1: FillerLevel(workers=1, batch_size=32, streams=1, sleep_ms=20),
        2: FillerLevel(workers=1, batch_size=64, streams=2, sleep_ms=10),
        3: FillerLevel(workers=2, batch_size=96, streams=2, sleep_ms=5),
        4: FillerLevel(workers=2, batch_size=128, streams=4, sleep_ms=0),
        5: FillerLevel(workers=3, batch_size=160, streams=4, sleep_ms=0),
        6: FillerLevel(workers=3, batch_size=192, streams=5, sleep_ms=0),
        7: FillerLevel(workers=4, batch_size=224, streams=6, sleep_ms=0),
        8: FillerLevel(workers=4, batch_size=256, streams=8, sleep_ms=0),
    })
    mps_caps_no_experiment: List[int] = field(default_factory=lambda: [0, 40, 60, 80, 90, 92, 94, 96, 98])
    mps_caps_experiment_active: List[int] = field(default_factory=lambda: [0, 5, 10, 20, 30, 35, 40, 45, 50])

    @property
    def max_major_level(self) -> int:
        return max(self.levels)

    @property
    def max_step(self) -> int:
        return self.max_major_level * self.sublevels_per_major

    def split_step(self, step: int) -> tuple[int, int]:
        clamped_step = max(0, min(step, self.max_step))
        return divmod(clamped_step, self.sublevels_per_major)

    def interpolate_mps_cap(self, step: int, experiment_active: bool) -> int:
        caps = self.mps_caps_experiment_active if experiment_active else self.mps_caps_no_experiment
        major_level, sublevel = self.split_step(step)
        if self.sublevels_per_major == 1 or major_level >= self.max_major_level:
            return caps[major_level]
        next_level = min(major_level + 1, self.max_major_level)
        ratio = sublevel / self.sublevels_per_major
        interpolated = caps[major_level] + ratio * (caps[next_level] - caps[major_level])
        return int(round(interpolated))

    def interpolate_level_config(self, step: int) -> FillerLevel:
        major_level, sublevel = self.split_step(step)
        base = self.levels[major_level]
        if self.sublevels_per_major == 1 or major_level >= self.max_major_level:
            return FillerLevel(
                workers=base.workers,
                batch_size=base.batch_size,
                streams=base.streams,
                sleep_ms=base.sleep_ms,
            )

        next_level = min(major_level + 1, self.max_major_level)
        nxt = self.levels[next_level]
        ratio = sublevel / self.sublevels_per_major
        return self._blend_level_configs(base, nxt, ratio)

    def _blend_level_configs(self, start: FillerLevel, end: FillerLevel, ratio: float) -> FillerLevel:
        workers = int(round(start.workers + ratio * (end.workers - start.workers)))
        batch_size = int(round(start.batch_size + ratio * (end.batch_size - start.batch_size)))
        streams = int(round(start.streams + ratio * (end.streams - start.streams)))
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
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManagerConfig":
        """Create config from dictionary."""
        config = cls()

        if 'gpu_id' in data:
            config.gpu_id = data['gpu_id']
        if 'poll_interval_sec' in data:
            config.poll_interval_sec = data['poll_interval_sec']
        if 'target_util_pct' in data:
            config.target_util_pct = data['target_util_pct']
        if 'enable_mps' in data:
            config.enable_mps = bool(data['enable_mps'])

        if 'thresholds' in data:
            config.thresholds = Thresholds(**data['thresholds'])

        if 'hysteresis' in data:
            config.hysteresis = HysteresisConfig(**data['hysteresis'])

        if 'filler' in data:
            filler_data = data['filler']
            if 'sublevels_per_major' in filler_data:
                config.filler.sublevels_per_major = int(filler_data['sublevels_per_major'])
            if 'levels' in filler_data:
                levels = {}
                for level_id, level_data in filler_data['levels'].items():
                    levels[int(level_id)] = FillerLevel(**level_data)
                config.filler.levels = levels
            if 'mps_caps_no_experiment' in filler_data:
                config.filler.mps_caps_no_experiment = filler_data['mps_caps_no_experiment']
            if 'mps_caps_experiment_active' in filler_data:
                config.filler.mps_caps_experiment_active = filler_data['mps_caps_experiment_active']

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'gpu_id': self.gpu_id,
            'poll_interval_sec': self.poll_interval_sec,
            'target_util_pct': self.target_util_pct,
            'enable_mps': self.enable_mps,
            'thresholds': {
                'low_boost_pct': self.thresholds.low_boost_pct,
                'target_floor_pct': self.thresholds.target_floor_pct,
                'healthy_high_pct': self.thresholds.healthy_high_pct,
                'reduce_pct': self.thresholds.reduce_pct,
                'emergency_reduce_pct': self.thresholds.emergency_reduce_pct,
                'critical_pause_pct': self.thresholds.critical_pause_pct,
            },
            'hysteresis': {
                'consecutive_polls': self.hysteresis.consecutive_polls,
                'min_dwell_sec': self.hysteresis.min_dwell_sec,
            },
            'filler': {
                'sublevels_per_major': self.filler.sublevels_per_major,
                'levels': {
                    k: {
                        'workers': v.workers,
                        'batch_size': v.batch_size,
                        'streams': v.streams,
                        'sleep_ms': v.sleep_ms,
                    }
                    for k, v in self.filler.levels.items()
                },
                'mps_caps_no_experiment': self.filler.mps_caps_no_experiment,
                'mps_caps_experiment_active': self.filler.mps_caps_experiment_active,
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

    if 'GPU_MANAGER_GPU_ID' in os.environ:
        config.gpu_id = int(os.environ['GPU_MANAGER_GPU_ID'])
    if 'GPU_MANAGER_POLL_INTERVAL' in os.environ:
        config.poll_interval_sec = float(os.environ['GPU_MANAGER_POLL_INTERVAL'])
    if 'GPU_MANAGER_TARGET_UTIL' in os.environ:
        config.target_util_pct = float(os.environ['GPU_MANAGER_TARGET_UTIL'])

    return config
