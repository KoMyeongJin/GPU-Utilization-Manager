import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.config import FillerLevel, ManagerConfig, level_work_proxy


def _legacy_interpolated_level(config: ManagerConfig, step: int) -> FillerLevel:
    filler = config.filler
    major_level, sublevel = filler.split_step(step)
    base = filler.levels[major_level]
    if (
        filler.sublevels_per_major == 1
        or major_level >= filler.max_major_level
        or sublevel == 0
    ):
        return FillerLevel(
            workers=base.workers,
            batch_size=base.batch_size,
            streams=base.streams,
            sleep_ms=base.sleep_ms,
        )

    next_level = filler.levels[min(major_level + 1, filler.max_major_level)]
    ratio = sublevel / filler.sublevels_per_major
    return FillerLevel(
        workers=base.workers,
        batch_size=int(
            round(base.batch_size + ratio * (next_level.batch_size - base.batch_size))
        ),
        streams=int(round(base.streams + ratio * (next_level.streams - base.streams))),
        sleep_ms=base.sleep_ms + ratio * (next_level.sleep_ms - base.sleep_ms),
    )


def _step_work_proxy(
    config: ManagerConfig, step: int, use_legacy: bool = False
) -> float:
    filler = config.filler
    level = (
        _legacy_interpolated_level(config, step)
        if use_legacy
        else filler.interpolate_level_config(step)
    )
    return level_work_proxy(level, filler.matrix_size_for_step(step))


class TestManagerConfig:
    def test_default_config(self):
        config = ManagerConfig()
        assert config.gpu_id == 0
        assert config.poll_interval_sec == 2.0
        assert config.target_util_pct == 70.0

    def test_thresholds(self):
        config = ManagerConfig()
        assert config.thresholds.low_boost_pct == 65.0
        assert config.thresholds.target_floor_pct == 70.0
        assert config.thresholds.emergency_reduce_pct == 95.0

    def test_filler_levels(self):
        config = ManagerConfig()
        assert 0 in config.filler.levels
        assert 8 in config.filler.levels
        assert config.filler.levels[0].workers == 0
        assert config.filler.levels[8].workers == 1

    def test_sublevels_default(self):
        config = ManagerConfig()
        assert config.filler.sublevels_per_major == 1
        assert config.filler.max_step == 8

    def test_split_step_and_interpolate_cap(self):
        config = ManagerConfig()
        config.filler.sublevels_per_major = 4
        assert config.filler.split_step(0) == (0, 0)
        assert config.filler.split_step(1) == (1, 0)
        assert config.filler.split_step(6) == (2, 1)
        assert config.filler.interpolate_mps_cap(6, False) == 65
        assert config.filler.interpolate_mps_cap(6, True) == 12

    def test_interpolate_level_config_preserves_active_ladder_endpoints(self):
        config = ManagerConfig()
        config.filler.sublevels_per_major = 4

        first_active_step = config.filler.first_active_step
        max_step = config.filler.max_step

        assert (
            config.filler.interpolate_level_config(first_active_step)
            == config.filler.levels[1]
        )
        assert (
            config.filler.interpolate_level_config(max_step) == config.filler.levels[8]
        )

    def test_target_work_proxy_uses_global_linear_active_ladder_curve(self):
        config = ManagerConfig()
        config.filler.sublevels_per_major = 4

        first_active_step = config.filler.first_active_step
        max_step = config.filler.max_step
        proxies = [
            config.filler.target_work_proxy(step)
            for step in range(first_active_step, max_step + 1)
        ]
        deltas = [proxies[idx + 1] - proxies[idx] for idx in range(len(proxies) - 1)]

        assert proxies[0] == pytest.approx(_step_work_proxy(config, first_active_step))
        assert proxies[-1] == pytest.approx(_step_work_proxy(config, max_step))
        assert max(deltas) == pytest.approx(min(deltas))

    def test_interpolate_level_config_keeps_workers_stable_within_major_interval(self):
        config = ManagerConfig()
        config.filler.sublevels_per_major = 4

        workers_by_step = {
            step: config.filler.interpolate_level_config(step).workers
            for step in (5, 6, 7, 8, 9)
        }

        assert workers_by_step[5] == 1
        assert workers_by_step[6] == 1
        assert workers_by_step[7] == 1
        assert workers_by_step[8] == 1
        assert workers_by_step[9] == 1

    def test_interpolate_level_config_keeps_streams_stable_within_major_interval(self):
        config = ManagerConfig()
        config.filler.sublevels_per_major = 4

        streams_by_step = {
            step: config.filler.interpolate_level_config(step).streams
            for step in (9, 10, 11, 12, 13, 17)
        }

        assert streams_by_step[9] == 1
        assert streams_by_step[10] == 1
        assert streams_by_step[11] == 1
        assert streams_by_step[12] == 1
        assert streams_by_step[13] == 1
        assert streams_by_step[17] == 2

    def test_level_zero_is_not_subdivided(self):
        config = ManagerConfig()
        config.filler.sublevels_per_major = 4

        assert config.filler.first_active_step == 1
        assert config.filler.interpolate_level_config(0) == config.filler.levels[0]
        assert config.filler.interpolate_level_config(1) == config.filler.levels[1]

    def test_interpolate_level_config_work_proxy_is_monotonic_across_full_ladder(self):
        config = ManagerConfig()
        config.filler.sublevels_per_major = 4

        proxies = [
            _step_work_proxy(config, step) for step in range(config.filler.max_step + 1)
        ]

        assert proxies == sorted(proxies)

    def test_total_work_driven_interpolation_is_non_decreasing_across_active_ladder(
        self,
    ):
        config = ManagerConfig()
        config.filler.sublevels_per_major = 4
        first_active_step = config.filler.first_active_step

        new_proxies = [
            _step_work_proxy(config, step) for step in range(config.filler.max_step + 1)
        ]
        new_deltas = [
            new_proxies[idx + 1] - new_proxies[idx]
            for idx in range(first_active_step, len(new_proxies) - 1)
            if new_proxies[idx + 1] > 0
        ]

        assert all(delta >= 0 for delta in new_deltas)

    def test_upper_ladder_growth_no_longer_has_extreme_post_boundary_cliff(self):
        config = ManagerConfig()
        config.filler.sublevels_per_major = 4

        upper_steps = list(range(12, 17))
        proxies = [_step_work_proxy(config, step) for step in upper_steps]
        growth_factors = [
            proxies[idx + 1] / proxies[idx] for idx in range(len(proxies) - 1)
        ]

        assert max(growth_factors) < 3.2
        assert proxies[1] / proxies[0] < 3.2

    def test_worker_rise_boundaries_keep_per_worker_growth_bounded(self):
        config = ManagerConfig()
        config.filler.sublevels_per_major = 4

        for major_level in range(1, config.filler.max_major_level + 1):
            prev_level = config.filler.levels[major_level - 1]
            curr_level = config.filler.levels[major_level]
            if curr_level.workers <= prev_level.workers or prev_level.workers == 0:
                continue

            boundary_step = major_level * config.filler.sublevels_per_major
            prior_step = boundary_step - 1
            prior_config = config.filler.interpolate_level_config(prior_step)
            boundary_config = config.filler.interpolate_level_config(boundary_step)
            prior_per_worker = (
                _step_work_proxy(config, prior_step) / prior_config.workers
            )
            boundary_per_worker = (
                _step_work_proxy(config, boundary_step) / boundary_config.workers
            )

            assert boundary_per_worker / prior_per_worker < 1.5

    def test_mps_caps(self):
        config = ManagerConfig()
        assert len(config.filler.mps_caps_no_experiment) == 9
        assert len(config.filler.mps_caps_experiment_active) == 9
        assert config.filler.mps_caps_no_experiment[8] == 98
        assert config.filler.mps_caps_experiment_active[8] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
