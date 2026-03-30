import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from config import ManagerConfig, FillerLevel, Thresholds, load_config


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
        assert config.filler.levels[8].workers == 4

    def test_mps_caps(self):
        config = ManagerConfig()
        assert len(config.filler.mps_caps_no_experiment) == 9
        assert len(config.filler.mps_caps_experiment_active) == 9
        assert config.filler.mps_caps_no_experiment[8] == 98
        assert config.filler.mps_caps_experiment_active[8] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
