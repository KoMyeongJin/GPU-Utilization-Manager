import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scaling_engine import ScalingConfig, ScalingEngine


class Sample:
    def __init__(self, util: float):
        self.gpu_util = util


class Status:
    def __init__(self, active: bool = False):
        self.active = active


class TestScalingEngine:
    def test_no_experiment_moves_one_sublevel(self):
        engine = ScalingEngine(ScalingConfig(
            sublevels_per_major=4,
            max_major_level=8,
            mps_caps_no_experiment=[0, 40, 60, 80, 90, 92, 94, 96, 98],
            mps_caps_experiment_active=[0, 5, 10, 20, 30, 35, 40, 45, 50],
        ))
        decision = engine.decide([Sample(0), Sample(0), Sample(0)], Status(False), 4, 0.0)
        assert decision.target_step == 5
        assert decision.filler_mps_cap_pct == 45

    def test_holding_range_keeps_current_step(self):
        engine = ScalingEngine(ScalingConfig(
            sublevels_per_major=4,
            max_major_level=8,
            mps_caps_no_experiment=[0, 40, 60, 80, 90, 92, 94, 96, 98],
            mps_caps_experiment_active=[0, 5, 10, 20, 30, 35, 40, 45, 50],
        ))
        decision = engine.decide([Sample(72), Sample(71), Sample(70)], Status(False), 9, 0.0)
        assert decision.target_step == 9

    def test_experiment_mode_uses_interpolated_cap(self):
        engine = ScalingEngine(ScalingConfig(
            sublevels_per_major=4,
            max_major_level=8,
            mps_caps_no_experiment=[0, 40, 60, 80, 90, 92, 94, 96, 98],
            mps_caps_experiment_active=[0, 5, 10, 20, 30, 35, 40, 45, 50],
        ))
        decision = engine.decide([Sample(0), Sample(0), Sample(0)], Status(True), 4, 0.0)
        assert decision.target_step == 5
        assert decision.filler_mps_cap_pct == 6
