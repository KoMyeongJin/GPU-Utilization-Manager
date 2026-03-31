import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.state_machine import FillerStateMachine, FillerState, StateMachineConfig, ScalingDecision


class TestFillerStateMachine:
    def test_initial_state(self):
        sm = FillerStateMachine()
        assert sm.current_state == FillerState.PAUSED
        assert sm.current_level == 0
        assert sm.current_step == 0

    def test_transition_to_valid(self):
        sm = FillerStateMachine(StateMachineConfig(min_dwell_sec=0))
        result = sm.transition_to(FillerState.LOW)
        assert result is True
        assert sm.current_state == FillerState.LOW

    def test_transition_with_dwell_time(self):
        sm = FillerStateMachine(StateMachineConfig(min_dwell_sec=1))
        sm.transition_to(FillerState.LOW)
        result = sm.transition_to(FillerState.MEDIUM)
        assert result is False
        assert sm.current_state == FillerState.LOW

    def test_upshift(self):
        sm = FillerStateMachine(StateMachineConfig(min_dwell_sec=0))
        sm.transition_to(FillerState.LOW)
        sm.upshift(1)
        assert sm.current_state == FillerState.MEDIUM

    def test_downshift(self):
        sm = FillerStateMachine(StateMachineConfig(min_dwell_sec=0))
        sm.transition_to(FillerState.HIGH)
        sm.downshift(1)
        assert sm.current_state == FillerState.MEDIUM

    def test_downshift_clamp(self):
        sm = FillerStateMachine(StateMachineConfig(min_dwell_sec=0))
        sm.downshift(1)
        assert sm.current_state == FillerState.PAUSED

    def test_process_decision_no_change(self):
        sm = FillerStateMachine(StateMachineConfig(min_dwell_sec=0))
        sm.transition_to(FillerState.MEDIUM)
        decision = ScalingDecision(target_step=2, filler_mps_cap_pct=60, reason="test")
        result = sm.process_decision(decision, hysteresis_met=True)
        assert result == 2

    def test_process_decision_with_change(self):
        sm = FillerStateMachine(StateMachineConfig(min_dwell_sec=0))
        sm.transition_to(FillerState.LOW)
        decision = ScalingDecision(target_step=3, filler_mps_cap_pct=80, reason="test")
        result = sm.process_decision(decision, hysteresis_met=True)
        assert result == 3
        assert sm.current_state == FillerState.HIGH

    def test_upshift_clamp_to_max(self):
        sm = FillerStateMachine(StateMachineConfig(min_dwell_sec=0))
        sm.transition_to(FillerState.HIGH)
        sm.upshift(10)
        assert sm.current_level == 8

    def test_sublevels_track_major_and_minor(self):
        sm = FillerStateMachine(StateMachineConfig(min_dwell_sec=0, sublevels_per_major=4))
        decision = ScalingDecision(target_step=6, filler_mps_cap_pct=50, reason="test")
        result = sm.process_decision(decision, hysteresis_met=True)
        assert result == 6
        assert sm.current_level == 1
        assert sm.current_sublevel == 2
        assert sm.current_state == FillerState.LOW


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
