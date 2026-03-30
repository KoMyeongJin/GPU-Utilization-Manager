from typing import List
from dataclasses import dataclass
from dcgm_monitor import GpuMetrics
from experiment_registry import ExperimentStatus
from state_machine import FillerStateMachine, ScalingDecision, FillerState


@dataclass
class ScalingConfig:
    target_util_pct: float = 70.0
    low_boost_pct: float = 65.0
    target_floor_pct: float = 70.0
    healthy_high_pct: float = 88.0
    reduce_pct: float = 92.0
    emergency_reduce_pct: float = 95.0
    critical_pause_pct: float = 98.0


class ScalingEngine:

    def __init__(self, config: ScalingConfig):
        self.config = config
        self._last_decision_time = 0.0

    def decide(
        self,
        samples: List[GpuMetrics],
        experiment: ExperimentStatus,
        current_level: int,
        trend: float = 0.0
    ) -> ScalingDecision:
        if not samples:
            return ScalingDecision(
                filler_level=0,
                filler_mps_cap_pct=0,
                reason="no samples"
            )

        util = self._smoothed_util(samples)

        if experiment.active:
            return self._decide_with_experiment(util, trend, current_level, experiment)
        else:
            return self._decide_no_experiment(util, trend, current_level)

    def _smoothed_util(self, samples: List[GpuMetrics]) -> float:
        recent = samples[-3:]
        avg_util = sum(sample.gpu_util for sample in recent) / len(recent)
        latest_util = recent[-1].gpu_util
        return (avg_util * 0.7) + (latest_util * 0.3)

    def _decide_no_experiment(
        self,
        util: float,
        trend: float,
        current_level: int
    ) -> ScalingDecision:
        mps_caps = [0, 40, 60, 80, 90, 92, 94, 96, 98]

        if util < self.config.low_boost_pct:
            new_level = min(current_level + 1, 8)
            return ScalingDecision(
                filler_level=new_level,
                filler_mps_cap_pct=mps_caps[new_level],
                reason=f"util {util:.1f}% < {self.config.low_boost_pct}% (no exp)"
            )

        if util < self.config.target_floor_pct:
            new_level = min(current_level + 1, 8)
            return ScalingDecision(
                filler_level=new_level,
                filler_mps_cap_pct=mps_caps[new_level],
                reason=f"util {util:.1f}% < {self.config.target_floor_pct}% (no exp)"
            )

        if self.config.target_floor_pct <= util <= self.config.healthy_high_pct:
            return ScalingDecision(
                filler_level=current_level,
                filler_mps_cap_pct=mps_caps[current_level],
                reason=f"util {util:.1f}% in healthy range"
            )

        if util > self.config.critical_pause_pct:
            new_level = max(current_level - 1, 0)
            return ScalingDecision(
                filler_level=new_level,
                filler_mps_cap_pct=mps_caps[new_level],
                reason=f"util {util:.1f}% > {self.config.critical_pause_pct}% CRITICAL"
            )

        if util > self.config.emergency_reduce_pct:
            new_level = max(current_level - 1, 0)
            return ScalingDecision(
                filler_level=new_level,
                filler_mps_cap_pct=mps_caps[new_level],
                reason=f"util {util:.1f}% > {self.config.emergency_reduce_pct}%"
            )

        if util > self.config.reduce_pct:
            new_level = max(current_level - 1, 0)
            return ScalingDecision(
                filler_level=new_level,
                filler_mps_cap_pct=mps_caps[new_level],
                reason=f"util {util:.1f}% > {self.config.reduce_pct}%"
            )

        return ScalingDecision(
            filler_level=current_level,
            filler_mps_cap_pct=mps_caps[current_level],
            reason=f"util {util:.1f}% - holding"
        )

    def _decide_with_experiment(
        self,
        util: float,
        trend: float,
        current_level: int,
        experiment: ExperimentStatus
    ) -> ScalingDecision:
        mps_caps = [0, 5, 10, 20, 30, 35, 40, 45, 50]

        if trend > 10:
            new_level = max(current_level - 1, 0)
            return ScalingDecision(
                filler_level=new_level,
                filler_mps_cap_pct=mps_caps[new_level],
                reason=f"experiment surge detected (trend +{trend:.1f}%)"
            )

        if util > self.config.emergency_reduce_pct:
            new_level = max(current_level - 1, 0)
            return ScalingDecision(
                filler_level=new_level,
                filler_mps_cap_pct=mps_caps[new_level],
                reason=f"util {util:.1f}% > {self.config.emergency_reduce_pct}% (exp active)"
            )

        if util > self.config.reduce_pct:
            new_level = max(current_level - 1, 0)
            return ScalingDecision(
                filler_level=new_level,
                filler_mps_cap_pct=mps_caps[new_level],
                reason=f"util {util:.1f}% > {self.config.reduce_pct}% (exp active)"
            )

        if util < self.config.target_floor_pct:
            new_level = min(current_level + 1, 8)
            return ScalingDecision(
                filler_level=new_level,
                filler_mps_cap_pct=mps_caps[new_level],
                reason=f"util {util:.1f}% < {self.config.target_floor_pct}% (exp active, boost)"
            )

        return ScalingDecision(
            filler_level=current_level,
            filler_mps_cap_pct=mps_caps[current_level],
            reason=f"util {util:.1f}% with exp active - holding"
        )

    def should_transition(
        self,
        decision: ScalingDecision,
        current_state: FillerState,
        state_machine: FillerStateMachine,
        consecutive_count: int,
        hysteresis_polls: int
    ) -> bool:
        if decision.filler_level == int(current_state):
            return True

        if consecutive_count < hysteresis_polls:
            return False

        return state_machine.can_transition(FillerState(decision.filler_level))
