from typing import List
from dataclasses import dataclass

try:
    from .dcgm_monitor import GpuMetrics
    from .experiment_registry import ExperimentStatus
    from .state_machine import FillerStateMachine, ScalingDecision, FillerState
except ImportError:
    from src.dcgm_monitor import GpuMetrics
    from src.experiment_registry import ExperimentStatus
    from src.state_machine import FillerStateMachine, ScalingDecision, FillerState


@dataclass
class ScalingConfig:
    target_util_pct: float = 70.0
    low_boost_pct: float = 65.0
    target_floor_pct: float = 70.0
    healthy_high_pct: float = 88.0
    reduce_pct: float = 92.0
    emergency_reduce_pct: float = 95.0
    critical_pause_pct: float = 98.0
    sublevels_per_major: int = 1
    max_major_level: int = 8
    mps_caps_no_experiment: List[int] | None = None
    mps_caps_experiment_active: List[int] | None = None


class ScalingEngine:
    def __init__(self, config: ScalingConfig):
        self.config = config
        self._last_decision_time = 0.0

    @property
    def max_step(self) -> int:
        return self.config.max_major_level * self.config.sublevels_per_major

    @property
    def level_step(self) -> int:
        return max(1, self.config.sublevels_per_major)

    def _upshift_step(self, current_step: int, util: float) -> int:
        step_delta = (
            self.level_step if util <= (self.config.target_util_pct - 10.0) else 1
        )
        return min(current_step + step_delta, self.max_step)

    def _downshift_step(self, current_step: int, util: float) -> int:
        step_delta = (
            self.level_step if util >= (self.config.target_util_pct + 10.0) else 1
        )
        return max(current_step - step_delta, 0)

    def decide(
        self,
        samples: List[GpuMetrics],
        experiment: ExperimentStatus,
        current_step: int,
        trend: float = 0.0,
    ) -> ScalingDecision:
        if not samples:
            return ScalingDecision(
                target_step=0, filler_mps_cap_pct=0, reason="no samples"
            )

        util = self._smoothed_util(samples)

        if experiment.active:
            return self._decide_with_experiment(util, trend, current_step, experiment)
        return self._decide_no_experiment(util, trend, current_step)

    def _interpolate_mps_cap(self, step: int, experiment_active: bool) -> int:
        caps = (
            self.config.mps_caps_experiment_active
            if experiment_active
            else self.config.mps_caps_no_experiment
        )
        if caps is None:
            return 0
        clamped_step = max(0, min(step, self.max_step))
        major_level, sublevel = divmod(clamped_step, self.config.sublevels_per_major)
        if (
            self.config.sublevels_per_major == 1
            or major_level >= self.config.max_major_level
        ):
            return caps[major_level]
        next_level = min(major_level + 1, self.config.max_major_level)
        ratio = sublevel / self.config.sublevels_per_major
        interpolated = caps[major_level] + ratio * (
            caps[next_level] - caps[major_level]
        )
        return int(round(interpolated))

    def _smoothed_util(self, samples: List[GpuMetrics]) -> float:
        recent = samples[-3:]
        avg_util = sum(sample.gpu_util for sample in recent) / len(recent)
        latest_util = recent[-1].gpu_util
        return (avg_util * 0.7) + (latest_util * 0.3)

    def _decide_no_experiment(
        self, util: float, trend: float, current_step: int
    ) -> ScalingDecision:
        if util < self.config.low_boost_pct:
            new_step = self._upshift_step(current_step, util)
            return ScalingDecision(
                target_step=new_step,
                filler_mps_cap_pct=self._interpolate_mps_cap(new_step, False),
                reason=f"util {util:.1f}% < {self.config.low_boost_pct}% (no exp)",
            )

        if util < self.config.target_floor_pct:
            new_step = self._upshift_step(current_step, util)
            return ScalingDecision(
                target_step=new_step,
                filler_mps_cap_pct=self._interpolate_mps_cap(new_step, False),
                reason=f"util {util:.1f}% < {self.config.target_floor_pct}% (no exp)",
            )

        if self.config.target_floor_pct <= util <= self.config.healthy_high_pct:
            return ScalingDecision(
                target_step=current_step,
                filler_mps_cap_pct=self._interpolate_mps_cap(current_step, False),
                reason=f"util {util:.1f}% in healthy range",
            )

        if util > self.config.critical_pause_pct:
            new_step = 0
            return ScalingDecision(
                target_step=new_step,
                filler_mps_cap_pct=self._interpolate_mps_cap(new_step, False),
                reason=f"util {util:.1f}% > {self.config.critical_pause_pct}% CRITICAL",
            )

        if util > self.config.emergency_reduce_pct:
            new_step = self._downshift_step(current_step, util)
            return ScalingDecision(
                target_step=new_step,
                filler_mps_cap_pct=self._interpolate_mps_cap(new_step, False),
                reason=f"util {util:.1f}% > {self.config.emergency_reduce_pct}%",
            )

        if util > self.config.reduce_pct:
            new_step = self._downshift_step(current_step, util)
            return ScalingDecision(
                target_step=new_step,
                filler_mps_cap_pct=self._interpolate_mps_cap(new_step, False),
                reason=f"util {util:.1f}% > {self.config.reduce_pct}%",
            )

        return ScalingDecision(
            target_step=current_step,
            filler_mps_cap_pct=self._interpolate_mps_cap(current_step, False),
            reason=f"util {util:.1f}% - holding",
        )

    def _decide_with_experiment(
        self, util: float, trend: float, current_step: int, experiment: ExperimentStatus
    ) -> ScalingDecision:
        if trend > 10:
            new_step = self._downshift_step(current_step, util)
            return ScalingDecision(
                target_step=new_step,
                filler_mps_cap_pct=self._interpolate_mps_cap(new_step, True),
                reason=f"experiment surge detected (trend +{trend:.1f}%)",
            )

        if util > self.config.emergency_reduce_pct:
            new_step = 0
            return ScalingDecision(
                target_step=new_step,
                filler_mps_cap_pct=self._interpolate_mps_cap(new_step, True),
                reason=f"util {util:.1f}% > {self.config.emergency_reduce_pct}% (exp active)",
            )

        if util > self.config.reduce_pct:
            new_step = self._downshift_step(current_step, util)
            return ScalingDecision(
                target_step=new_step,
                filler_mps_cap_pct=self._interpolate_mps_cap(new_step, True),
                reason=f"util {util:.1f}% > {self.config.reduce_pct}% (exp active)",
            )

        if util < self.config.target_floor_pct:
            new_step = self._upshift_step(current_step, util)
            return ScalingDecision(
                target_step=new_step,
                filler_mps_cap_pct=self._interpolate_mps_cap(new_step, True),
                reason=f"util {util:.1f}% < {self.config.target_floor_pct}% (exp active, boost)",
            )

        return ScalingDecision(
            target_step=current_step,
            filler_mps_cap_pct=self._interpolate_mps_cap(current_step, True),
            reason=f"util {util:.1f}% with exp active - holding",
        )

    def should_transition(
        self,
        decision: ScalingDecision,
        current_state: FillerState,
        state_machine: FillerStateMachine,
        consecutive_count: int,
        hysteresis_polls: int,
    ) -> bool:
        target_major_level = decision.target_step // self.config.sublevels_per_major
        if target_major_level == int(current_state):
            return True

        if consecutive_count < hysteresis_polls:
            return False

        return state_machine.can_transition(decision.target_step)
