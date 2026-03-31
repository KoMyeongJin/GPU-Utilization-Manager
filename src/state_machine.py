import time
from dataclasses import dataclass, field
from typing import List
from enum import IntEnum


class FillerState(IntEnum):
    PAUSED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    LEVEL4 = 4
    LEVEL5 = 5
    LEVEL6 = 6
    LEVEL7 = 7
    MAX = 8


@dataclass
class StateTransition:
    from_step: int
    to_step: int
    timestamp: float = field(default_factory=time.time)
    reason: str = ""


@dataclass
class StateMachineConfig:
    hysteresis_polls: int = 2
    min_dwell_sec: float = 8.0
    sublevels_per_major: int = 1


class FillerStateMachine:
    def __init__(self, config: StateMachineConfig | None = None):
        self.config = config or StateMachineConfig()
        self._current_step = 0
        self._last_transition_time = time.time() - self.config.min_dwell_sec
        self._consecutive_count = 0
        self._last_decision = None
        self._transitions: List[StateTransition] = []

    @property
    def max_step(self) -> int:
        return int(FillerState.MAX) * self.config.sublevels_per_major

    @property
    def current_state(self) -> FillerState:
        return FillerState(self.current_level)

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def current_level(self) -> int:
        return self._current_step // self.config.sublevels_per_major

    @property
    def current_sublevel(self) -> int:
        return self._current_step % self.config.sublevels_per_major

    def can_transition(self, new_step: int) -> bool:
        elapsed = time.time() - self._last_transition_time
        return elapsed >= self.config.min_dwell_sec

    def transition_to_step(self, new_step: int, reason: str = "") -> bool:
        clamped_step = max(0, min(new_step, self.max_step))
        if not self.can_transition(clamped_step):
            return False

        if self._current_step != clamped_step:
            transition = StateTransition(
                from_step=self._current_step,
                to_step=clamped_step,
                reason=reason
            )
            self._transitions.append(transition)
            self._current_step = clamped_step
            self._last_transition_time = time.time()
            self._consecutive_count = 0

        return True

    def transition_to(self, new_state: FillerState, reason: str = "") -> bool:
        return self.transition_to_step(int(new_state) * self.config.sublevels_per_major, reason)

    def process_decision(
        self,
        decision: 'ScalingDecision',
        hysteresis_met: bool = True
    ) -> int:
        target_step = max(0, min(decision.target_step, self.max_step))

        if target_step == self._current_step:
            self._consecutive_count = 0
            return self._current_step

        if not hysteresis_met:
            self._consecutive_count += 1
            if self._consecutive_count < self.config.hysteresis_polls:
                return self._current_step

        success = self.transition_to_step(target_step, decision.reason)
        return self._current_step if success else self._current_step

    def upshift(self, levels: int = 1, reason: str = "") -> bool:
        return self.transition_to_step(self._current_step + (levels * self.config.sublevels_per_major), reason)

    def downshift(self, levels: int = 1, reason: str = "") -> bool:
        return self.transition_to_step(self._current_step - (levels * self.config.sublevels_per_major), reason)

    def pause(self, reason: str = "") -> bool:
        return self.transition_to_step(0, reason)

    def resume(self, reason: str = "") -> bool:
        return self.transition_to(FillerState.LOW, reason)

    def get_transitions(self, limit: int = 100) -> List[StateTransition]:
        return self._transitions[-limit:]

    def get_state_duration(self) -> float:
        return time.time() - self._last_transition_time


@dataclass
class ScalingDecision:
    target_step: int
    filler_mps_cap_pct: int
    reason: str
