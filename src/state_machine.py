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
    from_state: FillerState
    to_state: FillerState
    timestamp: float = field(default_factory=time.time)
    reason: str = ""


@dataclass
class StateMachineConfig:
    hysteresis_polls: int = 2
    min_dwell_sec: float = 8.0


class FillerStateMachine:
    def __init__(self, config: StateMachineConfig | None = None):
        self.config = config or StateMachineConfig()
        self._current_state = FillerState.PAUSED
        self._last_transition_time = time.time() - self.config.min_dwell_sec
        self._consecutive_count = 0
        self._last_decision = None
        self._transitions: List[StateTransition] = []

    @property
    def current_state(self) -> FillerState:
        return self._current_state

    @property
    def current_level(self) -> int:
        return int(self._current_state)

    def can_transition(self, new_state: FillerState) -> bool:
        elapsed = time.time() - self._last_transition_time
        return elapsed >= self.config.min_dwell_sec

    def transition_to(self, new_state: FillerState, reason: str = "") -> bool:
        if not self.can_transition(new_state):
            return False

        if self._current_state != new_state:
            transition = StateTransition(
                from_state=self._current_state,
                to_state=new_state,
                reason=reason
            )
            self._transitions.append(transition)
            self._current_state = new_state
            self._last_transition_time = time.time()
            self._consecutive_count = 0

        return True

    def process_decision(
        self,
        decision: 'ScalingDecision',
        hysteresis_met: bool = True
    ) -> FillerState:
        target_state = FillerState(decision.filler_level)

        if target_state == self._current_state:
            self._consecutive_count = 0
            return self._current_state

        if not hysteresis_met:
            self._consecutive_count += 1
            if self._consecutive_count < self.config.hysteresis_polls:
                return self._current_state

        success = self.transition_to(target_state, decision.reason)
        return self._current_state if success else self._current_state

    def upshift(self, levels: int = 1, reason: str = "") -> bool:
        new_level = min(int(self._current_state) + levels, int(FillerState.MAX))
        return self.transition_to(FillerState(new_level), reason)

    def downshift(self, levels: int = 1, reason: str = "") -> bool:
        new_level = max(int(self._current_state) - levels, 0)
        return self.transition_to(FillerState(new_level), reason)

    def pause(self, reason: str = "") -> bool:
        return self.transition_to(FillerState.PAUSED, reason)

    def resume(self, reason: str = "") -> bool:
        return self.transition_to(FillerState.LOW, reason)

    def get_transitions(self, limit: int = 100) -> List[StateTransition]:
        return self._transitions[-limit:]

    def get_state_duration(self) -> float:
        return time.time() - self._last_transition_time


@dataclass
class ScalingDecision:
    filler_level: int
    filler_mps_cap_pct: int
    reason: str
