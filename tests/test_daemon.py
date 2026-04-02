import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.daemon import GPUtilizationManager


class TestDaemon:
    def test_disabled_mps_forces_inactive_experiment_status(self, monkeypatch):
        manager = GPUtilizationManager("config.yaml")
        manager.config.enable_mps = False

        monkeypatch.setattr(
            manager.registry,
            "get_status",
            lambda current_gpu_util_pct=None, filler_pids=None: (_ for _ in ()).throw(
                AssertionError(
                    "registry.get_status should not be used when MPS is disabled"
                )
            ),
        )
        status = manager._get_effective_experiment_status(current_gpu_util_pct=55.0)

        assert status.active is False
        assert status.active_pids == []

    def test_initialize_starts_at_step_32_when_initial_util_is_zero(self, monkeypatch):
        manager = GPUtilizationManager("config.yaml")
        manager.config.enable_mps = False
        manager.filler.initialize = Mock(return_value=True)
        manager.filler.apply_step = Mock(return_value=True)
        manager.monitor.read_sample = Mock(
            return_value=type("Sample", (), {"gpu_util": 0.0})()
        )
        manager.state_machine.transition_to_step = Mock(return_value=True)
        manager._start_socket_server = Mock()
        manager._setup_signal_handlers = Mock()

        assert manager.initialize() is True

        manager.state_machine.transition_to_step.assert_called_once_with(
            32, "startup initial util check"
        )
        manager.filler.apply_step.assert_called_once_with(32)

    def test_initialize_starts_paused_when_initial_util_is_nonzero(self, monkeypatch):
        manager = GPUtilizationManager("config.yaml")
        manager.config.enable_mps = False
        manager.filler.initialize = Mock(return_value=True)
        manager.filler.apply_step = Mock(return_value=True)
        manager.monitor.read_sample = Mock(
            return_value=type("Sample", (), {"gpu_util": 12.0})()
        )
        manager.state_machine.transition_to_step = Mock(return_value=True)
        manager._start_socket_server = Mock()
        manager._setup_signal_handlers = Mock()

        assert manager.initialize() is True

        manager.state_machine.transition_to_step.assert_called_once_with(
            0, "startup initial util check"
        )
        manager.filler.apply_step.assert_called_once_with(0)
