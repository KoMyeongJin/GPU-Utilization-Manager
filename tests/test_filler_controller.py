import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import FillerLevel, ManagerConfig
from src.filler_controller import FillerController


class TestFillerController:
    def test_apply_step_restarts_workers_when_effective_config_changes(self):
        config = ManagerConfig()
        controller = FillerController(config)
        controller.shm = Mock()
        controller._workers = {1234: Mock()}
        controller._current_level_config = FillerLevel(
            workers=2, batch_size=4, streams=2, sleep_ms=5.0
        )
        controller.stop_all = Mock()
        controller.ensure_workers = Mock(return_value=2)

        updated_level = FillerLevel(workers=2, batch_size=4, streams=2, sleep_ms=4.38)
        controller.config.filler.interpolate_level_config = Mock(
            return_value=updated_level
        )
        controller.config.filler.split_step = Mock(return_value=(3, 1))
        controller.config.filler.levels = {3: updated_level}

        assert controller.apply_step(8) is True

        controller.stop_all.assert_called_once_with()
        controller.ensure_workers.assert_called_once_with(2, updated_level)
        controller.shm.set_step.assert_called_once_with(3)
        controller.shm.resume.assert_called_once_with()

    def test_apply_step_does_not_restart_workers_when_config_is_unchanged(self):
        config = ManagerConfig()
        controller = FillerController(config)
        controller.shm = Mock()
        stable_level = FillerLevel(workers=2, batch_size=4, streams=2, sleep_ms=5.0)
        controller._workers = {1234: Mock()}
        controller._current_level_config = stable_level
        controller.stop_all = Mock()
        controller.ensure_workers = Mock(return_value=2)

        controller.config.filler.interpolate_level_config = Mock(
            return_value=stable_level
        )
        controller.config.filler.split_step = Mock(return_value=(3, 0))
        controller.config.filler.levels = {3: stable_level}

        assert controller.apply_step(8) is True

        controller.stop_all.assert_not_called()
        controller.ensure_workers.assert_called_once_with(2, stable_level)
