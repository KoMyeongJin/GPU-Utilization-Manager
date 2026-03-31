import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.daemon import GPUtilizationManager


class TestDaemon:
    def test_disabled_mps_forces_inactive_experiment_status(self, monkeypatch):
        manager = GPUtilizationManager('config.yaml')
        manager.config.enable_mps = False

        monkeypatch.setattr(
            manager.registry,
            'get_status',
            lambda current_gpu_util_pct=None, filler_pids=None: (_ for _ in ()).throw(AssertionError('registry.get_status should not be used when MPS is disabled')),
        )
        status = manager._get_effective_experiment_status(current_gpu_util_pct=55.0)

        assert status.active is False
        assert status.active_pids == []
