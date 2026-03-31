import sys
import os
import time
import signal
import json
import socket
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from .config import load_config
    from .dcgm_monitor import DCGMMonitor, MetricsAggregator
    from .experiment_registry import ExperimentRegistry, ExperimentStatus
    from .mps_adapter import MPSAdapter
    from .filler_controller import FillerController
    from .scaling_engine import ScalingEngine, ScalingConfig
    from .state_machine import FillerStateMachine, StateMachineConfig
except ImportError:
    from src.config import load_config
    from src.dcgm_monitor import DCGMMonitor, MetricsAggregator
    from src.experiment_registry import ExperimentRegistry, ExperimentStatus
    from src.mps_adapter import MPSAdapter
    from src.filler_controller import FillerController
    from src.scaling_engine import ScalingEngine, ScalingConfig
    from src.state_machine import FillerStateMachine, StateMachineConfig


class GPUtilizationManager:

    def _inactive_experiment_status(self) -> ExperimentStatus:
        return ExperimentStatus(
            active=False,
            active_pids=[],
            estimated_load_pct=0.0,
            experiments=[],
        )

    def _get_effective_experiment_status(self, current_gpu_util_pct: Optional[float] = None) -> ExperimentStatus:
        if not self.config.enable_mps:
            return self._inactive_experiment_status()

        return self.registry.get_status(
            current_gpu_util_pct=current_gpu_util_pct,
            filler_pids=set(self.filler.get_active_pids()),
        )

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.running = False
        self._shutdown_requested = False

        self.monitor = DCGMMonitor(self.config.gpu_id)
        self.aggregator = MetricsAggregator(window_size=10)
        self.registry = ExperimentRegistry(
            heartbeat_timeout_sec=self.config.experiment_heartbeat_timeout_sec,
            enable_process_detection=self.config.enable_process_detection
        )
        self.registry.set_gpu_id(self.config.gpu_id)

        self.mps = MPSAdapter(
            pipe_dir=self.config.mps_pipe_dir,
            log_dir=self.config.mps_log_dir
        )

        self.filler = FillerController(self.config)

        scaling_config = ScalingConfig(
            target_util_pct=self.config.target_util_pct,
            low_boost_pct=self.config.thresholds.low_boost_pct,
            target_floor_pct=self.config.thresholds.target_floor_pct,
            healthy_high_pct=self.config.thresholds.healthy_high_pct,
            reduce_pct=self.config.thresholds.reduce_pct,
            emergency_reduce_pct=self.config.thresholds.emergency_reduce_pct,
            critical_pause_pct=self.config.thresholds.critical_pause_pct,
            sublevels_per_major=self.config.filler.sublevels_per_major,
            max_major_level=self.config.filler.max_major_level,
            mps_caps_no_experiment=self.config.filler.mps_caps_no_experiment,
            mps_caps_experiment_active=self.config.filler.mps_caps_experiment_active,
        )
        self.scaling = ScalingEngine(scaling_config)

        state_config = StateMachineConfig(
            hysteresis_polls=self.config.hysteresis.consecutive_polls,
            min_dwell_sec=self.config.hysteresis.min_dwell_sec,
            sublevels_per_major=self.config.filler.sublevels_per_major,
        )
        self.state_machine = FillerStateMachine(state_config)

        self._sock = None
        self._poll_count = 0
        self._consecutive_matches = 0
        self._last_decision = None

    def _setup_signal_handlers(self):
        def handle_signal(signum, frame):
            self._shutdown_requested = True

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

    def _start_socket_server(self):
        try:
            if os.path.exists(self.config.socket_path):
                os.remove(self.config.socket_path)

            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.bind(self.config.socket_path)
            self._sock.listen(5)
            self._sock.settimeout(0.1)
        except Exception:
            pass

    def _handle_socket_requests(self):
        if self._sock is None:
            return

        try:
            conn, _ = self._sock.accept()
            data = conn.recv(1024)
            if data:
                msg = json.loads(data.decode())
                action = msg.get("action")
                pid = msg.get("pid", os.getpid())
                name = msg.get("name", "")
                metadata = msg.get("metadata", {})

                if action == "start":
                    self.registry.register(pid, name, metadata)
                    self._on_experiment_start()
                elif action == "stop":
                    self.registry.unregister(pid)
                    self._on_experiment_stop()
                elif action == "heartbeat":
                    self.registry.heartbeat(pid)

                conn.sendall(b'{"status": "ok"}')
            conn.close()
        except socket.timeout:
            pass
        except Exception:
            pass

    def _on_experiment_start(self):
        status = self._get_effective_experiment_status()
        if status.active:
            self.state_machine.downshift(2, "experiment started")
            mps_cap = self.config.filler.interpolate_mps_cap(self.state_machine.current_step, True)
            if self.config.enable_mps:
                self.mps.set_active_thread_percentage(mps_cap)
            self.filler.apply_step(self.state_machine.current_step)

    def _on_experiment_stop(self):
        status = self._get_effective_experiment_status()
        if not status.active:
            pass

    def initialize(self) -> bool:
        print("Initializing GPU Utilization Manager...")

        if self.config.enable_mps and not self.mps.start():
            print("Warning: MPS not started, continuing anyway")

        if not self.filler.initialize():
            print("Failed to initialize shared memory")
            return False

        self._start_socket_server()
        self._setup_signal_handlers()

        self.filler.apply_level(0)
        print(f"Initialized on GPU {self.config.gpu_id}")

        return True

    def _control_loop_iteration(self):
        sample = self.monitor.read_sample()
        self.aggregator.add(sample)

        util = self.aggregator.ema_util()
        trend = self.aggregator.trend()
        exp_status = self._get_effective_experiment_status(current_gpu_util_pct=util)

        decision = self.scaling.decide(
            self.aggregator.samples,
            exp_status,
            self.state_machine.current_step,
            trend
        )

        if decision.target_step == self.state_machine.current_step:
            self._consecutive_matches = 0
        else:
            self._consecutive_matches += 1

        hysteresis_met = self._consecutive_matches >= self.config.hysteresis.consecutive_polls

        if hysteresis_met:
            previous_step = self.state_machine.current_step
            new_step = self.state_machine.process_decision(decision, hysteresis_met)

            if new_step != previous_step:
                mps_cap = self.config.filler.interpolate_mps_cap(new_step, exp_status.active)
                if self.config.enable_mps:
                    self.mps.set_active_thread_percentage(mps_cap)
                self.filler.apply_step(new_step)

                print(f"Transitioned to step {new_step} (level {self.state_machine.current_level}+{self.state_machine.current_sublevel}/{self.config.filler.sublevels_per_major}): {decision.reason} "
                      f"(util={util:.1f}%, exp_active={exp_status.active})")

        self._last_decision = decision

    def run(self):
        self.running = True
        print("Starting control loop...")

        while self.running and not self._shutdown_requested:
            try:
                self._control_loop_iteration()

                if self._sock:
                    self._handle_socket_requests()

                time.sleep(self.config.poll_interval_sec)

            except Exception as e:
                print(f"Error in control loop: {e}")
                time.sleep(1)

        self.shutdown()

    def shutdown(self):
        print("Shutting down...")
        self.running = False

        self.filler.cleanup()

        if self._sock:
            self._sock.close()
            if os.path.exists(self.config.socket_path):
                os.remove(self.config.socket_path)

        if self.config.enable_mps:
            self.mps.stop()

        print("Shutdown complete")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    manager = GPUtilizationManager(config_path)

    if not manager.initialize():
        sys.exit(1)

    manager.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
