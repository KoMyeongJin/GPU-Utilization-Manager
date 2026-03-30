import sys
import os
import time
import signal
import json
import socket
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from dcgm_monitor import DCGMMonitor, MetricsAggregator
from experiment_registry import ExperimentRegistry
from mps_adapter import MPSAdapter
from filler_controller import FillerController
from scaling_engine import ScalingEngine, ScalingConfig
from state_machine import FillerStateMachine, StateMachineConfig


class GPUtilizationManager:

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
        )
        self.scaling = ScalingEngine(scaling_config)

        state_config = StateMachineConfig(
            hysteresis_polls=self.config.hysteresis.consecutive_polls,
            min_dwell_sec=self.config.hysteresis.min_dwell_sec
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
        status = self.registry.get_status()
        if status.active:
            self.state_machine.downshift(2, "experiment started")
            mps_cap = self.config.filler.mps_caps_experiment_active[
                min(self.state_machine.current_level, 4)
            ]
            self.mps.set_active_thread_percentage(mps_cap)
            self.filler.apply_level(self.state_machine.current_level)

    def _on_experiment_stop(self):
        status = self.registry.get_status()
        if not status.active:
            pass

    def initialize(self) -> bool:
        print("Initializing GPU Utilization Manager...")

        if not self.mps.start():
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

        exp_status = self.registry.get_status()

        util = self.aggregator.ema_util()
        trend = self.aggregator.trend()

        decision = self.scaling.decide(
            self.aggregator.samples,
            exp_status,
            self.state_machine.current_level,
            trend
        )

        if decision.filler_level == self.state_machine.current_level:
            self._consecutive_matches = 0
        else:
            self._consecutive_matches += 1

        hysteresis_met = self._consecutive_matches >= self.config.hysteresis.consecutive_polls

        if hysteresis_met:
            previous_state = self.state_machine.current_state
            new_state = self.state_machine.process_decision(decision, hysteresis_met)

            if new_state != previous_state:
                if exp_status.active:
                    mps_cap = self.config.filler.mps_caps_experiment_active[
                        min(int(new_state), 4)
                    ]
                else:
                    mps_cap = self.config.filler.mps_caps_no_experiment[
                        min(int(new_state), 4)
                    ]

                self.mps.set_active_thread_percentage(mps_cap)
                self.filler.apply_level(int(new_state))

                print(f"Transitioned to level {int(new_state)}: {decision.reason} "
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
