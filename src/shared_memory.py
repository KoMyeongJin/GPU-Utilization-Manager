"""Shared memory IPC for manager-worker communication."""

import mmap
import os
import struct
import time
import fcntl
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum


class ShmCommands(IntEnum):
    NONE = 0
    PAUSE = 1
    RESUME = 2
    SET_LEVEL = 3
    SHUTDOWN = 4


@dataclass
class ShmState:
    current_level: int = 0
    target_level: int = 0
    command: int = int(ShmCommands.NONE)
    command_param: int = 0
    timestamp: float = 0.0
    active: bool = False

    PACK_FORMAT = "iiii?d"
    PACK_SIZE = struct.calcsize(PACK_FORMAT)

    def pack(self) -> bytes:
        return struct.pack(
            self.PACK_FORMAT,
            self.current_level,
            self.target_level,
            self.command,
            self.command_param,
            self.active,
            self.timestamp,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ShmState":
        unpacked = struct.unpack(cls.PACK_FORMAT, data)
        return cls(
            current_level=unpacked[0],
            target_level=unpacked[1],
            command=unpacked[2],
            command_param=unpacked[3],
            active=unpacked[4],
            timestamp=unpacked[5],
        )


class SharedMemoryManager:
    def __init__(self, name: str = "/gpu_manager_shm", size: int = 1024):
        self.name = name
        self.size = size
        self.fd = None
        self.mm = None
        self._lock_file = f"/tmp{name}.lock"

    def create(self) -> None:
        shm_path = f"/dev/shm{self.name}"
        self.fd = os.open(shm_path, os.O_CREAT | os.O_RDWR)
        os.ftruncate(self.fd, self.size)
        self.mm = mmap.mmap(self.fd, self.size)

        state = ShmState(active=True, timestamp=time.time())
        self._write_state(state)

    def attach(self) -> None:
        shm_path = f"/dev/shm{self.name}"
        self.fd = os.open(shm_path, os.O_RDWR)
        self.mm = mmap.mmap(self.fd, self.size)

    def close(self) -> None:
        if self.mm:
            self.mm.close()
        if self.fd:
            os.close(self.fd)

    def _write_state(self, state: ShmState) -> None:
        mm = self.mm
        if mm is None:
            raise RuntimeError("shared memory not initialized")
        mm.seek(0)
        mm.write(state.pack())
        mm.flush()

    def _read_state(self) -> ShmState:
        mm = self.mm
        if mm is None:
            raise RuntimeError("shared memory not initialized")
        mm.seek(0)
        data = mm.read(ShmState.PACK_SIZE)
        return ShmState.unpack(data)

    def set_level(self, level: int) -> None:
        state = self._read_state()
        state.target_level = level
        state.timestamp = time.time()
        self._write_state(state)

    def set_step(self, step: int) -> None:
        self.set_level(step)

    def get_level(self) -> int:
        return self._read_state().target_level

    def get_target_step(self) -> int:
        return self.get_level()

    def send_command(self, command: ShmCommands, param: int = 0) -> None:
        state = self._read_state()
        state.command = command
        state.command_param = param
        state.timestamp = time.time()
        self._write_state(state)

    def pause(self) -> None:
        self.send_command(ShmCommands.PAUSE)

    def resume(self) -> None:
        self.send_command(ShmCommands.RESUME)

    def shutdown(self) -> None:
        self.send_command(ShmCommands.SHUTDOWN)

    def get_status(self) -> ShmState:
        return self._read_state()


class SharedMemoryClient:
    def __init__(self, name: str = "/gpu_manager_shm"):
        self.name = name
        self.fd = None
        self.mm = None

    def attach(self) -> bool:
        try:
            shm_path = f"/dev/shm{self.name}"
            self.fd = os.open(shm_path, os.O_RDWR)
            self.mm = mmap.mmap(self.fd, 1024)
            return True
        except (FileNotFoundError, OSError):
            return False

    def close(self) -> None:
        if self.mm:
            self.mm.close()
        if self.fd:
            os.close(self.fd)

    def _read_state(self) -> ShmState:
        mm = self.mm
        if mm is None:
            raise RuntimeError("shared memory not attached")
        mm.seek(0)
        data = mm.read(ShmState.PACK_SIZE)
        return ShmState.unpack(data)

    def update_current_level(self, level: int) -> None:
        mm = self.mm
        if mm is None:
            raise RuntimeError("shared memory not attached")
        state = self._read_state()
        state.current_level = level
        mm.seek(0)
        mm.write(state.pack())
        mm.flush()

    def update_current_step(self, step: int) -> None:
        self.update_current_level(step)

    def get_command(self) -> Tuple[ShmCommands, int]:
        state = self._read_state()
        return ShmCommands(state.command), state.command_param

    def get_target_level(self) -> int:
        return self._read_state().target_level

    def get_target_step(self) -> int:
        return self.get_target_level()

    def clear_command(self) -> None:
        mm = self.mm
        if mm is None:
            raise RuntimeError("shared memory not attached")
        mm.seek(struct.calcsize("ii"))
        mm.write(struct.pack("i", int(ShmCommands.NONE)))
        mm.flush()

    def is_shutdown_requested(self) -> bool:
        cmd, _ = self.get_command()
        return cmd == ShmCommands.SHUTDOWN

    def is_paused(self) -> bool:
        cmd, _ = self.get_command()
        return cmd == ShmCommands.PAUSE
