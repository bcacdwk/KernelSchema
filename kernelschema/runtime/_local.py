"""Private Linux durability/process primitives, shared with the exec supervisor."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path

from ..artifacts import durable_mkdir, durable_write, read_regular
from ..contracts import ProtocolError, jcs, loads


def read_json(directory: Path, name: str) -> dict:
    return loads(read_regular(directory, name)[0])


def immutable(directory: Path, name: str, value: dict):
    data = jcs(value)
    path = directory / name
    if path.exists() or path.is_symlink():
        if read_regular(directory, name)[0] != data:
            raise ProtocolError(f"conflicting immutable runtime fact: {name}")
    else:
        durable_write(path, data)


@contextmanager
def locked(directory: Path):
    durable_mkdir(directory)
    fd = os.open(directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def process_identity(pid: int | None = None) -> dict:
    pid = os.getpid() if pid is None else pid
    # comm may contain spaces and parentheses. starttime is stat field 22.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def process_alive(identity: dict) -> bool | None:
    try:
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != identity["boot_id"]:
            return False
        fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] == identity["start_ticks"] and fields[0] != "Z"
    except FileNotFoundError:
        return False
    except (OSError, KeyError, IndexError):
        return None
