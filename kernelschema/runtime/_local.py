"""Private Linux durability/process primitives, shared with the exec supervisor."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import time

from ..artifacts import durable_mkdir, durable_write, read_regular
from ..contracts import MAX_INTEGER, ProtocolError, jcs, loads


class DeadlineExpired(ProtocolError):
    """The owner's fixed launch window has elapsed."""


class DeadlineClockMismatch(ProtocolError):
    """A monotonic deadline cannot be used on a different boot."""


def current_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def validate_deadlines(value: dict):
    """Validate optional owner bounds, without granting or renewing a window."""
    if "deadline_utc_ms" in value:
        deadline = value["deadline_utc_ms"]
        if type(deadline) is not int or abs(deadline) > MAX_INTEGER:
            raise ProtocolError("deadline_utc_ms must be a strict safe integer")
    mono = "deadline_monotonic_ns" in value
    boot = "clock_boot_id" in value
    if mono != boot:
        raise ProtocolError("deadline_monotonic_ns and clock_boot_id must be supplied together")
    if mono:
        deadline = value["deadline_monotonic_ns"]
        if not isinstance(deadline, str) or not re.fullmatch(r"[0-9]{1,40}", deadline):
            raise ProtocolError("deadline_monotonic_ns must be an unsigned decimal string")
        if not isinstance(value["clock_boot_id"], str) or not value["clock_boot_id"]:
            raise ProtocolError("clock_boot_id must be a nonempty string")
        if value["clock_boot_id"] != current_boot_id():
            raise DeadlineClockMismatch("deadline clock_boot_id differs from the current boot")


def remaining_timeout_ms(value: dict, *, utc_ns: int | None = None, mono_ns: int | None = None) -> int:
    """Recheck persisted cutoffs; never turn them back into a fresh duration."""
    validate_deadlines(value)
    remaining = value["timeout_ms"]
    if "deadline_utc_ms" in value:
        now = time.time_ns() if utc_ns is None else utc_ns
        remaining = min(remaining, (value["deadline_utc_ms"] * 1_000_000 - now) // 1_000_000)
    if "deadline_monotonic_ns" in value:
        now = time.monotonic_ns() if mono_ns is None else mono_ns
        remaining = min(remaining, (int(value["deadline_monotonic_ns"]) - now) // 1_000_000)
    if remaining <= 0:
        raise DeadlineExpired("owner launch deadline has expired")
    return remaining


def freeze_launch_deadlines(reservation: dict, timeout_ms: int) -> dict:
    """Freeze the tighter duration/owner cutoff for supervisor and exec gates."""
    validate_deadlines(reservation)
    utc_ns, mono_ns = time.time_ns(), time.monotonic_ns()
    # Every new launch has one fixed window, including duration-only callers.
    # Keep the persisted epoch cutoff inside the protocol's safe integer range.
    timeout_ms = min(timeout_ms, MAX_INTEGER - utc_ns // 1_000_000)
    remaining = remaining_timeout_ms({**reservation, "timeout_ms": timeout_ms}, utc_ns=utc_ns, mono_ns=mono_ns)
    return {"timeout_ms": remaining,
            "deadline_utc_ms": utc_ns // 1_000_000 + remaining,
            "deadline_monotonic_ns": str(mono_ns + remaining * 1_000_000),
            "clock_boot_id": current_boot_id()}


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
            "boot_id": current_boot_id()}


def process_alive(identity: dict) -> bool | None:
    try:
        if current_boot_id() != identity["boot_id"]:
            return False
        fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] == identity["start_ticks"] and fields[0] != "Z"
    except FileNotFoundError:
        return False
    except (OSError, KeyError, IndexError):
        return None
