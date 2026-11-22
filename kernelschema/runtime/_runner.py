"""Detached one-invoke supervisor. Never retries, resumes, or admits an action.

The child records its PID/start/boot identity *before* exec of Claude. Killing
the calling Python process does not discard stdout, stderr or the exit receipt.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..artifacts import fsync_directory, read_regular
from ..contracts import sha256
from ._local import (DeadlineClockMismatch, DeadlineExpired, immutable, locked,
                     process_identity, read_json, remaining_timeout_ms)


def not_executed(directory: Path, launch: dict, *, stage: str, reason: str):
    """Written only on a path that has not called execve/started its child."""
    immutable(directory, "not-executed.json", {
        "operation_id": launch["operation_id"], "session_id": launch["session_id"],
        "stage": stage, "reason": reason, "process": process_identity(),
    })


def child(directory: Path):
    launch = read_json(directory, "launch.json")
    immutable(directory, "child.json", process_identity())
    try:
        remaining_timeout_ms(launch)
        executable = Path(launch["command"][0])
        if sha256(executable.read_bytes()) != launch["executable_sha256"]:
            raise RuntimeError("Claude executable changed before exec")
        os.chdir(launch["cwd"])
        immutable(directory, "exec-intent.json", {
            "operation_id": launch["operation_id"], "session_id": launch["session_id"],
            "process": process_identity(),
        })
        # Hashing, interpreter startup and fsync all consume the original window.
        remaining_timeout_ms(launch)
    except (DeadlineExpired, DeadlineClockMismatch) as error:
        not_executed(directory, launch, stage="child", reason=type(error).__name__)
        raise SystemExit(124) from None
    os.execve(str(executable), launch["command"], dict(os.environ))


def wait_for_child(proc, launch):
    if "deadline_utc_ms" not in launch and "deadline_monotonic_ns" not in launch:
        return proc.wait(timeout=launch["timeout_ms"] / 1000)
    while True:
        if proc.poll() is not None: return proc.returncode
        remaining = remaining_timeout_ms(launch)
        try:
            return proc.wait(timeout=min(remaining / 1000, 0.1))
        except subprocess.TimeoutExpired:
            pass


def run(directory: Path):
    with locked(directory):
        if (directory / "supervisor.json").exists():
            raise RuntimeError("a supervisor already owns this operation")
        launch = read_json(directory, "launch.json")
        immutable(directory, "supervisor.json", process_identity())
    started = time.monotonic_ns()
    code, timed_out, spawn_error = None, False, None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    out = os.open(directory / "events.jsonl", flags, 0o600)
    err = os.open(directory / "stderr.log", flags, 0o600)
    fsync_directory(directory)
    try:
        prompt, _ = read_regular(directory, "input.json")
        if sha256(prompt) != launch["input_sha256"]:
            raise RuntimeError("frozen role input changed before exec")
        with open(directory / "input.json", "rb") as stdin:
            try:
                remaining_timeout_ms(launch)
            except (DeadlineExpired, DeadlineClockMismatch) as error:
                not_executed(directory, launch, stage="supervisor", reason=type(error).__name__)
                raise
            proc = subprocess.Popen(
                [sys.executable, "-B", "-m", "kernelschema.runtime._runner", "child", str(directory)],
                stdin=stdin, stdout=out, stderr=err, start_new_session=True,
            )
            try:
                code = wait_for_child(proc, launch)
            except (subprocess.TimeoutExpired, DeadlineExpired, DeadlineClockMismatch):
                timed_out = True
                # SIGKILL and wait establish actual local process termination;
                # a mere timeout/cancellation request does not do so.
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                code = proc.wait()
    except Exception as error:
        spawn_error = type(error).__name__  # Never serialize auth/environment.
    finally:
        os.fsync(out); os.fsync(err)
        os.close(out); os.close(err)
    immutable(directory, "exit.json", {
        "returncode": code, "timed_out": timed_out, "spawn_error": spawn_error,
        "active_ms": (time.monotonic_ns() - started) // 1_000_000,
        "not_executed": (directory / "not-executed.json").exists(),
    })


if __name__ == "__main__":
    mode, root = sys.argv[1:]
    (child if mode == "child" else run)(Path(root))
