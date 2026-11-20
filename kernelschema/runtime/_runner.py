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
from ._local import immutable, locked, process_identity, read_json


def child(directory: Path):
    launch = read_json(directory, "launch.json")
    executable = Path(launch["command"][0])
    if sha256(executable.read_bytes()) != launch["executable_sha256"]:
        raise RuntimeError("Claude executable changed before exec")
    immutable(directory, "child.json", process_identity())
    os.chdir(launch["cwd"])
    os.execve(str(executable), launch["command"], dict(os.environ))


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
            proc = subprocess.Popen(
                [sys.executable, "-B", "-m", "kernelschema.runtime._runner", "child", str(directory)],
                stdin=stdin, stdout=out, stderr=err, start_new_session=True,
            )
            try:
                code = proc.wait(timeout=launch["timeout_ms"] / 1000)
            except subprocess.TimeoutExpired:
                timed_out = True
                # SIGKILL and wait establish actual local process termination;
                # a mere timeout/cancellation request does not do so.
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
    except Exception as error:
        spawn_error = type(error).__name__  # Never serialize auth/environment.
    finally:
        os.fsync(out); os.fsync(err)
        os.close(out); os.close(err)
    immutable(directory, "exit.json", {
        "returncode": code, "timed_out": timed_out, "spawn_error": spawn_error,
        "active_ms": (time.monotonic_ns() - started) // 1_000_000,
    })


if __name__ == "__main__":
    mode, root = sys.argv[1:]
    (child if mode == "child" else run)(Path(root))
