"""Detached local job supervisor; durable original terminal survives caller loss."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def durable(path, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    temp = path.with_suffix(".pending")
    with temp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)


def main():
    job = Path(sys.argv[1]).resolve()
    intent = json.loads((job / "intent.json").read_bytes())
    # A controller may be started at most once. Never replay after an ambiguous spawn.
    try:
        fd = os.open(job / "claimed", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    os.fsync(fd)
    os.close(fd)
    stage = Path(intent["stage"]["root"])
    manifest = intent["stage"]["manifest"]
    for name in ("_controller.py", "_worker.py"):
        if "sha256:" + hashlib.sha256((stage / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError("controller/worker staged identity changed")
    started = time.monotonic_ns()
    worker = None
    timeout = False
    error = None
    # Only these two writable descriptors reach the read-only candidate process.
    with (job / "stdout.raw").open("xb") as stdout, (job / "stderr.raw").open("xb") as stderr:
        try:
            def limits():
                resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            worker = subprocess.Popen([intent["python"], "-I", "-B", str(stage / "_worker.py"), str(stage), "execute"],
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True,
                env={"PATH": "/usr/bin:/bin", "HOME": str(job), "LANG": "C.UTF-8", "OMP_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                preexec_fn=limits)
            durable(job / "started.json", {"pid": worker.pid, "controller_pid": os.getpid(), "operation_id": intent["operation_id"]})
            try:
                worker.wait(timeout=intent["reservation"]["active_ms"] / 1000)
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    durable(job / "native-terminal.json", {
        "operation_id": intent["operation_id"], "submission_key": intent["submission_key"], "owner": intent["owner"],
        "profile": intent["profile"], "stage_sha256": intent["stage"]["sha256"],
        "execution_sha256": intent["execution_sha256"], "state": "terminal", "sequence": 1,
        "exit_code": None if worker is None else worker.returncode, "timeout": timeout, "error": error,
        "elapsed_ms": (time.monotonic_ns() - started + 999999) // 1000000,
        "usage": {"calls": 0, "evaluator_jobs": 1, "tokens": 0, "gpu_ms": 0},
        "outputs": {name: "sha256:" + hashlib.sha256((job / name).read_bytes()).hexdigest() for name in ("stdout.raw", "stderr.raw")}})


if __name__ == "__main__":
    main()
