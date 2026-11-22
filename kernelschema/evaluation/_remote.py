"""Agate dev entry, using explicitly selected pre-existing pinned tools.

No package install, alternate comparator or automatic replay. A missing tool or
namespace capability fails before candidate import. All stdout is framed and
hashed so gateway truncation cannot become a passing report.
"""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

MARKER = "KS_EVALUATOR_V1 "


def remaining_timeout(profile):
    """Queueing/preparation cannot renew the owner's original UTC cutoff.

    Cross-host UTC clock agreement is an execution-profile prerequisite. The
    host's monotonic cutoff is additionally checked before the original POST.
    """
    remaining = profile["timeout_s"]
    cutoff = profile.get("deadline_utc_ms")
    if cutoff is not None:
        if type(cutoff) is not int:
            raise ValueError("invalid owner UTC cutoff")
        remaining = min(remaining, (cutoff - time.time_ns() // 1000000) // 1000)
    if remaining < 1:
        raise TimeoutError("original owner deadline expired before remote execution")
    return remaining


def emit(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    print(MARKER + json.dumps({"sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                              "gzip_base64": base64.b64encode(gzip.compress(raw, mtime=0)).decode()}, separators=(",", ":")))


def main():
    root = Path(__file__).resolve().parent
    profile = json.loads((root / "remote.json").read_bytes())
    action = sys.argv[1]
    if profile.get("channel") in ("agate-dev-stage-ro-v1", "agate-dev-stage-ro-inherited-proc-v1"):
        helper = {"__name__": "ks_stage_ro"}
        path = root / "_stage_ro.py"
        exec(compile(path.read_bytes(), str(path), "exec"), helper)
        emit(helper["run"](root, profile, action, remaining_timeout(profile)))
        return
    if action == "probe":
        result = subprocess.run([profile["python"], "-I", "-B", str(root / "_worker.py"), str(root), "prepare-cuda"],
                                capture_output=True, timeout=remaining_timeout(profile))
        binary = Path(profile["bwrap"])
        emit({"action": "probe", "exit_code": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode(),
              "bwrap_sha256": "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else None,
              "bwrap_permission_qualification": "not_run"})
        return
    bwrap = Path(profile["bwrap"])
    if "sha256:" + hashlib.sha256(bwrap.read_bytes()).hexdigest() != profile["bwrap_sha256"]:
        raise RuntimeError("actual bwrap binary differs from selected execution profile")
    with tempfile.TemporaryDirectory(prefix="ks-compile-") as scratch:
        argv = [str(bwrap), "--die-with-parent", "--new-session", "--unshare-all", "--ro-bind", "/", "/",
                "--dev-bind", "/dev", "/dev", "--proc", "/proc", "--bind", scratch, scratch,
                "--chdir", scratch, "--cap-drop", "ALL", "--setenv", "HOME", scratch,
                "--setenv", "TMPDIR", scratch, "--setenv", "TRITON_CACHE_DIR", scratch + "/triton",
                "--setenv", "XDG_CACHE_HOME", scratch + "/cache",
                profile["python"], "-I", "-B", str(root / "_worker.py"), str(root), "execute"]
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            def limits():
                resource.setrlimit(resource.RLIMIT_FSIZE,(32 * 1024 * 1024,32 * 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_CORE,(0,0))
            result = subprocess.run(argv, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                    timeout=remaining_timeout(profile), preexec_fn=limits)
            stdout.seek(0); stderr.seek(0)
            emit({"action": "execute", "exit_code": result.returncode, "stdout": stdout.read().decode(),
                  "stderr": stderr.read().decode(), "bwrap_sha256": profile["bwrap_sha256"]})


if __name__ == "__main__":
    main()
