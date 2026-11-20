"""Agate dev entry, using a pre-existing pinned bwrap executable.

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

MARKER = "KS_EVALUATOR_V1 "


def emit(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    print(MARKER + json.dumps({"sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                              "gzip_base64": base64.b64encode(gzip.compress(raw, mtime=0)).decode()}, separators=(",", ":")))


def main():
    root = Path(__file__).resolve().parent
    profile = json.loads((root / "remote.json").read_bytes())
    action = sys.argv[1]
    if action == "probe":
        result = subprocess.run([profile["python"], "-I", "-B", str(root / "_worker.py"), str(root), "prepare-cuda"],
                                capture_output=True, timeout=profile["timeout_s"])
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
                                    timeout=profile["timeout_s"], preexec_fn=limits)
            stdout.seek(0); stderr.seek(0)
            emit({"action": "execute", "exit_code": result.returncode, "stdout": stdout.read().decode(),
                  "stderr": stderr.read().decode(), "bwrap_sha256": profile["bwrap_sha256"]})


if __name__ == "__main__":
    main()
