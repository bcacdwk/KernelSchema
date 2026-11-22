"""Finite FA4 launcher: two fresh original-stack processes in readonly bwrap.

Generated code is trusted not to attack the evaluator's Python address space.
The enforced OS boundary denies stage/root writes and network, permits CUDA
devices and a separate compile/output scratch. This is not hostile-native-code
isolation. Qualification must explicitly accept and exercise this boundary.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile


def sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def run(root, settings):
    outer = module(root / "_remote.py", "ks_frame")
    entry = module(root / "evaluator/entry.py", "ks_fa4_entry")
    manifest = json.loads((root / "stage.json").read_bytes())
    # remote.json is transport configuration, outside the immutable inner stage.
    for name, facts in manifest["files"].items():
        path = root / name
        if (path.is_symlink() or sha(path.read_bytes()) != facts["sha256"]
                or path.stat().st_size != facts["size"] or path.stat().st_mode & 0o7777 != int(facts["mode"], 8)):
            raise ValueError("uploaded stage bytes/mode differ: " + name)
    if sha(Path(settings["bwrap"]).read_bytes()) != settings["bwrap_sha256"]:
        raise ValueError("pinned bwrap differs")
    reports = {}
    for label, source_key in (("baseline", "baseline"), ("candidate", "source")):
        with tempfile.TemporaryDirectory(prefix="ks-fa4-stage-") as stage_dir, tempfile.TemporaryDirectory(prefix="ks-fa4-scratch-") as scratch_dir:
            stage, scratch = Path(stage_dir), Path(scratch_dir)
            files = {}
            for name, facts in manifest["files"].items():
                if name.startswith(source_key + "/"):
                    target = "source/" + name.split("/", 1)[1]
                elif name.startswith(("source/", "baseline/")) or name in ("_remote.py", "_fa4_remote.py"):
                    continue
                else:
                    target = name
                path = stage / target
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, path)
                path.chmod(0o644)
                files[target] = facts
            inner = {**{k: manifest[k] for k in ("version", "task", "schema", "scope", "evaluator", "inputs", "timing", "environment", "coverage")},
                     "source": manifest[source_key], "files": files}
            raw = json.dumps(inner, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            (stage / "stage.json").write_bytes(raw)
            (stage / "stage.json").chmod(0o644)
            output = scratch / "observation.json"
            argv = [settings["bwrap"], "--die-with-parent", "--new-session", "--unshare-all",
                    "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc",
                    "--bind", str(scratch), str(scratch), "--ro-bind", str(stage), str(stage),
                    "--chdir", str(scratch), "--cap-drop", "ALL", "--clearenv",
                    "--setenv", "PATH", "/usr/local/cuda/bin:/usr/bin:/bin",
                    "--setenv", "TMPDIR", str(scratch), "--setenv", "XDG_CACHE_HOME", str(scratch / "cache"),
                    "--setenv", "CUTE_DSL_CACHE_DIR", str(scratch / "cute"),
                    "--setenv", "TVM_FFI_CACHE_DIR", str(scratch / "ffi"),
                    settings["python"], "-I", "-B", str(stage / "evaluator/entry.py"),
                    "--stage", str(stage), "--stage-sha256", sha(raw), "--output", str(output), "--require-readonly"]
            def limits():
                resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                completed = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           timeout=outer.remaining_timeout(settings), preexec_fn=limits)
                stdout.seek(0); stderr.seek(0)
                raw_output = output.read_text() if output.is_file() else None
                entry.verify_stage(stage, sha(raw))
                reports[label] = {"exit_code": completed.returncode, "stdout": stdout.read().decode(),
                                  "stderr": stderr.read().decode(), "observation": raw_output,
                                  "stage_sha256": sha(raw), "files": files}
    return {"action": "fa4_execute", "exit_code": 0, "bwrap_sha256": settings["bwrap_sha256"],
            "stage_sha256": sha((root / "stage.json").read_bytes()), "loaded": manifest["files"],
            "boundary": "fa4-agate-f1-bwrap-v1", "reports": reports}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    framing = module(root / "_remote.py", "ks_frame_main")
    settings = json.loads((root / "remote.json").read_bytes())
    framing.remaining_timeout(settings)
    if sys.argv[1] == "fa4_probe":
        import torch
        entry = module(root / "evaluator/entry.py", "ks_probe_entry")
        framing.emit({"action": "fa4_probe", "exit_code": 0,
                      "environment": entry.environment(json.loads((root / "environment.json").read_bytes())["required"], torch),
                      "bwrap_sha256": sha(Path(settings["bwrap"]).read_bytes())})
    elif sys.argv[1] == "fa4_execute":
        framing.emit(run(root, settings))
    else:
        raise ValueError("unsupported FA4 action")
