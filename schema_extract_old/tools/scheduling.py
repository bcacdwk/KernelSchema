"""Start queued work and manage the optional macOS supervisor schedule."""

import hashlib
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from .codex_cli import write_json
from .records import StateError, optional_json, try_lock


INTERVAL_SECONDS = 15 * 60


def runner_busy(store):
    path = store.root / ".runner.lock"
    if not path.exists():
        return False
    with try_lock(path) as handle:
        return handle is None


def executable_path(value):
    found = shutil.which(str(value))
    if not found:
        raise StateError(f"executable not found: {value}")
    return str(Path(found).absolute())


def start_runner(store, run_id, codex):
    directory, plan = store.load_plan(run_id)
    control = optional_json(directory / "control.json")
    if control and control.get("mode"):
        return {"outcome": "control_requested", "mode": control["mode"]}
    states = [store.load(key) for key in plan["tasks"]]
    if runner_busy(store):
        return {"outcome": "runner_already_active"}
    if any(s.get("active_process") or s["status"] == "running" for s in states):
        return {"outcome": "active_work_requires_investigation"}
    if not any(s["status"] == "queued" for s in states):
        return {"outcome": "no_queued_work"}
    log = directory / "supervisor" / ("runner-" + uuid4().hex + ".log")
    command = [str(Path(sys.executable).resolve()), "-m", "schema_extract",
               "--repo", str(store.repo), "--state-root", str(store.root), "--codex", codex, "run", run_id]
    # Keep the existing runner entry point and its authoritative Store.lock.
    # A race with a manual/other-run launch is refused there and recorded here.
    with log.open("ab") as output:
        process = subprocess.Popen(command, cwd=store.repo, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"outcome": "requested", "pid": process.pid, "requested_at": time.time(), "argv": command, "log": str(log)}
    write_json(directory / "supervisor" / "runner.json", result)
    return result


def schedule_label(store, run_id):
    return "com.kernelschema.supervisor." + hashlib.sha256(
        (str(store.repo) + "\0" + str(store.root) + "\0" + run_id).encode()).hexdigest()[:20]


def schedule_config(store, run_id, codex):
    directory, _ = store.load_plan(run_id)
    label = schedule_label(store, run_id)
    return {"run_id": run_id, "repo": str(store.repo), "state_root": str(store.root),
            "python": str(Path(sys.executable).resolve()), "codex": executable_path(codex),
            "interval_seconds": INTERVAL_SECONDS, "label": label,
            "plist": str(Path.home() / "Library/LaunchAgents" / (label + ".plist")),
            "records": str(directory / "supervisor")}


def launchd_state(label):
    result = subprocess.run(["/bin/launchctl", "print", f"gui/{os.getuid()}/{label}"], text=True, capture_output=True)
    # A failed query can mean an absent service or a query error; do not claim
    # successful unloading from this result alone.
    return {"loaded": True if result.returncode == 0 else None,
            "query_exit_code": result.returncode, "stderr": result.stderr.strip()[:400]}


def launchd_plist(config):
    path = list(dict.fromkeys([str(Path(config["codex"]).parent), str(Path(config["python"]).parent),
                              "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]))
    return {"Label": config["label"], "ProgramArguments": [config["python"], "-m", "schema_extract.supervisor",
            "--repo", config["repo"], "--state-root", config["state_root"], "--codex", config["codex"], "tick", config["run_id"]],
            "WorkingDirectory": config["repo"], "StartInterval": INTERVAL_SECONDS,
            "RunAtLoad": False, "KeepAlive": False, "EnvironmentVariables": {"PATH": os.pathsep.join(path)},
            "StandardOutPath": str(Path(config["records"]) / "launchd.stdout.log"),
            "StandardErrorPath": str(Path(config["records"]) / "launchd.stderr.log")}


def schedule(store, run_id, codex, *, enable):
    if sys.platform != "darwin":
        raise StateError("enable/disable use macOS launchd; call tick from an external scheduler elsewhere")
    directory, _ = store.load_plan(run_id)
    directory = directory / "supervisor"
    directory.mkdir(exist_ok=True)
    with try_lock(directory / ".schedule.lock") as handle, try_lock(directory / ".tick.lock") as tick_handle:
        if handle is None:
            raise StateError("another schedule update is active")
        if tick_handle is None:
            raise StateError("supervisor tick is active; wait for it to exit or explicitly cancel the run first; schedule left unchanged")
        config_path = directory / "schedule.json"
        previous = optional_json(config_path)
        if not enable and previous is None:
            return {"outcome": "disabled", "run_id": run_id}
        config = schedule_config(store, run_id, codex) if enable else dict(previous)
        if config["repo"] != str(store.repo) or config["state_root"] != str(store.root) or config["run_id"] != run_id:
            raise StateError("saved schedule belongs to another run")
        if config["label"] != schedule_label(store, run_id):
            raise StateError("saved launchd label does not match this run")
        path = Path(config["plist"])
        expected_path = Path.home() / "Library/LaunchAgents" / (config["label"] + ".plist")
        if path != expected_path:
            raise StateError("saved launchd path is not this schedule's plist")
        if previous and previous["label"] != config["label"]:
            raise StateError("saved launchd label does not match this run")
        if path.exists() and (not previous or plistlib.loads(path.read_bytes()) != launchd_plist(previous)):
            raise StateError("existing plist is not the saved schedule; left unchanged")
        target = f"gui/{os.getuid()}/{config['label']}"
        outcomes = []
        loaded = launchd_state(config["label"])
        if loaded["loaded"] and not previous:
            raise StateError("launchd label already exists without an owned local configuration")
        if not enable and previous.get("outcome") == "disabled" and not path.exists() and not loaded["loaded"]:
            return {**previous, "launchd": loaded}
        if loaded["loaded"] or not enable:
            result = subprocess.run(["/bin/launchctl", "bootout", target], text=True, capture_output=True)
            outcomes.append({"action": "bootout", "exit_code": result.returncode, "stderr": result.stderr.strip()})
            if result.returncode:
                if not enable and path.exists():
                    path.unlink()  # Disable future login loading even if live unload was not confirmed.
                config.update(outcome="error", updated_at=time.time(), launchctl=outcomes, launchd=loaded)
                write_json(config_path, config)
                return config
        if enable:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".plist.tmp")
            temporary.write_bytes(plistlib.dumps(launchd_plist(config)))
            os.replace(temporary, path)
            result = subprocess.run(["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], text=True, capture_output=True)
            outcomes.append({"action": "bootstrap", "exit_code": result.returncode, "stderr": result.stderr.strip()})
            outcome = "enabled" if result.returncode == 0 else "error"
            if outcome == "error":
                path.unlink()  # A failed enable must not become active at next login.
        else:
            outcome = "disabled"
            if path.exists():
                path.unlink()
        config.update(outcome=outcome, updated_at=time.time(), launchctl=outcomes)
        write_json(config_path, config)
        return config
