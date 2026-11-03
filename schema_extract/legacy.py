"""Explicit, read-only import of terminated v1 checkpoints and their provenance."""

from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .common import ExtractError, manifest, read_json, regular_file, sha256
from .inputs import resolve_request
from .processes import owned_processes, process_table


def _v1_digest(value) -> str:
    # v1 identities tag JSON types; v2's ordinary canonical JSON is different.
    def exact(item):
        if isinstance(item, dict):
            return ["mapping", [[key, exact(item[key])] for key in sorted(item)]]
        if isinstance(item, list):
            return ["sequence", [exact(value) for value in item]]
        return [type(item).__name__, item]
    return sha256(json.dumps(exact(value), ensure_ascii=False, separators=(",", ":")).encode())


def _verify_inputs(task: dict, lock: dict) -> Path:
    if (task.get("format") != 1 or _v1_digest(lock["identity"]) != task["fingerprint"] or
            lock["fingerprint"] != task["fingerprint"] or lock["files"] != lock["identity"]["files"]):
        raise ExtractError("legacy task and frozen input identity do not match")
    snapshot = Path(lock["snapshot"]).resolve()
    for name, expected in lock["files"].items():
        if sha256(regular_file(snapshot, name).read_bytes()) != expected:
            raise ExtractError("legacy fixed input changed: " + name)
    for name in ("schema_extract/__init__.py", "schema_extract/__main__.py", "schema_extract/runner.py"):
        if name not in lock["files"]:
            raise ExtractError("legacy lock does not contain its actual frozen executor: " + name)
    return snapshot


def _frozen_status(source_task: Path, task: dict, snapshot: Path) -> tuple[dict, dict]:
    root = source_task.parent.parent
    plans = []
    for path in sorted((root / "runs").glob("*/plan.json")):
        plan = read_json(path)
        if task["key"] in plan.get("tasks", []):
            if plan.get("run_id") != path.parent.name:
                raise ExtractError("legacy run plan identity mismatch")
            plans.append(plan)
    if not plans:
        raise ExtractError("legacy task needs its original run plan for frozen status reading")
    plan = plans[0]
    argv = [sys.executable, "-B", "-m", "schema_extract", "--repo", str(snapshot),
            "--state-root", str(root), "status", plan["run_id"], "--full"]
    result = subprocess.run(argv, cwd=snapshot, env={**os.environ, "PYTHONPATH": str(snapshot),
                                                   "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ExtractError("legacy frozen status failed: " + result.stderr.strip())
    try:
        matches = [value for value in json.loads(result.stdout) if value.get("key") == task["key"]]
    except (ValueError, TypeError, AttributeError) as exc:
        raise ExtractError("legacy frozen executor did not return its task state") from exc
    if matches != [task]:
        raise ExtractError("legacy frozen status differs from the selected task record")
    return task, {"run_id": plan["run_id"], "argv": argv, "cwd": str(snapshot), "exit_code": result.returncode}


def _terminated(source_task: Path, task: dict) -> list[dict]:
    if task["status"] not in {"ended", "stopped"} or task.get("active_process"):
        raise ExtractError("legacy task is not stopped; confirm termination using its frozen executor first")
    confirmations = []
    for attempt in task["attempts"]:
        for call in attempt["calls"]:
            directory = Path(call).resolve()
            if not directory.is_relative_to(source_task / "attempts"):
                raise ExtractError("legacy invocation record leaves its task")
            result_path, recovery_path = directory / "result.json", directory / "recovery_termination.json"
            result = read_json(result_path).get("termination", {}) if result_path.is_file() else {}
            confirmation = read_json(recovery_path) if recovery_path.is_file() else result
            if confirmation.get("confirmed") is not True:
                raise ExtractError("legacy invocation termination is not confirmed: " + str(directory))
            process_path = directory / "process.json"
            if process_path.is_file():
                process = read_json(process_path)
                identity = process["identity"]
                table = process_table()
                group = [item for item in table.values() if item["pgid"] == identity["pid"] and not item["stat"].startswith("Z")]
                if group or owned_processes(identity, process.get("descendants", [])):
                    raise ExtractError("legacy invocation is live or its process group ownership is uncertain: " + str(directory))
            confirmations.append({"call": str(directory), "confirmation": confirmation})
    return confirmations


def import_legacy(repo: Path, state_root: Path, output: Path, *, source_task: Path,
                  checkpoint_id: str, source: str, bindings: Path | None = None,
                  config: dict | None = None, queued=False):
    """Create a v2 P1 draft; no old review, stage approval or eligible bit is reused."""
    from .runner import create_run
    source_task, state_root = source_task.resolve(), state_root.resolve()
    if source_task.parent.name != "tasks" or not re.fullmatch(r"[A-Za-z0-9_-]+", checkpoint_id):
        raise ExtractError("select a legacy tasks/<task-key> directory and its exact checkpoint ID")
    if state_root == source_task.parent.parent or state_root.is_relative_to(source_task):
        raise ExtractError("legacy imports must use a separate v2 state root")
    if output.resolve().is_relative_to(source_task.parent.parent):
        raise ExtractError("legacy import output must not overwrite the old runtime storage")
    lock_path = source_task.parent.parent / ".runner.lock"
    if not lock_path.is_file():
        raise ExtractError("legacy runner lock is missing; retain the original state root and confirm its frozen executor is stopped")
    with lock_path.open("r") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExtractError("legacy Runner is active; stop it through its frozen executor before importing") from exc
        task, old_lock = read_json(source_task / "task.json"), read_json(source_task / "inputs.lock.json")
        if task.get("key") != source_task.name:
            raise ExtractError("legacy task key does not match its directory")
        snapshot = _verify_inputs(task, old_lock)
        task, status_receipt = _frozen_status(source_task, task, snapshot)
        terminations = _terminated(source_task, task)
        if checkpoint_id not in task.get("history", []) or not task.get("checkpoint"):
            raise ExtractError("checkpoint is not part of this legacy task's accepted history")
        checkpoint = source_task / "checkpoints" / checkpoint_id
        record = read_json(checkpoint / "checkpoint.json")
        if (record.get("id") != checkpoint_id or record.get("fingerprint") != task["fingerprint"] or
                manifest(checkpoint / "artifacts") != record["files"]):
            raise ExtractError("legacy checkpoint identity or artifact contents changed")
        regular_file(checkpoint / "artifacts", "root.yaml")
        tree, normalized, names = resolve_request(repo, source=source, bindings=bindings)
        if normalized["id"] != task["operator_id"]:
            raise ExtractError("legacy and new operator IDs differ")
        # The descriptor may point at the old snapshot or an exact historical
        # tree. Every selected byte still has to appear in the old frozen lock.
        files = {name: sha256(tree.read(name)) for name in names}
        if any(old_lock["files"].get(name) != value for name, value in files.items()):
            raise ExtractError("import source must be the legacy frozen source bytes; new evidence needs a new extract run")
        origins = [{"repository_id": old_lock["repository_id"], "ref": old_lock["snapshot_commit"], "files": files}]
        for binding in task.get("source_bindings", []):
            subset = {name: files[name] for name in binding["paths"] if name in files}
            if subset:
                origins.append({"repository_id": binding["repository_id"], "ref": binding["commit"], "files": subset})
        provenance = {"format": "schema_extract.import.v1-to-v2", "task_key": task["key"],
                      "checkpoint": checkpoint_id, "fingerprint": task["fingerprint"],
                      "source_task": str(source_task), "files": record["files"],
                      "usage": deepcopy(task["usage"]), "usage_observations": deepcopy(task.get("usage_observations", {})),
                      "source_bindings": origins, "prior_issues": deepcopy(task.get("issues", [])),
                      "frozen_status": status_receipt, "termination_confirmations": terminations}
        seed = {"artifact_directory": checkpoint / "artifacts", "checkpoint_record": record,
                "input_lock": old_lock, "task_record": task, "provenance": provenance, "source_aliases": origins}
        if read_json(source_task / "task.json") != task:
            raise ExtractError("legacy task changed while preparing its import")
        return create_run(repo, state_root, output, source=source, bindings=bindings, config=config, import_seed=seed, queued=queued)
