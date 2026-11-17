"""Admit only the current run's fixed source, rules, executor and agent identity."""

from __future__ import annotations

import ast
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from .common import ExtractError, digest, json_bytes, read_json, rules_root, sha256
from .snapshots import frozen_invocation, verify_executor, verify_run_inputs


STATE_FORMAT = "schema_extract.run.v2"
EXECUTOR_CAPABILITIES = ("summary.v1", "state_guard.v1", "hold.v1", "scoped_restart.v1",
                         "scoped_recovery.v2", "audit.v1", "proposal.v1",
                         "budget_extension.v1", "frozen_execution.v1", "current_maintenance.v1",
                         "design_region_derivation.v1")
_BOUNDARY = ContextVar("schema_extract_execution_boundary", default=None)


def _key(store, state):
    return (store.root, state.get("run_id"), digest({key: state.get(key) for key in
            ("format", "input_fingerprint", "snapshot", "source_identity", "config", "executor")}))


def execution_boundary(function):
    """Share a fixed-input check across nested approval and actual-source checks."""
    @wraps(function)
    def checked(store, state, *args, **kwargs):
        key = _key(store, state)
        active = _BOUNDARY.get()
        if active and active["key"] == key:
            return function(store, state, *args, **kwargs)
        lock = verify_execution(store, state)
        token = _BOUNDARY.set({"key": key, "lock": lock})
        try:
            return function(store, state, *args, **kwargs)
        finally:
            _BOUNDARY.reset(token)
    return checked


def _literal(lock: dict, name: str, field: str):
    if name not in lock["rules"]["files"]:
        raise ExtractError("frozen executor is missing " + name)
    try:
        module = ast.parse((rules_root(lock) / name).read_text())
        values = [ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == field for target in node.targets)]
    except (OSError, SyntaxError, ValueError, TypeError) as exc:
        raise ExtractError("frozen executor has no valid literal " + field) from exc
    if len(values) != 1:
        raise ExtractError("frozen executor needs one literal " + field)
    return values[0]


def _executor(lock: dict) -> dict:
    version = _literal(lock, "schema_extract/tools/__init__.py", "__version__")
    capabilities = _literal(lock, "schema_extract/tools/runtime.py", "EXECUTOR_CAPABILITIES")
    if (not isinstance(version, str) or not version or
            not isinstance(capabilities, (tuple, list)) or
            not all(isinstance(value, str) and value for value in capabilities) or
            len(capabilities) != len(set(capabilities)) or "frozen_execution.v1" not in capabilities or
            _literal(lock, "schema_extract/tools/runtime.py", "STATE_FORMAT") != STATE_FORMAT or
            _literal(lock, "schema_extract/tools/store.py", "STATE_FORMAT") != STATE_FORMAT):
        raise ExtractError("frozen executor does not implement the current run protocol")
    return {"version": version, "rules_id": lock["rules"]["id"],
            "digest": digest({name: checksum for name, checksum in lock["rules"]["files"].items()
                              if name.startswith("schema_extract/") and name.endswith(".py")}),
            "capabilities": list(capabilities)}


def _attempt_identity(store, state, attempt):
    from .backends import exact_uuid
    from .stages import session_scope
    exact_uuid(attempt.get("attempt_id"))
    directory = store.run_dir(state["run_id"]) / "attempts" / attempt["attempt_id"]
    config = store.execution_config(state)
    if (attempt.get("run_id") != state["run_id"] or
            attempt.get("input_fingerprint") != state["input_fingerprint"] or
            attempt["attempt_id"] not in state["attempts"] or
            attempt.get("role") not in {"worker", "reviewer"} or
            any(attempt.get(key) != config[key] for key in ("backend", "model")) or
            attempt.get("executable") != config["agent_binary"] or
            attempt.get("effort") != config[attempt["role"] + "_effort"] or
            attempt.get("session_scope") != session_scope(attempt) or
            Path(attempt.get("directory", "")) != directory or
            Path(attempt.get("workspace", "")) != directory / "workspace" or
            Path(attempt.get("session_directory", "")) != store.run_dir(state["run_id"]) / "sessions" / digest(attempt["session_scope"]) or
            read_json(directory / "attempt.json") != attempt):
        raise ExtractError("attempt differs from its registered frozen run, backend or scope")
    if "session_id" not in attempt:
        raise ExtractError("attempt has no explicit session identity")
    if attempt["session_id"] is not None:
        exact_uuid(attempt["session_id"])
    if any(key in attempt for key in ("execution_binding", "semantic_revision")):
        raise ExtractError("attempt is not part of the current fixed execution protocol")


def verify_execution(store, state: dict, *, executor=False, attempt: dict | None = None,
                     attempts=None, evidence=None) -> dict:
    """Verify one fixed run; never adopt, migrate or infer missing identities.

    ``executor=True`` also requires these executing Python bytes to match the
    frozen implementation. Control entrypoints use the returned lock to launch
    that implementation before touching the run.
    """
    from .backends import backend_for
    if (state.get("format") != STATE_FORMAT or
            not {"selected_designs", "writing_plan", "active_attempts", "reopen_requests"} <= state.keys() or
            any(key in state for key in ("execution_binding", "execution_bindings", "semantic_revision",
                                         "semantic_revisions", "continuation", "imported_draft"))):
        raise ExtractError("unsupported run protocol; start a new extraction")
    config = store.execution_config(state)
    backend_for(config["backend"])
    active = _BOUNDARY.get()
    if evidence is None and active and active["key"] == _key(store, state):
        lock = active["lock"]
    else:
        lock_path = store.run_dir(state["run_id"]) / "inputs.lock.json"
        lock = read_json(lock_path)
        verify_run_inputs(state, lock)
        if state.get("source_identity") != {key: lock["source"][key]
                for key in ("id", "repository_id", "ref", "ref_kind")}:
            raise ExtractError("run source identity differs from its fixed input lock")
        if state.get("executor") != _executor(lock):
            raise ExtractError("run executor identity differs from its frozen implementation")
        if evidence:
            checksum = sha256(json_bytes(lock))
            evidence(lock_path, sha256=checksum)
            snapshot = Path(lock["snapshot"])
            evidence(snapshot / "inputs.lock.json", sha256=checksum)
            for kind in ("source", "rules"):
                for name, value in lock[kind]["files"].items():
                    evidence(snapshot / kind / name, sha256=value)
            if lock["source"]["ref_kind"] == "private_snapshot_commit":
                for path in sorted((snapshot / "source.git").rglob("*")):
                    if path.is_file():
                        evidence(path)
    for registered in ([attempt] if attempt is not None else attempts or []):
        _attempt_identity(store, state, registered)
    if executor:
        verify_executor(lock)
    return lock


def frozen_run_invocation(store, state: dict, arguments: list[str]):
    command, _, environment = frozen_invocation(verify_execution(store, state), arguments)
    return command, store.run_dir(state["run_id"]), environment
