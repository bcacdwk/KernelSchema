"""Reviewed runtime replacements without changing a run's semantic input lock.

The state pointer is the only commit. Sealed migration records retain the exact
pre-upgrade state, report, review and target lock, including successive upgrades.
"""

from __future__ import annotations

import ast
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

from .common import (ExtractError, digest, fsync_directory, json_bytes, read_json,
                     rules_root, seal, sha256, source_root, sync_tree, verify_manifest, write_json)
from .compatibility import verify_report, verify_review
from .inputs import SourceTree
from .snapshots import freeze_inputs, frozen_invocation, verify_executor, verify_run_inputs


CAPABILITY = "runtime_binding.v1"
EXECUTOR_CAPABILITIES = ("summary.v1", "state_guard.v1", "hold.v1", "scoped_restart.v1",
                         "scoped_recovery.v2", "continuation.v1", "audit.v1", "proposal.v1",
                         "budget_extension.v1", "runtime_binding.v1", "semantic_revision.v1")
STATE_FORMAT = "schema_extract.run.v2.runtime.v1"
BINDING_FORMAT = "schema_extract.execution_binding.v1"
INPUT_POLICY = "source_lock_for_source_config_semantic_lock_for_specs_checks_and_prompt_materials"
LEGACY_INPUT_POLICY = "original_lock_for_source_specs_checks_and_prompt_materials"
_BOUNDARY = ContextVar("schema_extract_execution_boundary", default=None)


def execution_boundary(function):
    """Share one deep check across a call's nested approval/ancestor checks."""
    @wraps(function)
    def checked(store, state, *args, **kwargs):
        if not state.get("execution_binding") and not state.get("execution_bindings") and not state.get("semantic_revision") and state.get("format") != STATE_FORMAT:
            return function(store, state, *args, **kwargs)
        active = _BOUNDARY.get()
        key = (store.root, state["run_id"], state.get("execution_binding"), state.get("semantic_revision"))
        if active and active["key"] == key:
            return function(store, state, *args, **kwargs)
        verified = {}
        verify_execution(store, state, _verified=verified)
        token = _BOUNDARY.set({"key": key, **verified})
        try:
            return function(store, state, *args, **kwargs)
        finally:
            _BOUNDARY.reset(token)
    return checked


def _identity(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ExtractError("invalid execution binding ID")
    return value


def binding_directory(store, run_id: str, binding_id: str) -> Path:
    return store.run_dir(run_id) / "runtime" / _identity(binding_id)


def _history(state: dict) -> list[str]:
    from .revisions import STATE_FORMAT as SEMANTIC_STATE_FORMAT
    history = state.get("execution_bindings", [])
    if not isinstance(history, list):
        raise ExtractError("run execution binding history is not a list")
    for identity in history:
        _identity(identity)
    if (len(set(history)) != len(history) or
            state.get("execution_binding") != (history[-1] if history else None) or
            (bool(history) != (state["format"] == STATE_FORMAT) and state["format"] != SEMANTIC_STATE_FORMAT)):
        raise ExtractError("run execution binding history or format is inconsistent")
    return history


class _BasisStore:
    """Verify an archived report against its sealed state, never a later state."""
    def __init__(self, store, before):
        self.store, self.before = store, before
        self.root = store.root

    def load(self, run_id):
        if run_id != self.before["run_id"]:
            raise ExtractError("compatibility basis belongs to another run")
        return deepcopy(self.before)

    def run_dir(self, run_id):
        return self.store.run_dir(run_id)

    def verify_checkpoint(self, state, checkpoint_id=None):
        return self.store.verify_checkpoint(state, checkpoint_id)


def _record(store, state, binding_id, prefix, lock, evidence=None):
    directory = binding_directory(store, state["run_id"], binding_id)
    record = read_json(directory / "binding.json")
    if (record.get("format") != BINDING_FORMAT or record.get("id") != binding_id or
            digest({key: value for key, value in record.items() if key != "id"}) != binding_id):
        raise ExtractError("execution binding digest or format mismatch")
    payloads = {name: read_json(directory / name) for name in
                ("before.json", "report.json", "review.json", "target.lock.json", "inputs.lock.json")}
    hashes = {"binding.json": sha256(json_bytes(record)),
              **{name: sha256(json_bytes(value)) for name, value in payloads.items()}}
    verify_manifest(directory, hashes)
    for path in (directory, *directory.rglob("*")):
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise ExtractError("execution binding evidence is not immutable")
    before, report, review, target = (payloads[name] for name in
                                     ("before.json", "report.json", "review.json", "target.lock.json"))
    from .revisions import semantic_rules_lock
    semantic = semantic_rules_lock(store, before)
    if (record["run_id"] != state["run_id"] or record["input_fingerprint"] != state["input_fingerprint"] or
            payloads["inputs.lock.json"] != lock or record["inputs_lock_sha256"] != digest(lock) or
            record["before_sha256"] != digest(before) or record["report_id"] != report["id"] or
            record["review_sha256"] != digest(review) or record["target_lock_sha256"] != digest(target) or
            record["predecessor"] != (prefix[-1] if prefix else None) or _history(before) != prefix or
            record["semantic_rules_id"] != semantic["rules"]["id"] or
            record["input_policy"] not in ({INPUT_POLICY} if before.get("semantic_revision") else {INPUT_POLICY, LEGACY_INPUT_POLICY}) or
            record["target_fingerprint"] != target["fingerprint"] or
            set(record["attempt_sha256"]) != set(before["attempts"])):
        raise ExtractError("execution binding source, target, history or inputs changed")
    verify_report(_BasisStore(store, before), report, target_lock=target)
    if verify_review(report, review)["outcome"] != "compatible_runtime":
        raise ExtractError("execution binding does not have compatible independent review")
    expected = _executor(target)
    if record["executor"] != expected:
        raise ExtractError("execution binding executor identity changed")
    if evidence:
        for name, checksum in hashes.items():
            evidence(directory / name, sha256=checksum)
        artifacts, checkpoint = store.verify_checkpoint(before, report["basis"]["checkpoint_id"])
        evidence(artifacts.parent / "checkpoint.json", sha256=sha256(json_bytes(checkpoint)))
        for name, checksum in checkpoint["files"].items():
            evidence(artifacts / name, sha256=checksum)
    return record, before, target


def verify_execution(store, state: dict, *, executor=False, attempt: dict | None = None,
                     attempts=None, evidence=None, _verified=None, _semantic_records=None) -> dict:
    """One gate for execution, approvals, publication and explicit deep audit.

    Return the precise execution lock. Domain rules are selected independently
    by revisions.resolve_context; the original lock remains source authority. ``evidence(path, sha256=...)`` collects sealed audit bytes.
    """
    from .revisions import verify_revisions
    semantic_records = verify_revisions(store, state, evidence=evidence) if _semantic_records is None else _semantic_records
    history = _history(state)
    active = _BOUNDARY.get()
    if evidence is None and active and active["key"] == (store.root, state["run_id"], state.get("execution_binding"), state.get("semantic_revision")):
        selected, records = active["selected"], active["records"]
    else:
        lock = read_json(store.run_dir(state["run_id"]) / "inputs.lock.json")
        verify_run_inputs(state, lock)
        selected, records = lock, []
        for index, binding_id in enumerate(history):
            record, before, selected = _record(store, state, binding_id, history[:index], lock, evidence)
            records.append((record, before))
            if evidence:
                snapshot = Path(selected["snapshot"])
                evidence(snapshot / "inputs.lock.json", sha256=sha256(json_bytes(selected)))
                for kind in ("source", "rules"):
                    for name, checksum in selected[kind]["files"].items():
                        evidence(snapshot / kind / name, sha256=checksum)
                if selected["source"].get("ref_kind") == "private_snapshot_commit":
                    for path in sorted((snapshot / "source.git").rglob("*")):
                        if path.is_file():
                            evidence(path)
    if semantic_records and semantic_records[-1]["record"].get("execution_binding") == state.get("execution_binding"):
        selected = semantic_records[-1]["target_lock"]
    if semantic_records and "semantic_revision.v1" not in _executor(selected)["capabilities"]:
        raise ExtractError("executor cannot read the current semantic state")
    if (records or semantic_records) and state.get("executor") != _executor(selected):
        raise ExtractError("run executor differs from its registered execution context")
    selected_for_attempt = None
    for attempt in ([attempt] if attempt is not None else attempts or []):
        if (attempt.get("run_id") != state["run_id"] or
                attempt.get("input_fingerprint") != state["input_fingerprint"] or
                attempt.get("attempt_id") not in state["attempts"]):
            raise ExtractError("attempt does not belong to the bound run")
        expected = history[-1] if history else None
        for record, before in records:
            if attempt["attempt_id"] in record["attempt_sha256"]:
                if digest(attempt) != record["attempt_sha256"][attempt["attempt_id"]]:
                    raise ExtractError("pre-upgrade attempt identity changed")
                expected = before.get("execution_binding")
                break
        if attempt.get("execution_binding") != expected:
            raise ExtractError("attempt executor differs from its registered execution interval")
        # The first later semantic/runtime record freezes the exact executor
        # that owned this attempt; later upgrades cannot relabel its evidence.
        candidates = []
        for semantic in semantic_records:
            if attempt["attempt_id"] in semantic["record"]["attempt_sha256"]:
                candidates.append(semantic["before"])
                break
        for record, before in records:
            if attempt["attempt_id"] in record["attempt_sha256"]:
                candidates.append(before)
                break
        if candidates:
            basis = min(candidates, key=lambda item: len(item["attempts"]) + len(item.get("semantic_revisions", [])) + len(item.get("execution_bindings", [])))
            if basis.get("semantic_revision") or basis.get("execution_binding"):
                selected_for_attempt = verify_execution(store, basis)
            else:
                selected_for_attempt = read_json(store.run_dir(state["run_id"]) / "inputs.lock.json")
        else:
            selected_for_attempt = selected
    if executor:
        verify_executor(selected)
    if _verified is not None:
        _verified.update(selected=selected, records=records, semantic_records=semantic_records)
    return selected_for_attempt or selected


def frozen_run_invocation(store, state: dict, arguments: list[str]):
    command, _, environment = frozen_invocation(verify_execution(store, state), arguments)
    return command, store.run_dir(state["run_id"]), environment


def _literal(lock: dict, name: str, field: str):
    if name not in lock["rules"]["files"]:
        raise ExtractError("target executor is missing " + name)
    try:
        module = ast.parse((rules_root(lock) / name).read_text())
        values = [ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == field for target in node.targets)]
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ExtractError("target executor has no valid literal " + field) from exc
    if len(values) != 1:
        raise ExtractError("target executor needs one literal " + field)
    return values[0]


def _executor(lock: dict) -> dict:
    if "schema_extract/tools/runtime.py" not in lock["rules"]["files"]:
        raise ExtractError("target executor does not support registered runtime bindings")
    version = _literal(lock, "schema_extract/tools/__init__.py", "__version__")
    capabilities = _literal(lock, "schema_extract/tools/runtime.py", "EXECUTOR_CAPABILITIES")
    if not isinstance(version, str):
        raise ExtractError("target executor has no literal version identity")
    if (not isinstance(capabilities, (tuple, list)) or not all(isinstance(value, str) for value in capabilities) or
            len(capabilities) != len(set(capabilities)) or CAPABILITY not in capabilities or
            _literal(lock, "schema_extract/tools/runtime.py", "STATE_FORMAT") != STATE_FORMAT):
        raise ExtractError("target executor does not declare this binding format and its capabilities")
    return {"version": version, "rules_id": lock["rules"]["id"],
            "digest": digest({name: checksum for name, checksum in lock["rules"]["files"].items()
                              if name.startswith("schema_extract/") and name.endswith(".py")}),
            "capabilities": list(capabilities)}


def _quiescent(store, state, ledger):
    from .monitor import _alive, _processes
    if state.get("active_attempts") or state.get("unconfirmed_attempts") or state.get("unsettled_attempts"):
        raise ExtractError("upgrade-runtime requires all old attempts and accounting to be settled")
    launch = ledger["starts"].get(state["run_id"], {}).get("launch", {})
    identities = []
    if launch.get("status") in {"launching", "claimed", "termination_unknown"}:
        identities.append(launch.get("identity") or launch.get("launcher"))
    if state.get("runner"):
        identities.append(state["runner"].get("identity"))
    if identities:
        table, error = _processes()
        if any(_alive(identity, table) is not False for identity in identities):
            raise ExtractError("old writer or launch is live or unconfirmed" + (": " + error if error else ""))
    if state["run_id"] in ledger["continuations"]:
        raise ExtractError("a run with a reserved semantic successor cannot upgrade its runtime")


def _retire_sessions(store, state, binding, *, reason="reviewed_runtime_upgrade"):
    for group, roles in state["sessions"].items():
        for role, session in roles.items():
            if session is None:
                continue
            attempts = (read_json(store.run_dir(state["run_id"]) / "attempts" / identity / "attempt.json")
                        for identity in reversed(state["attempts"]))
            attempt = next((item for item in attempts if item["session_group"] == group and item["role"] == role), None)
            if attempt is None:
                raise ExtractError("active role session has no registered source attempt")
            checkpoint = (state["nodes"][attempt["node"]].get("draft_checkpoint")
                          if attempt["stage"] == "P5" else state["checkpoint"]) or state["checkpoint"]
            store.verify_checkpoint(state, checkpoint)
            state.setdefault("session_handoffs", []).append({
                "attempt_id": attempt["attempt_id"], "session_group": group, "stage": attempt["stage"],
                "node": attempt["node"], "role": role, "previous_session_id": session,
                "next_attempt_id": None, "next_session_id": None, "checkpoint": checkpoint,
                "input_fingerprint": state["input_fingerprint"],
                "execution_binding": binding["id"] if reason == "reviewed_runtime_upgrade" else binding.get("execution_binding"),
                "reason": reason, "at": binding["registered_at"],
                **({"semantic_revision": binding["id"]} if reason == "semantic_revision" else {})})
            roles[role] = None


@execution_boundary
def _retained_approvals(store, state):
    from .stages import effective_nodes
    for name, stage in state["stages"].items():
        if name != "P5" and stage.get("status") == "approved":
            store.verify_approval(state, stage["approval"])
    for node_id in effective_nodes(state):
        store.verify_node(state, node_id)


def upgrade_runtime(store, run_id: str, report: dict, review: dict, *, target_repo: Path | None = None) -> dict:
    """Register only. Never dispatch, clear controls, reset budgets or reapprove."""
    with store.writer(run_id):
        state = store.load(run_id)
        store.assert_constructed(run_id)
        verify_execution(store, state)
        history = _history(state)
        for binding_id in history:
            existing = read_json(binding_directory(store, run_id, binding_id) / "binding.json")
            if existing["report_id"] == report.get("id"):
                if (existing["review_sha256"] != digest(review) or
                        read_json(binding_directory(store, run_id, binding_id) / "report.json") != report):
                    raise ExtractError("this upgrade is already registered with a different report or review")
                return {"run_id": run_id, "binding": existing, "reused": True, "started": False}
        observed = store.read_supervisor()
        from .revision_control import revision_barrier, maintenance_active
        if revision_barrier(store, state, ledger=observed) or maintenance_active(state):
            raise ExtractError("runtime upgrade cannot overlap pending semantic revision or maintenance")
        _quiescent(store, state, observed)
        _retained_approvals(store, state)
        if report.get("basis", {}).get("run_id") != run_id:
            raise ExtractError("upgrade report belongs to another run")
        if verify_review(report, review)["outcome"] != "compatible_runtime":
            raise ExtractError("semantic or unknown changes require continue and independent re-review")
        verify_report(store, report, target_repo)
        lock = read_json(store.run_dir(run_id) / "inputs.lock.json")
        target = freeze_inputs(store.root, SourceTree(source_root(lock)), lock["source"]["bindings"],
                               sorted(lock["source"]["files"]), lock["config"],
                               rules_repo=(target_repo or rules_root()).resolve())
        verify_report(store, report, target_lock=target)
        from .revisions import semantic_rules_lock, STATE_FORMAT as SEMANTIC_STATE_FORMAT
        semantic = semantic_rules_lock(store, state)
        record = {"format": BINDING_FORMAT, "run_id": run_id, "input_fingerprint": state["input_fingerprint"],
                  "inputs_lock_sha256": digest(lock), "predecessor": history[-1] if history else None,
                  "before_sha256": digest(state), "report_id": report["id"], "review_sha256": digest(review),
                  "target_lock_sha256": digest(target), "target_fingerprint": target["fingerprint"],
                  "semantic_rules_id": semantic["rules"]["id"], "input_policy": INPUT_POLICY,
                  "executor": _executor(target), "registered_at": time.time(),
                  "attempt_sha256": {identity: digest(read_json(store.run_dir(run_id) / "attempts" / identity / "attempt.json"))
                                     for identity in state["attempts"]}}
        if state.get("semantic_revision") and "semantic_revision.v1" not in record["executor"]["capabilities"]:
            raise ExtractError("runtime target cannot read the current semantic state")
        record["id"] = digest(record)
        updated = deepcopy(state)
        _retire_sessions(store, updated, record)
        updated.update(format=SEMANTIC_STATE_FORMAT if state.get("semantic_revision") else STATE_FORMAT, execution_binding=record["id"],
                       execution_bindings=[*history, record["id"]], executor=record["executor"])
        root = store.run_dir(run_id) / "runtime"
        root.mkdir(exist_ok=True)
        temporary = root / (".building-" + uuid4().hex)
        temporary.mkdir()
        try:
            for name, value in (("binding.json", record), ("before.json", state), ("report.json", report),
                                ("review.json", review), ("target.lock.json", target), ("inputs.lock.json", lock)):
                write_json(temporary / name, value)
            seal(temporary)
            sync_tree(temporary)
            os.rename(temporary, root / record["id"])
            fsync_directory(root)
            verify_execution(store, updated)
            # Freeze and process inspection happen outside this short control
            # transaction. Every launch must check its observed binding here too.
            with store.supervisor() as ledger:
                if (ledger["starts"].get(run_id, {}).get("launch") != observed["starts"].get(run_id, {}).get("launch") or
                        run_id in ledger["continuations"] or store.load(run_id) != state or
                        revision_barrier(store, state, ledger=ledger)):
                    raise ExtractError("run or launch changed before runtime binding registration")
                store.save(updated)
            return {"run_id": run_id, "binding": record, "reused": False, "started": False}
        finally:
            if temporary.exists():
                for path in temporary.rglob("*"):
                    path.chmod(0o755 if path.is_dir() else 0o644)
                temporary.chmod(0o755)
                shutil.rmtree(temporary)
