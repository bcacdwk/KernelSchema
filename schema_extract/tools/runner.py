"""Concurrent agent calls with one authoritative, serial commit loop per operator."""

from __future__ import annotations

from copy import deepcopy
from uuid import UUID
from pathlib import Path
import os
import json
import queue
import shutil
import stat
import threading
import time

from .agent import AgentCLI
from .backends import backend_for
from .artifacts import (CandidateRejected, UpstreamRequired, approval_projection, artifact_manifest, check_stage_snapshot,
                        normalize_candidate, source_context, validate_artifact_sources,
                        validate_artifacts, validate_plan)
from .comparators import validate_comparisons
from .common import (STATE_GUARD, ExtractError, StaleResult, SupervisorBusy, copy_artifacts, digest, json_bytes, manifest,
                     read_json, seal, sha256, write_json, write_text, rules_root)
from .inputs import resolve_request
from .prompts import build_prompt
from .receipts import (ReceiptFormatError, check_proposal_identity, evidence_identity,
                       normalize_response, proposal_schema, provider_request_details, validate_receipt)
from .recovery import (HARD_RECOVERY_STOPS, account, attempt_cancelled, complete_pending_result, confirm_termination, consume_controls, elapsed_on_disk,
                       handoff_session, recover_result, release_unstarted_handoff, retry_classification, retry_delay, termination_observed, native_exit)
from .snapshots import freeze_inputs, verify_run_inputs
from .runtime import execution_boundary, verify_execution
from . import maintenance
from .stages import (CALL_STOPS, CALL_RECOVERY, add_calls, call_exhaustion, NODE_STOPS, advance, affected_nodes, call_view, effective_nodes, final_eligible,
                     invalidate_final, invalidate_node, invalidate_stage, next_calls, owner_for, resume_progress, selection_impact, selection_signature, scope_for, scope_key,
                     session_group, settle, writing_signature)
from .store import Store, summary


DEFAULT_CONFIG = {"backend": "claude", "worker_effort": "max", "reviewer_effort": "max", "model": "qwen3.8-max[1m]",
                  "agent_binary": "claude", "revision_limit": 5, "max_calls": 32}


def normalize_config(config: dict | None = None) -> dict:
    """Validate and resolve new-run configuration before reserving its output."""
    requested = config or {}
    config = {**DEFAULT_CONFIG, **requested}
    if set(config) != set(DEFAULT_CONFIG):
        raise ExtractError("unknown extraction configuration field")
    if config["backend"] not in {"claude", "codex"}:
        raise ExtractError("backend must be claude or codex")
    if config["backend"] == "codex":
        if any(key not in requested for key in ("model", "worker_effort", "reviewer_effort")):
            raise ExtractError("Codex requires explicit model and role efforts")
        if "agent_binary" not in requested:
            config["agent_binary"] = "codex"
    if not isinstance(config["model"], str) or not config["model"].strip():
        raise ExtractError("model must be explicitly registered")
    for role in ("worker", "reviewer"):
        if config[f"{role}_effort"] not in {"low", "medium", "high", "xhigh", "max"}:
            raise ExtractError(f"invalid {role} effort")
        backend_for(config["backend"]).validate_effort(config[f"{role}_effort"])
    for key in ("revision_limit", "max_calls"):
        if type(config[key]) is not int or config[key] < 1:
            raise ExtractError(f"{key} must be a positive integer")
    if config["max_calls"] < 4:
        raise ExtractError("max_calls must reserve a worker, independent review, adjudication and signoff")
    config["retry_limit"] = 2
    executable = shutil.which(config["agent_binary"])
    if executable is None:
        raise ExtractError(f"Agent CLI executable is missing: {config['agent_binary']}")
    config["agent_binary"] = str(Path(executable).resolve())
    return config


def create_run(repo: Path, state_root: Path, output: Path, *, operator_id: str | None = None,
               source: str | None = None, bindings: Path | None = None,
               config: dict | None = None, queued=False,
               run_id: str | None = None, created_at: float | None = None) -> tuple[Store, dict, dict]:
    config = normalize_config(config)
    repo, state_root, output = repo.resolve(), state_root.resolve(), output.resolve()
    if output == repo or output.is_relative_to(state_root) or state_root.is_relative_to(output):
        raise ExtractError("output must be separate from repository root and v2 runtime storage")
    if output.exists():
        raise ExtractError("output must not exist; choose a new destination")
    tree, normalized, names = resolve_request(repo, operator_id=operator_id, source=source, bindings=bindings)
    source_directory = tree.root / normalized["directory"]
    if state_root.is_relative_to(source_directory) and ".schema_extract" not in state_root.relative_to(source_directory).parts:
        raise ExtractError("runtime storage inside selected sources must be under .schema_extract, or use a separate directory")
    store = Store(state_root)
    store.initialize()
    lock = freeze_inputs(state_root, tree, normalized, names, config)
    return store, store.create(lock, output, queued=queued,
                               run_id=run_id, created_at=created_at), lock


class _AuthorityChanged(ExtractError):
    """An external state edit must preserve the last Runner-authored bytes."""


class _StateGuard:
    def __init__(self, store: Store, state: dict):
        self.directory = store.run_dir(state["run_id"])
        self.path = self.directory / "state.json"
        self.thread_id = threading.get_ident()
        self.state_bytes = json_bytes(state)

    def verify(self, state: dict | None = None) -> None:
        if threading.get_ident() != self.thread_id:
            raise _AuthorityChanged("only the Runner's owning thread may access its state guard")
        try:
            # O_NONBLOCK also prevents a substituted FIFO from stalling the
            # Runner. No directory enumeration or historical hash cache is used.
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise OSError("state.json is not a regular file")
                data = stream.read()
                after = self.path.lstat()
                if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                    raise OSError("state.json changed during reading")
        except OSError as exc:
            raise _AuthorityChanged("Runner state changed externally: " + str(exc)) from exc
        if data != self.state_bytes:
            raise _AuthorityChanged("Runner state changed externally: state.json differs from the last committed bytes")

    def written(self, data: bytes) -> None:
        if threading.get_ident() != self.thread_id:
            raise _AuthorityChanged("only the Runner's owning thread may commit state")
        self.state_bytes = data

    def read_state(self) -> dict:
        self.verify()
        return json.loads(self.state_bytes)


def _release(state: dict, attempt: dict) -> None:
    key = scope_key(attempt["stage"], attempt["node"])
    if state["active_attempts"].get(key) == attempt["attempt_id"]:
        state["active_attempts"].pop(key)
    state["unconfirmed_attempts"].pop(attempt["attempt_id"], None)


def _session(state: dict, attempt: dict, result: dict, *, required=True) -> str | None:
    if result.get("session_status") != "ready":
        if required:
            raise ExtractError("agent session is not available for an exact scoped resume")
        return None
    session_id = result.get("session_id")
    if not session_id and not required:
        return attempt["session_id"]
    try:
        if str(UUID(session_id)) != session_id:
            raise ValueError()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ExtractError("agent did not return an exact session UUID") from exc
    if any(item["previous_session_id"] == session_id for item in state.get("session_handoffs", [])):
        raise ExtractError("an interrupted session cannot be reused as the new recovery context")
    if attempt["session_id"] and session_id != attempt["session_id"]:
        raise ExtractError("agent returned a different session than the exact requested UUID")
    group = session_group(state, attempt)
    if attempt.get("session_group", group) != group:
        raise ExtractError("attempt belongs to a different role/stage session group")
    for known_group, roles in state["sessions"].items():
        for role, known_id in roles.items():
            if known_id == session_id and (known_group, role) != (group, attempt["role"]):
                raise ExtractError("each new stage/node and worker/reviewer must have independent sessions")
    known = state["sessions"].get(group, {}).get(attempt["role"])
    if known not in {None, session_id}:
        raise ExtractError("role session identity changed")
    return session_id


def _remember_session(state: dict, attempt: dict, session_id: str | None) -> None:
    if session_id:
        state["sessions"].setdefault(attempt["session_group"], {"worker": None, "reviewer": None})[attempt["role"]] = session_id
        for handoff in state.get("session_handoffs", []):
            if handoff.get("next_attempt_id") == attempt["attempt_id"]:
                handoff["next_session_id"] = session_id


def _rebuild_directory(directory: Path) -> None:
    """A replay may discard derived working copies, never immutable checkpoints."""
    if directory.is_symlink():
        raise ExtractError("derived working directory became a symlink")
    if directory.exists():
        shutil.rmtree(directory)


def _artifact_gate(store: Store, state: dict, lock: dict, attempt: dict, before: Path, after: Path,
                   decision: str | None, *, role: str | None = None) -> dict:
    context = store.context(state, attempt)
    parser_root = rules_root(context["rules_lock"])
    entry = next((item for item in state.get("selected_designs") or [] if item["id"] == attempt["node"]), None)
    validation = validate_artifacts(attempt["stage"], role or attempt["role"], before, after,
        parser_root / "schema_template/starter.yaml",
        node_entry=entry, revision_scope=state["revision_scope"],
        first_draft=attempt["stage"] == "P5" and not state["nodes"][attempt["node"]].get("authored", False),
        maintenance_fields=maintenance.task_for(state, attempt).get("fields", []) if attempt.get("purpose") == "maintenance" else None)
    validate_artifact_sources(after, context["source_lock"], decision=decision, parser_root=parser_root,
                              completed_nodes=None if attempt["stage"] == "P6" else [attempt["node"]])
    validation.update({key: attempt[key] for key in ("maintenance_task_id", "purpose", "maintenance_step") if key in attempt})
    return validation


def _normalize(store: Store, state: dict, attempt: dict, before: Path, original: Path,
               destination: Path, decision: str, *, plan=None, adjudication=None) -> dict:
    context = store.context(state, attempt)
    parser_root = rules_root(context["rules_lock"])
    entry = next((item for item in state.get("selected_designs") or [] if item["id"] == attempt["node"]), None)
    _rebuild_directory(destination)
    baseline_materials = (store.verify_checkpoint(state, attempt["baseline_context_checkpoint"])[0]
                          if attempt.get("baseline_context_checkpoint") else None)
    try:
        validation = normalize_candidate(before, original, destination, parser_root / "schema_template/starter.yaml",
            stage=attempt["stage"], role="runner" if adjudication else "worker", decision=decision,
            selected_designs=(plan if attempt["stage"] == "P2" else attempt["selected_designs"] if attempt["stage"] == "P5" else state.get("selected_designs")),
            node_entry=entry if attempt["stage"] == "P5" else None, revision_scope=state["revision_scope"],
            first_draft=attempt["stage"] == "P5" and not state["nodes"][attempt["node"]].get("authored", False),
            maintenance_fields=maintenance.task_for(state, attempt).get("fields", []) if attempt.get("purpose") == "maintenance" else None,
            adjudication=adjudication, baseline_materials=baseline_materials)
    except CandidateRejected as exc:
        write_json(Path(attempt["directory"]) / "normalization.json", exc.validation)
        raise
    try:
        validate_artifact_sources(destination, context["source_lock"], decision=decision, parser_root=parser_root,
                                  completed_nodes=None if attempt["stage"] == "P6" else [attempt["node"]])
        validation["source_check"] = {"kind": "frozen_sources", "exit_code": 0}
    except ExtractError as exc:
        validation["source_check"] = {"kind": "frozen_sources", "exit_code": 1, "diagnostic": str(exc)}
        validation.update(status="rejected", diagnostic=str(exc))
        write_json(Path(attempt["directory"]) / "normalization.json", validation)
        raise CandidateRejected(str(exc), validation) from exc
    write_json(Path(attempt["directory"]) / "normalization.json", validation)
    return validation


def _draft(store: Store, state: dict, attempt: dict, workspace: Path, *, files, plan=None) -> str:
    checkpoint_id = store.checkpoint(state, workspace, attempt_id=attempt["attempt_id"],
                                     stage=attempt["stage"], node=attempt["node"], parent=attempt["base_checkpoint"], expected_files=files)
    if attempt.get("purpose") == "maintenance":
        maintenance.task_for(state, attempt)["draft_checkpoint"] = checkpoint_id
    elif attempt["stage"] != "P5":
        state["checkpoint"] = checkpoint_id
    state["recoverable_checkpoint"] = checkpoint_id
    if attempt["stage"] == "P1":
        store.shared_drafts(state, checkpoint_id)
    if attempt["attempt_id"] in state.get("attempt_usage", {}):
        state["attempt_usage"][attempt["attempt_id"]]["result_checkpoint"] = checkpoint_id
    node = state["nodes"][attempt["node"]]
    node["draft_checkpoint"] = checkpoint_id
    if attempt["stage"] == "P5":
        node["authored"] = True
    scope_for(state, attempt)["candidates"].append({"checkpoint": checkpoint_id, "plan": deepcopy(plan),
                                                 "generation": attempt["generation"], "node_signature": attempt["node_signature"]})
    return checkpoint_id


def _record_check_failure(store: Store, state: dict, attempt: dict, receipt: dict,
                          receipt_hash: str, workspace: Path, validation: dict, session_id: str, *, accept_receipt=True) -> dict:
    updated = deepcopy(state)
    _remember_session(updated, attempt, session_id)
    code = validation["check"]["exit_code"]
    scope, owner = scope_for(updated, attempt), owner_for(updated, attempt)
    scope["check_failures" if code == 1 else "tool_failures"] += 1
    if attempt["role"] == "worker" or attempt.get("mode") == "adjudicate":
        _draft(store, updated, attempt, workspace, files=validation["artifact_digests"], plan=receipt.get("plan"))
    _release(updated, attempt)
    if accept_receipt:
        updated["accepted_receipts"][attempt["attempt_id"]] = receipt_hash
    owner["reason"] = {"code": "checker_input_error" if code == 1 else "checker_tool_error",
                         "summary": "检查器退出1；诊断已交当前修改者。" if code == 1 else "检查器退出2；需要修复工具/环境后按原调用计数恢复。",
                         "check": validation["check"], "recovery_condition": "修复记录中的检查诊断并重新核对本阶段。"}
    write_json(Path(attempt["directory"]) / "validated.json", {"receipt_sha256": receipt_hash, **validation})
    if attempt.get("purpose") == "maintenance":
        maintenance.block(updated, attempt, owner["reason"], status="tool_error" if code == 2 else "blocked")
        settle(updated)
        store.save(updated)
        return updated
    if attempt.get("mode") in {"adjudicate", "signoff"}:
        return _stop_once(store, updated, attempt, owner["reason"])
    if code == 2:
        owner["status"], scope["status"] = "tool_error", "tool_error"
    elif attempt["stage"] == "P6" and not updated["revision_scope"]:
        owner["status"], scope["status"] = "blocked", "blocked"
        owner["reason"]["summary"] = "终审目录检查失败且没有可授权的局部修改范围；需定位上游问题。"
    else:
        owner["role"], scope["status"] = "worker", "draft"
        if attempt["stage"] == "P5":
            owner["status"] = "draft"
        owner["feedback"] = [owner["reason"], *receipt["issues"]]
    settle(updated)
    store.save(updated)
    return updated


def _stop_once(store: Store, state: dict, attempt: dict, reason: dict) -> dict:
    """Adjudication and the extra signoff have exactly one invocation and two exits."""
    scope = scope_for(state, attempt)
    scope["status"] = "blocked"
    record = "signoff" if attempt.get("mode") == "signoff" else "adjudication"
    scope[record] = {**(scope.get(record) or {}), "attempt_id": attempt["attempt_id"],
                     "decision": "blocked", "reason": reason}
    owner = owner_for(state, attempt)
    owner["reason"], owner["status"] = reason, "blocked"
    if attempt["stage"] == "P5":
        node = state["nodes"][attempt["node"]]
        node["status"], node["reason"] = "blocked", reason
    settle(state)
    store.save(state)
    return state


@execution_boundary
def _final_inputs(store: Store, state: dict, workspace: Path) -> None:
    if not final_eligible(state):
        raise ExtractError("P6 prerequisites are not all current and approved")
    parser_root = rules_root(store.context(state)["rules_lock"])
    validate_plan("P2", state["selected_designs"], parser_root=parser_root)
    validate_plan("P4", state["writing_plan"], parser_root=parser_root, selected_designs=state["selected_designs"])
    files = artifact_manifest(workspace, expected={"dispatch.md", *(entry["id"] + ".yaml" for entry in state["selected_designs"])})
    if {name for name in files if name.endswith(".yaml")} != {entry["id"] + ".yaml" for entry in state["selected_designs"]}:
        raise ExtractError("P6 requires exactly the selected raw YAML collection")
    for stage in ("P1", "P2", "P3", "P4"):
        approval = state["stages"][stage]["approval"]
        receipt, _ = store.verify_approval(state, approval, directory=workspace)
        if approval["generation"] != state["stages"][stage]["generation"]:
            raise ExtractError("upstream approval belongs to a superseded stage generation")
        if stage == "P2" and selection_signature(receipt["plan"]) != state["selection_signature"]:
            raise ExtractError("current selection differs from P2 approval")
        if stage == "P4" and writing_signature(receipt["plan"]) != state["writing_signature"]:
            raise ExtractError("current writing arrangement differs from P4 approval")
    for entry in state["selected_designs"]:
        node_id, node = entry["id"], state["nodes"][entry["id"]]
        store.verify_node(state, node_id)
        _, approved_checkpoint = store.verify_approval_origin(state, node["approval"])
        if approved_checkpoint["files"].get(node["file"]) != node["approved_sha256"]:
            raise ExtractError("node completion marker differs from its accepted review/checkpoint")
        if node_id not in state["pending_rechecks"] and files.get(node["file"]) != node["approved_sha256"]:
            raise ExtractError("a node changed without an effective review or explicit P6 recheck")
        if node_id not in state["pending_rechecks"]:
            current = source_context(workspace, parser_root=parser_root, nodes=[node_id])["nodes"][node_id]
            if current["effective_identity"] != store.source_identity(state, node_id):
                raise ExtractError("a node's actual sources changed without an explicit P6 recheck")


def _adjudicated_workspace(store: Store, state: dict, lock: dict, attempt: dict, receipt: dict) -> tuple[Path, dict]:
    parser_root = rules_root(store.context(state, attempt)["rules_lock"])
    action = receipt["adjudication"]
    before = Path(attempt["workspace"])
    source = before
    if action["action"] == "select":
        candidates = {item["checkpoint"]: item for item in attempt["adjudication_candidates"]}
        if action["checkpoint"] not in candidates:
            raise ExtractError("adjudication selected a candidate outside its saved scope")
        selected = candidates[action["checkpoint"]]
        source, _ = store.verify_checkpoint(state, action["checkpoint"])
        if artifact_manifest(source, expected=selected["files"]) != selected["files"]:
            raise ExtractError("adjudication candidate changed after it was provided")
        if attempt["stage"] in {"P2", "P4"} and validate_plan(attempt["stage"], receipt["plan"],
                parser_root=parser_root, selected_designs=state["selected_designs"] if attempt["stage"] == "P4" else None) != validate_plan(
                attempt["stage"], selected["plan"], parser_root=parser_root,
                selected_designs=state["selected_designs"] if attempt["stage"] == "P4" else None):
            raise ExtractError("selected candidate and approved stage decision differ")
    elif action["checkpoint"] != attempt["base_checkpoint"]:
        raise ExtractError("adjudication patch must target the current candidate")
    original = Path(attempt["directory"]) / "adjudicated-original"
    _rebuild_directory(original)
    artifact_manifest(source, expected=selected["files"] if action["action"] == "select" else attempt["dependency_digests"])
    copy_artifacts(source, original)
    for path in original.iterdir():
        path.chmod(0o644)
    original.chmod(0o755)
    if action["action"] == "patch":
        from .common import regular_file
        for patch in action["patches"]:
            path = regular_file(original, patch["path"])
            text = path.read_text(encoding="utf-8")
            if not patch["old"] or text.count(patch["old"]) != 1:
                raise ExtractError("adjudication old text must match exactly once")
            write_text(path, text.replace(patch["old"], patch["new"], 1))
        destination = Path(attempt["directory"]) / "adjudicated"
        validation = _normalize(store, state, attempt, before, original, destination, "approve",
                                plan=receipt.get("plan"), adjudication=receipt)
    else:
        destination = original
        preparation = selected["normalization"]
        if (preparation.get("status") != "landed" or preparation.get("check", {}).get("exit_code") != 0 or
                preparation.get("baseline_artifact_digests") != attempt["dependency_digests"] or
                preparation.get("artifact_digests") != artifact_manifest(destination, expected=selected["files"])):
            raise ExtractError("selected candidate lacks its exact pre-review normalization and scope proof")
        validation = {"changed_files": preparation["changed_files"], "artifact_digests": preparation["artifact_digests"],
                      "scope": preparation["scope"], "candidate_preparation": deepcopy(preparation)}
        if preparation.get("shared_metadata_sync"):
            validation["shared_metadata_sync"] = deepcopy(preparation["shared_metadata_sync"])
        validate_artifact_sources(destination, lock, decision="approve", parser_root=parser_root,
                                  completed_nodes=None if attempt["stage"] == "P6" else [attempt["node"]])
        validation["check"] = check_stage_snapshot(destination, parser_root=parser_root, stage=attempt["stage"],
            selected_designs=(receipt["plan"] if attempt["stage"] == "P2" else attempt["selected_designs"] if attempt["stage"] == "P5" else state["selected_designs"]), node=attempt["node"])
        if validation["check"]["exit_code"]:
            raise ExtractError("adjudication candidate was not a strict completed snapshot")
    validation["adjudication_changed"] = action["action"] == "patch" and bool(validation["changed_files"])
    if attempt["stage"] == "P1":
        validate_comparisons(receipt, lock, destination, parser_root=parser_root)
    if attempt["stage"] == "P6" and not {Path(name).stem for name in validation["changed_files"]
            if name.endswith(".yaml")} <= set(receipt["affected_nodes"]):
        raise ExtractError("P6 adjudication omitted changed nodes from its recheck scope")
    return destination, validation


def _begin_reopen(store: Store, state: dict, attempt: dict, receipt: dict) -> dict:
    """Return one concrete responsibility without reopening unrelated approvals."""
    request = receipt["reopen"]
    owner, scope = owner_for(state, attempt), scope_for(state, attempt)
    evidence = sorted({digest(evidence_identity(receipt["source_evidence"][index])):
                       evidence_identity(receipt["source_evidence"][index]) for index in request["evidence"]}.values(), key=digest)
    key = digest({"issue_id": request["issue_id"].strip()})
    def blocked(code, summary):
        owner["status"] = scope["status"] = "blocked"
        owner["reason"] = {"code": code, "summary": summary, "issues": receipt["issues"]}
        settle(state)
        return state
    if request["target"] == "external":
        return blocked("external_confirmation", "任务或固定来源发生变化；保留本次证据，交Supervisor另建运行。")
    if key in state["reopen_requests"]:
        return blocked("reopen_already_used", "同一稳定问题已经返回处理；改变引文范围不会重新授予预算或一次决定。")
    target = request["target"]
    prior = state["stages"][target].get("approval")
    if prior:
        store.verify_approval_origin(state, prior)
    source_checkpoint = (state["merged_checkpoint"] or state["checkpoint"]) if attempt["stage"] == "P5" else state["checkpoint"]
    source, source_info = store.verify_checkpoint(state, source_checkpoint)
    record = {"key": key, "status": "open", "issue_id": request["issue_id"].strip(), "target": target,
              "reason": request["reason"], "source_evidence": evidence, "attempt_id": attempt["attempt_id"],
              "artifact_evidence": deepcopy(receipt["artifact_evidence"]), "issues": deepcopy(receipt["issues"]),
              "from_stage": attempt["stage"], "from_node": attempt["node"],
              "previous_selection": deepcopy(state["selected_designs"]), "previous_writing_plan": deepcopy(state["writing_plan"]),
              "previous_merged_checkpoint": state["merged_checkpoint"], "previous_plan_revision": state["plan_revision"]}
    state["reopen_requests"][key] = deepcopy(record)
    state["reopening"] = record
    state["plan_revision"] += 1
    invalidate_stage(state, target, require_worker=True)
    invalidate_final(state)
    if attempt["stage"] == "P5":
        owner.update(status="pending" if attempt["role"] == "worker" else "reviewing", feedback=[request, *receipt["issues"]])
    state.update(stage=target, node="root", role="worker", mode="normal", status="running",
                 feedback=[request, *receipt["issues"]], reason=None)
    state["checkpoint"] = store.checkpoint(state, source, attempt_id=attempt["attempt_id"], parent=source_checkpoint,
                                            expected_files=source_info["files"])
    root = state["nodes"]["root"]
    root["last_draft_checkpoint"] = root.get("draft_checkpoint")
    root["draft_checkpoint"] = state["checkpoint"]
    return state


def _begin_provider_reopen(store: Store, state: dict, attempt: dict, receipt: dict, details: dict) -> dict:
    """Pause actual users while a definition is repaired; classify impact after the edit."""
    request, provider = receipt["reopen"], details["provider"]
    evidence = sorted({digest(evidence_identity(receipt["source_evidence"][index])):
                       evidence_identity(receipt["source_evidence"][index]) for index in request["evidence"]}.values(), key=digest)
    key = digest({"issue_id": request["issue_id"].strip()})
    owner, scope = owner_for(state, attempt), scope_for(state, attempt)
    if key in state["reopen_requests"]:
        owner.update(status="blocked", reason={"code": "reopen_already_used", "summary": "同一provider问题已请求过修复；原调用计数与决定继续有效。", "issues": receipt["issues"]})
        scope["status"] = "blocked"
        settle(state)
        return state
    node = state["nodes"][provider]
    if not node.get("approval") or provider not in effective_nodes(state):
        raise ExtractError("provider approval is no longer current")
    store.verify_node(state, provider)
    prior = deepcopy(node["approval"])
    if details.get("consumer_context") and attempt["stage"] == "P5":
        state["nodes"][attempt["node"]]["source_context"] = deepcopy(details["consumer_context"])
    affected = affected_nodes(state, [provider], regions=details["regions"])
    record = {"key": key, "kind": "provider", "status": "open", "issue_id": request["issue_id"].strip(),
              "target": request["target"], "provider": provider, "reason": request["reason"],
              "source_evidence": evidence, "artifact_evidence": deepcopy(receipt["artifact_evidence"]),
              "issues": deepcopy(receipt["issues"]), "regions": deepcopy(details["regions"]),
              "source_manifest": deepcopy(details["source_manifest"]), "effective_identity": details["effective_identity"],
              "consumer_identity": details["consumer_identity"], "attempt_id": attempt["attempt_id"],
              "from_stage": attempt["stage"], "from_node": attempt["node"], "previous_generation": node["generation"],
              "previous_approval": prior, "previous_draft_checkpoint": node.get("draft_checkpoint"),
              "previous_merged_checkpoint": state["merged_checkpoint"], "plan_revision": state["plan_revision"],
              "affected_nodes": affected}
    state["reopen_requests"][key] = record
    node["provider_reopen"] = key
    invalidate_node(node, require_worker=True)
    node["feedback"] = [{"code": "provider_repair", "reopen_key": key, "reason": request["reason"],
                         "regions": deepcopy(details["regions"]), "source_evidence": evidence,
                         "artifact_evidence": deepcopy(receipt["artifact_evidence"]), "issues": deepcopy(receipt["issues"])}]
    record["repair_generation"] = node["generation"]
    if attempt["stage"] == "P5" and attempt["node"] != provider:
        owner.update(status="pending" if attempt["role"] == "worker" else "reviewing", feedback=[request])
    invalidate_final(state, affected)
    state.update(stage="P5", node="root", role=None, mode="normal", status="running", reason=None)
    settle(state)
    return state


def _merge_check(store: Store, state: dict, merged: Path, *, node_id: str, parser_root: Path) -> dict:
    nodes = effective_nodes(state) | {node_id}
    hashes = artifact_manifest(merged)
    changed = hashes[node_id + ".yaml"] != state["nodes"][node_id].get("approved_sha256")
    superseded = set(affected_nodes(state, [node_id])) - {node_id} if changed else set()
    nodes -= superseded
    checked = check_stage_snapshot(merged, parser_root=parser_root, stage="P5",
                                  selected_designs=state["selected_designs"], node=node_id, checked_nodes=nodes)
    if manifest(merged) != hashes:
        raise ExtractError("serial merge check changed the raw artifacts")
    return {**checked, "checked_files": {name: hashes[name] for name in hashes
                                        if name.removesuffix(".yaml") in checked["checked_nodes"]},
            "superseded_dependents": sorted(superseded)}


def accept_result(store: Store, state: dict, lock: dict, attempt: dict, result: dict) -> dict:
    """Capture original evidence, normalize a new candidate, then commit one checked result."""
    state = store.load(state["run_id"])
    checking_started = time.monotonic()
    if attempt["attempt_id"] in state["accepted_receipts"]:
        raise ExtractError("duplicate receipt: this attempt was already accepted")
    store.validate_attempt(state, attempt)
    context = store.context(state, attempt)
    lock, semantic = context["source_lock"], context["rules_lock"]
    parser_root = rules_root(semantic)
    response_validation = result.get("response_validation", {})
    if (result.get("outcome") != "completed" or result.get("exit_code") != 0 or
            result.get("termination", {}).get("confirmed") is not True or
            result.get("native_status", {}).get("success") is not True or
            not all(response_validation.get(key) is True for key in ("schema_valid", "nonce_valid", "native_output_matches")) or
            result.get("backend") != attempt["backend"] or result.get("session_status") != "ready"):
        raise ExtractError("agent result lacks a verified successful native completion or response identity")
    session_id = _session(state, attempt, result)
    verify_run_inputs(state, lock)
    before, checkpoint = store.verify_checkpoint(state, attempt["base_checkpoint"])
    if checkpoint["files"] != attempt["dependency_digests"]:
        raise ExtractError("attempt input digests do not match its immutable checkpoint")
    directory = Path(attempt["directory"])
    original = Path(attempt["workspace"])
    if attempt["role"] == "reviewer":
        _artifact_gate(store, state, lock, attempt, before, original, "continue")
    proposal = result.get("response")
    check_proposal_identity(proposal, attempt)
    selecting = attempt["mode"] == "adjudicate" and (proposal.get("adjudication") or {}).get("action") == "select"
    original_expected = checkpoint["files"]
    if selecting:
        chosen = next((entry for entry in attempt["adjudication_candidates"]
                       if entry["checkpoint"] == proposal["adjudication"]["checkpoint"]), None)
        if chosen is None:
            raise ExtractError("adjudication selected an unprovided candidate")
        original = directory / "materials" / "candidates" / chosen["checkpoint"]
        original_expected = chosen["files"]
        if artifact_manifest(original, expected=original_expected) != chosen["files"]:
            raise ExtractError("actually read candidate bytes differ from the invocation")
    original_hashes = artifact_manifest(original, expected=original_expected,
        additions={"dispatch.md"} if (attempt["stage"], attempt["role"]) == ("P2", "worker") else ())
    receipt = normalize_response(proposal, lock, attempt, original)
    receipt_hash = digest(receipt)
    write_json(directory / "receipt.json", receipt)
    evidence_directory = directory / "original"
    if evidence_directory.exists():
        if artifact_manifest(evidence_directory, expected=original_hashes) != original_hashes:
            raise ExtractError("original evidence changed during result replay")
    else:
        evidence_directory.mkdir()
        for name in original_hashes:
            copy_artifacts(original / name, evidence_directory / name)
        seal(evidence_directory)
    if (artifact_manifest(evidence_directory, expected=original_hashes) != original_hashes or
            artifact_manifest(original, expected=original_hashes) != original_hashes):
        raise ExtractError("original artifacts changed during evidence capture")
    original_changed = sorted(name for name in checkpoint["files"].keys() | original_hashes.keys()
                              if checkpoint["files"].get(name) != original_hashes.get(name))
    if selecting:
        original_changed = []
    workspace = original
    decision = receipt["decision"]
    complete = decision in {"complete", "approve"}
    if attempt["role"] == "worker":
        workspace = directory / "normalized"
        try:
            validation = _normalize(store, state, attempt, before, original, workspace,
                                    decision, plan=receipt.get("plan"))
        except CandidateRejected as exc:
            failed_check = exc.validation.get("check")
            if not failed_check or failed_check.get("exit_code") not in {1, 2}:
                raise
            failed = deepcopy(exc.validation)
            validation = _normalize(store, state, attempt, before, original, workspace, "continue", plan=receipt.get("plan"))
            validation.update(check=failed_check, rejected_completion=failed)
            return _record_check_failure(store, state, attempt, receipt, receipt_hash, workspace,
                                         validation, session_id, accept_receipt=False)
    else:
        validation = _artifact_gate(store, state, lock, attempt, before, Path(attempt["workspace"]),
                                    "continue" if attempt["mode"] == "adjudicate" else decision)
        validation.update(original_artifact_digests=original_hashes, original_changed_files=original_changed)
    if attempt["role"] == "reviewer":
        validate_artifact_sources(workspace, lock, decision=decision, parser_root=parser_root,
                                  completed_nodes=None if attempt["stage"] == "P6" else [attempt["node"]])
    receipt = validate_receipt(receipt, state, lock, attempt, original, original_changed,
                               rules_lock=semantic, semantic_artifacts=workspace)
    if attempt["mode"] == "adjudicate" and decision == "approve":
        workspace, validation = _adjudicated_workspace(store, state, lock, attempt, receipt)
    elif attempt["role"] == "reviewer" and decision in {"approve", "revise"}:
        validation["check"] = check_stage_snapshot(workspace, parser_root=parser_root, stage=attempt["stage"],
            selected_designs=(receipt.get("plan") if attempt["stage"] == "P2" else attempt["selected_designs"] if attempt["stage"] == "P5" else state["selected_designs"]), node=attempt["node"])
    if attempt["stage"] == "P6":
        _final_inputs(store, state, workspace)
    graph = validation.get("check", {}).get("design_resolution")
    if complete and attempt["stage"] == "P5":
        validation.update(store.validate_sources(state, attempt, workspace, graph=graph))
    elif graph:
        validation["design_resolution"] = graph
    provider_details = (provider_request_details(receipt, state, attempt, workspace, parser_root=parser_root)
                        if decision == "reopen" and receipt["reopen"]["target"].startswith("P5:") else None)
    if provider_details:
        validation["provider_reopen"] = provider_details
        if provider_details.get("consumer_context"):
            validation["design_resolution"] = {"nodes": {attempt["node"]: provider_details["consumer_context"]}}
    validation["node_signature"] = attempt["node_signature"]
    validation["invocation_applicability"] = {"selection_signature": state.get("selection_signature"),
                                             "writing_signature": state.get("writing_signature"),
                                             "node_signature": attempt["node_signature"], "generation": attempt["generation"]}
    validation.setdefault("dependency_digests", {})
    write_json(directory / "check.json", validation["check"])
    if validation["check"]["exit_code"] and not (decision == "revise" and validation["check"]["exit_code"] == 1):
        return _record_check_failure(store, state, attempt, receipt, receipt_hash, workspace, validation, session_id)
    updated = deepcopy(state)
    if validation["check"]["exit_code"] == 1:
        scope_for(updated, attempt)["check_failures"] += 1
    _remember_session(updated, attempt, session_id)
    checkpoint_id = attempt["base_checkpoint"]
    if attempt["role"] == "worker" or attempt["mode"] == "adjudicate" and decision == "approve":
        checkpoint_id = store.checkpoint(state, workspace, attempt_id=attempt["attempt_id"],
                                        stage=attempt["stage"], node=attempt["node"], parent=attempt["base_checkpoint"],
                                        expected_files=validation["artifact_digests"])
        updated["recoverable_checkpoint"] = checkpoint_id
        if attempt["stage"] == "P5" and (validation["changed_files"] or decision == "complete"):
            updated["nodes"][attempt["node"]]["authored"] = True
    if attempt["stage"] == "P5" and decision == "approve" and not validation.get("adjudication_changed"):
        _rebuild_directory(directory / "merged")
        merged = store.merged_workspace(state, workspace, directory / "merged", node_id=attempt["node"])
        merge_check = _merge_check(store, state, merged, node_id=attempt["node"], parser_root=parser_root)
        write_json(directory / "merge_check.json", merge_check)
        validation["merge_check"] = merge_check
        if merge_check["exit_code"]:
            validation["check"] = merge_check
            return _record_check_failure(store, state, attempt, receipt, receipt_hash, workspace, validation, session_id)
        validation.update(store.validate_sources(state, attempt, merged, graph=merge_check["design_resolution"],
                                                  reviewed_graph=validation["reviewed_design_resolution"]))
        workspace = merged
        _, merged_info = store.verify_checkpoint(state, state["merged_checkpoint"])
        checkpoint_id = store.checkpoint(state, merged, attempt_id=attempt["attempt_id"], stage="P5", node=attempt["node"],
                                        parent=state["merged_checkpoint"], expected_files=set(merged_info["files"]) | {attempt["node"] + ".yaml"})
    if attempt["stage"] == "P2" and decision == "approve":
        validation["invalidated_nodes"] = sorted(selection_impact(state, receipt["plan"]))
    approving = decision == "approve" and not validation.get("adjudication_changed")
    if attempt["stage"] in {"P1", "P2", "P3", "P4"} and approving:
        destination = directory / "upstream-merged"
        _rebuild_directory(destination)
        selected = receipt["plan"] if attempt["stage"] == "P2" else state["selected_designs"]
        expected_files = set(validation["artifact_digests"])
        base = ((state.get("maintenance") or {}).get("working_checkpoint") if attempt.get("purpose") == "maintenance" else None) or state.get("merged_checkpoint")
        if base:
            expected_files.update(store.verify_checkpoint(state, base)[1]["files"])
        if selected is not None:
            expected_files &= {"dispatch.md", *(entry["id"] + ".yaml" for entry in selected)}
        workspace = maintenance.merge_upstream(store, state, attempt, workspace, destination, selected=selected)
        checked = check_stage_snapshot(workspace, parser_root=parser_root, stage=attempt["stage"], selected_designs=selected)
        validation["merge_check"] = checked
        if checked["exit_code"]:
            validation["check"] = checked
            return _record_check_failure(store, state, attempt, receipt, receipt_hash, workspace, validation, session_id)
        if checked.get("design_resolution"):
            validation["design_resolution"] = checked["design_resolution"]
        checkpoint_id = store.checkpoint(state, workspace, attempt_id=attempt["attempt_id"], stage=attempt["stage"],
                                        node=attempt["node"], parent=state["merged_checkpoint"] or attempt["base_checkpoint"],
                                        expected_files=expected_files)
        updated["merged_checkpoint"] = checkpoint_id
    validation["artifact_digests"] = manifest(workspace)
    if decision == "approve":
        validation["approval_projection"] = approval_projection(attempt["stage"], workspace,
            parser_root=parser_root, plan=receipt.get("plan"), selected_designs=state["selected_designs"], node=attempt["node"])
    validation.update({key: attempt[key] for key in ("maintenance_task_id", "purpose", "maintenance_step") if key in attempt})
    if attempt["attempt_id"] in updated.get("attempt_usage", {}):
        updated["attempt_usage"][attempt["attempt_id"]]["result_checkpoint"] = checkpoint_id
    updated = advance(updated, receipt, checkpoint_id, receipt_hash, validation=validation)
    if attempt["stage"] == "P1":
        store.shared_drafts(updated, checkpoint_id)
    if approving and attempt["stage"] in {"P1", "P2", "P3", "P4", "P5"}:
        store.refresh_approvals(updated, workspace, checkpoint_id, attempt, validation)
        if maintenance.active(updated):
            maintenance.settle(updated)
        elif attempt["stage"] != "P5" or validation["invalidated_uses"]:
            resume_progress(updated)
    validation["checking"] = {"elapsed_seconds": time.monotonic() - checking_started}
    write_json(directory / "validated.json", {"receipt_sha256": receipt_hash, **validation})
    if decision == "reopen" and attempt.get("purpose") != "maintenance":
        updated = (_begin_provider_reopen(store, updated, attempt, receipt, provider_details) if provider_details else
                   _begin_reopen(store, updated, attempt, receipt))
    if store.load(state["run_id"]) != state:
        raise ExtractError("authoritative state changed during result validation")
    store.save(updated)
    return updated


def _recover_call(store: Store, state: dict, lock: dict, attempt: dict, result: dict, *, kind: str, reason: str,
                  fresh_context=False) -> dict:
    """A finite same-role repair preserves safe edits and an already established UUID."""
    updated = deepcopy(state)
    if kind == "receipt_repair":
        check_proposal_identity(result.get("response"), attempt)
    session_id = _session(state, attempt, result, required=kind == "receipt_repair")
    verify_run_inputs(state, lock)
    before, _ = store.verify_checkpoint(state, attempt["base_checkpoint"])
    original = Path(attempt["workspace"])
    workspace = original
    if attempt["role"] == "worker":
        workspace = Path(attempt["directory"]) / "recovered-normalized"
        validation = _normalize(store, state, attempt, before, original, workspace, "continue")
    else:
        validation = _artifact_gate(store, state, lock, attempt, before, workspace, "continue")
    _remember_session(updated, attempt, session_id)
    if attempt["role"] == "worker" and validation["changed_files"]:
        _draft(store, updated, attempt, workspace, files=validation["artifact_digests"])
        if attempt["stage"] == "P6":
            updated["pending_rechecks"] = affected_nodes(updated, [*updated["pending_rechecks"],
                *(Path(name).stem for name in validation["changed_files"] if name.endswith(".yaml"))])
    if fresh_context or result.get("session_status") == "retired" or (attempt.get("recovery_handoff") and session_id is None):
        from .prompts import render_operation
        handoff_session(updated, attempt, result, render_operation("recovery", {
            "reason": reason, "checkpoint": call_view(updated, attempt)["checkpoint"],
            "stage": attempt["stage"], "node": attempt["node"]}, rules_lock=store.context(updated, attempt)["rules_lock"]))
    _release(updated, attempt)
    scope, owner = scope_for(updated, attempt), owner_for(updated, attempt)
    key = "receipt_repairs" if kind == "receipt_repair" else "cli_failures"
    scope[key] += 1
    if kind == "receipt_repair":
        scope["invalid_receipts"] += 1
    recorded = Path(attempt["directory"]) / "result.json"
    observed = time.time()
    owner["reason"] = {"code": kind, "summary": reason, "checkpoint": call_view(updated, attempt)["checkpoint"],
                       "classification": retry_classification(kind, result), "attempt_id": attempt["attempt_id"],
                       "scope": scope_key(attempt["stage"], attempt["node"]), "observed_at": observed,
                       "next_due_at": observed + retry_delay(scope[key]),
                       "result_sha256": digest(read_json(recorded) if recorded.is_file() else result)}
    owner["feedback"] = [*owner["feedback"], owner["reason"]]
    limit = updated["config"]["retry_limit"]
    if scope[key] > limit:
        owner["status"] = "invalid_receipt" if kind == "receipt_repair" else "cli_error"
        scope["status"] = owner["status"]
        owner["reason"]["code"] += "_limit_reached"
        owner["reason"].pop("classification", None)
    elif owner["reason"].get("classification") == "transient_execution":
        # Temporary failures remain visible for an explicit scope retry.
        owner["status"] = scope["status"] = "cli_error"
    else:
        owner["status"] = "draft" if attempt["stage"] == "P5" else "running"
    write_json(Path(attempt["directory"]) / "repair.json", {**owner["reason"], "validation": validation})
    if attempt.get("purpose") == "maintenance":
        maintenance.block(updated, attempt, owner["reason"], status="invalid_receipt" if kind == "receipt_repair" else "cli_error")
    settle(updated)
    store.save(updated)
    return updated


def _retire_stale(store: Store, state: dict, attempt: dict, result: dict | None, reason: str) -> dict:
    confirmed = bool(result and result.get("termination", {}).get("confirmed"))
    record = {"attempt_id": attempt["attempt_id"], "node": attempt["node"], "reason": reason,
              "generation": attempt["generation"], "termination_confirmed": confirmed,
              "workspace": attempt["workspace"], "base_checkpoint": attempt["base_checkpoint"],
              "recovery_condition": "旧依赖正文仅作只读线索；当前worker在新依赖和合法范围内显式采用后重新检查审阅。"}
    state["stale_results"][attempt["attempt_id"]] = record
    if confirmed:
        _release(state, attempt)
    else:
        state["unconfirmed_attempts"][attempt["attempt_id"]] = record
    write_json(Path(attempt["directory"]) / "stale.json", record)
    settle(state)
    store.save(state)
    return state


def _receive(store: Store, state: dict, lock: dict, attempt: dict, result: dict | None, error: Exception | None,
             *, recovering=False) -> dict:
    try:
        store.validate_attempt(state, attempt)
        if (result and result.get("outcome") in {"completed", "receipt_error"} and
                result.get("termination", {}).get("confirmed") is True and result.get("exit_code") == 0 and
                result.get("native_status", {}).get("success") is True and result.get("backend") == attempt["backend"]):
            # Native session completion survives a later proposal rejection.
            # Commit its exact scoped UUID before any receipt/artifact checks.
            _remember_session(state, attempt, _session(state, attempt, result))
            store.save(state)
        if error:
            raise error
        if not result or not result.get("termination", {}).get("confirmed"):
            raise ExtractError("agent termination is unconfirmed; this attempt remains fenced")
        outcome = result.get("outcome")
        if outcome == "receipt_error":
            raise ReceiptFormatError("Agent CLI response did not satisfy the scoped proposal schema")
        if outcome == "cli_error" and attempt["mode"] == "normal":
            return _recover_call(store, state, lock, attempt, result, kind="cli_retry",
                                 reason=f"CLI {outcome}；合法草稿与原计数已保留，按原调用计数重试。",
                                 fresh_context=recovering)
        if outcome != "completed":
            raise ExtractError("Agent CLI call did not complete: " + str(outcome))
        return accept_result(store, state, lock, attempt, result)
    except _AuthorityChanged:
        raise
    except StaleResult as exc:
        return _retire_stale(store, state, attempt, result, str(exc))
    except ReceiptFormatError as exc:
        confirmed = result is not None and result.get("termination", {}).get("confirmed")
        if confirmed and attempt["mode"] == "normal":
            try:
                return _recover_call(store, state, lock, attempt, result, kind="receipt_repair", reason=str(exc))
            except _AuthorityChanged:
                raise
            except (ExtractError, OSError) as unsafe:
                return _reject(store, state, attempt, result, unsafe)
        return _reject(store, state, attempt, result, exc)
    except (ExtractError, OSError) as exc:
        return _reject(store, state, attempt, result, exc)


def _interrupted(store: Store, state: dict, lock: dict, attempt: dict,
                 result: dict | None, termination: dict, reason: str, *, recovery_request_id=None, resume_scope=True) -> dict:
    """Save only a valid draft; an interrupted decision cannot approve anything."""
    if not termination["confirmed"]:
        owner, scope = owner_for(state, attempt), scope_for(state, attempt)
        owner["status"] = scope["status"] = "termination_unconfirmed"
        owner["reason"] = {"code": "termination_unconfirmed", "summary": reason,
                           "termination": termination, "recovery_condition": "确认记录中的旧进程及其子进程终止后重试resume。"}
        state["unconfirmed_attempts"][attempt["attempt_id"]] = owner["reason"]
        settle(state)
        store.save(state)
        return state
    try:
        store.validate_attempt(state, attempt)
    except StaleResult as exc:
        handoff_session(state, attempt, result, reason, termination=termination)
        return _retire_stale(store, state, attempt, {"termination": termination}, str(exc))
    validation, discarded = None, None
    if attempt["role"] == "worker":
        try:
            before, _ = store.verify_checkpoint(state, attempt["base_checkpoint"])
            workspace = Path(attempt["directory"]) / "interrupted-normalized"
            validation = _normalize(store, state, attempt, before, Path(attempt["workspace"]), workspace, "continue")
            if validation["changed_files"]:
                _draft(store, state, attempt, workspace, files=validation["artifact_digests"])
                if attempt["stage"] == "P6":
                    state["pending_rechecks"] = affected_nodes(state, [*state["pending_rechecks"],
                        *(Path(name).stem for name in validation["changed_files"] if name.endswith(".yaml"))])
        except (ExtractError, OSError) as exc:
            discarded = str(exc)
    _release(state, attempt)
    handoff_session(state, attempt, result, reason, termination=termination)
    record = {"attempt_id": attempt["attempt_id"], "reason": reason, "termination": termination,
              "validation": validation, "discarded_draft": discarded,
              "checkpoint": call_view(state, attempt)["checkpoint"]}
    if recovery_request_id:
        record["recovery_request_id"] = recovery_request_id
    state["recovered_attempts"][attempt["attempt_id"]] = record
    write_json(Path(attempt["directory"]) / "recovery.json", record)
    if attempt["mode"] in {"adjudicate", "signoff"}:
        return _stop_once(store, state, attempt, {"code": "interrupted_single_decision", "summary": reason,
                          "recovery_condition": "本范围的一次裁决/签收已经使用；保留原决定和调用计数，不能重新授予。"})
    owner = owner_for(state, attempt)
    if attempt.get("purpose") == "maintenance":
        maintenance.block(state, attempt, {"code": "maintenance_interrupted", "summary": reason,
                                          "checkpoint": record["checkpoint"], "discarded_draft": discarded})
        settle(state)
        store.save(state)
        return state
    if resume_scope:
        owner["status"] = "draft" if attempt["stage"] == "P5" else "running"
        scope_for(state, attempt)["status"] = "reviewing" if attempt["role"] == "reviewer" else "draft"
        owner["reason"] = None
    else:
        owner["status"] = scope_for(state, attempt)["status"] = "cli_error"
        owner["reason"] = {"code": "scope_recovery_pending", "summary": reason, "attempt_id": attempt["attempt_id"],
                           "recovery_request_id": recovery_request_id}
    settle(state)
    store.save(state)
    return state


def _reconcile(store: Store, state: dict, lock: dict, *, terminate_only=False, scopes=None,
               attempt_ids=None) -> tuple[dict, list]:
    """Reconcile only state-registered fences and unfinished accounting commits."""
    pending = []
    unsettled = state.setdefault("unsettled_attempts", {})
    registered = dict.fromkeys([*state["active_attempts"].values(), *state["unconfirmed_attempts"]])
    if not terminate_only:
        registered.update(dict.fromkeys(unsettled))
    for attempt_id in registered:
        if attempt_ids is not None and attempt_id not in attempt_ids:
            continue
        directory = store.run_dir(state["run_id"]) / "attempts" / attempt_id
        attempt = read_json(directory / "attempt.json")
        if scopes is not None and scope_key(attempt["stage"], attempt["node"]) not in scopes:
            continue
        if attempt["attempt_id"] != attempt_id or attempt["run_id"] != state["run_id"]:
            raise ExtractError("saved attempt identity differs from the run ledger")
        if Path(attempt["directory"]) != directory or Path(attempt["workspace"]) != directory / "workspace":
            raise ExtractError("saved invocation path leaves its run")
        if terminate_only:
            try:
                result = read_json(directory / "result.json") if (directory / "result.json").is_file() else None
            except (ExtractError, OSError):
                # A damaged result cannot prevent cancellation from checking
                # and terminating the recorded process and its descendants.
                result = None
            termination = confirm_termination(attempt, result, launch_state=unsettled.get(attempt_id),
                                              cancel=terminate_only or attempt_cancelled(state, attempt))
            if termination["confirmed"]:
                # Cancellation grants no new draft/approval authority. Resume
                # still accounts for this attempt and retires interrupted UUIDs.
                state["unconfirmed_attempts"].pop(attempt_id, None)
            else:
                state["unconfirmed_attempts"][attempt_id] = termination
            continue
        active = state["active_attempts"].get(scope_key(attempt["stage"], attempt["node"])) == attempt_id
        try:
            result = None
            if (directory / "result.json").is_file():
                result = read_json(directory / "result.json")
                if not isinstance(result, dict):
                    raise ExtractError("saved result is not a JSON object")
        except (ExtractError, OSError) as exc:
            if state.get("control", {}).get("action") != "cancel":
                raise
            termination = confirm_termination(attempt, None, launch_state=unsettled.get(attempt_id), cancel=True)
            reason = {"code": "unreadable_result", "summary": str(exc), "termination": termination,
                      "recovery_condition": "先保留原调用及坏result；修复结果记录后按原调用计数resume对账。"}
            owner = owner_for(state, attempt)
            owner["reason"] = reason
            if active:
                owner["status"] = "invalid_receipt" if termination["confirmed"] else "termination_unconfirmed"
                scope_for(state, attempt)["status"] = owner["status"]
                if termination["confirmed"]:
                    state["unconfirmed_attempts"].pop(attempt_id, None)
                else:
                    state["unconfirmed_attempts"][attempt_id] = reason
            # Keep active/unsettled identities: cancellation proves exit only,
            # and cannot turn a damaged result into an accepted or settled one.
            store.save(state)
            continue
        # Close a previous checking interval before replaying, so a repeated
        # resume cannot count the same interrupted check again.
        processing = directory / "processing.json"
        old = None
        if processing.is_file():
            old = read_json(processing)
            if old.get("started_at") is not None:
                estimate = max(0.0, time.time() - old["started_at"])
                old["elapsed_seconds"] += estimate
                old["estimated_seconds"] = old.get("estimated_seconds", 0.0) + estimate
                old["started_at"] = None
                write_json(processing, old)
        elapsed = elapsed_on_disk(attempt, result)
        account(state, attempt, elapsed, result=result, processing=old)
        if not active:
            if not (directory / "finished.json").exists():
                write_json(directory / "finished.json", {"elapsed_seconds": elapsed, "status": state["status"], "recovered": True})
            state["unsettled_attempts"].pop(attempt_id, None)
            account(state, attempt, elapsed, result=result, processing=old, final=True)
            continue
        termination = confirm_termination(attempt, result, launch_state=unsettled.get(attempt_id),
                                              cancel=terminate_only or attempt_cancelled(state, attempt))
        if not termination["confirmed"]:
            if not attempt_cancelled(state, attempt) and attempt_id not in state["unconfirmed_attempts"]:
                continue
            state = _interrupted(store, state, lock, attempt, result, termination, "恢复未能确认旧调用终止；仅该范围保留fence。")
            continue
        result = complete_pending_result(state, attempt, result, termination)
        elapsed = max(elapsed, (result or {}).get("elapsed_seconds", 0) + (old or {}).get("elapsed_seconds", 0))
        state["unconfirmed_attempts"].pop(attempt_id, None)
        owner = owner_for(state, attempt)
        if owner["status"] == "termination_unconfirmed":
            owner["status"] = "draft" if attempt["stage"] == "P5" else "running"
        if result and result.get("outcome") not in {"cancelled", "interrupted", "termination_unconfirmed"}:
            result = {**result, "termination": termination}
            result = recover_result(attempt, result, termination)
            event = read_json(directory / "event.json") if (directory / "event.json").is_file() else {}
            error = ExtractError(event["error"]) if event.get("error") else None
            pending.append((attempt, result, error, (result or {}).get("elapsed_seconds", elapsed), True))
        else:
            state = _interrupted(store, state, lock, attempt, result, termination, "旧调用已确认终止；从当前阶段页与安全checkpoint重建上下文。")
            write_json(directory / "finished.json", {"elapsed_seconds": elapsed, "status": state["status"], "recovered": True})
            state["unsettled_attempts"].pop(attempt_id, None)
            account(state, attempt, elapsed, result=result, processing=old, final=True)
    settle(state)
    store.save(state)
    return state, pending


def _resume_stopped(state: dict) -> None:
    control = state["control"]
    if state["status"] in {"paused", "cancelled"}:
        request = control["requests"].get(control.get("last_request_id"), {})
        if request.get("stopped_at") is not None:
            request.setdefault("resumed_at", time.time())
        state["status"] = control["resume_status"] or "running"
    if state["status"] == "queued":
        state["status"] = "running"
    control["action"], control["resume_status"] = None, None
    control["hold_id"] = None
    # Hold/queued/retry entries only restore their stopped control boundary.
    # Explicit manual resume separately authorizes ordinary failed scopes.
    if state["stage"] == "P5":
        settle(state)


def _resume_maintenance_receipts(store: Store, state: dict) -> None:
    """Retry a settled proposal rejection in its unchanged maintenance role/scope."""
    if not maintenance.active(state) or any(state.get(key) for key in
            ("active_attempts", "unconfirmed_attempts", "unsettled_attempts")):
        return
    for task in state["maintenance"]["tasks"].values():
        reason = task.get("reason") or {}
        if (task["status"] != "invalid_receipt" or task["mode"] != "normal" or
                reason.get("code") not in {"proposal_rejected", "receipt_repair", "receipt_repair_limit_reached"}):
            continue
        attempt_id = reason.get("attempt_id")
        attempts = [item["attempt_id"] for item in task["attempts"] if item["step"] == task["step"]]
        usage = state.get("attempt_usage", {}).get(attempt_id, {})
        if (not attempts or attempt_id != attempts[-1] or attempt_id in state["accepted_receipts"] or
                usage.get("settled") is not True):
            continue
        directory = store.run_dir(state["run_id"]) / "attempts" / attempt_id
        attempt, result = read_json(directory / "attempt.json"), read_json(directory / "result.json")
        if any(usage.get("evidence", {}).get(name) != digest(value) for name, value in
               (("attempt.json", attempt), ("result.json", result))):
            raise ExtractError("maintenance receipt retry differs from its settled invocation evidence")
        store.context(state, attempt)
        current = task["draft_checkpoint"]
        if current != attempt["base_checkpoint"]:
            if attempt["role"] != "worker" or usage.get("result_checkpoint") != current:
                raise StaleResult("maintenance receipt retry draft is not this worker's saved result")
            _, saved = store.verify_checkpoint(state, current)
            repair = read_json(directory / "repair.json")
            if (any(saved.get(key) != attempt[key] for key in ("attempt_id", "stage", "node")) or
                    saved.get("parent") != attempt["base_checkpoint"] or
                    usage["evidence"].get("repair.json") != digest(repair) or
                    repair.get("validation", {}).get("artifact_digests") != saved["files"]):
                raise StaleResult("maintenance receipt retry draft lacks its original checkpoint and repair evidence")
        maintenance.validate_attempt(state, attempt, status="invalid_receipt", current_checkpoint=current)
        _, checkpoint = store.verify_checkpoint(state, attempt["base_checkpoint"])
        if checkpoint["files"] != attempt["dependency_digests"]:
            raise ExtractError("maintenance receipt retry candidate differs from its fixed checkpoint")
        if (result.get("outcome") not in {"completed", "receipt_error"} or result.get("exit_code") != 0 or
                result.get("termination", {}).get("confirmed") is not True or
                result.get("native_status", {}).get("success") is not True or
                result.get("backend") != attempt["backend"] or not termination_observed(attempt, result)):
            continue
        session_id = _session(state, attempt, result)
        if state["sessions"].get(attempt["session_group"], {}).get(attempt["role"]) != session_id:
            continue
        if not call_exhaustion(call_view(state, attempt)):
            if not task["feedback"] or task["feedback"][-1] != reason:
                task["feedback"].append(deepcopy(reason))
            task.update(status="pending", reason=None)


def _resume_failed_scopes(state: dict) -> None:
    """Manual retry keeps counters and hard decisions; call stops need available credit."""
    if maintenance.active(state):
        for task in state["maintenance"]["tasks"].values():
            if task["status"] != "call_limit_reached":
                continue
            continued = {item["attempt_id"] for item in task.get("history", [])
                         if item.get("step") == task["step"] and item.get("decision") == "continue"}
            if any(item.get("step") == task["step"] and item["attempt_id"] not in continued
                   for item in task.get("attempts", [])):
                continue
            call = {**task, "purpose": "maintenance", "maintenance_task_id": task["task_id"],
                    "maintenance_step": task["step"], "mode": "normal"}
            if not call_exhaustion(call_view(state, call)):
                task["status"], task["reason"] = "pending", None
        settle(state)
        return
    calls = [{"stage": "P5", "node": entry["id"], **{key: state["nodes"][entry["id"]][key] for key in ("role", "mode")}}
             for entry in state.get("selected_designs") or [] if entry["id"] != "root"] if state["stage"] == "P5" else [state]
    for call in calls:
        owner, scope = owner_for(state, call), scope_for(state, call)
        reason = owner.get("reason") or {}
        if reason.get("code") in CALL_STOPS:
            if (any(item.endswith(":single_decision_used") for item in reason.get("scopes", [])) or
                    call_exhaustion(call_view(state, call))):
                continue
        if (owner["status"] in NODE_STOPS - {"blocked", "termination_unconfirmed"} and
                owner.get("mode", "normal") == "normal" and reason.get("code") not in HARD_RECOVERY_STOPS):
            owner["status"] = "draft" if owner is not state else "running"
            scope["status"] = "reviewing" if owner.get("role") == "reviewer" else "draft"
            if reason and (not owner["feedback"] or owner["feedback"][-1] != reason):
                owner["feedback"].append(reason)
            owner["reason"] = None
    if state["stage"] == "P5":
        settle(state)


def _settle_scope_retry(store, state, lock, request):
    """Close one exited fence without accepting an old failed proposal."""
    attempt_id = request["attempt_id"]
    directory = store.run_dir(state["run_id"]) / "attempts" / attempt_id
    attempt = read_json(directory / "attempt.json")
    result = read_json(directory / "result.json") if (directory / "result.json").is_file() else None
    termination = confirm_termination(attempt, result,
                                      launch_state=state.get("unsettled_attempts", {}).get(attempt_id))
    if not termination["confirmed"]:
        raise ExtractError("target termination could not be confirmed at recovery consumption")
    processing_path = directory / "processing.json"
    processing = read_json(processing_path) if processing_path.is_file() else None
    if processing and processing.get("started_at") is not None:
        elapsed = max(0.0, time.time() - processing["started_at"])
        processing["elapsed_seconds"] += elapsed
        processing["estimated_seconds"] = processing.get("estimated_seconds", 0.0) + elapsed
        processing["started_at"] = None
        write_json(processing_path, processing)
    elapsed = elapsed_on_disk(attempt, result)
    account(state, attempt, elapsed, result=result, processing=processing)
    result = complete_pending_result(state, attempt, result, termination)
    elapsed = max(elapsed, (result or {}).get("elapsed_seconds", 0) + (processing or {}).get("elapsed_seconds", 0))
    if state["active_attempts"].get(request["scope"]) == attempt_id:
        state = _interrupted(store, state, lock, attempt, result, termination,
                             "单范围恢复已确认旧调用退出；从安全checkpoint重建上下文。",
                             recovery_request_id=request["id"], resume_scope=False)
    write_json(directory / "finished.json", {"elapsed_seconds": elapsed, "status": state["status"], "recovered": True})
    state["unsettled_attempts"].pop(attempt_id, None)
    state["unconfirmed_attempts"].pop(attempt_id, None)
    account(state, attempt, elapsed, result=result, processing=processing, final=True)
    store.save(state)
    return state


def _consume_scope_retries(store, state, lock, running) -> tuple[dict, float | None]:
    """The sole run writer consumes the existing ledger, including after replay."""
    from .monitor import check_restart_authority, restart_evidence
    requests = [value for value in store.read_supervisor()["retries"].values()
                if value["run_id"] == state["run_id"] and value["status"] == "authorized"]
    next_due = None
    for request in requests[:8]:
        if request["attempt_id"] in running:
            continue
        stage, _, node = request["scope"].partition(":")
        call = {"stage": stage, "node": node or "root"}
        try:
            scope = scope_for(state, call)
            replayed = scope.get("last_recovery", {}).get("request_id") == request["id"]
            recovered = state.get("recovered_attempts", {}).get(request["attempt_id"], {})
            settled = recovered.get("recovery_request_id") == request["id"]
            if not replayed:
                consume_controls(store, state)
                check_restart_authority(store.read_supervisor(), request, state)
                target = request.get("target", {})
                if (target.get("input_fingerprint") != state["input_fingerprint"] or
                        target.get("executor_digest") != state["executor"]["digest"]):
                    raise ExtractError("scope recovery frozen input or executor identity changed")
                if not settled:
                    due = request.get("next_due_at", 0)
                    if time.time() < due:
                        next_due = min(next_due, due) if next_due is not None else due
                        continue
                    evidence = restart_evidence(store, state, request["scope"])
                    if any(evidence.get(key) != request.get(key) for key in ("failure_id", "run_id", "scope", "attempt_id")):
                        raise ExtractError("scope recovery target changed after authorization")
                    if evidence["settlement_required"]:
                        state = _settle_scope_retry(store, state, lock, request)
                elif request["attempt_id"] in state.get("unsettled_attempts", {}):
                    state = _settle_scope_retry(store, state, lock, request)
            with store.control_transaction() as ledger:
                current = ledger["retries"][request["id"]]
                if current["status"] != "authorized":
                    continue
                if not replayed:
                    check_restart_authority(ledger, current, state)
                    owner, scope = owner_for(state, call), scope_for(state, call)
                    owner["status"] = "draft" if stage == "P5" else "running"
                    scope["status"] = "reviewing" if owner.get("role") == "reviewer" else "draft"
                    owner["reason"] = None
                    scope["last_recovery"] = {"request_id": request["id"], "attempt_id": request["attempt_id"], "at": time.time()}
                    if stage == "P5":
                        state["status"] = "running"
                        settle(state)
                    # State commits first. The scope reference makes a crash
                    # before the control-ledger commit an idempotent replay.
                    store.save(state)
                consumed = scope["last_recovery"]["at"]
                current.update(status="consumed", consumed_at=consumed,
                               recovery_wait_seconds=max(0.0, consumed - current["created_at"]))
        except (_AuthorityChanged, SupervisorBusy):
            raise
        except (ExtractError, OSError, KeyError, TypeError, ValueError) as exc:
            with store.control_transaction() as ledger:
                current = ledger["retries"].get(request["id"], {})
                if current.get("status") == "authorized":
                    current.update(status="superseded", completed_at=time.time(), reason=str(exc))
    return state, next_due


def _authority_failure(store: Store, guard: _StateGuard, exc: Exception) -> dict:
    # Restore only Runner-authored bytes. Keep every active fence and unfinished
    # accounting record, even when the current event already confirms its exit.
    state = json.loads(guard.state_bytes)
    state["status"] = "invalid_receipt"
    state["reason"] = {"code": "authority_changed", "summary": str(exc), "path": "state.json",
                       "expected_sha256": sha256(guard.state_bytes),
                       "recovery_condition": "核对state外改原因及最后可信checkpoint；resume保留原调用计数并确认旧调用终止。"}
    token = STATE_GUARD.set(None)
    try:
        store.save(state)
    finally:
        STATE_GUARD.reset(token)
    return state


def _reconcile_orphans(store, state, lock, running, waiting):
    """Observe orphaned scopes; only explicit cancellation permits killing a live CLI."""
    pending = []
    for scope, attempt_id in list(state["active_attempts"].items()):
        if attempt_id in running or attempt_id in waiting:
            continue
        directory = store.run_dir(state["run_id"]) / "attempts" / attempt_id
        attempt = read_json(directory / "attempt.json")
        result = read_json(directory / "result.json") if (directory / "result.json").is_file() else None
        if (attempt_cancelled(state, attempt) or native_exit(attempt) is not None or
                termination_observed(attempt, result, launch_state=state.get("unsettled_attempts", {}).get(attempt_id))):
            state, recovered = _reconcile(store, state, lock, scopes={scope})
            pending.extend(recovered)
    return state, pending


def run(store: Store, run_id: str, *, transport=None, resume=False, queued_only=False,
        launch_id: str | None = None) -> dict:
    with store.writer(run_id):
        state = store.load(run_id)
        guard, events, running, deferred = _StateGuard(store, state), queue.Queue(), {}, []
        write_token = STATE_GUARD.set(guard)
        cancel_event = threading.Event()

        def invoke(attempt, prompt, schema):
            result, error, started = None, None, time.monotonic()
            try:
                result = cli.run(attempt=attempt, prompt=prompt, schema=schema, cancel_event=cancel_event)
                if result is not None:
                    write_json(Path(attempt["directory"]) / "result.json", result)
            except BaseException as exc:
                error = exc if isinstance(exc, (ExtractError, OSError)) else ExtractError(
                    f"invocation thread {type(exc).__name__}: {exc}")
            finally:
                elapsed = time.monotonic() - started
                event_path = Path(attempt["directory"]) / "event.json"
                try:
                    write_json(event_path, {"elapsed_seconds": elapsed, "received_at": time.time(),
                                           "error": str(error) if error else None})
                except BaseException as exc:
                    reason = f"cannot persist {event_path}: {type(exc).__name__}: {exc}"
                    error = ExtractError(f"{error}; {reason}" if error else reason)
                finally:
                    # Completion is an in-memory lifecycle signal. Audit IO or
                    # a thread exit must not strand the scheduler's running set.
                    events.put((attempt, result, error, elapsed, False))

        try:
            guard.verify()
            if queued_only and state["status"] != "queued":
                raise ExtractError("queued work changed before startup; inspect cannot resume paused or interrupted work")
            lock = read_json(store.run_dir(run_id) / "inputs.lock.json")
            verify_run_inputs(state, lock)
            verify_execution(store, state, executor=True)
            consume_controls(store, state)
            state = maintenance.consume_requests(store, state)
            initial_ledger = store.read_supervisor()
            initial_entry = initial_ledger["starts"].get(run_id, {}).get("launch", {})
            scope_entry = bool(launch_id and initial_entry.get("mode") == "retry")
            if scope_entry and initial_entry.get("retry_id") not in initial_ledger["retries"]:
                raise ExtractError("scope recovery has no failure-bound authorization")
            with store.control_transaction() as ledger:
                entry = ledger["starts"].get(run_id, {}).get("launch", {})
                if launch_id and (entry.get("id") != launch_id or entry.get("status") != "claimed"):
                    raise ExtractError("run launch authorization was replaced")
                if resume:
                    store.assert_startable(run_id, ledger)
                    if launch_id and entry.get("control_revision") != ledger["controls"].get(run_id, {}).get("revision", 0):
                        resume = False  # A later pause/cancel wins the launch window.
                    if launch_id and set(state["control"]["requests"]) - set(entry.get("request_ids", [])):
                        resume = False  # Request persistence may precede its ledger commit.
                    if entry.get("mode") == "hold" and launch_id:
                        hold_id = entry["hold_id"]
                        member = ledger["holds"][hold_id]["members"][run_id]
                        latest = ledger["controls"].get(run_id, {})
                        if (state["status"] != "paused" or state["control"].get("hold_id") != hold_id or
                                member.get("request_id") != latest.get("id") or state["active_attempts"] or
                                state.get("unconfirmed_attempts") or state.get("unsettled_attempts")):
                            raise ExtractError("hold no longer owns a confirmed pause")
                elif ledger["active_hold"]:
                    store.hold_member(ledger, state)
            if state["status"] == "queued" and not resume and not state["control"]["action"] and not maintenance.barrier(store, state):
                return state
            if resume:
                if launch_id:
                    add_calls(state, entry.get("add_calls", 0), identity=launch_id, origin="resume")
                _resume_stopped(state)
                if not launch_id or entry.get("mode") == "resume":
                    _resume_maintenance_receipts(store, state)
                    _resume_failed_scopes(state)
                store.save(state)
            consume_controls(store, state)
            state = maintenance.consume_requests(store, state)
            from .monitor import _identity
            state["runner"] = {"identity": _identity(), "launch_id": launch_id, "observed_at": time.time()}
            store.save(state)
            if ((state.get("reason") or {}).get("code") == "authority_changed" and
                    state["control"]["action"] == "cancel"):
                reason = deepcopy(state["reason"])
                state, _ = _reconcile(store, state, lock, terminate_only=True)
                state["status"], state["reason"] = "cancelled", reason
                store.save(state)
                return state
            held_status = state["status"] if state["status"] in {"paused", "cancelled"} else None
            draining = maintenance.barrier(store, state)
            if scope_entry and not draining:
                state, _ = _consume_scope_retries(store, state, lock, running)
            if draining:
                state, pending = _reconcile_orphans(store, state, lock, running, set())
                inactive = set(state.get("unsettled_attempts", {})) - set(state["active_attempts"].values())
                state, recovered = _reconcile(store, state, lock, attempt_ids=inactive)
                pending.extend(recovered)
            else:
                state, pending = _reconcile(store, state, lock, scopes=set() if scope_entry else None)
            if held_status and not resume:
                state["status"] = "cancelled" if state["control"]["action"] == "cancel" else held_status
                store.save(state)
                if not pending:
                    if draining and not any(state.get(key) for key in ("active_attempts", "unconfirmed_attempts", "unsettled_attempts")):
                        state = maintenance.mark_drained(store, state)
                    return state
            cli = transport or AgentCLI(backend=state["config"]["backend"],
                                        executable=state["config"]["agent_binary"], model=state["config"]["model"])
            next_orphan_observation = 0.0
            while True:
                consume_controls(store, state)
                state = maintenance.consume_requests(store, state)
                draining = maintenance.barrier(store, state)
                retry_due = None
                if not draining:
                    state, retry_due = _consume_scope_retries(store, state, lock, running)
                if time.monotonic() >= next_orphan_observation:
                    waiting = {item[0]["attempt_id"] for item in [*pending, *deferred]}
                    state, recovered = _reconcile_orphans(store, state, lock, running, waiting)
                    pending.extend(recovered)
                    next_orphan_observation = time.monotonic() + 1.0
                if state["control"]["action"] == "cancel":
                    cancel_event.set()
                if draining and not running and not pending and not deferred and not any(
                        state.get(key) for key in ("active_attempts", "unconfirmed_attempts", "unsettled_attempts")):
                    state = maintenance.mark_drained(store, state)
                    draining = maintenance.barrier(store, state)
                    if state["control"]["action"] or draining:
                        break
                # Every ready sibling is launched before waiting. Control is
                # checked at each call boundary, including between siblings.
                for call in next_calls(state) if not state["control"]["action"] and not draining else []:
                    reason = owner_for(state, call).get("reason") or {}
                    due = reason.get("next_due_at", 0)
                    if reason.get("code") in {"cli_retry", "receipt_repair"} and time.time() < due:
                        retry_due = min(retry_due, due) if retry_due is not None else due
                        continue
                    with store.control_transaction() as ledger:
                        if ledger["active_hold"]:
                            store.hold_member(ledger, state)
                    consume_controls(store, state)
                    state = maintenance.consume_requests(store, state)
                    if state["control"]["action"] or maintenance.barrier(store, state):
                        break
                    if call["stage"] == "P6":
                        try:
                            current, _ = store.verify_checkpoint(state)
                            _final_inputs(store, state, current)
                        except ExtractError as exc:
                            state["status"] = state["stages"]["P6"]["status"] = "blocked"
                            state["reason"] = {"code": "invalid_final_prerequisites", "summary": str(exc),
                                               "recovery_condition": "重新核对已审节点/上游验收的固定回执和产物后才能终审。"}
                            store.save(state)
                            break
                    view = call_view(state, call)
                    exhausted = call_exhaustion(view)
                    if exhausted:
                        owner, scope = owner_for(state, call), scope_for(state, call)
                        owner["status"] = scope["status"] = "blocked" if call["mode"] != "normal" else "call_limit_reached"
                        owner["reason"] = {"code": "call_limit_reached", "summary": exhausted, "checkpoint": view["checkpoint"],
                                           "recovery_condition": CALL_RECOVERY.format(run_id=state["run_id"])}
                        settle(state)
                        store.save(state)
                        continue
                    guard.verify(state)
                    final_started = state["final_review_started"]
                    writing_started = deepcopy(state["nodes"][call["node"]].get("writing_started"))
                    attempt = store.prepare_attempt(state, call["session_id"], call=call)
                    context = store.context(state, attempt)
                    prompt = build_prompt(state, context["source_lock"], attempt,
                                          rules_lock=context["rules_lock"], executor_lock=context["executor_lock"])
                    schema = proposal_schema(attempt, source_files=lock["source"]["files"])
                    thread = threading.Thread(target=invoke, args=(attempt, prompt, schema), daemon=True)
                    with store.control_transaction() as ledger:
                        if ledger["active_hold"]:
                            store.hold_member(ledger, state)
                        consume_controls(store, state)
                        denied = bool(state["control"]["action"] or ledger["active_hold"] or
                                      maintenance.barrier(store, state, ledger=ledger))
                        if not denied:
                            state["unsettled_attempts"][attempt["attempt_id"]] = "dispatched"
                            store.save(state)
                            guard.verify(state)
                            running[attempt["attempt_id"]] = thread
                            thread.start()
                    if denied:
                        _release(state, attempt)
                        state["unsettled_attempts"].pop(attempt["attempt_id"], None)
                        # Preparation is durable evidence, but a gate-denied
                        # invocation used no model call or one-shot decision.
                        scope = scope_for(state, attempt)
                        scope["calls"] -= 1
                        if attempt["mode"] in {"adjudicate", "signoff"}:
                            scope["adjudications" if attempt["mode"] == "adjudicate" else "signoffs"] -= 1
                        if attempt["stage"] == "P6" and not final_started:
                            state["final_review_started"] = False
                            state["p6_starts"] -= 1
                        if attempt["stage"] == "P5":
                            node = state["nodes"][attempt["node"]]
                            if writing_started is None:
                                node.pop("writing_started", None)
                            else:
                                node["writing_started"] = writing_started
                        release_unstarted_handoff(state, attempt)
                        if attempt.get("purpose") == "maintenance":
                            task = maintenance.task_for(state, attempt)
                            task["attempts"] = [item for item in task.get("attempts", []) if item["attempt_id"] != attempt["attempt_id"]]
                        write_json(Path(attempt["directory"]) / "termination.json", {
                            "confirmed": True, "actions": [], "survivors": [], "reason": "control gate denied dispatch"})
                        write_json(Path(attempt["directory"]) / "finished.json", {"elapsed_seconds": 0.0, "status": "not_dispatched"})
                        consume_controls(store, state)
                        store.save(state)
                        break
                if deferred and (draining or not state.get("reopening") or state["status"] != "running"):
                    event = deferred.pop(0)
                elif pending:
                    event = pending.pop(0)
                elif running:
                    try:
                        event = events.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    running.pop(event[0]["attempt_id"]).join()
                elif not state["control"]["action"] and (retry_due is not None or
                        any(item not in state.get("unconfirmed_attempts", {}) for item in state["active_attempts"].values())):
                    cancel_event.wait(min(0.25, max(0.01, retry_due - time.time())) if retry_due is not None else 0.25)
                    continue
                else:
                    break
                attempt, result, error, elapsed, recovering = event
                guard.verify(state)
                if attempt["stage"] == "P5" and state.get("reopening") and state["status"] == "running" and not error and not draining:
                    deferred.append(event)
                    continue
                account(state, attempt, elapsed, result=result)
                store.save(state)
                processing_path = Path(attempt["directory"]) / "processing.json"
                processing = read_json(processing_path) if processing_path.is_file() else {"elapsed_seconds": 0.0}
                processing["started_at"] = time.time()
                write_json(processing_path, processing)
                started = time.monotonic()
                if result and not error and (result.get("outcome") == "cancelled" or
                                               result.get("termination", {}).get("confirmed") is not True):
                    state = _interrupted(store, state, lock, attempt, result, result["termination"],
                                         "调用终止仍待确认。" if not result["termination"]["confirmed"] else
                                         "cancel请求终止了本次调用；已保存可验证草稿。")
                else:
                    state = _receive(store, state, lock, attempt, result, error, recovering=recovering)
                processing["elapsed_seconds"] += time.monotonic() - started
                processing["started_at"] = None
                processing["completed_at"] = time.time()
                write_json(processing_path, processing)
                elapsed += processing["elapsed_seconds"]
                write_json(Path(attempt["directory"]) / "finished.json", {"elapsed_seconds": elapsed, "status": state["status"]})
                if attempt["attempt_id"] not in state["unconfirmed_attempts"]:
                    state["unsettled_attempts"].pop(attempt["attempt_id"], None)
                account(state, attempt, elapsed, result=result, processing=processing, final=True)
                store.save(state)
            with store.control_transaction() as ledger:
                if ledger["active_hold"]:
                    store.hold_member(ledger, state)
            consume_controls(store, state)
            guard.verify(state)
            action = state["control"]["action"]
            if action:
                state["control"]["resume_status"] = state["status"]
                state["status"] = "cancelled" if action == "cancel" else "paused"
                state["control"]["stopped_at"] = time.time()
                request = state["control"]["requests"].get(state["control"].get("last_request_id"))
                if request is not None:
                    request.setdefault("stopped_at", state["control"]["stopped_at"])
                store.save(state)
            elif state["status"] == "ready" and not maintenance.barrier(store, state):
                from .publication import publish
                state = publish(store, state, Path(state["output"]))
            return state
        except _AuthorityChanged as exc:
            cancel_event.set()
            return _authority_failure(store, guard, exc)
        finally:
            cancel_event.set()
            try:
                if launch_id:
                    with store.supervisor() as ledger:
                        entry = ledger["starts"].get(run_id, {}).get("launch", {})
                        if entry.get("id") == launch_id:
                            entry.update(status="termination_unknown" if any(thread.is_alive() for thread in running.values()) else "finished",
                                         finished_at=time.time())
            except SupervisorBusy:
                pass
            finally:
                STATE_GUARD.reset(write_token)


def _reject(store: Store, state: dict, attempt: dict, result: dict | None, exc: Exception) -> dict:
    confirmed = result is not None and result.get("termination", {}).get("confirmed")
    owner = owner_for(state, attempt)
    owner["status"] = "invalid_receipt" if confirmed else "termination_unconfirmed"
    scope_for(state, attempt)["invalid_receipts"] += 1
    scope_for(state, attempt)["status"] = owner["status"]
    owner["reason"] = {"code": "proposal_rejected", "summary": str(exc), "attempt_id": attempt["attempt_id"],
                       "recovery_condition": "核对拒绝原因与固定输入；确认旧进程终止后按原调用计数恢复。"}
    if confirmed and isinstance(exc, UpstreamRequired):
        owner["status"] = scope_for(state, attempt)["status"] = "blocked"
        owner["reason"].update(code="invalid_root_prerequisite", target=exc.target,
                              recovery_condition=f"在P4提供固定来源及root产物锚点，按现有reopen协议请求{exc.target}修复并重新审阅；P4/P5不可改root。")
        owner["feedback"] = [*owner["feedback"], owner["reason"]]
    if confirmed:
        _release(state, attempt)
    else:
        state["unconfirmed_attempts"][attempt["attempt_id"]] = owner["reason"]
    write_json(Path(attempt["directory"]) / "rejected.json", owner["reason"])
    if confirmed and attempt.get("mode") in {"adjudicate", "signoff"}:
        return _stop_once(store, state, attempt, owner["reason"])
    settle(state)
    store.save(state)
    return state


def export_checkpoint(store: Store, run_id: str, output: Path) -> dict:
    """Explicit alternate delivery never changes the original output destination."""
    from .publication import publish
    with store.writer(run_id):
        state = store.load(run_id)
        guard = _StateGuard(store, state)
        token = STATE_GUARD.set(guard)
        try:
            guard.verify()
            verify_execution(store, state, executor=True)
            return publish(store, state, output, export=True)["exports"][-1]
        except _AuthorityChanged as exc:
            _authority_failure(store, guard, exc)
            raise
        finally:
            STATE_GUARD.reset(token)
