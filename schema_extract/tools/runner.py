"""Concurrent agent calls with one authoritative, serial commit loop per operator."""

from __future__ import annotations

from copy import deepcopy
from uuid import UUID
from pathlib import Path
import os
import inspect
import json
import queue
import shutil
import stat
import threading
import time

from .agent import CodexCLI
from .artifacts import (UpstreamRequired, _same, check_complete, load_yaml, validate_artifact_sources, validate_artifacts,
                        validate_plan, write_yaml)
from .comparators import validate_comparisons
from .budget import (BUDGET_STOPS, RECOVERY as BUDGET_RECOVERY, call_budget as _budget,
                     check_timeout as _check_timeout, extend as extend_budget)
from .common import (STATE_GUARD, ExtractError, StaleResult, copy_artifacts, digest, json_bytes, manifest,
                     read_json, sha256, write_json, write_text, rules_root)
from .inputs import resolve_request
from .prompts import build_prompt
from .receipts import (ReceiptFormatError, check_proposal_identity, evidence_identity,
                       normalize_response, proposal_schema, validate_receipt)
from .recovery import account, confirm_termination, consume_controls, elapsed_on_disk, handoff_session
from .snapshots import freeze_inputs, verify_executor, verify_run_inputs
from .stages import (NODE_STOPS, advance, affected_nodes, call_view, effective_nodes, final_eligible,
                     next_calls, node_signature, owner_for, plan_impact, plan_signature, scope_for, scope_key,
                     session_group, settle)
from .store import Store, summary


DEFAULT_CONFIG = {"worker_effort": "xhigh", "reviewer_effort": "xhigh", "model": None,
                  "codex_binary": "codex", "revision_limit": 5, "operator_seconds": 7200,
                  "stage_seconds": 3600, "call_seconds": 1800, "max_calls": 32}


def create_run(repo: Path, state_root: Path, output: Path, *, operator_id: str | None = None,
               source: str | None = None, bindings: Path | None = None,
               config: dict | None = None, import_seed: dict | None = None, queued=False,
               run_id: str | None = None, created_at: float | None = None) -> tuple[Store, dict, dict]:
    config = {**DEFAULT_CONFIG, **(config or {})}
    if set(config) != set(DEFAULT_CONFIG):
        raise ExtractError("unknown extraction configuration field")
    for role in ("worker", "reviewer"):
        if config[f"{role}_effort"] not in {"low", "medium", "high", "xhigh", "max"}:
            raise ExtractError(f"invalid {role} effort")
    for key in ("revision_limit", "operator_seconds", "stage_seconds", "call_seconds", "max_calls"):
        if type(config[key]) is not int or config[key] < 1:
            raise ExtractError(f"{key} must be a positive integer")
    if config["max_calls"] < 3:
        raise ExtractError("max_calls must be at least 3: P6 needs review, adjudication and signoff allowances")
    config["adjudication_seconds"] = min(config["call_seconds"], config["stage_seconds"] / 4,
                                         config["operator_seconds"] / 12)
    config["retry_limit"] = 2
    executable = shutil.which(config["codex_binary"])
    if executable is None:
        raise ExtractError(f"Codex CLI executable is missing: {config['codex_binary']}")
    config["codex_binary"] = str(Path(executable).resolve())
    repo, state_root, output = repo.resolve(), state_root.resolve(), output.resolve()
    if output == repo or output.is_relative_to(state_root) or state_root.is_relative_to(output):
        raise ExtractError("output must be separate from repository root and v2 runtime storage")
    tree, normalized, names = resolve_request(repo, operator_id=operator_id, source=source, bindings=bindings)
    source_directory = tree.root / normalized["directory"]
    if state_root.is_relative_to(source_directory) and ".schema_extract" not in state_root.relative_to(source_directory).parts:
        raise ExtractError("runtime storage inside selected sources must be under .schema_extract, or use a separate directory")
    store = Store(state_root)
    store.initialize()
    lock = freeze_inputs(state_root, tree, normalized, names, config,
                         source_aliases=import_seed.get("source_aliases") if import_seed else None)
    return store, store.create(lock, output, seed=import_seed, queued=queued,
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


def _scope_at(state: dict, attempt: dict) -> dict:
    return state["nodes"][attempt["node"]]["stage"] if attempt["stage"] == "P5" else state["stages"][attempt["stage"]]


def _session(state: dict, attempt: dict, result: dict, *, required=True) -> str | None:
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


def _artifact_gate(state: dict, lock: dict, attempt: dict, before: Path, after: Path,
                   decision: str | None, *, role: str | None = None) -> dict:
    entry = next((item for item in state.get("plan") or [] if item["id"] == attempt["node"]), None)
    validation = validate_artifacts(attempt["stage"], role or attempt["role"], before, after,
        rules_root(lock) / "schema_template/starter.yaml", decision=decision,
        node_entry=entry, revision_scope=state["revision_scope"],
        first_draft=attempt["stage"] == "P5" and not state["nodes"][attempt["node"]].get("authored", False),
        root_revision=attempt["stage"] == "P3" and bool(state.get("reopening")))
    validate_artifact_sources(after, lock, decision=decision)
    return validation


def _draft(store: Store, state: dict, attempt: dict, workspace: Path, *, plan=None) -> str:
    checkpoint_id = store.checkpoint(state, workspace, attempt_id=attempt["attempt_id"],
                                     stage=attempt["stage"], node=attempt["node"], parent=attempt["base_checkpoint"])
    if attempt["stage"] != "P5":
        state["checkpoint"] = checkpoint_id
    state["recoverable_checkpoint"] = checkpoint_id
    if attempt["attempt_id"] in state.get("attempt_usage", {}):
        state["attempt_usage"][attempt["attempt_id"]]["result_checkpoint"] = checkpoint_id
    node = state["nodes"][attempt["node"]]
    node["draft_checkpoint"] = checkpoint_id
    if attempt["stage"] == "P5":
        node["authored"] = True
    _scope_at(state, attempt)["candidates"].append({"checkpoint": checkpoint_id, "plan": deepcopy(plan)})
    return checkpoint_id


def _record_check_budget(store: Store, state: dict, attempt: dict, receipt: dict,
                        receipt_hash: str, workspace: Path, validation: dict, session_id: str) -> dict:
    updated = deepcopy(state)
    _remember_session(updated, attempt, session_id)
    if attempt["role"] == "worker" or attempt.get("mode") == "adjudicate":
        _draft(store, updated, attempt, workspace, plan=receipt.get("plan"))
    _release(updated, attempt)
    updated["accepted_receipts"][attempt["attempt_id"]] = receipt_hash
    owner = owner_for(updated, attempt)
    owner["status"] = scope_for(updated, attempt)["status"] = "budget_exhausted"
    owner["reason"] = {"code": "checker_budget_exhausted", "summary": "已保存合法草稿；检查预算已耗尽，未动用后续审阅/裁决/签收预留。",
                       "checkpoint": call_view(updated, attempt)["checkpoint"], "recovery_condition": BUDGET_RECOVERY.format(run_id=state["run_id"])}
    validation["check"] = {"kind": "budget", "exit_code": None, "not_run": True}
    write_json(Path(attempt["directory"]) / "validated.json", {"receipt_sha256": receipt_hash, **validation})
    if attempt.get("mode") in {"adjudicate", "signoff"}:
        return _stop_once(store, updated, attempt, owner["reason"])
    settle(updated)
    store.save(updated)
    return updated


def _record_check_failure(store: Store, state: dict, attempt: dict, receipt: dict,
                          receipt_hash: str, workspace: Path, validation: dict, session_id: str) -> dict:
    updated = deepcopy(state)
    _remember_session(updated, attempt, session_id)
    code = validation["check"]["exit_code"]
    scope, owner = scope_for(updated, attempt), owner_for(updated, attempt)
    scope["check_failures" if code == 1 else "tool_failures"] += 1
    if attempt["role"] == "worker" or attempt.get("mode") == "adjudicate":
        _draft(store, updated, attempt, workspace, plan=receipt.get("plan"))
    _release(updated, attempt)
    updated["accepted_receipts"][attempt["attempt_id"]] = receipt_hash
    owner["reason"] = {"code": "checker_input_error" if code == 1 else "checker_tool_error",
                         "summary": "检查器退出1；诊断已交当前修改者。" if code == 1 else "检查器退出2；需要修复工具/环境后按原预算恢复。",
                         "check": validation["check"], "recovery_condition": "修复记录中的检查诊断并重新核对本阶段。"}
    write_json(Path(attempt["directory"]) / "validated.json", {"receipt_sha256": receipt_hash, **validation})
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
    scope = _scope_at(state, attempt)
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


def _final_inputs(store: Store, state: dict, workspace: Path) -> None:
    if not final_eligible(state):
        raise ExtractError("P6 prerequisites are not all current and approved")
    validate_plan(state["plan"], workspace, rules_root(state), require_files=True)
    files = manifest(workspace)
    for stage in ("P1", "P2", "P3", "P4"):
        approval = state["stages"][stage]["approval"]
        receipt, _ = store.verify_approval(state, approval)
        if approval["generation"] != state["stages"][stage]["generation"]:
            raise ExtractError("upstream approval belongs to a superseded stage generation")
        if stage == "P4" and receipt["plan"] != state["plan"]:
            raise ExtractError("current plan differs from P4 approval")
    for entry in state["plan"]:
        node_id, node = entry["id"], state["nodes"][entry["id"]]
        store.verify_node(state, node_id)
        _, approved_checkpoint = store.verify_approval(state, node["approval"])
        if approved_checkpoint["files"].get(node["file"]) != node["approved_sha256"]:
            raise ExtractError("node completion marker differs from its accepted review/checkpoint")
        if node_id not in state["pending_rechecks"] and files.get(node["file"]) != node["approved_sha256"]:
            raise ExtractError("a node changed without an effective review or explicit P6 recheck")
        if node_id not in state["pending_rechecks"]:
            parent = node["parent"]
            while parent:
                ancestor = state["nodes"][parent]
                if approved_checkpoint["files"].get(ancestor["file"]) != files.get(ancestor["file"]):
                    raise ExtractError("a node's approved ancestor dependency changed without P6 recheck")
                parent = ancestor["parent"]
    root_approval = state["stages"]["P3"]["approval"]
    if files["root.yaml"] != root_approval["artifact_digests"]["root.yaml"]:
        raise ExtractError("P6 cannot change the approved root prerequisites; upstream reopening is required")


def _adjudicated_workspace(store: Store, state: dict, lock: dict, attempt: dict, receipt: dict) -> tuple[Path, dict]:
    adjudication = receipt["adjudication"]
    source = Path(attempt["workspace"])
    choice = adjudication["checkpoint"]
    candidates = {item["checkpoint"]: item for item in scope_for(state, attempt)["candidates"]}
    if adjudication["action"] == "select":
        if choice not in candidates:
            raise ExtractError("adjudication selected a checkpoint outside this stage/node's saved candidates")
        source, _ = store.verify_checkpoint(state, choice)
        if attempt["stage"] == "P4" and plan_signature(receipt["plan"]) != plan_signature(candidates[choice]["plan"]):
            raise ExtractError("selected P4 checkpoint and structured plan do not match")
    elif choice != attempt["base_checkpoint"]:
        raise ExtractError("adjudication patch must be based on the current checkpoint")
    destination = Path(attempt["directory"]) / "adjudicated"
    _rebuild_directory(destination)
    copy_artifacts(source, destination)
    for path in destination.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    destination.chmod(0o755)
    before = Path(attempt["workspace"])
    if adjudication["action"] == "patch":
        from .common import regular_file
        for patch in adjudication["patches"]:
            path = regular_file(destination, patch["path"])
            text = path.read_text(encoding="utf-8")
            if not patch["old"] or text.count(patch["old"]) != 1:
                raise ExtractError("adjudication patch old text must match exactly once")
            write_text(path, text.replace(patch["old"], patch["new"], 1))
    else:
        # Selecting an old draft restores content, never a historical state/budget/version.
        for name in manifest(destination):
            if not name.endswith(".yaml"):
                continue
            current = load_yaml(before, name, parser_root=rules_root(lock))
            selected = load_yaml(destination, name, parser_root=rules_root(lock))
            left, right = deepcopy(current), deepcopy(selected)
            left["meta_data"].pop("version", None)
            right["meta_data"].pop("version", None)
            if not _same(left, right):
                selected["meta_data"]["version"] = current["meta_data"]["version"] + 1
                write_yaml(destination / name, selected, source=destination / name,
                           parser_root=rules_root(lock))
            elif selected["meta_data"]["version"] < current["meta_data"]["version"]:
                selected["meta_data"]["version"] = current["meta_data"]["version"]
                write_yaml(destination / name, selected, source=destination / name,
                           parser_root=rules_root(lock))
    validation = _artifact_gate(state, lock, attempt, before, destination, "approve", role="worker")
    if attempt["stage"] == "P1":
        validate_comparisons(receipt, lock, destination)
    validation["adjudication_changed"] = adjudication["action"] == "patch" and bool(validation["changed_files"])
    if attempt["stage"] == "P4":
        validate_plan(receipt["plan"], destination, rules_root(lock))
    if attempt["stage"] == "P6":
        changed_nodes = {Path(name).stem for name in validation["changed_files"] if name.endswith(".yaml")}
        if not changed_nodes <= set(receipt["affected_nodes"]):
            raise ExtractError("P6 adjudication did not identify every changed node")
    return destination, validation


def _plan_workspace(store: Store, state: dict, attempt: dict, receipt: dict, workspace: Path, validation: dict) -> Path:
    """Install a reviewed index onto only those previous merges still proved valid."""
    old_plan = {entry["id"]: entry for entry in state.get("plan") or []}
    root_entry = next(entry for entry in receipt["plan"] if entry["id"] == "root")
    reopening = state.get("reopening")
    if reopening and reopening["target"] == "P4" and node_signature(root_entry) != node_signature(old_plan["root"]):
        raise ExtractError("changing the root selection or design requires P2/P3 upstream review")
    previous, old_info = (store.verify_checkpoint(state, state["merged_checkpoint"])
                          if state["merged_checkpoint"] else (None, {"files": {}}))
    files = manifest(workspace)
    invalidated = plan_impact(state, receipt["plan"], root_changed=bool(previous) and
                              old_info["files"].get("root.yaml") != files["root.yaml"])
    destination = Path(attempt["directory"]) / "planned"
    _rebuild_directory(destination)
    copy_artifacts(workspace, destination)
    for path in destination.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    destination.chmod(0o755)
    for entry in receipt["plan"]:
        node_id = entry["id"]
        if node_id != "root" and node_id not in invalidated and node_id in effective_nodes(state):
            store.verify_node(state, node_id, merged_info=old_info)
            copy_artifacts(previous / entry["file"], destination / entry["file"])
    validate_plan(receipt["plan"], destination, rules_root(state))
    validation["invalidated_nodes"] = sorted(invalidated)
    validation["artifact_digests"] = manifest(destination)
    return destination


def _begin_reopen(store: Store, state: dict, attempt: dict, receipt: dict) -> dict:
    """One evidence-bound entry for every upstream reopening, never a state reset."""
    request = receipt["reopen"]
    owner, scope = owner_for(state, attempt), scope_for(state, attempt)
    evidence = sorted({digest(evidence_identity(receipt["source_evidence"][index])):
                       evidence_identity(receipt["source_evidence"][index]) for index in request["evidence"]}.values(), key=digest)
    key = digest({"issue_id": request["issue_id"].strip(), "evidence": evidence})
    def blocked(code: str, summary: str) -> dict:
        owner["status"] = scope["status"] = "blocked"
        owner["reason"] = {"code": code, "summary": summary, "issues": receipt["issues"]}
        settle(state)
        return state
    if request["target"] == "external":
        return blocked("external_confirmation", "外部任务要求发生变化；需要任务方明确确认回执中列出的要求与恢复条件。")
    if key in state["reopen_requests"]:
        return blocked("reopen_already_used", "同一问题与同一固定源码证据已经重开过；原预算和决定继续有效。")
    target = request["target"]
    prior = state["stages"][target]["approval"]
    if prior:
        prior_receipt, _ = store.verify_approval(state, prior)
        old_evidence = {digest(evidence_identity(item)) for item in prior_receipt["source_evidence"]}
        artifact_correction = any(issue["kind"] in {"transcription", "source_transcription", "schema_conformance"} and
                                  any(anchor["path"] in issue["scope"] for anchor in receipt["artifact_evidence"])
                                  for issue in receipt["issues"])
        if all(digest(item) in old_evidence for item in evidence) and not artifact_correction:
            return blocked("reopen_requires_new_evidence", "请求没有提供新的固定源码发现，或带当前产物锚点的具体转录/规范表达差异。")
    source_checkpoint = state["merged_checkpoint"] or state["checkpoint"]
    source, _ = store.verify_checkpoint(state, source_checkpoint)
    record = {"key": key, "issue_id": request["issue_id"].strip(), "target": target, "reason": request["reason"],
              "source_evidence": evidence, "attempt_id": attempt["attempt_id"], "from_stage": attempt["stage"],
              "artifact_evidence": deepcopy(receipt["artifact_evidence"]), "issues": deepcopy(receipt["issues"]),
              "from_node": attempt["node"], "previous_plan": deepcopy(state["plan"]),
              "previous_merged_checkpoint": state["merged_checkpoint"], "previous_plan_revision": state["plan_revision"]}
    state["reopen_requests"][key] = deepcopy(record)
    state["reopening"] = record
    state["plan_revision"] += 1
    state["final_review_started"] = False
    for stage in ("P1", "P2", "P3", "P4", "P6"):
        if int(stage[1:]) >= int(target[1:]):
            stage_scope = state["stages"][stage]
            stage_scope["last_approval"] = stage_scope["approval"] or stage_scope.get("last_approval")
            stage_scope["status"], stage_scope["approval"] = "pending", None
            stage_scope["generation"] += 1
    if target in {"P1", "P2", "P3"}:
        root = state["nodes"]["root"]
        root["generation"] += 1
        root["status"] = "pending"
    if attempt["stage"] == "P5":
        owner.update(role="worker", mode="normal", status="pending", feedback=[request])
    state.update(stage=target, node="root", role="worker", mode="normal", status="running",
                 feedback=[request, *receipt["issues"]], reason=None)
    destination = Path(attempt["directory"]) / "upstream"
    _rebuild_directory(destination)
    destination.mkdir()
    for name in ("root.yaml", "dispatch.md"):
        if (source / name).is_file():
            copy_artifacts(source / name, destination / name)
    state["checkpoint"] = store.checkpoint(state, destination, attempt_id=attempt["attempt_id"], parent=source_checkpoint)
    state["nodes"]["root"]["draft_checkpoint"] = state["checkpoint"]
    return state


def accept_result(store: Store, state: dict, lock: dict, attempt: dict, result: dict) -> dict:
    """Validate a result and commit once; late/repeated proposals never move pointers."""
    state = store.load(state["run_id"])
    if attempt["attempt_id"] in state["accepted_receipts"]:
        raise ExtractError("duplicate receipt: this attempt was already accepted")
    store.validate_attempt(state, attempt)
    if result.get("outcome") != "completed" or not result.get("termination", {}).get("confirmed"):
        raise ExtractError("agent call did not complete with confirmed termination")
    session_id = _session(state, attempt, result)
    verify_run_inputs(state, lock)
    before, checkpoint = store.verify_checkpoint(state, attempt["base_checkpoint"])
    if checkpoint["files"] != attempt["dependency_digests"]:
        raise ExtractError("attempt dependency digests do not match the fixed checkpoint")
    workspace = Path(attempt["workspace"])
    proposal = result.get("response")
    check_proposal_identity(proposal, attempt)
    # An adjudicator reads the current draft but may select/patch a different
    # final artifact. Enforce read-only scope now and completion on that result.
    gate_decision = "continue" if attempt.get("mode") == "adjudicate" else proposal.get("decision")
    validation = _artifact_gate(state, lock, attempt, before, workspace, gate_decision)
    receipt = normalize_response(proposal, lock, attempt, workspace)
    receipt = validate_receipt(receipt, state, lock, attempt, workspace, validation["changed_files"])
    receipt_hash = digest(receipt)
    directory = Path(attempt["directory"])
    write_json(directory / "receipt.json", receipt)
    if attempt.get("mode") == "adjudicate" and receipt["decision"] == "approve":
        workspace, validation = _adjudicated_workspace(store, state, lock, attempt, receipt)
    if attempt["stage"] == "P6":
        _final_inputs(store, state, workspace)
    validation["node_signature"] = attempt["node_signature"]
    validation["dependency_digests"] = {name: value for name, value in attempt["dependency_digests"].items()
                                         if name != state["nodes"][attempt["node"]]["file"]}
    if attempt["stage"] in {"P3", "P5", "P6"} and receipt["decision"] in {"complete", "approve", "revise"}:
        check_timeout = _check_timeout(call_view(state, attempt))
        if check_timeout <= 0:
            return _record_check_budget(store, state, attempt, receipt, receipt_hash, workspace, validation, session_id)
        validation["check"] = check_complete(workspace, rules_root(lock),
            filename=state["nodes"][attempt["node"]]["file"], timeout=check_timeout)
        write_json(directory / "check.json", validation["check"])
        # A review can send an unfinished draft back to its author. Its real
        # exit-1 diagnostics accompany that counted revision; approval still
        # requires exit 0, and exit 2 remains a separate tool failure.
        if validation["check"]["exit_code"] and not (
                receipt["decision"] == "revise" and validation["check"]["exit_code"] == 1):
            write_json(directory / "validated.json", {"receipt_sha256": receipt_hash, **validation})
            return _record_check_failure(store, state, attempt, receipt, receipt_hash, workspace, validation, session_id)
    updated = deepcopy(state)
    if validation.get("check", {}).get("exit_code") == 1:
        scope_for(updated, attempt)["check_failures"] += 1
    _remember_session(updated, attempt, session_id)
    checkpoint_id = attempt["base_checkpoint"]
    if attempt["role"] == "worker" or (attempt.get("mode") == "adjudicate" and receipt["decision"] == "approve"):
        checkpoint_id = store.checkpoint(state, workspace, attempt_id=attempt["attempt_id"],
                                         stage=attempt["stage"], node=attempt["node"], parent=attempt["base_checkpoint"])
        updated["recoverable_checkpoint"] = checkpoint_id
        if attempt["stage"] == "P5" and (validation["changed_files"] or receipt["decision"] == "complete"):
            updated["nodes"][attempt["node"]]["authored"] = True
    if attempt["stage"] == "P5" and receipt["decision"] == "approve":
        _rebuild_directory(directory / "merged")
        merged = store.merged_workspace(state, workspace, directory / "merged", node_id=attempt["node"])
        check_timeout = _check_timeout(call_view(state, attempt))
        if check_timeout <= 0:
            return _record_check_budget(store, state, attempt, receipt, receipt_hash, workspace, validation, session_id)
        merge_check = check_complete(merged, rules_root(lock), timeout=check_timeout)
        write_json(directory / "merge_check.json", merge_check)
        validation["merge_check"] = merge_check
        if merge_check["exit_code"]:
            validation["check"] = merge_check
            return _record_check_failure(store, state, attempt, receipt, receipt_hash, workspace, validation, session_id)
        checkpoint_id = store.checkpoint(state, merged, attempt_id=attempt["attempt_id"], stage="P5", node=attempt["node"],
                                         parent=state["merged_checkpoint"])
        validation["artifact_digests"] = manifest(merged)
    if attempt["stage"] == "P4" and receipt["decision"] == "approve":
        planned = _plan_workspace(store, state, attempt, receipt, workspace, validation)
        checkpoint_id = store.checkpoint(state, planned, attempt_id=attempt["attempt_id"], parent=state["merged_checkpoint"])
    write_json(directory / "validated.json", {"receipt_sha256": receipt_hash, **validation})
    if attempt["attempt_id"] in updated.get("attempt_usage", {}):
        updated["attempt_usage"][attempt["attempt_id"]]["result_checkpoint"] = checkpoint_id
    updated = advance(updated, receipt, checkpoint_id, receipt_hash, validation=validation)
    if receipt["decision"] == "reopen":
        updated = _begin_reopen(store, updated, attempt, receipt)
    if attempt["stage"] in {"P2", "P3", "P4"} and receipt["decision"] == "approve":
        store.handoff(updated, attempt, receipt["handoff"], lock)
    if store.load(state["run_id"]) != state:
        raise ExtractError("agent modified the authoritative state during result validation")
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
    workspace = Path(attempt["workspace"])
    validation = _artifact_gate(state, lock, attempt, before, workspace, "continue")
    _remember_session(updated, attempt, session_id)
    if attempt["role"] == "worker" and validation["changed_files"]:
        _draft(store, updated, attempt, workspace)
        if attempt["stage"] == "P6":
            updated["pending_rechecks"] = affected_nodes(updated, [*updated["pending_rechecks"],
                *(Path(name).stem for name in validation["changed_files"] if name.endswith(".yaml"))])
    if fresh_context:
        handoff_session(updated, attempt, result, "恢复磁盘上的未完成CLI调用；保留失败计数和预算，从本阶段页与安全checkpoint建立新上下文。")
    _release(updated, attempt)
    scope, owner = scope_for(updated, attempt), owner_for(updated, attempt)
    key = "receipt_repairs" if kind == "receipt_repair" else "cli_failures"
    scope[key] += 1
    if kind == "receipt_repair":
        scope["invalid_receipts"] += 1
    owner["reason"] = {"code": kind, "summary": reason, "checkpoint": call_view(updated, attempt)["checkpoint"]}
    if kind == "cli_retry" and result.get("outcome") == "cli_error" and result.get("exit_code") == 75:
        # EX_TEMPFAIL is an explicit process-level temporary-failure signal.
        # Missing responses, timeouts and generic nonzero exits are not one.
        recorded = Path(attempt["directory"]) / "result.json"
        owner["reason"].update(classification="transient_execution", attempt_id=attempt["attempt_id"],
                               scope=scope_key(attempt["stage"], attempt["node"]),
                               result_sha256=digest(read_json(recorded) if recorded.is_file() else result))
    owner["feedback"] = [*owner["feedback"], owner["reason"]]
    limit = updated["config"]["retry_limit"]
    if scope[key] > limit:
        owner["status"] = "invalid_receipt" if kind == "receipt_repair" else "cli_error"
        scope["status"] = owner["status"]
        owner["reason"]["code"] += "_budget_exhausted"
        owner["reason"].pop("classification", None)
    elif owner["reason"].get("classification") == "transient_execution":
        # A positively identified temporary failure becomes an observable
        # stopped scope. Only an explicit, budgeted monitor authorization may
        # retry it; independent ready scopes continue in the same Runner.
        owner["status"] = scope["status"] = "cli_error"
    else:
        owner["status"] = "draft" if attempt["stage"] == "P5" else "running"
    write_json(Path(attempt["directory"]) / "repair.json", {**owner["reason"], "validation": validation})
    settle(updated)
    store.save(updated)
    return updated


def _retire_stale(store: Store, state: dict, attempt: dict, result: dict | None, reason: str) -> dict:
    confirmed = bool(result and result.get("termination", {}).get("confirmed"))
    record = {"attempt_id": attempt["attempt_id"], "node": attempt["node"], "reason": reason,
              "generation": attempt["generation"], "termination_confirmed": confirmed}
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
        if error:
            raise error
        if not result or not result.get("termination", {}).get("confirmed"):
            raise ExtractError("agent termination is unconfirmed; this attempt remains fenced")
        outcome = result.get("outcome")
        if outcome == "receipt_error":
            raise ReceiptFormatError("Codex CLI completed but its response was not valid JSON")
        if outcome in {"cli_error", "timeout"} and attempt["mode"] == "normal":
            return _recover_call(store, state, lock, attempt, result, kind="cli_retry",
                                 reason=f"CLI {outcome}；合法草稿与原计数已保留，按剩余预算重试。",
                                 fresh_context=recovering)
        if outcome != "completed":
            raise ExtractError("Codex CLI call did not complete: " + str(outcome))
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
                 result: dict | None, termination: dict, reason: str) -> dict:
    """Save only a valid draft; an interrupted decision cannot approve anything."""
    if not termination["confirmed"]:
        owner, scope = owner_for(state, attempt), _scope_at(state, attempt)
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
        handoff_session(state, attempt, result, reason)
        return _retire_stale(store, state, attempt, {"termination": termination}, str(exc))
    validation, discarded = None, None
    if attempt["role"] == "worker":
        try:
            before, _ = store.verify_checkpoint(state, attempt["base_checkpoint"])
            workspace = Path(attempt["workspace"])
            validation = _artifact_gate(state, lock, attempt, before, workspace, "continue")
            if validation["changed_files"]:
                _draft(store, state, attempt, workspace)
                if attempt["stage"] == "P6":
                    state["pending_rechecks"] = affected_nodes(state, [*state["pending_rechecks"],
                        *(Path(name).stem for name in validation["changed_files"] if name.endswith(".yaml"))])
        except (ExtractError, OSError) as exc:
            discarded = str(exc)
    _release(state, attempt)
    handoff_session(state, attempt, result, reason)
    record = {"attempt_id": attempt["attempt_id"], "reason": reason, "termination": termination,
              "validation": validation, "discarded_draft": discarded,
              "checkpoint": call_view(state, attempt)["checkpoint"]}
    state["recovered_attempts"][attempt["attempt_id"]] = record
    write_json(Path(attempt["directory"]) / "recovery.json", record)
    if attempt["mode"] in {"adjudicate", "signoff"}:
        return _stop_once(store, state, attempt, {"code": "interrupted_single_decision", "summary": reason,
                          "recovery_condition": "本范围的一次裁决/签收已经使用；保留原决定和预算，不能重新授予。"})
    owner = owner_for(state, attempt)
    owner["status"] = "draft" if attempt["stage"] == "P5" else "running"
    _scope_at(state, attempt)["status"] = "reviewing" if attempt["role"] == "reviewer" else "draft"
    owner["reason"] = None
    settle(state)
    store.save(state)
    return state


def _reconcile(store: Store, state: dict, lock: dict, *, terminate_only=False) -> tuple[dict, list]:
    """Reconcile only state-registered fences and unfinished accounting commits."""
    pending = []
    unsettled = state.setdefault("unsettled_attempts", {})
    registered = dict.fromkeys([*state["active_attempts"].values(), *state["unconfirmed_attempts"]])
    if not terminate_only:
        registered.update(dict.fromkeys(unsettled))
    for attempt_id in registered:
        directory = store.run_dir(state["run_id"]) / "attempts" / attempt_id
        attempt = read_json(directory / "attempt.json")
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
            termination = confirm_termination(attempt, result, launch_state=unsettled.get(attempt_id))
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
            termination = confirm_termination(attempt, None, launch_state=unsettled.get(attempt_id))
            reason = {"code": "unreadable_result", "summary": str(exc), "termination": termination,
                      "recovery_condition": "先保留原调用及坏result；修复结果记录后按原预算resume对账。"}
            owner = owner_for(state, attempt)
            owner["reason"] = reason
            if active:
                owner["status"] = "invalid_receipt" if termination["confirmed"] else "termination_unconfirmed"
                _scope_at(state, attempt)["status"] = owner["status"]
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
        termination = confirm_termination(attempt, result, launch_state=unsettled.get(attempt_id))
        if not termination["confirmed"]:
            state = _interrupted(store, state, lock, attempt, result, termination, "恢复未能确认旧调用终止；仅该范围保留fence。")
            continue
        state["unconfirmed_attempts"].pop(attempt_id, None)
        owner = owner_for(state, attempt)
        if owner["status"] == "termination_unconfirmed":
            owner["status"] = "draft" if attempt["stage"] == "P5" else "running"
        if result and result.get("outcome") != "cancelled":
            result = {**result, "termination": termination}
            if (result.get("outcome") == "termination_unconfirmed" and result.get("exit_code") == 0 and
                    "turn.completed" in result.get("event_types", []) and result.get("response") and
                    result.get("session_id") and not result.get("parse_error_lines")):
                result["outcome"] = "completed"
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
        state["status"] = control["resume_status"] or "running"
    if state["status"] == "queued":
        state["status"] = "running"
    control["action"], control["resume_status"] = None, None
    control["hold_id"] = None
    # Hold/queued/retry entries only restore their stopped control boundary.
    # Explicit manual resume separately authorizes ordinary failed scopes.
    if state["stage"] == "P5":
        settle(state)


def _resume_failed_scopes(state: dict) -> None:
    """Manual retry keeps counters and hard decisions; budget stops need available credit."""
    hard = {"external_confirmation", "adjudication_already_used", "interrupted_single_decision",
            "reopen_already_used"}
    calls = [{"stage": "P5", "node": entry["id"], **{key: state["nodes"][entry["id"]][key] for key in ("role", "mode")}}
             for entry in state.get("plan") or [] if entry["id"] != "root"] if state["stage"] == "P5" else [state]
    for call in calls:
        owner, scope = owner_for(state, call), scope_for(state, call)
        reason = owner.get("reason") or {}
        if reason.get("code") in BUDGET_STOPS:
            if (any(item.endswith(":single_decision_used") for item in reason.get("scopes", [])) or
                    _budget(call_view(state, call))[1]):
                continue
        if (owner["status"] in NODE_STOPS and owner.get("mode", "normal") == "normal" and
                reason.get("code") not in hard):
            owner["status"] = "draft" if owner is not state else "running"
            scope["status"] = "reviewing" if owner.get("role") == "reviewer" else "draft"
            if reason and (not owner["feedback"] or owner["feedback"][-1] != reason):
                owner["feedback"].append(reason)
            owner["reason"] = None
    if state["stage"] == "P5":
        settle(state)


def _authority_failure(store: Store, guard: _StateGuard, exc: Exception) -> dict:
    # Restore only Runner-authored bytes. Keep every active fence and unfinished
    # accounting record, even when the current event already confirms its exit.
    state = json.loads(guard.state_bytes)
    state["status"] = "invalid_receipt"
    state["reason"] = {"code": "authority_changed", "summary": str(exc), "path": "state.json",
                       "expected_sha256": sha256(guard.state_bytes),
                       "recovery_condition": "核对state外改原因及最后可信checkpoint；resume保留原预算并确认旧调用终止。"}
    token = STATE_GUARD.set(None)
    try:
        store.save(state)
    finally:
        STATE_GUARD.reset(token)
    return state


def run(store: Store, run_id: str, *, transport=None, resume=False, queued_only=False,
        launch_id: str | None = None) -> dict:
    with store.writer(run_id):
        state = store.load(run_id)
        guard, events, running, deferred = _StateGuard(store, state), queue.Queue(), {}, []
        write_token = STATE_GUARD.set(guard)
        cancel_event = threading.Event()

        def invoke(attempt, prompt, schema, effort, timeout):
            result, error, started = None, None, time.monotonic()
            try:
                extra = {"cancel_event": cancel_event} if accepts_cancel else {}
                result = cli.run(attempt=attempt, prompt=prompt, schema=schema, effort=effort, timeout=timeout, **extra)
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
            verify_executor(lock)
            consume_controls(store, state)
            initial_ledger = store.read_supervisor()
            initial_entry = initial_ledger["starts"].get(run_id, {}).get("launch", {})
            retry_evidence = None
            if launch_id and initial_entry.get("mode") == "retry":
                from .monitor import restart_evidence
                retry = initial_ledger["retries"].get(initial_entry.get("retry_id"))
                if not retry:
                    raise ExtractError("automatic restart has no failure-bound authorization")
                retry_evidence = restart_evidence(store, state, retry["scope"])
                if any(retry_evidence.get(key) != retry.get(key) for key in ("failure_id", "run_id", "scope", "attempt_id", "state_revision")):
                    raise ExtractError("automatic restart target changed after authorization")
            with store.supervisor() as ledger:
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
                    if retry_evidence:
                        request = ledger["retries"][entry["retry_id"]]
                        if (not resume or request["status"] != "authorized" or
                                request["control_revision"] != ledger["controls"].get(run_id, {}).get("revision", 0)):
                            raise ExtractError("automatic restart was consumed or superseded by manual control")
                        request.update(status="consumed", consumed_at=time.time())
                        stage, _, node = request["scope"].partition(":")
                        target = {"stage": stage, "node": node or "root"}
                        owner = owner_for(state, target)
                        owner["status"] = "draft" if stage == "P5" else "running"
                        scope_for(state, target)["status"] = "reviewing" if owner.get("role") == "reviewer" else "draft"
                        if stage == "P5":
                            state["status"] = "running"
                            settle(state)
                elif ledger["active_hold"]:
                    store.hold_member(ledger, state)
                if run_id in ledger["continuations"] and not state["control"]["action"]:
                    raise ExtractError("this run has a reserved successor")
            if state["status"] == "queued" and not resume and not state["control"]["action"]:
                return state
            if resume:
                if launch_id:
                    extend_budget(state, entry.get("budget_addition", {}), identity=launch_id, origin="resume")
                _resume_stopped(state)
                if not launch_id or entry.get("mode") == "resume":
                    _resume_failed_scopes(state)
                store.save(state)
            consume_controls(store, state)
            from .monitor import _identity
            state["runner"] = {"identity": _identity(), "observed_at": time.time()}
            store.save(state)
            if ((state.get("reason") or {}).get("code") == "authority_changed" and
                    state["control"]["action"] == "cancel"):
                reason = deepcopy(state["reason"])
                state, _ = _reconcile(store, state, lock, terminate_only=True)
                state["status"], state["reason"] = "cancelled", reason
                store.save(state)
                return state
            held_status = state["status"] if state["status"] in {"paused", "cancelled"} else None
            state, pending = _reconcile(store, state, lock)
            if held_status and not resume:
                state["status"] = "cancelled" if state["control"]["action"] == "cancel" else held_status
                store.save(state)
                return state
            cli = transport or CodexCLI(state["config"]["codex_binary"], state["config"]["model"])
            accepts_cancel = "cancel_event" in inspect.signature(cli.run).parameters
            while True:
                consume_controls(store, state)
                if state["control"]["action"] == "cancel":
                    cancel_event.set()
                # Every ready sibling is launched before waiting. Control is
                # checked at each call boundary, including between siblings.
                for call in next_calls(state) if not state["control"]["action"] else []:
                    with store.supervisor() as ledger:
                        if ledger["active_hold"]:
                            store.hold_member(ledger, state)
                    consume_controls(store, state)
                    if state["control"]["action"]:
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
                    timeout, exhausted = _budget(view)
                    if exhausted:
                        owner, scope = owner_for(state, call), scope_for(state, call)
                        owner["status"] = scope["status"] = "blocked" if call["mode"] != "normal" else "budget_exhausted"
                        owner["reason"] = {"code": "budget_exhausted", "summary": exhausted, "checkpoint": view["checkpoint"],
                                           "recovery_condition": BUDGET_RECOVERY.format(run_id=state["run_id"])}
                        settle(state)
                        store.save(state)
                        continue
                    guard.verify(state)
                    attempt = store.prepare_attempt(state, call["session_id"], call=call, timeout_seconds=timeout)
                    prompt = build_prompt(state, lock, attempt)
                    schema = proposal_schema(attempt, source_files=lock["source"]["files"])
                    thread = threading.Thread(target=invoke, args=(attempt, prompt, schema,
                        state["config"][f"{attempt['role']}_effort"], timeout), daemon=True)
                    with store.supervisor() as ledger:
                        denied = bool(ledger["active_hold"] or run_id in ledger["continuations"])
                        if ledger["active_hold"]:
                            store.hold_member(ledger, state)
                        if not denied:
                            state["unsettled_attempts"][attempt["attempt_id"]] = "dispatched"
                            store.save(state)
                            guard.verify(state)
                            running[attempt["attempt_id"]] = thread
                            thread.start()
                    if denied:
                        _release(state, attempt)
                        state["unsettled_attempts"].pop(attempt["attempt_id"], None)
                        write_json(Path(attempt["directory"]) / "termination.json", {
                            "confirmed": True, "actions": [], "survivors": [], "reason": "control gate denied dispatch"})
                        write_json(Path(attempt["directory"]) / "finished.json", {"elapsed_seconds": 0.0, "status": "not_dispatched"})
                        consume_controls(store, state)
                        store.save(state)
                        break
                if deferred and (not state.get("reopening") or state["status"] != "running"):
                    event = deferred.pop(0)
                elif pending:
                    event = pending.pop(0)
                elif running:
                    try:
                        event = events.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    running.pop(event[0]["attempt_id"]).join()
                else:
                    break
                attempt, result, error, elapsed, recovering = event
                guard.verify(state)
                if attempt["stage"] == "P5" and state.get("reopening") and state["status"] == "running" and not error:
                    deferred.append(event)
                    continue
                account(state, attempt, elapsed, result=result)
                store.save(state)
                processing_path = Path(attempt["directory"]) / "processing.json"
                processing = read_json(processing_path) if processing_path.is_file() else {"elapsed_seconds": 0.0}
                processing["started_at"] = time.time()
                write_json(processing_path, processing)
                started = time.monotonic()
                if result and result.get("outcome") == "cancelled" and not error:
                    state = _interrupted(store, state, lock, attempt, result, result["termination"], "cancel请求终止了本次调用；已保存可验证草稿。")
                else:
                    state = _receive(store, state, lock, attempt, result, error, recovering=recovering)
                processing["elapsed_seconds"] += time.monotonic() - started
                processing["started_at"] = None
                processing["completed_at"] = time.time()
                write_json(processing_path, processing)
                elapsed += processing["elapsed_seconds"]
                write_json(Path(attempt["directory"]) / "finished.json", {"elapsed_seconds": elapsed, "status": state["status"]})
                state["unsettled_attempts"].pop(attempt["attempt_id"], None)
                account(state, attempt, elapsed, result=result, processing=processing, final=True)
                store.save(state)
            with store.supervisor() as ledger:
                if ledger["active_hold"]:
                    store.hold_member(ledger, state)
            consume_controls(store, state)
            guard.verify(state)
            action = state["control"]["action"]
            if action:
                state["control"]["resume_status"] = state["status"]
                state["status"] = "cancelled" if action == "cancel" else "paused"
                state["control"]["stopped_at"] = time.time()
                store.save(state)
            elif state["status"] == "ready":
                from .publication import publish
                state = publish(store, state, Path(state["output"]))
            return state
        except _AuthorityChanged as exc:
            cancel_event.set()
            return _authority_failure(store, guard, exc)
        finally:
            cancel_event.set()
            for thread in running.values():
                thread.join(timeout=3)
            try:
                if launch_id:
                    with store.supervisor() as ledger:
                        entry = ledger["starts"].get(run_id, {}).get("launch", {})
                        if entry.get("id") == launch_id:
                            entry.update(status="termination_unknown" if any(thread.is_alive() for thread in running.values()) else "finished",
                                         finished_at=time.time())
            finally:
                STATE_GUARD.reset(write_token)


def _reject(store: Store, state: dict, attempt: dict, result: dict | None, exc: Exception) -> dict:
    confirmed = result is not None and result.get("termination", {}).get("confirmed")
    owner = owner_for(state, attempt)
    owner["status"] = "invalid_receipt" if confirmed else "termination_unconfirmed"
    _scope_at(state, attempt)["invalid_receipts"] += 1
    _scope_at(state, attempt)["status"] = owner["status"]
    owner["reason"] = {"code": "proposal_rejected", "summary": str(exc), "attempt_id": attempt["attempt_id"],
                       "recovery_condition": "核对拒绝原因与固定输入；确认旧进程终止后按原计数和预算恢复。"}
    if confirmed and isinstance(exc, UpstreamRequired):
        owner["status"] = _scope_at(state, attempt)["status"] = "blocked"
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
            verify_executor(read_json(store.run_dir(run_id) / "inputs.lock.json"))
            return publish(store, state, output, export=True)["exports"][-1]
        except _AuthorityChanged as exc:
            _authority_failure(store, guard, exc)
            raise
        finally:
            STATE_GUARD.reset(token)
