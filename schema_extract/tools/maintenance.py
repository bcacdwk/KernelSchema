"""Finite semantic maintenance; ordinary progress and all scope budgets stay intact."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import time
from uuid import uuid4

from .common import ExtractError, StaleResult, copy_artifacts, manifest


FINISHED = {"done", "waiting", "removed"}


def active(state: dict) -> bool:
    return (state.get("maintenance") or {}).get("status") in {"active", "blocked"}


def task_for(state: dict, call: dict) -> dict:
    task = (state.get("maintenance") or {}).get("tasks", {}).get(call.get("maintenance_task_id"))
    if not task or call.get("purpose") != "maintenance":
        raise StaleResult("maintenance invocation has no current task")
    return task


def _scope(state: dict, task: dict) -> dict:
    return state["nodes"][task["node"]]["stage"] if task["stage"] == "P5" else state["stages"][task["stage"]]


def _complete_candidate(state: dict, task: dict) -> bool:
    if "complete_candidate" in task:
        return bool(task["complete_candidate"])
    if "complete_candidate" in task.get("basis", {}):
        return bool(task["basis"]["complete_candidate"])
    scope = _scope(state, task)
    if scope.get("approval") or scope.get("last_approval"):
        return True
    cursor = state["maintenance"]["normal_cursor"]
    owner = state["nodes"][task["node"]] if task["stage"] == "P5" else cursor
    return (scope.get("status") in {"reviewing", "revision", "adjudicating", "signoff", "approved"}
            or (owner.get("stage", task["stage"]) == task["stage"] and owner.get("role") == "reviewer"))


def _initialize(state: dict) -> None:
    maintenance = state["maintenance"]
    for key, task in maintenance["tasks"].items():
        task.setdefault("id", key)
        task.setdefault("task_id", key)
        task["purpose"] = "maintenance"
        task.setdefault("step", "patch" if task["action"] in {"patch", "reextract"} else "review")
        task.setdefault("role", "worker" if task["step"] == "patch" else "reviewer")
        task.setdefault("mode", "normal")
        if not task.get("feedback"):
            task["feedback"] = [item for item in [task.get("reason"), *task.get("issues", [])] if item]
        if isinstance(task.get("reason"), str):
            task["plan_reason"], task["reason"] = task["reason"], None
        task.setdefault("history", [])
        task.setdefault("complete_candidate", _complete_candidate(state, task))
        if task.get("basis", {}).get("retired"):
            task["status"] = "removed"
        if task.get("status") in FINISHED:
            continue
        if task["stage"] == "P6" or not task["complete_candidate"]:
            # A partial draft is still ordinary authoring. Its new requirements
            # accompany that author; a maintenance approval cannot complete it.
            task["status"] = "waiting"
            task["waiting_reason"] = "normal_final_review" if task["stage"] == "P6" else "unfinished_authoring"
        elif task["action"] == "keep":
            from .revisions import approval_is_current
            approval = _scope(state, task).get("approval")
            if approval and approval_is_current(state, approval):
                task["status"] = "done"
                if key not in maintenance["completed"]:
                    maintenance["completed"].append(key)
            else:
                task.update(action="review", step="review", role="reviewer", status="pending")


def _dependencies_ready(state: dict, task: dict) -> bool:
    from .revisions import approval_is_current
    from .stages import effective_nodes
    for stage in ("P1", "P2", "P3", "P4"):
        if int(stage[1:]) >= int(task["stage"][1:]):
            break
        scope = state["stages"][stage]
        if scope["status"] != "approved" or not approval_is_current(state, scope.get("approval")):
            return False
    if task["stage"] == "P5":
        node = state["nodes"].get(task["node"])
        return bool(node and node.get("parent") in effective_nodes(state))
    return True


def next_calls(state: dict) -> list[dict]:
    from .stages import STAGES, session_group
    if state.get("status") != "running" or state.get("active_attempts"):
        return []
    tasks = state["maintenance"]["tasks"].values()
    for task in sorted(tasks, key=lambda item: (int(item["stage"][1:]), item["scope"])):
        if task.get("status") != "pending" or not _dependencies_ready(state, task):
            continue
        call = {"stage": task["stage"], "node": task["node"], "role": task["role"], "mode": "normal",
                "purpose": "maintenance", "maintenance_task_id": task["task_id"], "maintenance_step": task["step"]}
        group = session_group(state, call)
        return [{**call, "page": STAGES[task["stage"]]["page"], "session_group": group,
                 "session_id": state["sessions"].get(group, {}).get(task["role"])}]
    return []


def _normal_feedback(state: dict, task: dict) -> None:
    entry = {"code": "semantic_revision_continuation", "maintenance_task_id": task["task_id"],
             "action": task["action"], "summary": task.get("plan_reason", task.get("reason")), "issues": task.get("issues", []),
             "fields": task.get("fields", [])}
    owner = state["nodes"][task["node"]] if task["stage"] == "P5" else state
    if entry not in owner.setdefault("feedback", []):
        owner["feedback"].append(entry)


def _resume_normal(state: dict) -> None:
    from .revisions import approval_is_current
    from .stages import final_eligible, settle as settle_stages
    maintenance = state["maintenance"]
    cursor = maintenance["normal_cursor"]
    deferred = any(task["status"] == "waiting" and task.get("complete_candidate") and task["stage"] != "P6"
                   for task in maintenance["tasks"].values())
    maintenance["status"] = "waiting" if deferred else "complete"
    if not deferred:
        maintenance["completed_at"] = time.time()
    for key in ("stage", "node", "role", "mode", "feedback", "reason"):
        if key in cursor:
            state[key] = deepcopy(cursor[key])
    state["status"] = "running"
    state["reason"] = None
    state["checkpoint"] = maintenance.get("working_checkpoint") or state["checkpoint"]
    for stage in ("P1", "P2", "P3", "P4"):
        scope = state["stages"][stage]
        if scope["status"] == "approved" and approval_is_current(state, scope.get("approval")):
            continue
        same = cursor.get("stage") == stage
        role = cursor.get("role") if same else ("reviewer" if scope["status"] in {"reviewing", "revision", "adjudicating"} else "worker")
        state.update(stage=stage, node="root", role=role or "worker", mode=cursor.get("mode", "normal") if same else "normal")
        break
    else:
        if final_eligible(state):
            # P6 remains a current-context final review. Preserve an unfinished
            # final patch and every consumed one-shot decision when resuming it.
            same = cursor.get("stage") == "P6"
            state.update(stage="P6", node="root", role=(cursor.get("role") or "reviewer") if same else "reviewer",
                         mode=cursor.get("mode", "normal") if same else "normal")
            state["stages"]["P5"]["status"] = "approved" if len(state["plan"]) > 1 else "skipped"
            state["stages"]["P6"]["status"] = "reviewing" if state["role"] == "reviewer" else "draft"
            if state["role"] is None:
                state["role"] = "reviewer"
        else:
            state.update(stage="P5", node="root", role=None, mode="normal")
            settle_stages(state)
    for task in maintenance["tasks"].values():
        if task["status"] == "waiting" and (task["stage"] == "P5" or task["stage"] == state["stage"]):
            _normal_feedback(state, task)


def resume_deferred(state: dict) -> None:
    """Ordinary authoring can unlock a saved maintenance candidate later."""
    from .revisions import approval_is_current
    maintenance = state.get("maintenance") or {}
    if maintenance.get("status") != "waiting":
        return
    ready = False
    for task in maintenance["tasks"].values():
        if task["status"] != "waiting" or task["stage"] == "P6":
            continue
        scope = _scope(state, task)
        if scope["status"] == "approved" and approval_is_current(state, scope.get("approval")):
            task["status"] = "done"
        elif task.get("complete_candidate") and _dependencies_ready(state, task):
            task.update(status="pending", prepared=False)
            ready = True
    if ready:
        maintenance["normal_cursor"] = {key: deepcopy(state.get(key)) for key in
                                        ("stage", "node", "role", "mode", "status", "checkpoint", "feedback", "reason")}
        maintenance["working_checkpoint"] = state["checkpoint"]
        maintenance["status"] = "active"
        settle(state)
    elif not any(task["status"] == "waiting" and task.get("complete_candidate") and task["stage"] != "P6"
                 for task in maintenance["tasks"].values()):
        maintenance["status"], maintenance["completed_at"] = "complete", time.time()


def settle(state: dict) -> None:
    from .revisions import approval_is_current
    _initialize(state)
    if state.get("active_attempts"):
        return
    maintenance = state["maintenance"]
    tasks = list(maintenance["tasks"].values())
    stopped = [task for task in tasks if task["status"] not in FINISHED | {"pending"}]
    if stopped:
        maintenance["status"] = "blocked"
        state["status"] = "blocked"
        state["reason"] = {"code": "maintenance_blocked", "summary": "维护已保留当前稿；以下有界动作需要明确修复或预算追加。",
                           "tasks": {task["task_id"]: task.get("reason") for task in stopped}}
        return
    # A downstream complete candidate cannot be reviewed before an unfinished
    # ordinary prerequisite. Defer it together with its original progress.
    unfinished = [task for task in tasks if task["status"] == "waiting" or
                  not task.get("complete_candidate") and not approval_is_current(state, _scope(state, task).get("approval"))]
    waiting_stages = {task["stage"] for task in unfinished if task["stage"] != "P6"}
    waiting_nodes = {task["node"] for task in unfinished if task["stage"] == "P5"}
    for task in tasks:
        if task["status"] != "pending":
            continue
        if any(int(stage[1:]) < int(task["stage"][1:]) for stage in waiting_stages if stage != "P5"):
            task.update(status="waiting", waiting_reason="unfinished_prerequisite")
        elif task["stage"] == "P5":
            parent = state["nodes"][task["node"]].get("parent")
            while parent:
                if parent in waiting_nodes:
                    task.update(status="waiting", waiting_reason="unfinished_parent")
                    break
                parent = state["nodes"][parent].get("parent")
    if all(task["status"] in FINISHED for task in tasks):
        _resume_normal(state)
        return
    maintenance["status"] = "active"
    if state["status"] not in {"paused", "cancelled"}:
        state["status"], state["reason"] = "running", None
        if not next_calls(state):
            maintenance["status"], state["status"] = "blocked", "blocked"
            state["reason"] = {"code": "maintenance_dependencies", "summary": "维护候选缺少当前适用的上游批准；已保留所有稿件。",
                               "scopes": [task["scope"] for task in tasks if task["status"] == "pending"]}


def validate_attempt(state: dict, attempt: dict) -> None:
    task = task_for(state, attempt)
    expected = {"stage": task["stage"], "node": task["node"], "role": task["role"], "mode": "normal",
                "maintenance_step": task["step"], "base_checkpoint": task["draft_checkpoint"]}
    if not active(state) or task["status"] != "pending" or any(attempt.get(key) != value for key, value in expected.items()):
        raise StaleResult("maintenance task, step or draft changed while the invocation ran")
    generation = state["nodes"][task["node"]]["generation"] if task["stage"] == "P5" else _scope(state, task)["generation"]
    if attempt.get("generation") != generation:
        raise StaleResult("maintenance scope generation changed")


def _copy_local(source: Path, destination: Path) -> None:
    destination.mkdir()
    for name in ("root.yaml", "dispatch.md"):
        if (source / name).is_file():
            copy_artifacts(source / name, destination / name)


def prepare_workspace(store, state: dict, call: dict) -> None:
    """Refresh only read-only inputs, always keeping a node's own authored file."""
    maintaining = call.get("purpose") == "maintenance"
    task = task_for(state, call) if maintaining else None
    if maintaining:
        maintenance = state["maintenance"]
        if not maintenance.get("working_checkpoint"):
            working = state["merged_checkpoint"] or state["checkpoint"]
            current_id = maintenance["normal_cursor"].get("checkpoint", state["checkpoint"])
            current, _ = store.verify_checkpoint(state, current_id)
            if state["merged_checkpoint"] and maintenance["normal_cursor"].get("stage") in {"P1", "P2", "P3", "P4"}:
                temporary = store.run_dir(state["run_id"]) / (".maintaining-" + uuid4().hex)
                source, _ = store.verify_checkpoint(state, working)
                try:
                    copy_artifacts(source, temporary)
                    for path in temporary.rglob("*"):
                        path.chmod(0o755 if path.is_dir() else 0o644)
                    temporary.chmod(0o755)
                    for name in ("root.yaml", "dispatch.md"):
                        if (current / name).is_file():
                            copy_artifacts(current / name, temporary / name)
                    working = store.checkpoint(state, temporary, attempt_id=None, stage=call["stage"], node=call["node"], parent=working, call=call)
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)
            maintenance["working_checkpoint"] = working
        if call["stage"] == "P5":
            store.start_node(state, call["node"], call=call)
            task["draft_checkpoint"] = state["nodes"][call["node"]]["draft_checkpoint"]
            return
        if task.get("prepared"):
            return
        source_id = maintenance["working_checkpoint"]
    elif call["stage"] in {"P1", "P2", "P3", "P4"}:
        source_id = state["checkpoint"]
    else:
        return
    source, info = store.verify_checkpoint(state, source_id)
    if not maintaining and set(info["files"]) <= {"root.yaml", "dispatch.md"}:
        return
    temporary = store.run_dir(state["run_id"]) / (".maintaining-" + uuid4().hex)
    try:
        _copy_local(source, temporary)
        checkpoint = store.checkpoint(state, temporary, attempt_id=None, stage=call["stage"], node=call["node"], parent=source_id, call=call)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    if maintaining:
        task["draft_checkpoint"], task["prepared"] = checkpoint, True
    else:
        state["checkpoint"] = checkpoint


def merge_upstream(store, state: dict, attempt: dict, workspace: Path, destination: Path) -> Path:
    """A local P1-P4 checkpoint overlays the complete tree; it never replaces it."""
    base = ((state.get("maintenance") or {}).get("working_checkpoint")
            if attempt.get("purpose") == "maintenance" else None) or state["merged_checkpoint"]
    if base:
        source, _ = store.verify_checkpoint(state, base)
        copy_artifacts(source, destination)
    else:
        destination.mkdir()
    for path in destination.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    destination.chmod(0o755)
    for name in ("root.yaml", "dispatch.md"):
        if (workspace / name).is_file():
            copy_artifacts(workspace / name, destination / name)
    return destination


def block(state: dict, call: dict, reason: dict, *, status="blocked") -> None:
    task = task_for(state, call)
    task["status"], task["reason"] = status, deepcopy(reason)
    task.setdefault("feedback", []).append(deepcopy(reason))


def _patch_fields(task: dict, receipt: dict, state: dict) -> list[dict]:
    from .stages import stage_files
    allowed = set(stage_files(task["stage"], node_entry=state["nodes"].get(task["node"]),
                              revision_scope=state.get("revision_scope")))
    requested = receipt.get("revision_scope") or task.get("fields", [])
    ceiling = {item["path"]: item["fields"] for item in task.get("fields", [])}
    anchored = {anchor["path"] for anchor in receipt.get("artifact_evidence", [])}
    result, seen = [], set()
    for item in requested:
        path, fields = item["path"], item["fields"]
        if path not in allowed & anchored or path in seen or not fields:
            raise ExtractError("maintenance revision scope needs unique stage files and current artifact evidence")
        seen.add(path)
        if path == "dispatch.md":
            if fields != ["*"]:
                raise ExtractError("dispatch.md maintenance scope uses fields ['*']")
        elif any(not field or any(not part or part == "*" for part in field.split(".")) for field in fields):
            raise ExtractError("maintenance revision scope requires explicit YAML fields")
        if ceiling and (path not in ceiling or any(not any(limit == "*" or field == limit or field.startswith(limit + ".")
                                                         for limit in ceiling[path]) for field in fields)):
            raise ExtractError("reviewer maintenance scope exceeds the sealed plan's field boundary")
        result.append({"path": path, "fields": list(fields)})
    return result


def _reconsider_dependents(state: dict, task: dict, validation: dict) -> None:
    from .stages import affected_nodes
    changed = validation.get("maintenance_changed_files", [])
    if not changed and not validation.get("invalidated_nodes"):
        return
    descendants = set(affected_nodes(state, [task["node"]])) if task["stage"] == "P5" else set()
    candidates = set(validation.get("invalidated_nodes", []))
    planned = {entry["id"] for entry in state.get("plan") or []}
    for other in state["maintenance"]["tasks"].values():
        if other is task or other["stage"] == "P6":
            continue
        if other["stage"] == "P5" and other["node"] not in planned:
            other["status"] = "removed"
            continue
        affected = ((task["stage"] in {"P1", "P2", "P3"} and int(other["stage"][1:]) > int(task["stage"][1:]))
                    or (other["stage"] == "P5" and other["node"] in candidates | descendants))
        if not affected or not other.get("complete_candidate"):
            continue
        if other["status"] in {"done", "waiting"}:
            if other.get("history"):
                # The finite task already ran. Changed dependencies now need a
                # new concrete plan, rather than buying another review loop.
                block(state, {"purpose": "maintenance", "maintenance_task_id": other["task_id"]},
                      {"code": "maintenance_dependency_changed_after_review", "summary": "依赖在本次有限复核后改变；保留稿件并重新评估具体范围。"})
                continue
            other.update(action="review", step="review", role="reviewer", status="pending", prepared=False)
            if other["task_id"] in state["maintenance"]["completed"]:
                state["maintenance"]["completed"].remove(other["task_id"])
            _scope(state, other)["status"] = "reviewing"
            if other["stage"] == "P5":
                state["nodes"][other["node"]]["status"] = "reviewing"


def advance(state: dict, receipt: dict, checkpoint_id: str, receipt_hash: str, validation: dict) -> dict:
    from .stages import _approve, scope_key, plan_signature
    result = deepcopy(state)
    identity = receipt["identity"]
    attempt_id = identity["attempt_id"]
    key = scope_key(identity["stage"], identity["node"])
    if attempt_id in result["accepted_receipts"] or result["active_attempts"].get(key) != attempt_id:
        raise StaleResult("maintenance receipt is duplicate or no longer active")
    task = task_for(result, identity)
    decision, step = receipt["decision"], task["step"]
    result["accepted_receipts"][attempt_id] = receipt_hash
    result["active_attempts"].pop(key)
    task["draft_checkpoint"] = checkpoint_id
    task["history"].append({"attempt_id": attempt_id, "step": step, "decision": decision,
                            "checkpoint": checkpoint_id, "receipt_sha256": receipt_hash})
    result["recoverable_checkpoint"] = checkpoint_id
    if identity["stage"] == "P5":
        result["nodes"][identity["node"]]["draft_checkpoint"] = checkpoint_id
    if identity["role"] == "worker":
        _scope(result, task)["candidates"].append({"checkpoint": checkpoint_id, "plan": deepcopy(receipt.get("plan"))})
        if decision == "continue":
            # One bounded patch can require several ordinary authoring calls.
            # Keep its logical step and saved draft; normal call/time credit
            # still reserves the independent verification that follows.
            _scope(result, task)["continues"] += 1
            task.update(status="pending", reason=None, feedback=[receipt["summary"], *receipt["issues"]])
        elif decision == "complete":
            task.update(step="verify", role="reviewer", status="pending", reason=None,
                        feedback=[receipt["summary"], *receipt["issues"]])
            if identity["stage"] == "P4":
                result["plan_draft"] = deepcopy(receipt["plan"])
        else:
            block(result, identity, {"code": "maintenance_patch_incomplete", "summary": receipt["summary"],
                                     "issues": receipt["issues"], "checkpoint": checkpoint_id})
    elif decision == "approve":
        if not task["complete_candidate"]:
            raise ExtractError("maintenance cannot approve an unfinished ordinary draft")
        changed_plan = identity["stage"] == "P4" and plan_signature(result.get("plan")) != plan_signature(receipt["plan"])
        _approve(result, receipt, checkpoint_id, receipt_hash, validation)
        if changed_plan:
            result["plan_revision"] += 1
        task.update(status="done", reason=None)
        result["maintenance"]["completed"].append(task["task_id"])
        result["maintenance"]["working_checkpoint"] = checkpoint_id
        _reconsider_dependents(result, task, validation)
    elif decision == "revise" and step == "review":
        fields = _patch_fields(task, receipt, result)
        if fields:
            task.update(step="patch", role="worker", status="pending", fields=fields,
                        reason=None, feedback=receipt["issues"])
        else:
            block(result, identity, {"code": "maintenance_patch_scope_missing", "summary": "复核已定位问题；需要有产物锚点的明确文件/字段范围才能执行一次修补。",
                                     "issues": receipt["issues"], "checkpoint": checkpoint_id})
    else:
        block(result, identity, {"code": "maintenance_verify_failed" if step == "verify" else "maintenance_review_blocked",
                                 "summary": receipt["summary"], "issues": receipt["issues"], "checkpoint": checkpoint_id})
    settle(result)
    return result
