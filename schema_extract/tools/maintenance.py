"""Bounded maintenance of one frozen run, behind the existing single writer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import time
from uuid import uuid4

from .common import ExtractError, StaleResult, copy_artifacts, digest, manifest, read_json, write_json
from .stages import scope_for


PENDING = {"requested", "draining"}
FINISHED = {"done", "waiting", "removed"}


def active(state: dict) -> bool:
    return (state.get("maintenance") or {}).get("status") in {"active", "blocked"}


def _request(ledger: dict, run_id: str) -> dict | None:
    return ledger.get("maintenance_requests", {}).get(run_id)


def barrier(store, state: dict, ledger=None) -> bool:
    request = _request(store.read_supervisor() if ledger is None else ledger, state["run_id"])
    return bool(request and request["status"] in PENDING)


def publication_barrier(store, state: dict, ledger=None) -> bool:
    return barrier(store, state, ledger) or (state.get("maintenance") or {}).get("status") in {"active", "blocked", "waiting"}


def _validate_tasks(state: dict, tasks) -> list[dict]:
    from .stages import scope_key, stage_files
    if not isinstance(tasks, list) or not tasks or len(tasks) > 14:
        raise ExtractError("maintenance needs one to fourteen explicit current scope tasks")
    selected = {entry["id"] for entry in state.get("selected_designs") or []} | {"root"}
    allowed_fields = {
        "P1": ("META_DATA.Name", "META_DATA.Describe", "META_DATA.Source", "CONTRACT.Algorithm_requirement", "CONTRACT.Hardware_specification.Execution"),
        "P2": ("META_DATA.Source", "META_DATA.Dispatch"),
        "P3": ("DESIGN", "REALIZATION", "META_DATA.Dispatch.when"),
        "P4": (), "P5": ("DESIGN", "REALIZATION", "META_DATA.Source", "META_DATA.Dispatch.when"),
    }
    result, seen = [], set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"stage", "node", "action", "fields", "reason"}:
            raise ExtractError("maintenance task requires stage/node/action/fields/reason")
        stage, node = task["stage"], task["node"]
        if stage not in allowed_fields or node not in selected or (stage == "P5") != (node != "root"):
            raise ExtractError("maintenance targets a selected P1-P5 responsibility; P6 remains the normal final review")
        key = scope_key(stage, node)
        if key in seen or task["action"] not in {"review", "patch"} or not isinstance(task["reason"], str) or not task["reason"].strip():
            raise ExtractError("maintenance needs unique scopes, review/patch, and a concrete reason")
        seen.add(key)
        fields = task["fields"]
        if not isinstance(fields, list) or task["action"] == "patch" and not fields:
            raise ExtractError("maintenance patch requires explicit fields")
        paths = set(stage_files(stage, node_entry=state["nodes"].get(node)))
        cited = set()
        for entry in fields:
            if not isinstance(entry, dict) or set(entry) != {"path", "fields"} or entry["path"] not in paths or entry["path"] in cited:
                raise ExtractError("maintenance fields exceed the stage file boundary")
            cited.add(entry["path"])
            values = entry["fields"]
            if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values) or len(set(values)) != len(values):
                raise ExtractError("maintenance fields must be a nonempty unique list")
            if entry["path"] == "dispatch.md":
                if values != ["*"]:
                    raise ExtractError("dispatch.md uses fields ['*']")
            elif any(not any(value == prefix or value.startswith(prefix + ".") for prefix in allowed_fields[stage])
                     or any(not part or part == "*" for part in value.split(".")) for value in values):
                raise ExtractError("maintenance field is outside its defining stage")
        result.append({**deepcopy(task), "reason": task["reason"].strip()})
    return result


def submit_request(store, run_id: str, *, issue_id: str, tasks: list) -> dict:
    """Seal an intention in the supervisor ledger; never write the run state here."""
    if not isinstance(issue_id, str) or not issue_id.strip():
        raise ExtractError("maintenance requires a stable nonempty issue_id")
    state = store.load(run_id)
    store.context(state)
    tasks = _validate_tasks(state, tasks)
    body = {"run_id": run_id, "input_fingerprint": state["input_fingerprint"], "issue_id": issue_id.strip(), "tasks": tasks}
    with store.supervisor() as ledger:
        state = store.load(run_id)
        prior = _request(ledger, run_id)
        history = [*prior.get("history", []), {key: value for key, value in prior.items() if key != "history"}] if prior else []
        same = next((item for item in history if item["issue_id"] == body["issue_id"]), None)
        if same:
            if same["request_sha256"] != digest(body):
                raise ExtractError("maintenance issue_id is already bound to another concrete request")
            return {"run_id": run_id, "status": same["status"], "request": deepcopy(same), "reused": True, "delivered": False}
        if prior and prior["status"] in PENDING or active(state) or (state.get("maintenance") or {}).get("status") == "waiting":
            raise ExtractError("finish the pending current-run maintenance before submitting another request")
        if state["published"] or state.get("exports") or ledger.get("commits", {}).get(run_id):
            raise ExtractError("maintenance cannot change an already delivered run")
        if body["input_fingerprint"] != state["input_fingerprint"]:
            raise ExtractError("the frozen run changed while submitting maintenance")
        request = {**body, "id": str(uuid4()), "request_sha256": digest(body), "status": "requested",
                   "created_at": time.time(), "history": history}
        write_json(store.root / "maintenance_requests" / run_id / (request["id"] + ".json"),
                   {key: value for key, value in request.items() if key not in {"history", "status"}})
        ledger.setdefault("maintenance_requests", {})[run_id] = request
    return {"run_id": run_id, "status": "requested", "request": deepcopy(request), "reused": False, "delivered": False}


def consume_requests(store, state: dict) -> dict:
    if not barrier(store, state):
        return state
    with store.control_transaction() as ledger:
        request = _request(ledger, state["run_id"])
        if not request or request["status"] not in PENDING:
            return state
        body = {key: request[key] for key in ("run_id", "input_fingerprint", "issue_id", "tasks")}
        sealed = read_json(store.root / "maintenance_requests" / state["run_id"] / (request["id"] + ".json"))
        expected = {**body, **{key: request[key] for key in ("id", "request_sha256", "created_at")}}
        if sealed != expected or digest(body) != request["request_sha256"] or request["input_fingerprint"] != state["input_fingerprint"]:
            raise ExtractError("maintenance request differs from its sealed current-run identity")
        if (state.get("maintenance") or {}).get("request_id") == request["id"]:
            request.update(status="applied", applied_at=state["maintenance"]["started_at"])
            return state
        request.update(status="draining", observed_at=request.get("observed_at", time.time()))
        projection = {key: request[key] for key in ("id", "issue_id", "request_sha256", "status")}
    if state.get("maintenance_request") != projection:
        state["maintenance_request"] = projection
        store.save(state)
    return state


def mark_drained(store, state: dict) -> dict:
    """The writer also proves its local running/pending/deferred queues empty."""
    from .stages import invalidate_final, invalidate_node, invalidate_stage, scope_key
    if any(state.get(name) for name in ("active_attempts", "unconfirmed_attempts", "unsettled_attempts")):
        raise ExtractError("maintenance drain still has unsettled or unterminated calls")
    request = _request(store.read_supervisor(), state["run_id"])
    if not request or request["status"] not in PENDING:
        return state
    store.context(state)
    try:
        tasks = _validate_tasks(state, request["tasks"])
    except ExtractError as exc:
        with store.control_transaction() as ledger:
            current = _request(ledger, state["run_id"])
            if not current or current["id"] != request["id"]:
                raise ExtractError("maintenance request changed during drain") from exc
            current.update(status="superseded", reason=str(exc), completed_at=time.time())
        state["maintenance_request"] = {"id": request["id"], "issue_id": request["issue_id"],
                                        "request_sha256": request["request_sha256"], "status": "superseded"}
        state.update(status="blocked", reason={"code": "maintenance_scope_changed", "summary": str(exc),
            "request_id": request["id"], "recovery_condition": "提交符合当前选定集合的明确维护任务；原稿、已用次数与决定保持。"})
        store.save(state)
        return state
    cursor = {key: deepcopy(state.get(key)) for key in ("stage", "node", "role", "mode", "status", "feedback", "reason", "checkpoint", "plan_draft")}
    working = state["checkpoint"]
    records = {}
    for task in tasks:
        key = scope_key(task["stage"], task["node"])
        task_id = request["id"] + ":" + key
        scope = scope_for(state, task)
        complete = bool(scope.get("approval") or scope["status"] in {"reviewing", "revision", "adjudicating", "signoff"})
        entry = {**deepcopy(task), "task_id": task_id, "scope": key, "purpose": "maintenance", "complete_candidate": complete,
                 "status": "pending" if complete else "waiting", "step": "patch" if task["action"] == "patch" else "review",
                 "role": "worker" if task["action"] == "patch" else "reviewer", "mode": "normal",
                 "plan_reason": task["reason"], "reason": None, "feedback": [task["reason"]], "history": [], "attempts": [],
                 "draft_checkpoint": working, "prepared": False}
        if complete:
            if task["stage"] == "P5":
                invalidate_node(state["nodes"][task["node"]], require_worker=task["action"] == "patch")
            else:
                invalidate_stage(state, task["stage"], require_worker=task["action"] == "patch")
        records[task_id] = entry
    invalidate_final(state)
    state["maintenance"] = {"request_id": request["id"], "issue_id": request["issue_id"], "status": "active",
                            "started_at": time.time(), "normal_cursor": cursor, "working_checkpoint": working,
                            "tasks": records, "completed": []}
    settle(state)
    if cursor["status"] in {"paused", "cancelled"}:
        state["status"] = cursor["status"]
    store.save(state)
    with store.control_transaction() as ledger:
        current = _request(ledger, state["run_id"])
        if not current or current["id"] != request["id"]:
            raise ExtractError("maintenance request changed during writer installation")
        current.update(status="applied", applied_at=state["maintenance"]["started_at"])
    return state


def task_for(state: dict, call: dict) -> dict:
    task = (state.get("maintenance") or {}).get("tasks", {}).get(call.get("maintenance_task_id"))
    if not task or call.get("purpose") != "maintenance":
        raise StaleResult("maintenance invocation has no current task")
    return task


def _dependencies_ready(state: dict, task: dict) -> bool:
    from .stages import approval_is_current, dependencies_ready
    for stage in ("P1", "P2", "P3", "P4"):
        if int(stage[1:]) >= int(task["stage"][1:]):
            break
        scope = state["stages"][stage]
        if scope["status"] != "approved" or not approval_is_current(state, scope.get("approval")):
            return False
    return task["stage"] != "P5" or dependencies_ready(state, task["node"])


def next_calls(state: dict) -> list[dict]:
    from .stages import STAGES, session_group
    if state["status"] != "running" or state["active_attempts"]:
        return []
    for task in sorted(state["maintenance"]["tasks"].values(), key=lambda item: (item["stage"], item["scope"])):
        if task["status"] != "pending" or not _dependencies_ready(state, task):
            continue
        call = {key: task[key] for key in ("stage", "node", "role", "mode")}
        call.update(purpose="maintenance", maintenance_task_id=task["task_id"], maintenance_step=task["step"])
        group = session_group(state, call)
        return [{**call, "page": STAGES[task["stage"]]["page"], "session_group": group,
                 "session_id": state["sessions"].get(group, {}).get(task["role"])}]
    return []


def _normal_feedback(state: dict, task: dict) -> None:
    owner = state["nodes"][task["node"]] if task["stage"] == "P5" else state
    entry = {"code": "current_run_maintenance", "maintenance_task_id": task["task_id"], "summary": task["plan_reason"], "fields": task["fields"]}
    if entry not in owner.setdefault("feedback", []):
        owner["feedback"].append(entry)


def _resume_normal(state: dict) -> None:
    from .stages import resume_progress
    maintenance = state["maintenance"]
    pending = any(task["status"] == "waiting" and task["complete_candidate"] for task in maintenance["tasks"].values())
    maintenance["status"] = "waiting" if pending else "complete"
    if not pending:
        maintenance["completed_at"] = time.time()
    for key, value in maintenance["normal_cursor"].items():
        state[key] = deepcopy(value)
    if state["stage"] in {"P2", "P4"} and state["stages"][state["stage"]]["status"] == "approved":
        state["plan_draft"] = None
    state["checkpoint"] = maintenance["working_checkpoint"]
    state["status"], state["reason"] = "running", None
    resume_progress(state)
    for task in maintenance["tasks"].values():
        if task["status"] == "waiting" and (task["stage"] == "P5" or task["stage"] == state["stage"]):
            _normal_feedback(state, task)


def resume_deferred(state: dict) -> None:
    from .stages import approval_is_current
    maintenance = state.get("maintenance") or {}
    if maintenance.get("status") != "waiting":
        return
    ready = False
    for task in maintenance["tasks"].values():
        if task["status"] != "waiting":
            continue
        scope = scope_for(state, task)
        if scope["status"] == "approved" and approval_is_current(state, scope.get("approval")):
            task["status"] = "done"
        elif task["complete_candidate"] and _dependencies_ready(state, task):
            task.update(status="pending", prepared=False)
            ready = True
    if ready:
        maintenance["normal_cursor"] = {key: deepcopy(state.get(key)) for key in ("stage", "node", "role", "mode", "status", "checkpoint", "feedback", "reason", "plan_draft")}
        maintenance.update(status="active", working_checkpoint=state["checkpoint"])
    elif not any(task["status"] == "waiting" and task["complete_candidate"] for task in maintenance["tasks"].values()):
        maintenance.update(status="complete", completed_at=time.time())


def settle(state: dict) -> None:
    if state["active_attempts"]:
        return
    maintenance = state["maintenance"]
    tasks = list(maintenance["tasks"].values())
    stopped = [task for task in tasks if task["status"] not in FINISHED | {"pending"}]
    if stopped:
        maintenance["status"], state["status"] = "blocked", "blocked"
        state["reason"] = {"code": "maintenance_blocked", "summary": "有限维护已停止；原稿、已用次数和决定保持。",
                           "tasks": {task["task_id"]: task.get("reason") for task in stopped}}
        return
    state["status"], state["reason"] = "running", None
    maintenance["status"] = "active"
    if next_calls(state):
        return
    for task in tasks:
        if task["status"] == "pending":
            task.update(status="waiting", waiting_reason="normal_prerequisite")
    _resume_normal(state)


def validate_attempt(state: dict, attempt: dict, *, status="pending", current_checkpoint=None) -> None:
    task = task_for(state, attempt)
    expected = {"stage": task["stage"], "node": task["node"], "role": task["role"], "mode": "normal",
                "maintenance_step": task["step"]}
    checkpoint = attempt["base_checkpoint"] if current_checkpoint is None else current_checkpoint
    if (not active(state) or task["status"] != status or checkpoint != task["draft_checkpoint"] or
            any(attempt.get(key) != value for key, value in expected.items())):
        raise StaleResult("maintenance task, step or draft changed")
    generation = state["nodes"][task["node"]]["generation"] if task["stage"] == "P5" else scope_for(state, task)["generation"]
    if attempt.get("generation") != generation:
        raise StaleResult("maintenance scope generation changed")


def prepare_workspace(store, state: dict, call: dict) -> None:
    task = task_for(state, call) if call.get("purpose") == "maintenance" else None
    if task and task["step"] == "review" and call["stage"] in {"P2", "P4"}:
        cursor = state["maintenance"]["normal_cursor"]
        plan = (cursor["plan_draft"] if cursor["stage"] == call["stage"] and cursor["plan_draft"] is not None
                else state["selected_designs"] if call["stage"] == "P2" else state["writing_plan"])
        state["plan_draft"] = deepcopy(plan)
    if task and call["stage"] == "P5":
        store.start_node(state, call["node"], call=call)
        task["draft_checkpoint"] = state["nodes"][call["node"]]["draft_checkpoint"]
        return
    if task and task["prepared"]:
        return
    source_id = state["maintenance"]["working_checkpoint"] if task else state["checkpoint"]
    if call["stage"] == "P1" and call["role"] == "worker":
        source_id = store.prepare_shared_metadata(state, source_id, call)
        if not task:
            state["checkpoint"] = source_id
    if call["stage"] != "P2":
        if task:
            task.update(draft_checkpoint=source_id, prepared=True)
        return
    source, info = store.verify_checkpoint(state, source_id)
    if set(info["files"]) <= {"root.yaml", "dispatch.md"}:
        if task:
            task.update(draft_checkpoint=source_id, prepared=True)
        return
    temporary = store.run_dir(state["run_id"]) / (".maintaining-" + uuid4().hex)
    temporary.mkdir()
    try:
        for name in ("root.yaml", "dispatch.md"):
            if (source / name).is_file():
                copy_artifacts(source / name, temporary / name)
        checkpoint = store.checkpoint(state, temporary, attempt_id=None, stage=call["stage"], node=call["node"],
                                     parent=source_id, call=call, expected_files=set(info["files"]) & {"root.yaml", "dispatch.md"})
    finally:
        shutil.rmtree(temporary)
    if task:
        task.update(draft_checkpoint=checkpoint, prepared=True)
    else:
        state["checkpoint"] = checkpoint


def merge_upstream(store, state: dict, attempt: dict, workspace: Path, destination: Path, *, selected=None) -> Path:
    """Merge reviewed bytes, including P1's already checked metadata copies."""
    base = ((state.get("maintenance") or {}).get("working_checkpoint") if attempt.get("purpose") == "maintenance" else None) or state.get("merged_checkpoint")
    if base:
        source, _ = store.verify_checkpoint(state, base)
        copy_artifacts(source, destination)
    else:
        destination.mkdir()
    for path in destination.iterdir():
        path.chmod(0o644)
    destination.chmod(0o755)
    for name in ("root.yaml", "dispatch.md"):
        if (workspace / name).is_file():
            copy_artifacts(workspace / name, destination / name)
    if attempt["stage"] == "P1":
        # P1's worker could write only root. The candidate transaction derived
        # these shared fields before review; merging must never regenerate them.
        for name in manifest(workspace):
            if name.endswith(".yaml") and name != "root.yaml":
                copy_artifacts(workspace / name, destination / name)
    if selected is not None:
        planned = {entry["id"] + ".yaml" for entry in selected}
        for name in manifest(destination):
            if name.endswith(".yaml") and name not in planned:
                (destination / name).unlink()
    return destination


def block(state: dict, call: dict, reason: dict, *, status="blocked") -> None:
    task = task_for(state, call)
    task["status"], task["reason"] = status, deepcopy(reason)
    task["feedback"].append(deepcopy(reason))


def _patch_fields(task: dict, receipt: dict, state: dict) -> list[dict]:
    requested = receipt.get("revision_scope") or task["fields"]
    ceiling = {item["path"]: item["fields"] for item in task["fields"]}
    anchored = {item["path"] for item in receipt["artifact_evidence"]}
    _validate_tasks(state, [{"stage": task["stage"], "node": task["node"], "action": "patch", "fields": requested, "reason": task["plan_reason"]}])
    for item in requested:
        if item["path"] not in ceiling or item["path"] not in anchored or any(not any(limit == "*" or field == limit or field.startswith(limit + ".")
                for limit in ceiling[item["path"]]) for field in item["fields"]):
            raise ExtractError("maintenance revision exceeds its sealed fields or lacks current artifact evidence")
    return deepcopy(requested)


def advance(state: dict, receipt: dict, checkpoint_id: str, receipt_hash: str, validation: dict) -> dict:
    from .stages import _approve, scope_key
    result = deepcopy(state)
    identity = receipt["identity"]
    attempt_id, key = identity["attempt_id"], scope_key(identity["stage"], identity["node"])
    if attempt_id in result["accepted_receipts"] or result["active_attempts"].get(key) != attempt_id:
        raise StaleResult("maintenance receipt is duplicate or inactive")
    task, decision = task_for(result, identity), receipt["decision"]
    result["accepted_receipts"][attempt_id] = receipt_hash
    result["active_attempts"].pop(key)
    task["draft_checkpoint"] = checkpoint_id
    task["history"].append({"attempt_id": attempt_id, "step": task["step"], "decision": decision, "checkpoint": checkpoint_id, "receipt_sha256": receipt_hash})
    result["recoverable_checkpoint"] = checkpoint_id
    if identity["stage"] == "P5":
        result["nodes"][identity["node"]]["draft_checkpoint"] = checkpoint_id
    if identity["role"] == "worker":
        scope_for(result, task)["candidates"].append({"checkpoint": checkpoint_id, "plan": deepcopy(receipt.get("plan")),
                                                 "generation": identity["generation"], "node_signature": identity.get("node_signature")})
        if decision == "continue":
            scope_for(result, task)["continues"] += 1
            task.update(status="pending", feedback=[receipt["summary"], *receipt["issues"]])
        elif decision == "complete":
            if identity["stage"] == "P5":
                result["nodes"][identity["node"]]["authored_generation"] = result["nodes"][identity["node"]]["generation"]
            task.update(step="verify", role="reviewer", status="pending", complete_candidate=True, feedback=[receipt["summary"]])
            if identity["stage"] in {"P2", "P4"}:
                result["plan_draft"] = deepcopy(receipt["plan"])
        else:
            block(result, identity, {"code": "maintenance_patch_incomplete", "summary": receipt["summary"], "issues": receipt["issues"]})
    elif decision == "approve":
        if not task["complete_candidate"]:
            raise ExtractError("maintenance cannot approve an unfinished ordinary draft")
        _approve(result, receipt, checkpoint_id, receipt_hash, validation)
        task.update(status="done", reason=None)
        result["maintenance"]["completed"].append(task["task_id"])
        result["maintenance"]["working_checkpoint"] = checkpoint_id
        if identity["stage"] in {"P2", "P4"}:
            result["plan_draft"] = None
    elif decision == "revise" and task["step"] == "review":
        fields = _patch_fields(task, receipt, result)
        task.update(step="patch", role="worker", status="pending", fields=fields, feedback=receipt["issues"])
        if identity["stage"] == "P1":
            task["prepared"] = False
        scope_for(result, task)["revisions"] += 1
    else:
        block(result, identity, {"code": "maintenance_verify_failed" if task["step"] == "verify" else "maintenance_review_blocked", "summary": receipt["summary"], "issues": receipt["issues"]})
    settle(result)
    return result
