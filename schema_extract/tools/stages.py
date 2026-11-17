"""Pure stage and dependency decisions; only the Runner performs IO and commits."""

from __future__ import annotations

from copy import deepcopy

from .common import ExtractError, digest


DESIGN_READS = ["schema_docs/0.common.md", "schema_docs/3.design.math_reshape.md",
                "schema_docs/4.design.work_schedule.md", "schema_docs/5.design.tile_pipeline.md",
                "schema_docs/6.realization.md", "schema_template/README.md"]
STAGES = {
    "P1": {"page": "01_contract.md", "writes": ["root.yaml"],
           "evidence": ["implementation", "io", "input", "reference", "acceptance", "scope"],
           "conclusions": ["contract", "coverage"],
           "reads": ["schema_docs/0.common.md", "schema_docs/1.contract.algorithm_requirement.md",
                     "schema_template/starter.yaml", "schema_template/README.md"]},
    "P2": {"page": "02_dispatch_draft.md", "writes": ["root.yaml", "dispatch.md"],
           "evidence": ["host"], "conclusions": ["mechanisms", "root_choice", "selection", "coverage"],
           "reads": ["schema_docs/0.common.md"]},
    "P3": {"page": "03_root.md", "writes": ["root.yaml"],
           "evidence": ["implementation", "host"], "conclusions": ["design", "walkthrough", "dependencies"],
           "reads": DESIGN_READS},
    "P4": {"page": "04_dispatch_plan.md", "writes": ["dispatch.md"],
           "evidence": ["host"], "conclusions": ["directory", "writing_plan", "coverage"],
           "reads": ["schema_docs/0.common.md"]},
    "P5": {"page": "05_branch.md", "writes": [],
           "evidence": ["implementation", "host"],
           "conclusions": ["differences", "shared_design", "walkthrough"],
           "reads": ["schema_extract/prompt/phases/03_root.md", *DESIGN_READS]},
    "P6": {"page": "06_final.md", "writes": [],
           "evidence": ["host"], "conclusions": ["directory", "selection", "coverage", "references", "unresolved"],
           "reads": ["schema_docs/0.common.md", "schema_template/README.md"]},
}

DOCUMENT_CONCLUSIONS = (
    "document_self_contained", "document_layering",
    "document_consistency", "document_relevance",
)


def required_conclusions(attempt: dict) -> list[str]:
    result = list(STAGES[attempt["stage"]]["conclusions"])
    if attempt["role"] == "reviewer":
        result.extend(DOCUMENT_CONCLUSIONS)
    return result


def read_specs(attempt: dict) -> list[str]:
    result = list(STAGES[attempt["stage"]]["reads"])
    if attempt["role"] == "reviewer":
        result.extend(["schema_extract/prompt/roles/reviewer.md", "schema_docs/0.common.md",
                       "schema_docs/1.contract.algorithm_requirement.md",
                       "schema_template/README.md"])
        if attempt["stage"] not in {"P1", "P2"}:
            result.extend(DESIGN_READS)
    return list(dict.fromkeys(result))


def new_scope() -> dict:
    return {"status": "pending", "revisions": 0, "continues": 0, "invalid_receipts": 0,
            "receipt_repairs": 0, "cli_failures": 0, "check_failures": 0, "tool_failures": 0,
            "calls": 0, "elapsed_seconds": 0.0, "approval": None, "started_at": None,
            "deadline": None, "baseline_checkpoint": None, "candidates": [],
            "adjudications": 0, "signoffs": 0, "adjudication": None, "signoff": None,
            "generation": 0}


def scope_key(stage: str, node: str = "root") -> str:
    return "P5:" + node if stage == "P5" else stage


def scope_for(state: dict, call: dict | None = None) -> dict:
    call = call or state
    return state["nodes"][call["node"]]["stage"] if call["stage"] == "P5" else state["stages"][call["stage"]]


def owner_for(state: dict, call: dict) -> dict:
    if call.get("purpose") == "maintenance":
        from .maintenance import task_for
        return task_for(state, call)
    return state["nodes"][call["node"]] if call["stage"] == "P5" else state


def session_group(state: dict, call: dict | None = None) -> str:
    call = call or state
    if call.get("purpose") == "maintenance":
        return "maintenance:" + call["maintenance_task_id"] + ":" + call.get("maintenance_step", "review")
    group = scope_key(call["stage"], call["node"])
    return group if call.get("mode", "normal") == "normal" else group + ":" + call["mode"]


def session_scope(call: dict) -> dict:
    operation = call.get("mode", "normal")
    if call.get("purpose") == "maintenance":
        operation = "maintenance:" + call["maintenance_task_id"] + ":" + call["maintenance_step"]
    return {"stage": call["stage"], "node": call["node"], "role": call["role"], "operation": operation}


def stage_files(stage: str, *, node_entry: dict | None = None, revision_scope: list | None = None) -> list[str]:
    """One stage file policy for workspace permissions and artifact validation."""
    if stage == "P5":
        return [node_entry["id"] + ".yaml"] if node_entry else []
    if stage == "P6":
        return sorted({item["path"] for item in revision_scope or []})
    return STAGES[stage]["writes"]


def writable_files(state: dict, call: dict | None = None) -> list[str]:
    call = call or state
    if call["role"] != "worker" or call.get("mode", "normal") != "normal":
        return []
    allowed = stage_files(call["stage"], node_entry=state["nodes"].get(call["node"]),
                          revision_scope=state["revision_scope"])
    if call.get("purpose") == "maintenance":
        from .maintenance import task_for
        paths = {item["path"] for item in task_for(state, call).get("fields", [])}
        return [name for name in allowed if name in paths]
    return allowed


def call_view(state: dict, call: dict) -> dict:
    """Read a node's own cursor without changing the operator's stage cursor."""
    value = dict(state)
    value.update({key: call[key] for key in ("stage", "node", "role", "mode")})
    if call["stage"] == "P5":
        node = state["nodes"][call["node"]]
        value["checkpoint"] = node["draft_checkpoint"] or state["merged_checkpoint"]
        value["feedback"] = node["feedback"]
    if call.get("purpose") == "maintenance":
        from .maintenance import task_for
        task = task_for(state, call)
        value.update(purpose="maintenance", maintenance_task_id=task["task_id"],
                     maintenance_step=task["step"], checkpoint=task["draft_checkpoint"],
                     feedback=task.get("feedback", []))
    value["active_attempt"] = state["active_attempts"].get(scope_key(call["stage"], call["node"]))
    return value


def node_signature(entry: dict) -> str:
    return digest({"id": entry["id"], "instance": entry["instance"]})


def selection_signature(selected: list | None) -> str:
    return digest({entry["id"]: entry for entry in selected or []})


def writing_signature(plan: list | None) -> str:
    return digest({entry["id"]: sorted(entry["wait_for"]) for entry in plan or []})


def approval_is_current(state: dict, approval: dict | None) -> bool:
    if not isinstance(approval, dict) or approval.get("input_fingerprint") != state["input_fingerprint"]:
        return False
    stage, node = approval.get("stage"), approval.get("applies_to_node", approval.get("node"))
    selected = {entry["id"]: entry for entry in state.get("selected_designs") or []}
    if stage == "P5" or approval.get("applies_to_node"):
        scope = state.get("nodes", {}).get(node, {})
        generation = approval.get("applies_generation", approval.get("generation"))
    else:
        scope = state.get("stages", {}).get(stage, {})
        generation = approval.get("generation")
    if generation != scope.get("generation"):
        return False
    if stage in {"P3", "P5"} or approval.get("applies_to_node"):
        return node in selected and approval.get("node_signature") == node_signature(selected[node])
    if stage in {"P2", "P4", "P6"} and approval.get("selection_signature") != selection_signature(state.get("selected_designs")):
        return False
    return stage not in {"P4", "P6"} or approval.get("writing_signature") == writing_signature(state.get("writing_plan"))


def actual_sources(node: dict) -> set[str]:
    context = node.get("source_context") or node.get("approval") or {}
    origins = context.get("origins", context.get("design_origins", {}))
    return {source for origin in origins.values() for source in origin.get("chain", [])[1:]}


def effective_nodes(state: dict) -> set[str]:
    planned = {entry["id"] for entry in state.get("selected_designs") or []}
    available = {node_id for node_id in planned if (node := state["nodes"].get(node_id))
            and node.get("status") == "approved" and node.get("merged") and node.get("approval")
            and not node.get("provider_reopen")
            and approval_is_current(state, node["approval"])
            and node.get("approved_generation") == node.get("generation")}
    while True:
        valid = {node_id for node_id in available if
                 (actual_sources(state["nodes"][node_id]) | ({"root"} if node_id != "root" else set())) <= available}
        if valid == available:
            return valid
        available = valid


def final_eligible(state: dict) -> bool:
    if not state.get("selected_designs") or state.get("writing_plan") is None or not all(state["stages"][stage]["status"] == "approved"
                                       and approval_is_current(state, state["stages"][stage].get("approval"))
                                       for stage in ("P1", "P2", "P3", "P4")):
        return False
    planned = {item["id"] for item in state["selected_designs"]}
    return planned == effective_nodes(state) and not state.get("reopening")


def affected_nodes(state: dict, seeds, *, regions=None, contract=False) -> list[str]:
    """Follow recorded actual definitions, never writing-order preferences."""
    affected, definitions = set(seeds), set(seeds)
    for entry in state.get("selected_designs") or []:
        context = state["nodes"].get(entry["id"], {}).get("source_context") or {}
        origins = context.get("origins", {})
        if contract or any((regions is None or region in regions) and
                           set(origin.get("chain", [])[1:]) & definitions for region, origin in origins.items()):
            affected.add(entry["id"])
    return sorted(affected)


def writing_order(state: dict, node_id: str) -> dict:
    waits = next((item["wait_for"] for item in state.get("writing_plan") or [] if item["id"] == node_id), [])
    return {"instance": state["nodes"][node_id]["instance"], "wait_for": sorted(waits)}


def dependencies_ready(state: dict, node_id: str) -> bool:
    approved = effective_nodes(state)
    node = state["nodes"][node_id]
    selected = {entry["id"] for entry in state.get("selected_designs") or []}
    # A removed definition must be replaced explicitly by this consumer's
    # worker. It is never an approved source or a permanent wait target.
    sources = actual_sources(node)
    if node.get("role") == "worker":
        sources &= selected
    order = writing_order(state, node_id)
    # A satisfied writing order is not a continuing content dependency.
    waits = set(order["wait_for"]) if node.get("writing_started") != order else set()
    return (sources | waits | {"root"}) <= approved


NODE_STOPS = {"blocked", "budget_exhausted", "tool_error", "invalid_receipt", "cli_error", "termination_unconfirmed"}


def next_calls(state: dict) -> list[dict]:
    from .maintenance import active, next_calls as maintenance_calls
    if active(state):
        return maintenance_calls(state)
    if state["status"] != "running":
        return []
    if state["stage"] not in STAGES:
        raise ExtractError("the current stage has no implementation")
    if state["stage"] == "P6" and not final_eligible(state):
        raise ExtractError("P6 requires the selected complete, valid merged node set and P1-P4 approvals")
    candidates = []
    if state["stage"] == "P5":
        if not all(state["stages"][stage]["status"] == "approved" for stage in ("P1", "P2", "P3", "P4")):
            return []
        for entry in state["selected_designs"]:
            node = state["nodes"][entry["id"]]
            if entry["id"] == "root" or node["status"] in NODE_STOPS | {"approved", "removed"}:
                continue
            if dependencies_ready(state, entry["id"]):
                candidates.append({"stage": "P5", "node": entry["id"], "role": node["role"], "mode": node["mode"]})
    else:
        candidates.append({key: state[key] for key in ("stage", "node", "role", "mode")})
    calls = []
    for call in candidates:
        if scope_key(call["stage"], call["node"]) in state["active_attempts"]:
            continue
        group = session_group(state, call)
        calls.append({**call, "page": STAGES[call["stage"]]["page"], "session_group": group,
                      "session_id": state["sessions"].get(group, {}).get(call["role"])})
    return calls


def next_call(state: dict) -> dict | None:
    """Single-call convenience for callers preparing an upstream phase."""
    return next(iter(next_calls(state)), None)


def _blocked(result: dict, receipt: dict, code="missing_input") -> None:
    call = receipt["identity"]
    scope_for(result, call)["status"] = "blocked"
    reason = {"code": code, "summary": receipt["summary"], "issues": receipt["issues"]}
    if call["stage"] == "P5":
        node = result["nodes"][call["node"]]
        node["status"], node["reason"] = "blocked", reason
    else:
        result["status"], result["reason"] = "blocked", reason


def settle(result: dict) -> None:
    """Settle P5 only after considering *all* runnable nodes and active fences."""
    from .maintenance import active, resume_deferred, settle as settle_maintenance
    resume_deferred(result)
    if active(result):
        settle_maintenance(result)
        return
    if result["stage"] != "P5":
        return
    result["checkpoint"] = result["merged_checkpoint"]
    result.update(node="root", role=None, mode="normal", feedback=[], status="running", reason=None)
    unconfirmed = result.get("unconfirmed_attempts", {})
    if next_calls(result) or any(attempt_id not in unconfirmed for attempt_id in result["active_attempts"].values()):
        result["stages"]["P5"]["status"] = "running"
    elif result["active_attempts"]:
        result["status"] = "termination_unconfirmed"
        result["reason"] = {"code": "termination_unconfirmed", "summary": "旧调用尚未确认终止；同节点保持调用身份，不可重发。",
                            "attempts": unconfirmed}
    elif final_eligible(result):
        result["stages"]["P5"]["status"] = "approved" if len(result["selected_designs"]) > 1 else "skipped"
        result.update(stage="P6", node="root", role="reviewer", status="running", reason=None)
    else:
        result["stages"]["P5"]["status"] = "blocked"
        result["status"], result["role"] = "blocked", None
        result["reason"] = {"code": "blocked_nodes", "summary": "所有可继续的节点已处理；仍有必需节点阻塞或等待已定安排。",
                            "nodes": {node: value.get("reason")
                                      for node, value in result["nodes"].items()
                                      if node in {entry["id"] for entry in result["selected_designs"]} and value["status"] != "approved"}}


def new_node(entry: dict) -> dict:
    return {"id": entry["id"], "instance": entry["instance"], "file": entry["id"] + ".yaml",
            "status": "pending", "merged": False, "draft_checkpoint": None,
            "approval": None, "approved_sha256": None,
            "approved_generation": None, "generation": 0, "stage": new_scope(),
            "role": "worker", "mode": "normal", "feedback": [], "reason": None, "authored": False}


def needs_authoring(node: dict) -> bool:
    return bool(node.get("provider_reopen")) and (
        node.get("authored_generation") != node.get("generation"))


def invalidate_node(node: dict, *, require_worker=False) -> None:
    """Retain content, approval history and accounting while fencing old decisions."""
    complete = bool(node.get("approval") or node.get("role") == "reviewer")
    node["generation"] += 1
    node["last_draft_checkpoint"] = node.get("draft_checkpoint") or node.get("last_draft_checkpoint")
    node["last_approval"] = node.get("approval") or node.get("last_approval")
    worker = require_worker or needs_authoring(node) or not complete
    node.update(approval=None, approved_generation=None, approved_sha256=None, merged=False,
                status="pending" if worker else "reviewing",
                role="worker" if worker else "reviewer", mode="normal", reason=None)
    node["stage"]["last_approval"] = node["stage"].get("approval") or node["stage"].get("last_approval")
    node["stage"].update(status=node["status"], approval=None)
    node.pop("applicability", None)


def invalidate_final(state: dict, nodes=()) -> None:
    scope = state["stages"]["P6"]
    scope["last_approval"] = scope.get("approval") or scope.get("last_approval")
    changed = scope.get("approval") is not None or scope.get("status") not in {"pending", "draft"}
    scope.update(status="pending", approval=None, generation=scope["generation"] + int(changed))
    state["final_review_started"] = False
    state["pending_rechecks"] = sorted(set(state.get("pending_rechecks", [])) | set(nodes))


def selection_impact(state: dict, selected: list, *, root_changed=False) -> set[str]:
    old = {entry["id"]: entry for entry in state.get("selected_designs") or []}
    new = {entry["id"]: entry for entry in selected}
    changed = {node for node in old.keys() | new.keys()
               if node not in old or node not in new or node_signature(old[node]) != node_signature(new[node])}
    if root_changed:
        changed.add("root")
    return set(affected_nodes(state, changed))


def install_selection(state: dict, selected: list, invalidated: set[str]) -> None:
    """Retire identities without deleting their budgets, drafts or used decisions."""
    planned = {entry["id"]: entry for entry in selected}
    old_plan = {entry["id"]: entry for entry in state.get("selected_designs") or []}
    for node_id, node in state["nodes"].items():
        if node_id == "root":
            continue
        changed_instance = node_id in planned and node_signature(planned[node_id]) != node_signature(node)
        restored = node_id in planned and node.get("status") == "removed"
        removed_sources = actual_sources(node) - planned.keys()
        if node_id in invalidated and (node_id not in planned or changed_instance or removed_sources or restored):
            invalidate_node(node, require_worker=bool(changed_instance or removed_sources))
        if node_id not in planned:
            node["status"] = "removed"
            key = node.pop("provider_reopen", None)
            if key:
                state["reopen_requests"][key].update(status="retired", retired_reason="provider removed by P2",
                                                     retired_plan_revision=state.get("plan_revision", 0))
    for node_id, entry in planned.items():
        node = state["nodes"].setdefault(node_id, new_node(entry))
        node.update(id=node_id, instance=entry["instance"], file=node_id + ".yaml")
    state["selected_designs"] = deepcopy(selected)
    state["selection_signature"] = selection_signature(selected)
    if old_plan and selection_signature(list(old_plan.values())) != state["selection_signature"]:
        invalidate_stage(state, "P4", require_worker=True)
        if "root" in old_plan and node_signature(old_plan["root"]) != node_signature(planned["root"]):
            invalidate_stage(state, "P3", require_worker=True)
    if old_plan and invalidated:
        invalidate_final(state, invalidated & planned.keys())
    state["pending_rechecks"] = [node for node in state["pending_rechecks"] if node in planned]


def invalidate_stage(state: dict, stage: str, *, require_worker=False) -> None:
    scope = state["stages"][stage]
    complete = bool(scope.get("approval") or scope.get("last_approval") or scope.get("status") in {"reviewing", "approved"})
    scope["last_approval"] = scope.get("approval") or scope.get("last_approval")
    scope.update(approval=None, status="pending" if require_worker or not complete else "reviewing",
                 generation=scope["generation"] + 1)
    if stage == "P3":
        node = state["nodes"]["root"]
        node["last_approval"] = node.get("approval") or node.get("last_approval")
        node.update(approval=None, status=scope["status"], merged=False,
                    approved_generation=None, generation=node["generation"] + 1)
        node.pop("applicability", None)


def resume_progress(state: dict) -> None:
    """Return to the first genuinely unfinished responsibility, preserving its budget."""
    for stage in ("P1", "P2", "P3", "P4"):
        scope = state["stages"][stage]
        if scope["status"] == "approved" and approval_is_current(state, scope.get("approval")):
            continue
        role = "reviewer" if scope["status"] == "reviewing" else "worker"
        if state.get("stage") == stage and state.get("role") and scope["status"] in {"draft", "revision", "adjudicating", "signoff"}:
            return
        if stage in {"P2", "P4"} and role == "reviewer" and (state.get("stage") != stage or state.get("plan_draft") is None):
            state["plan_draft"] = deepcopy(state["selected_designs"] if stage == "P2" else state["writing_plan"])
        state.update(stage=stage, node="root", role=role, mode="normal", status="running", reason=None)
        return
    if state.get("reopening"):
        record = state["reopening"]
        record.update(status="closed", closing_checkpoint=state.get("merged_checkpoint"))
        state["reopen_requests"][record["key"]].update(record)
        state["plan_history"].append(deepcopy(record))
        state["reopening"] = None
    state.update(stage="P5", node="root", role=None, mode="normal", status="running", reason=None)
    settle(state)


def _approve(result: dict, receipt: dict, checkpoint_id: str, receipt_hash: str, validation: dict) -> None:
    identity = receipt["identity"]
    stage, node_id = identity["stage"], identity["node"]
    if stage == "P5":
        node = result["nodes"][node_id]
        if needs_authoring(node):
            raise ExtractError("provider repair requires a current-generation worker draft before independent approval")
        if identity.get("generation", node["generation"]) != node["generation"]:
            raise ExtractError("approval belongs to an obsolete node generation")
    scope = scope_for(result, identity)
    if stage == "P2":
        install_selection(result, receipt["plan"], set(validation.get("invalidated_nodes", [])))
    approval = {"attempt_id": identity["attempt_id"], "receipt_sha256": receipt_hash,
                "checkpoint": checkpoint_id, "input_fingerprint": identity["input_fingerprint"],
                "plan_revision": identity["plan_revision"], "stage": stage, "node": node_id,
                "generation": result["nodes"][node_id]["generation"] if stage == "P5" else scope["generation"],
                "node_signature": validation.get("node_signature"),
                "selection_signature": result.get("selection_signature") if stage == "P2" else identity.get("selection_signature"),
                "writing_signature": identity.get("writing_signature") if stage in {"P5", "P6"} else None,
                "projection": validation.get("approval_projection"),
                "dependency_digests": validation.get("dependency_digests", {}),
                "artifact_digests": validation.get("artifact_digests") or
                                    {item["path"]: item["sha256"] for item in receipt["artifacts"]},
                "conclusions": receipt["conclusions"], "issues": receipt["issues"]}
    approval.update({key: identity[key] for key in ("maintenance_task_id", "purpose", "maintenance_step")
                     if key in identity})
    resolution = validation.get("design_resolution", {}).get("nodes", {}).get(node_id)
    if stage in {"P3", "P5"} and resolution:
        approval.update(effective_identity=resolution["effective_identity"], source_manifest=resolution["source_manifest"],
                        design_origins=resolution["origins"])
        result["nodes"][node_id]["source_context"] = deepcopy(resolution)
    scope["status"], scope["approval"] = "approved", approval
    owner = owner_for(result, identity)
    owner["feedback"], owner["mode"] = [], "normal"
    maintaining = identity.get("purpose") == "maintenance"
    if maintaining and stage in {"P1", "P2", "P3", "P4"}:
        result["merged_checkpoint"] = checkpoint_id
    if stage in {"P1", "P2"} and not maintaining:
        result["plan_draft"] = None
    elif stage == "P3":
        root = result["nodes"]["root"]
        root.update(status="approved", merged=True, approval=approval, approved_sha256=approval["artifact_digests"]["root.yaml"],
                    approved_generation=root["generation"])
        result["merged_checkpoint"] = checkpoint_id
        result["pending_rechecks"] = [node for node in result["pending_rechecks"] if node != "root"]
    elif stage == "P4":
        result["writing_plan"] = deepcopy(receipt["plan"])
        result["writing_signature"] = writing_signature(receipt["plan"])
        approval["writing_signature"] = result["writing_signature"]
        result["plan_draft"] = None
        result["merged_checkpoint"] = checkpoint_id
        result["nodes"]["root"]["merged"] = True
    elif stage == "P5":
        node = result["nodes"][node_id]
        node.update(status="approved", merged=True, approval=approval,
                    approved_sha256=approval["artifact_digests"][node["file"]],
                    approved_generation=node["generation"], reason=None)
        result["merged_checkpoint"] = checkpoint_id
        result["pending_rechecks"] = [item for item in result["pending_rechecks"] if item != node_id]
        key = node.pop("provider_reopen", None)
        if key:
            result["reopen_requests"][key].update(status="closed", closing_approval=deepcopy(approval),
                                                closed_generation=node["generation"])
    elif stage == "P6":
        for node_id in result["pending_rechecks"]:
            node = result["nodes"][node_id]
            current = validation["design_resolution"]["nodes"][node_id]
            node["source_context"] = deepcopy(current)
            node["approval"] = {**approval, "effective_identity": current["effective_identity"],
                                "source_manifest": current["source_manifest"], "design_origins": current["origins"],
                                "applies_to_node": node_id, "applies_generation": node["generation"],
                                "node_signature": node_signature(node)}
            node["approved_sha256"] = approval["artifact_digests"][node["file"]]
            node["approved_generation"] = node["generation"]
        result["pending_rechecks"] = []
        result["merged_checkpoint"] = checkpoint_id
        result["status"], result["role"] = "ready", None
    if stage in {"P1", "P2", "P3", "P4"} and not maintaining:
        resume_progress(result)


def advance(state: dict, receipt: dict, checkpoint_id: str, receipt_hash: str, *, validation: dict | None = None) -> dict:
    """No IO or model calls; identical saved inputs yield identical decisions."""
    if receipt["identity"].get("purpose") == "maintenance":
        from .maintenance import advance as advance_maintenance
        return advance_maintenance(state, receipt, checkpoint_id, receipt_hash, validation or {})
    result = deepcopy(state)
    attempt_id = receipt["identity"]["attempt_id"]
    if attempt_id in state["accepted_receipts"]:
        raise ExtractError("duplicate receipt cannot advance state")
    identity = receipt["identity"]
    key = scope_key(identity["stage"], identity["node"])
    if attempt_id != state["active_attempts"].get(key):
        raise ExtractError("inactive receipt cannot advance state")
    validation = validation or {}
    stage, role, mode, decision = identity["stage"], identity["role"], identity.get("mode", "normal"), receipt["decision"]
    scope, owner = scope_for(result, identity), owner_for(result, identity)
    if stage != "P5":
        result["checkpoint"] = checkpoint_id
    result["nodes"][identity["node"]]["draft_checkpoint"] = checkpoint_id
    result["accepted_receipts"][attempt_id] = receipt_hash
    result["active_attempts"].pop(key)
    owner["reason"] = None
    if role == "worker":
        scope["candidates"].append({"checkpoint": checkpoint_id, "plan": deepcopy(receipt.get("plan")),
                                    "generation": identity.get("generation"), "node_signature": identity.get("node_signature")})
        resolution = validation.get("design_resolution", {}).get("nodes", {}).get(identity["node"])
        if resolution:
            result["nodes"][identity["node"]]["source_context"] = deepcopy(resolution)
    if mode == "adjudicate":
        scope["adjudication"] = {"attempt_id": attempt_id, "receipt_sha256": receipt_hash,
                                  "action": receipt["adjudication"]["action"], "checkpoint": checkpoint_id,
                                  "decision": decision, "check": validation.get("check")}
    if mode == "signoff":
        scope["signoff"] = {"attempt_id": attempt_id, "receipt_sha256": receipt_hash,
                             "decision": decision, "checkpoint": checkpoint_id,
                             "check": validation.get("check")}
    if decision == "reopen":
        scope["status"] = "reopen_requested"
        return result
    if decision == "blocked" and role == "reviewer":
        _blocked(result, receipt, "adjudication_blocked" if mode == "adjudicate" else "missing_input")
    elif mode == "adjudicate" and validation.get("adjudication_changed"):
        owner["mode"], owner["role"] = "signoff", "reviewer"
        scope["status"] = "signoff"
        owner["feedback"] = [{"operation": "signoff", "adjudication_attempt": attempt_id}]
        if stage in {"P2", "P4"}:
            result["plan_draft"] = deepcopy(receipt["plan"])
        if stage == "P6":
            result["pending_rechecks"] = affected_nodes(result, [*result["pending_rechecks"], *receipt["affected_nodes"]])
    elif decision == "approve" and (role == "reviewer" or mode in {"adjudicate", "signoff"}):
        _approve(result, receipt, checkpoint_id, receipt_hash, validation)
    elif role == "worker" and decision == "continue":
        scope["continues"] += 1
        scope["status"] = "draft"
        owner["feedback"] = [receipt["summary"]]
    elif role == "worker" and decision in {"complete", "blocked"}:
        if stage == "P5" and decision == "complete":
            owner["authored_generation"] = owner["generation"]
        owner["role"] = "reviewer"
        scope["status"] = "reviewing"
        owner["feedback"] = [receipt["summary"], *receipt["issues"]] if decision == "blocked" else []
        if stage in {"P2", "P4"}:
            result["plan_draft"] = deepcopy(receipt["plan"])
        if stage == "P6":
            result["pending_rechecks"] = affected_nodes(result, [*result["pending_rechecks"], *receipt["affected_nodes"]])
    elif role == "reviewer" and decision == "revise":
        scope["revisions"] += 1
        scope["status"] = "revision"
        owner["role"], owner["feedback"] = "worker", receipt["issues"]
        if validation.get("check", {}).get("exit_code") == 1:
            owner["feedback"] = [{"code": "checker_input_error", "check": validation["check"]}, *owner["feedback"]]
        if stage == "P6":
            result["revision_scope"] = deepcopy(receipt["revision_scope"])
            result["pending_rechecks"] = affected_nodes(result, [*result["pending_rechecks"], *receipt["affected_nodes"]])
        if scope["revisions"] >= state["config"]["revision_limit"]:
            if scope["adjudications"]:
                _blocked(result, receipt, "adjudication_already_used")
            else:
                owner["mode"], owner["role"] = "adjudicate", "reviewer"
                scope["status"] = "adjudicating"
    else:
        raise ExtractError("invalid stage/role/decision transition")
    if stage == "P5" and owner["status"] not in NODE_STOPS | {"approved", "removed"}:
        owner["status"] = scope["status"]
    settle(result)
    return result
