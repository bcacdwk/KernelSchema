"""Pure stage and dependency decisions; only the Runner performs IO and commits."""

from __future__ import annotations

from copy import deepcopy

from .common import ExtractError, digest
from .snapshots import PHASE_PAGES


DESIGN_READS = ["schema_docs/0.common.md", "schema_docs/3.design.math_reshape.md",
                "schema_docs/4.design.work_schedule.md", "schema_docs/5.design.tile_pipeline.md",
                "schema_docs/6.realization.md", "schema_template/README.md"]
STAGES = {
    "P1": {"page": PHASE_PAGES["P1"], "session_group": "P1_P2", "writes": ["root.yaml"],
           "evidence": ["implementation", "io", "input", "reference", "acceptance", "scope"],
           "conclusions": ["contract", "coverage"],
           "reads": ["schema_docs/0.common.md", "schema_docs/1.contract.algorithm_requirement.md",
                     "schema_template/starter.yaml", "schema_template/README.md"]},
    "P2": {"page": PHASE_PAGES["P2"], "session_group": "P1_P2", "writes": ["root.yaml", "dispatch.md"],
           "evidence": ["host"], "conclusions": ["root_choice", "paths", "coverage"],
           "reads": ["schema_docs/0.common.md"]},
    "P3": {"page": PHASE_PAGES["P3"], "session_group": "P3", "writes": ["root.yaml"],
           "evidence": ["implementation", "host"], "conclusions": ["design", "walkthrough", "dependencies"],
           "reads": DESIGN_READS},
    "P4": {"page": PHASE_PAGES["P4"], "session_group": "P4", "writes": ["dispatch.md"],
           "evidence": ["host"], "conclusions": ["tree", "paths", "coverage"],
           "reads": ["schema_docs/0.common.md"]},
    "P5": {"page": PHASE_PAGES["P5"], "session_group": "P5", "writes": [],
           "evidence": ["implementation", "host"],
           "conclusions": ["differences", "inherited_design", "walkthrough"],
           "reads": ["schema_extract/phases/03_root.md", *DESIGN_READS]},
    "P6": {"page": PHASE_PAGES["P6"], "session_group": "P6", "writes": [],
           "evidence": ["host"], "conclusions": ["tree", "selection", "coverage", "references", "unresolved"],
           "reads": ["schema_docs/0.common.md", "schema_template/README.md"]},
}


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
    return state["nodes"][call["node"]] if call["stage"] == "P5" else state


def session_group(state: dict, call: dict | None = None) -> str:
    call = call or state
    if call.get("mode") == "signoff":
        return "P6_signoff"
    if call["stage"] == "P5":
        return "P5:" + call["node"]
    return STAGES[call["stage"]]["session_group"]


def writable_files(state: dict, call: dict | None = None) -> list[str]:
    call = call or state
    if call["role"] != "worker" or call.get("mode", "normal") != "normal":
        return []
    if call["stage"] == "P5":
        return [state["nodes"][call["node"]]["file"]]
    if call["stage"] == "P6":
        return sorted({item["path"] for item in state["revision_scope"]})
    return STAGES[call["stage"]]["writes"]


def call_view(state: dict, call: dict) -> dict:
    """Read a node's own cursor without changing the operator's stage cursor."""
    value = dict(state)
    value.update({key: call[key] for key in ("stage", "node", "role", "mode")})
    if call["stage"] == "P5":
        node = state["nodes"][call["node"]]
        value["checkpoint"] = node["draft_checkpoint"] or state["merged_checkpoint"]
        value["feedback"] = node["feedback"]
    value["active_attempt"] = state["active_attempts"].get(scope_key(call["stage"], call["node"]))
    return value


def node_signature(entry: dict) -> str:
    # Evidence indices are local to a receipt. Selection text and every other
    # semantic part of an approved entry remain part of the dependency identity.
    return digest({key: value for key, value in entry.items() if key != "evidence"})


def plan_signature(plan: list | None) -> str:
    return digest({entry["id"]: node_signature(entry) for entry in plan or []})


def effective_nodes(state: dict) -> set[str]:
    planned = {entry["id"] for entry in state.get("plan") or []}
    return {node_id for node_id in planned if (node := state["nodes"].get(node_id))
            and node.get("status") == "approved" and node.get("merged") and node.get("approval")
            and node.get("approved_generation") == node.get("generation")}


def final_eligible(state: dict) -> bool:
    if not state.get("plan") or not all(state["stages"][stage]["status"] == "approved"
                                       for stage in ("P1", "P2", "P3", "P4")):
        return False
    planned = {item["id"] for item in state["plan"]}
    return planned == effective_nodes(state) and not state.get("reopening")


def affected_nodes(state: dict, seeds) -> list[str]:
    """Final edits also recheck descendants whose copied design relied on a changed node."""
    affected = set(seeds)
    while True:
        expanded = affected | {entry["id"] for entry in state.get("plan") or [] if entry["parent"] in affected}
        if expanded == affected:
            return sorted(affected)
        affected = expanded


NODE_STOPS = {"blocked", "budget_exhausted", "tool_error", "invalid_receipt", "cli_error", "termination_unconfirmed"}


def next_calls(state: dict) -> list[dict]:
    if state["status"] != "running":
        return []
    if state["stage"] not in STAGES:
        raise ExtractError("the current stage has no implementation")
    if state["stage"] == "P6" and not final_eligible(state):
        raise ExtractError("P6 requires the current plan's complete, valid merged node set and P1-P4 approvals")
    candidates = []
    if state["stage"] == "P5":
        if not all(state["stages"][stage]["status"] == "approved" for stage in ("P1", "P2", "P3", "P4")):
            return []
        approved = effective_nodes(state)
        for entry in state["plan"]:
            node = state["nodes"][entry["id"]]
            if entry["id"] == "root" or node["status"] in NODE_STOPS | {"approved", "removed"}:
                continue
            if entry["parent"] in approved:
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
        result["stages"]["P5"]["status"] = "approved" if len(result["plan"]) > 1 else "skipped"
        result.update(stage="P6", node="root", role="reviewer", status="running", reason=None)
    else:
        result["stages"]["P5"]["status"] = "blocked"
        result["status"], result["role"] = "blocked", None
        result["reason"] = {"code": "blocked_nodes", "summary": "所有可继续的节点已处理；仍有必需节点阻塞或等待父节点。",
                            "nodes": {node: value.get("reason", {"parent": value.get("parent")})
                                      for node, value in result["nodes"].items()
                                      if node in {entry["id"] for entry in result["plan"]} and value["status"] != "approved"}}


def new_node(entry: dict) -> dict:
    return {"file": entry["file"], "parent": entry["parent"],
            "status": "pending", "merged": False, "draft_checkpoint": None,
            "approved_checkpoints": {}, "approval": None, "approved_sha256": None,
            "approved_generation": None, "generation": 0, "stage": new_scope(),
            "role": "worker", "mode": "normal", "feedback": [], "reason": None, "authored": False}


def plan_impact(state: dict, plan: list, *, root_changed=False) -> set[str]:
    old = {entry["id"]: entry for entry in state.get("plan") or []}
    new = {entry["id"]: entry for entry in plan}
    changed = {node for node in old.keys() | new.keys()
               if node not in old or node not in new or node_signature(old[node]) != node_signature(new[node])}
    if root_changed:
        changed.add("root")
    while True:
        expanded = changed | {entry["id"] for entry in [*old.values(), *new.values()] if entry["parent"] in changed}
        if expanded == changed:
            return changed
        changed = expanded


def install_plan(state: dict, plan: list, invalidated: set[str]) -> None:
    """Retire identities without deleting their budgets, drafts or used decisions."""
    planned = {entry["id"]: entry for entry in plan}
    for node_id, node in state["nodes"].items():
        if node_id == "root":
            continue
        if node_id in invalidated:
            node["generation"] += 1
            node["last_draft_checkpoint"] = node["draft_checkpoint"] or node.get("last_draft_checkpoint")
            node["last_approval"] = node["approval"] or node.get("last_approval")
            node.update(approval=None, approved_generation=None, approved_sha256=None,
                        draft_checkpoint=None, merged=False, status="pending",
                        role="worker", mode="normal", feedback=[], reason=None)
            node["stage"]["status"], node["stage"]["approval"] = "pending", None
        if node_id not in planned:
            node["status"] = "removed"
    for node_id, entry in planned.items():
        if node_id == "root":
            continue
        node = state["nodes"].setdefault(node_id, new_node(entry))
        node.update(file=entry["file"], parent=entry["parent"])
    state["plan"] = deepcopy(plan)
    state["pending_rechecks"] = [node for node in state["pending_rechecks"] if node in planned]


def _approve(result: dict, receipt: dict, checkpoint_id: str, receipt_hash: str, validation: dict) -> None:
    identity = receipt["identity"]
    stage, node_id = identity["stage"], identity["node"]
    scope = scope_for(result, identity)
    approval = {"attempt_id": identity["attempt_id"], "receipt_sha256": receipt_hash,
                "checkpoint": checkpoint_id, "input_fingerprint": identity["input_fingerprint"],
                "plan_revision": identity["plan_revision"], "stage": stage, "node": node_id,
                "generation": result["nodes"][node_id]["generation"] if stage == "P5" else scope["generation"],
                "node_signature": validation.get("node_signature"),
                "dependency_digests": validation.get("dependency_digests", {}),
                "artifact_digests": validation.get("artifact_digests") or
                                    {item["path"]: item["sha256"] for item in receipt["artifacts"]},
                "conclusions": receipt["conclusions"], "issues": receipt["issues"]}
    scope["status"], scope["approval"] = "approved", approval
    result["nodes"][node_id]["approved_checkpoints"][stage] = checkpoint_id
    owner = owner_for(result, identity)
    owner["feedback"], owner["mode"] = [], "normal"
    if stage in {"P1", "P2"}:
        result["stage"], result["role"] = ("P2" if stage == "P1" else "P3"), "worker"
    elif stage == "P3":
        root = result["nodes"]["root"]
        root.update(status="approved", approval=approval, approved_sha256=approval["artifact_digests"]["root.yaml"],
                    approved_generation=root["generation"])
        result["stage"], result["role"] = "P4", "worker"
    elif stage == "P4":
        install_plan(result, receipt["plan"], set(validation.get("invalidated_nodes", [])))
        result["merged_checkpoint"] = checkpoint_id
        result["nodes"]["root"]["merged"] = True
        if result.get("reopening"):
            result["reopening"]["approved_plan_checkpoint"] = checkpoint_id
            result["reopening"]["new_plan"] = deepcopy(receipt["plan"])
            result["reopening"]["invalidated_nodes"] = validation.get("invalidated_nodes", [])
            result["plan_history"].append(deepcopy(result["reopening"]))
        result["reopening"] = None
        result.update(stage="P5", node="root", role=None, status="running")
    elif stage == "P5":
        node = result["nodes"][node_id]
        node.update(status="approved", merged=True, approval=approval,
                    approved_sha256=approval["artifact_digests"][node["file"]],
                    approved_generation=node["generation"], reason=None)
        result["merged_checkpoint"] = checkpoint_id
    elif stage == "P6":
        for node_id in result["pending_rechecks"]:
            node = result["nodes"][node_id]
            node["approval"] = approval
            node["approved_sha256"] = approval["artifact_digests"][node["file"]]
            node["approved_generation"] = node["generation"]
        result["pending_rechecks"] = []
        result["merged_checkpoint"] = checkpoint_id
        result["status"], result["role"] = "ready", None


def advance(state: dict, receipt: dict, checkpoint_id: str, receipt_hash: str, *, validation: dict | None = None) -> dict:
    """No IO or model calls; identical saved inputs yield identical decisions."""
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
        scope["candidates"].append({"checkpoint": checkpoint_id, "plan": deepcopy(receipt.get("plan"))})
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
    elif mode == "adjudicate" and stage == "P6" and validation.get("adjudication_changed"):
        owner["mode"], owner["role"] = "signoff", "reviewer"
        scope["status"] = "signoff"
        owner["feedback"] = [{"detail": "独立签收裁决后的最终改动及受影响结论，仅可 approve 或 blocked。"}]
        result["pending_rechecks"] = affected_nodes(result, [*result["pending_rechecks"], *receipt["affected_nodes"]])
    elif mode in {"adjudicate", "signoff"} and decision == "approve":
        _approve(result, receipt, checkpoint_id, receipt_hash, validation)
    elif role == "worker" and decision == "continue":
        scope["continues"] += 1
        scope["status"] = "draft"
        owner["feedback"] = [receipt["summary"]]
    elif role == "worker" and decision in {"complete", "blocked"}:
        owner["role"] = "reviewer"
        scope["status"] = "reviewing"
        owner["feedback"] = [receipt["summary"], *receipt["issues"]] if decision == "blocked" else []
        if stage == "P4":
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
    elif role == "reviewer" and decision == "approve":
        _approve(result, receipt, checkpoint_id, receipt_hash, validation)
    else:
        raise ExtractError("invalid stage/role/decision transition")
    if stage == "P5" and owner["status"] not in NODE_STOPS | {"approved", "removed"}:
        owner["status"] = scope["status"]
    settle(result)
    return result
