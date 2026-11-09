"""Append-only semantic contexts and evidence for continuing an unfinished run.

The input lock remains the source/config authority. Rules and executor locks are
separate real snapshots; a revision never manufactures an input fingerprint.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

from .common import (ExtractError, StaleResult, digest, fsync_directory, json_bytes,
                     read_json, rules_root, seal, sha256, source_root, sync_tree,
                     verify_manifest, write_json)
from .snapshots import freeze_inputs, verify_run_inputs, verify_snapshot


CAPABILITY = "semantic_revision.v1"
STATE_FORMAT = "schema_extract.run.v2.semantic.v1"
RECORD_FORMAT = "schema_extract.semantic_revision.v1"
PLAN_FORMAT = "schema_extract.revision_plan.v1"
CONTEXT_FIELDS = ("semantic_revision", "maintenance_task_id", "purpose", "maintenance_step")


def _id(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ExtractError("invalid semantic revision identity")
    return value


def revision_directory(store, run_id, revision_id):
    return store.run_dir(run_id) / "revisions" / _id(revision_id)


def _history(state):
    history = state.get("semantic_revisions", [])
    if not isinstance(history, list):
        raise ExtractError("semantic history is not a list")
    for identity in history:
        _id(identity)
    if (len(set(history)) != len(history) or state.get("semantic_revision") != (history[-1] if history else None)
            or bool(history) != (state.get("format") == STATE_FORMAT)):
        raise ExtractError("semantic history or state format is inconsistent")
    return history


def _preserved(source, target):
    verify_snapshot(target)
    if (target["source"]["files"] != source["source"]["files"] or
            target["source"].get("bindings") != source["source"].get("bindings") or
            target["config"] != source["config"] or target["check_policy"] != source["check_policy"]):
        raise ExtractError("semantic revision must preserve source bytes, role bindings, config and check policy")


def freeze_target(store, state, target_repo=None):
    from .inputs import SourceTree
    source = read_json(store.run_dir(state["run_id"]) / "inputs.lock.json")
    verify_run_inputs(state, source)
    return freeze_inputs(store.root, SourceTree(source_root(source)), source["source"]["bindings"],
                         sorted(source["source"]["files"]), source["config"],
                         rules_repo=(target_repo or rules_root()).resolve())


def verify_revisions(store, state, *, evidence=None):
    """Verify sealed context records; historical receipts are never rewritten."""
    from .runtime import _BOUNDARY, _executor
    active = _BOUNDARY.get()
    key = (getattr(store, "root", None), state["run_id"], state.get("execution_binding"), state.get("semantic_revision"))
    if evidence is None and active and active["key"] == key and "semantic_records" in active:
        return active["semantic_records"]
    source = read_json(store.run_dir(state["run_id"]) / "inputs.lock.json")
    verify_run_inputs(state, source)
    history, records = _history(state), []
    for index, identity in enumerate(history):
        directory = revision_directory(store, state["run_id"], identity)
        names = ("revision.json", "before.json", "plan.json", "target.lock.json", "inputs.lock.json")
        payload = {name: read_json(directory / name) for name in names}
        hashes = {name: sha256(json_bytes(value)) for name, value in payload.items()}
        verify_manifest(directory, hashes)
        for path in (directory, *directory.rglob("*")):
            if path.is_symlink() or path.stat().st_mode & 0o222:
                raise ExtractError("semantic revision evidence is not immutable")
        record, before, plan, target = (payload[name] for name in names[:4])
        if (record.get("format") != RECORD_FORMAT or record.get("id") != identity or
                digest({key: value for key, value in record.items() if key != "id"}) != identity or
                record.get("run_id") != state["run_id"] or payload["inputs.lock.json"] != source or
                record.get("inputs_lock_sha256") != digest(source) or
                record.get("before_sha256") != digest(before) or record.get("plan_id") != plan.get("id") or
                plan.get("id") != digest({key: value for key, value in plan.items() if key != "id"}) or
                record.get("target_lock_sha256") != digest(target) or
                record.get("predecessor") != (history[index - 1] if index else None) or
                _history(before) != history[:index] or
                set(record.get("attempt_sha256", {})) != set(before["attempts"])):
            raise ExtractError("semantic revision source, plan, history or target changed")
        if (before["run_id"] != state["run_id"] or before["input_fingerprint"] != state["input_fingerprint"] or
                plan.get("basis", {}).get("semantic_revision") != record["predecessor"]):
            raise ExtractError("semantic revision belongs to another source context")
        _preserved(source, target)
        verify_run_inputs(before, source)
        from .compatibility import _changes, _report
        from .inputs import SourceTree
        prior_rules = records[-1]["target_lock"] if records else source
        old = {name: SourceTree(rules_root(prior_rules)).read(name) for name in prior_rules["rules"]["files"]}
        new = {name: SourceTree(rules_root(target)).read(name) for name in target["rules"]["files"]}
        _, checkpoint = store.verify_checkpoint(before)
        report_target = {"files": target["rules"]["files"], "manifest_sha256": digest(target["rules"]["files"]), "missing_files": []}
        expected_report = _report(before, source, checkpoint, report_target, _changes(old, new), rules_lock=prior_rules)
        if (plan.get("format") != PLAN_FORMAT or plan.get("report") != expected_report or
                plan.get("target_lock_sha256") != digest(target) or not plan.get("reviewed") or
                not _independent(plan.get("proposal")) or record.get("executor") != _executor(target) or
                CAPABILITY not in record["executor"]["capabilities"] or
                record.get("execution_binding") != before.get("execution_binding")):
            raise ExtractError("semantic revision has no complete source-bound independent plan")
        expected_scopes = {(stage, "root") for stage in ("P1", "P2", "P3", "P4", "P6")} | {("P5", node) for node in before["nodes"] if node != "root"}
        if {(item.get("stage"), item.get("node")) for item in plan["scopes"]} != expected_scopes or len(plan["scopes"]) != len(expected_scopes):
            raise ExtractError("semantic plan does not cover every existing scope")
        for item in plan["scopes"]:
            if item["basis"] != _scope_basis(store, before, item["stage"], item["node"], prior_rules):
                raise ExtractError("semantic plan draft or dependency basis changed")
            if item["action"] == "keep" and item["basis"]["approval"] and not item["basis"]["retired"] and not _adoptable(item, plan["proposal"], plan["report"]):
                raise ExtractError("semantic plan keep has no independent adoption evidence")
        if evidence:
            for name, checksum in hashes.items():
                evidence(directory / name, sha256=checksum)
            snapshot = Path(target["snapshot"])
            evidence(snapshot / "inputs.lock.json", sha256=sha256(json_bytes(target)))
            for kind in ("source", "rules"):
                for name, checksum in target[kind]["files"].items():
                    evidence(snapshot / kind / name, sha256=checksum)
            if target["source"].get("ref_kind") == "private_snapshot_commit":
                for path in sorted((snapshot / "source.git").rglob("*")):
                    if path.is_file():
                        evidence(path)
        records.append({"record": record, "before": before, "plan": plan, "target_lock": target})
    return records


def semantic_rules_lock(store, state, *, records=None):
    records = verify_revisions(store, state) if records is None else records
    return records[-1]["target_lock"] if records else read_json(store.run_dir(state["run_id"]) / "inputs.lock.json")


def resolve_context(store, state, attempt=None):
    """Return source_lock, rules_lock, executor_lock and their explicit identity.

    Historical attempts select the interval in which they were registered. Their
    archived digest prevents a caller from relabelling an old response as new.
    """
    from .runtime import verify_execution
    source = read_json(store.run_dir(state["run_id"]) / "inputs.lock.json")
    verify_run_inputs(state, source)
    records = verify_revisions(store, state)
    selected_revision = state.get("semantic_revision")
    if attempt is not None:
        if (attempt.get("run_id") != state["run_id"] or attempt.get("input_fingerprint") != source["fingerprint"]
                or attempt.get("attempt_id") not in state["attempts"]):
            raise ExtractError("attempt does not belong to the semantic context")
        for entry in records:
            hashes = entry["record"]["attempt_sha256"]
            if attempt["attempt_id"] in hashes:
                if hashes[attempt["attempt_id"]] != digest(attempt):
                    raise ExtractError("pre-revision attempt identity changed")
                selected_revision = entry["record"]["predecessor"]
                break
        if attempt.get("semantic_revision") != selected_revision:
            raise ExtractError("attempt semantic revision differs from its registered interval")
    rules = next((entry["target_lock"] for entry in records if entry["record"]["id"] == selected_revision), source)
    executor = verify_execution(store, state, attempt=attempt, _semantic_records=records)
    identity = {"semantic_revision": selected_revision, "source_fingerprint": source["fingerprint"],
                "rules_id": rules["rules"]["id"], "executor_rules_id": executor["rules"]["id"],
                "execution_binding": attempt.get("execution_binding") if attempt is not None else state.get("execution_binding")}
    return {"source_lock": source, "rules_lock": rules, "executor_lock": executor, "identity": identity}


def approval_is_current(state, approval):
    """Cheap eligibility only; Store's effective verification is the IO gate."""
    if not isinstance(approval, dict):
        return False
    if approval.get("stage") == "P5":
        node = state["nodes"].get(approval.get("node"), {})
        if approval.get("generation") != node.get("generation"):
            return False
    current = state.get("semantic_revision")
    if approval.get("semantic_revision") == current:
        return True
    adoption = state.get("approval_adoptions", {}).get(digest(approval))
    return bool(adoption and adoption.get("semantic_revision") == current)


def _scope_owner(state, stage, node):
    return state["nodes"][node]["stage"] if stage == "P5" else state["stages"][stage]


def _scope_approval(state, stage, node):
    return state["nodes"][node].get("approval") if stage == "P5" else state["stages"][stage].get("approval")


def _material_digest(material):
    """Hash parser Decimals exactly; leave all existing JSON-only hashes intact."""
    from decimal import Decimal

    def contains_decimal(value):
        if isinstance(value, Decimal):
            return True
        if isinstance(value, dict):
            return any(contains_decimal(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_decimal(item) for item in value)
        return False

    def typed(value):
        if isinstance(value, Decimal):
            if not value.is_finite():
                raise ExtractError("non-finite Decimal in semantic material")
            sign, digits, exponent = value.as_tuple()
            digits = list(digits)
            if not any(digits):
                digits, exponent = [0], 0
            else:
                while digits[-1] == 0:
                    digits.pop()
                    exponent += 1
            return ["decimal", sign, digits, exponent]
        if isinstance(value, dict):
            return ["mapping", [[key, typed(item)] for key, item in sorted(value.items())]]
        if isinstance(value, list):
            return ["sequence", [typed(item) for item in value]]
        return [type(value).__name__, value]

    if not contains_decimal(material):
        return digest(material)
    return sha256(b"schema_extract.material.decimal.v1\0" + json_bytes(typed(material)))


def _material(store, state, stage, node, checkpoint, lock):
    """Small stage-specific projections allow lawful later-stage additions."""
    from .artifacts import load_yaml
    directory, info = store.verify_checkpoint(state, checkpoint)
    root = load_yaml(directory, "root.yaml", parser_root=rules_root(lock))
    meta = root.get("meta_data", {})
    if stage == "P1":
        material = {"algorithm_requirement": root.get("contract", {}).get("algorithm_requirement"),
                    "metadata": {key: value for key, value in meta.items() if key not in {"version", "dispatch", "id", "parent"}}}
    elif stage == "P2":
        # P3 owns when; P4 replaces the draft index and its presentation status.
        dispatch = meta.get("dispatch")
        material = {"dispatch": {key: value for key, value in dispatch.items() if key not in {"when", "status"}}
                    if isinstance(dispatch, dict) else dispatch}
    elif stage == "P3":
        material = {key: root.get(key) for key in ("design", "realization")}
        material["when"] = meta.get("dispatch", {}).get("when") if isinstance(meta.get("dispatch"), dict) else None
    elif stage == "P4":
        from .stages import plan_signature
        material = {"plan": plan_signature(state.get("plan")), "dispatch": info["files"].get("dispatch.md")}
    elif stage == "P5":
        material = {"file": info["files"].get(state["nodes"][node]["file"])}
    else:
        material = info["files"]
    return _material_digest(material)


def _premises(store, state, stage, node, checkpoint, lock):
    upstream = {"P2": ("P1",), "P3": ("P1", "P2"), "P4": ("P1", "P2", "P3")}.get(stage, ())
    return {name: _material(store, state, name, node, checkpoint, lock) for name in upstream}


def _retained_node_checkpoint(store, state, node, fallback):
    """Bind preparation to the same own raw that Store.start_node will use."""
    data = state["nodes"][node]
    candidates = [data.get("draft_checkpoint"), data.get("last_draft_checkpoint"),
                  (data.get("approval") or {}).get("checkpoint"),
                  (data.get("last_approval") or {}).get("checkpoint"), fallback]
    for checkpoint in dict.fromkeys(value for value in candidates if value):
        _, info = store.verify_checkpoint(state, checkpoint)
        if data["file"] in info["files"]:
            return checkpoint
    return fallback


def _node_identity_migration(store, state, node, checkpoint, lock, *, entry=None, root_directory=None):
    """Compare retained own raw with the current identity, without editing either."""
    from .artifacts import load_yaml, _same
    data = state["nodes"][node]
    directory, info = store.verify_checkpoint(state, checkpoint)
    if data["file"] not in info["files"]:
        return None
    if entry is None:
        entry = next((item for item in state.get("plan") or [] if item["id"] == node), None)
        if entry is None or data.get("status") == "removed":
            return None
    if root_directory is None:
        # The retained plan and its approved root define identity. A current
        # upstream workspace may still be an unapproved revision candidate.
        plan_scope = state["stages"]["P4"]
        plan_approval = plan_scope.get("approval") or plan_scope.get("last_approval")
        if not plan_approval:
            return None
        root_directory, _ = store.verify_checkpoint(state, plan_approval["checkpoint"])
    raw = load_yaml(directory, data["file"], parser_root=rules_root(lock))
    root = load_yaml(root_directory, "root.yaml", parser_root=rules_root(lock))
    meta, current = raw.get("meta_data", {}), root.get("meta_data", {})
    dispatch = meta.get("dispatch", {})
    dispatch = dispatch if isinstance(dispatch, dict) else {}
    expected = {"id": node, "parent": (entry or data)["parent"],
                "name": current.get("name"), "describe": current.get("describe")}
    previous = {"id": dispatch.get("id"), "parent": dispatch.get("parent"),
                "name": meta.get("name"), "describe": meta.get("describe")}
    changed = [key for key in expected if not _same(previous[key], expected[key])]
    if not changed:
        return None
    return {"old_checkpoint": checkpoint, "own_sha256": info["files"][data["file"]],
            "root_sha256": sha256((root_directory / "root.yaml").read_bytes()),
            "previous": previous, "expected": expected, "fields": changed,
            "reason": "Retained own raw has a prior identity; worker must explicitly migrate it before review."}


def _scope_basis(store, state, stage, node, lock):
    from .stages import needs_authoring, node_signature, scope_key
    owner = _scope_owner(state, stage, node)
    approval = _scope_approval(state, stage, node)
    node_data = state["nodes"][node]
    maintenance = state.get("maintenance") or {}
    maintaining = maintenance.get("status") in {"active", "blocked"}
    canonical = maintenance.get("working_checkpoint") if maintaining else None
    previous_task = next((task for task in maintenance.get("tasks", {}).values()
                          if task["stage"] == stage and task["node"] == node and
                          task.get("status") not in {"done", "waiting"}), None) if maintaining else None
    checkpoint = ((node_data.get("draft_checkpoint") or canonical or state.get("merged_checkpoint")) if stage == "P5"
                  else canonical or (state.get("merged_checkpoint") if state["stage"] in {"P5", "P6"} else state["checkpoint"]))
    if stage == "P5":
        checkpoint = _retained_node_checkpoint(store, state, node, checkpoint or state["checkpoint"])
    if previous_task and previous_task.get("history") and previous_task.get("draft_checkpoint"):
        checkpoint = previous_task["draft_checkpoint"]
    checkpoint = checkpoint or state["checkpoint"]
    _, info = store.verify_checkpoint(state, checkpoint)
    role = node_data.get("role") if stage == "P5" else state.get("role") if state["stage"] == stage else None
    complete = bool(approval or role == "reviewer" or owner.get("status") in {"reviewing", "revision", "adjudicating"})
    # A retired approval is evidence that these exact raw bytes were complete,
    # never a current approval or permission to skip independent target review.
    historical = node_data.get("last_approval") if stage == "P5" else None
    if not complete and historical:
        _, historical_info = store.verify_approval_origin(state, historical)
        if (historical.get("stage") == "P5" and historical.get("node") == node and
                info["files"].get(node_data["file"]) is not None and
                info["files"].get(node_data["file"]) == historical_info["files"].get(node_data["file"])):
            complete = True
            role = "reviewer"
    if stage == "P5" and (needs_authoring(node_data) or
                          _node_identity_migration(store, state, node, checkpoint, lock)):
        complete, role, approval = False, "worker", None
    exists = bool(approval or owner.get("calls") or owner.get("started_at") or
                  (node_data.get("authored") if stage == "P5" else stage == state["stage"]))
    files = ({node_data["file"]} if stage == "P5" else {"root.yaml"} if stage in {"P1", "P3"}
             else set(info["files"]) if stage == "P6" else {"root.yaml", "dispatch.md"})
    dependencies = {}
    if stage == "P5":
        _, merged_info = store.verify_checkpoint(state, state.get("merged_checkpoint") or checkpoint)
        parent = node_data["parent"]
        while parent:
            ancestor = state["nodes"][parent]
            dependencies[ancestor["file"]] = merged_info["files"].get(ancestor["file"])
            parent = ancestor["parent"]
    entry = next((item for item in state.get("plan") or [] if item["id"] == node), None)
    origin_material = None
    if approval:
        origin_state = state
        invocation = read_json(store.run_dir(state["run_id"]) / "attempts" / approval["attempt_id"] / "attempt.json")
        origin_lock = lock
        if invocation.get("semantic_revision") != state.get("semantic_revision"):
            previous = invocation.get("semantic_revision")
            origin_lock = (read_json(revision_directory(store, state["run_id"], previous) / "target.lock.json") if previous
                           else read_json(store.run_dir(state["run_id"]) / "inputs.lock.json"))
        if stage == "P4":
            receipt = read_json(store.run_dir(state["run_id"]) / "attempts" / approval["attempt_id"] / "receipt.json")
            origin_state = {**state, "plan": receipt["plan"]}
        origin_material = _material(store, origin_state, stage, node, approval["checkpoint"], origin_lock)
    retired = stage == "P5" and (node_data.get("status") == "removed" or not entry)
    return {"scope": scope_key(stage, node), "checkpoint": checkpoint, "retired": retired,
            "files": {name: info["files"].get(name) for name in sorted(files)},
            "material_sha256": _material(store, state, stage, node, checkpoint, lock),
            "origin_material_sha256": origin_material,
            "prerequisite_digests": _premises(store, state, stage, node, checkpoint, lock),
            "dependency_digests": dependencies, "approval": deepcopy(approval),
            "approval_sha256": digest(approval) if approval else None,
            "generation": node_data.get("generation", 0) if stage == "P5" else owner["generation"],
            "node_signature": node_signature(entry) if entry and stage == "P5" else None,
            "status": owner["status"], "normal_role": role, "complete_candidate": complete, "exists": exists}


def _independent(proposal):
    reviewer = proposal.get("reviewer", {}) if isinstance(proposal, dict) else {}
    if not isinstance(reviewer, dict):
        return False
    return (reviewer.get("role") in {"Supervisor", "Reviewer"} and reviewer.get("independent_review") is True and
            all(isinstance(reviewer.get(key), str) and reviewer[key].strip() for key in ("identity", "record")) and
            isinstance(proposal.get("reason"), str) and bool(proposal["reason"].strip()))


def _adoptable(item, proposal, report):
    assessment = item.get("assessment") or {}
    if not isinstance(assessment, dict) or any(
            not isinstance(assessment.get(key, []), list) or
            not all(isinstance(value, str) for value in assessment.get(key, []))
            for key in ("covered_changes", "covered_issues")):
        return False
    issues = [str(issue.get("id", issue.get("issue_id", digest(issue)))) if isinstance(issue, dict) else str(issue)
              for issue in [*proposal.get("issues", []), *item.get("issues", [])]]
    return bool(item["basis"]["approval"] and _independent(proposal) and
                item["basis"]["material_sha256"] == item["basis"]["origin_material_sha256"] and
                assessment.get("decision") == "adopt" and
                all(assessment.get(key) is True for key in ("unchanged_content", "requirements_covered", "dependencies_compatible")) and
                isinstance(assessment.get("reason"), str) and assessment["reason"].strip() and
                isinstance(assessment.get("evidence"), list) and assessment["evidence"] and
                set(assessment.get("covered_changes", [])) == {change["path"] for change in report["changes"]} and
                set(assessment.get("covered_issues", [])) == set(issues))


def normalize_plan(store, state, target_lock, proposal=None, report=None):
    """Bind all decisions to the drained frontier and complete byte differences."""
    from .compatibility import _changes, _report
    from .inputs import SourceTree
    context = resolve_context(store, state)
    source, rules = context["source_lock"], context["rules_lock"]
    _preserved(source, target_lock)
    _, checkpoint = store.verify_checkpoint(state)
    old = {name: SourceTree(rules_root(rules)).read(name) for name in rules["rules"]["files"]}
    new = {name: SourceTree(rules_root(target_lock)).read(name) for name in target_lock["rules"]["files"]}
    target = {"files": target_lock["rules"]["files"], "manifest_sha256": digest(target_lock["rules"]["files"]), "missing_files": []}
    actual_report = _report(state, source, checkpoint, target, _changes(old, new), rules_lock=rules)
    if report is not None and report != actual_report:
        raise ExtractError("revision report no longer matches the drained frontier or complete target")
    proposal = deepcopy(proposal) if proposal is not None else {}
    if not isinstance(proposal, dict) or not isinstance(proposal.get("scopes", []), list):
        raise ExtractError("revision proposal must contain a scopes list")
    if any(not isinstance(proposal.get(key, []), list) for key in ("issues", "evidence")):
        raise ExtractError("revision issues and evidence must be complete lists")
    indexed = {}
    for item in proposal.get("scopes", []):
        if not isinstance(item, dict) or not isinstance(item.get("scope"), str) or item["scope"] in indexed:
            raise ExtractError("revision proposal has invalid or duplicate scopes")
        indexed[item["scope"]] = item
    scopes = [(stage, "root") for stage in ("P1", "P2", "P3", "P4", "P6")]
    scopes += [("P5", node) for node in sorted(state["nodes"]) if node != "root"]
    normalized = []
    for stage, node in scopes:
        basis = _scope_basis(store, state, stage, node, rules)
        supplied = deepcopy(indexed.pop(basis["scope"], {}))
        if supplied.get("basis") is not None and supplied["basis"] != basis:
            raise ExtractError("scope draft or dependency changed; refresh the revision plan: " + basis["scope"])
        action = supplied.get("action", "review" if basis["exists"] else "keep")
        if action not in {"keep", "review", "patch", "reextract"}:
            raise ExtractError("unknown revision action")
        item = {**supplied, "scope": basis["scope"], "stage": stage, "node": node, "action": action, "basis": basis,
                "reason": supplied.get("reason") or "Existing conclusions need current-rule review; retain the complete draft.",
                "issues": supplied.get("issues", []), "evidence": supplied.get("evidence", []),
                "fields": supplied.get("fields", []), "assessment": supplied.get("assessment")}
        if any(not isinstance(item[key], list) for key in ("issues", "evidence", "fields")):
            raise ExtractError("scope issues, evidence and fields must be lists")
        from .stages import stage_files
        allowed_files = set(stage_files(stage, node_entry=state["nodes"].get(node), revision_scope=state["revision_scope"]))
        for field_scope in item["fields"]:
            if (not isinstance(field_scope, dict) or field_scope.get("path") not in allowed_files or
                    not isinstance(field_scope.get("fields"), list) or not field_scope["fields"] or
                    not all(isinstance(field, str) and field.strip() for field in field_scope["fields"])):
                raise ExtractError("revision write scope must identify the stage file and explicit fields")
        if action in {"patch", "reextract"} and (not supplied.get("reason") or not item["evidence"] or not item["fields"]):
            raise ExtractError("patch/reextract needs concrete evidence and explicit file/field scope")
        if stage == "P6" and basis["approval"]:
            item["action"] = "review"
            item["reason"] = "Final delivery requires a new Reviewer approval in the current semantic context."
        if basis["retired"]:
            item["action"] = "keep"
            item["reason"] = "Retain this removed node and its historical accounting; no current approval is adopted."
        if stage == "P5" and (state["nodes"][node].get("identity_migration") or state["nodes"][node].get("provider_reopen")) and not basis["retired"]:
            if item["action"] == "keep":
                item["action"] = "review"
                item["reason"] += " Pending provider repair/identity migration still requires its current-generation independent approval."
        if item["action"] == "keep" and basis["approval"] and not basis["retired"] and not _adoptable(item, proposal, actual_report):
            item["action"] = "review"
            item["reason"] += " Approval adoption lacks a complete independent applicability assessment."
        if item["action"] == "keep" and basis["approval"] and not basis["retired"]:
            store.verify_effective_approval(state, basis["approval"])
        normalized.append(item)
    if indexed:
        raise ExtractError("revision proposal names nonexistent scopes: " + ", ".join(sorted(indexed)))
    basis = {"run_id": state["run_id"], "semantic_revision": state.get("semantic_revision"),
             "input_fingerprint": state["input_fingerprint"], "inputs_lock_sha256": digest(source),
             "rules_lock_sha256": digest(rules), "executor": deepcopy(state["executor"]),
             "execution_binding": state.get("execution_binding"), "plan_revision": state["plan_revision"],
             "cursor": deepcopy((state.get("maintenance") or {}).get("normal_cursor"))
             if (state.get("maintenance") or {}).get("status") in {"active", "blocked"}
             else {key: deepcopy(state.get(key)) for key in ("stage", "node", "role", "mode", "status", "checkpoint", "feedback")}}
    result = {"format": PLAN_FORMAT, "basis": basis, "target_lock_sha256": digest(target_lock),
              "target_rules_id": target_lock["rules"]["id"], "report": actual_report,
              "proposal": proposal, "reviewed": bool(_independent(proposal)), "scopes": normalized}
    return {**result, "id": digest(result)}


def _action_evidence(value):
    if isinstance(value, dict):
        return {key: _action_evidence(item) for key, item in value.items()
                if key not in {"note", "reason", "summary", "detail"}}
    if isinstance(value, list):
        return [_action_evidence(item) for item in value]
    return value


def initialize_maintenance(state, plan, revision_id):
    tasks, completed = {}, []
    previous = state.get("maintenance") or {}
    action_history = deepcopy(previous.get("action_history", {}))
    for identity, task in previous.get("tasks", {}).items():
        if task.get("history") or task.get("attempts"):
            action_history[identity] = {"revision_id": previous.get("revision_id"),
                                        "status": task["status"], "history": deepcopy(task.get("history", []))}
    for item in plan["scopes"]:
        basis = item["basis"]
        task_id = digest({"target_rules": plan["report"]["target"]["manifest_sha256"], "scope": item["scope"],
                          "issues": _action_evidence([*plan["proposal"].get("issues", []), *item["issues"]]),
                          "evidence": _action_evidence([*plan["proposal"].get("evidence", []), *item["evidence"]]), "action": item["action"]})
        step = "review" if item["action"] == "review" else "patch"
        status = "done" if item["action"] == "keep" else "waiting" if item["stage"] == "P6" or not basis["complete_candidate"] else "pending"
        tasks[task_id] = {**deepcopy(item), "id": task_id, "task_id": task_id, "purpose": "maintenance",
                          "step": step, "role": "reviewer" if step == "review" else "worker", "status": status,
                          "generation": basis["generation"], "draft_checkpoint": basis["checkpoint"],
                          "complete_candidate": basis["complete_candidate"], "normal_role": basis["normal_role"],
                          "feedback": [], "attempts": []}
        if task_id in action_history and status == "pending":
            tasks[task_id].update(status="blocked", previous_action=action_history[task_id],
                                 reason={"code": "maintenance_action_already_used",
                                         "summary": "The same target, scope, issue/evidence and action already consumed its finite maintenance pass."})
        if status == "done":
            completed.append(task_id)
    return {"revision_id": revision_id, "status": "active", "tasks": tasks,
            "normal_cursor": deepcopy(plan["basis"]["cursor"]), "completed": completed,
            "action_history": action_history,
            "working_checkpoint": previous.get("working_checkpoint") if previous.get("status") in {"active", "blocked"} else None,
            "superseded_maintenance": {"revision_id": previous.get("revision_id"), "status": "superseded",
                                       "unfinished_tasks": [identity for identity, task in previous.get("tasks", {}).items()
                                                            if task.get("status") not in {"done", "completed"}]} if previous else None,
            "created_at": time.time()}


def verify_effective_approval(store, state, approval):
    receipt, checkpoint = store.verify_approval_origin(state, approval)
    if approval.get("semantic_revision") == state.get("semantic_revision"):
        if state.get("semantic_revision"):
            rules = semantic_rules_lock(store, state)
            current = _scope_basis(store, state, approval["stage"], approval["node"], rules)
            if (current["material_sha256"] != _material(store, state, approval["stage"], approval["node"], approval["checkpoint"], rules) or
                    approval["generation"] != current["generation"] or
                    current["prerequisite_digests"] != _premises(store, state, approval["stage"], approval["node"], approval["checkpoint"], rules) or
                    any(checkpoint["files"].get(name) != checksum for name, checksum in current["dependency_digests"].items())):
                raise ExtractError("current approval content or dependencies changed")
        return receipt, checkpoint
    if not approval_is_current(state, approval):
        raise ExtractError("historical approval needs current review or evidenced adoption")
    adoption = state["approval_adoptions"][digest(approval)]
    records = verify_revisions(store, state)
    latest = records[-1]
    item = next((item for item in latest["plan"]["scopes"] if item["scope"] == adoption.get("scope")), None)
    if (item is None or item["action"] != "keep" or item["basis"]["approval"] != approval or
            adoption != {"semantic_revision": state["semantic_revision"], "scope": item["scope"],
                         "adoption_id": digest({"plan_id": latest["plan"]["id"], "scope": item["scope"], "approval": digest(approval)})} or
            not _adoptable(item, latest["plan"]["proposal"], latest["plan"]["report"])):
        raise ExtractError("approval adoption does not match sealed independent evidence")
    old_context = semantic_rules_lock(store, latest["before"])
    current = _scope_basis(store, state, item["stage"], item["node"], old_context)
    for key in ("material_sha256", "dependency_digests", "prerequisite_digests", "generation", "node_signature"):
        if current[key] != item["basis"][key]:
            raise ExtractError("adopted scope content or dependencies changed: " + item["scope"])
    return receipt, checkpoint


def effective_dependency_digests(store, state, approval, checkpoint):
    """An adopted node uses the explicitly reviewed current parent identities."""
    if approval.get("semantic_revision") == state.get("semantic_revision"):
        return checkpoint["files"]
    adoption = state.get("approval_adoptions", {}).get(digest(approval))
    if not adoption:
        raise ExtractError("missing current dependency adoption")
    latest = verify_revisions(store, state)[-1]
    item = next(item for item in latest["plan"]["scopes"] if item["scope"] == adoption["scope"])
    return item["basis"]["dependency_digests"]


def apply_revision(store, run_id, plan, target_lock, *, request_id=None):
    """Seal first, then commit the sole semantic pointer under the writer lock."""
    from .runtime import _executor, _quiescent, _retire_sessions
    with store.writer(run_id):
        state = store.load(run_id)
        records = verify_revisions(store, state)
        for entry in records:
            record = entry["record"]
            if ((request_id is not None and record.get("request_id") == request_id) or
                    record["plan_id"] == plan.get("id")):
                if record["plan_id"] != plan.get("id") or record["target_lock_sha256"] != digest(target_lock):
                    raise ExtractError("revision request is already bound to a different plan")
                return {"run_id": run_id, "revision": record["id"], "revision_record": record,
                        "maintenance": state.get("maintenance"), "reused": True, "started": False}
        ledger = store.read_supervisor()
        request = ledger.get("semantic_requests", {}).get(run_id, {})
        requested = bool(request_id and request.get("id") == request_id and request.get("eligible_at_request") is True and
                         request.get("source_semantic_revision") == state.get("semantic_revision") and
                         request.get("status") in {"requested", "draining", "drained", "prepared"})
        if request_id and not requested:
            raise ExtractError("revision request does not match the current pending semantic interval")
        if request.get("status") in {"requested", "draining", "drained", "prepared"} and not requested:
            raise ExtractError("apply must consume the registered pending revision request")
        if requested and (request.get("target_fingerprint") != target_lock["fingerprint"] or
                          request.get("target_rules_id") != target_lock["rules"]["id"]):
            raise ExtractError("revision target differs from the registered request")
        if (state.get("published") or state.get("exports") or ledger.get("commits", {}).get(run_id) or
                (state.get("publication") or {}).get("status") == "committed"):
            raise ExtractError("completed or exported runs cannot enter semantic revision")
        if state["status"] in {"ready", "published"} and not (state["status"] == "ready" and requested):
            raise ExtractError("completed runs cannot enter semantic revision")
        unfinished_maintenance = (state.get("maintenance") or {}).get("status") in {"active", "blocked", "waiting"}
        safely_stopped = (state.get("status") in {"paused", "cancelled"} and
                          not any(state.get(key) for key in ("active_attempts", "unconfirmed_attempts", "unsettled_attempts")))
        if unfinished_maintenance and not safely_stopped:
            raise ExtractError("finish or safely pause the active maintenance before another semantic revision")
        _quiescent(store, state, ledger)
        expected = normalize_plan(store, state, target_lock, plan.get("proposal"))
        if plan != expected:
            raise ExtractError("revision frontier changed; prepare the plan again")
        if not plan["reviewed"]:
            raise ExtractError("revision apply needs the complete independent plan assessment")
        executor = _executor(target_lock)
        if CAPABILITY not in executor["capabilities"]:
            raise ExtractError("target executor cannot read semantic revisions")
        source = read_json(store.run_dir(run_id) / "inputs.lock.json")
        record = {"format": RECORD_FORMAT, "run_id": run_id, "request_id": request_id,
                  "predecessor": state.get("semantic_revision"), "before_sha256": digest(state),
                  "inputs_lock_sha256": digest(source), "plan_id": plan["id"],
                  "target_lock_sha256": digest(target_lock), "executor": executor,
                  "execution_binding": state.get("execution_binding"), "registered_at": time.time(),
                  "supersedes_maintenance": (state.get("maintenance") or {}).get("revision_id") if unfinished_maintenance else None,
                  "attempt_sha256": {identity: digest(read_json(store.run_dir(run_id) / "attempts" / identity / "attempt.json"))
                                     for identity in state["attempts"]}}
        record["id"] = digest(record)
        updated = deepcopy(state)
        _retire_sessions(store, updated, record, reason="semantic_revision")
        updated.update(format=STATE_FORMAT, semantic_revision=record["id"],
                       semantic_revisions=[*state.get("semantic_revisions", []), record["id"]], executor=executor)
        updated["maintenance"] = initialize_maintenance(state, plan, record["id"])
        updated["approval_adoptions"] = {}
        for item in plan["scopes"]:
            if item["action"] == "keep" and item["basis"]["approval"] and not item["basis"]["retired"]:
                approved = item["basis"]["approval_sha256"]
                updated["approval_adoptions"][approved] = {
                    "semantic_revision": record["id"], "scope": item["scope"],
                    "adoption_id": digest({"plan_id": plan["id"], "scope": item["scope"], "approval": approved})}
        if updated["status"] == "ready":
            updated.update(status="running", stage="P6", node="root", role="reviewer", mode="normal")
        root = store.run_dir(run_id) / "revisions"
        root.mkdir(exist_ok=True)
        temporary = root / (".building-" + uuid4().hex)
        temporary.mkdir()
        try:
            for name, value in (("revision.json", record), ("before.json", state), ("plan.json", plan),
                                ("target.lock.json", target_lock), ("inputs.lock.json", source)):
                write_json(temporary / name, value)
            seal(temporary)
            sync_tree(temporary)
            os.rename(temporary, root / record["id"])
            fsync_directory(root)
            resolve_context(store, updated)
            with store.supervisor() as current_ledger:
                if (current_ledger.get("semantic_requests", {}).get(run_id, {}) != request or store.load(run_id) != state or
                        current_ledger.get("commits", {}).get(run_id) or
                        current_ledger.get("starts", {}).get(run_id, {}).get("launch") != ledger.get("starts", {}).get(run_id, {}).get("launch")):
                    raise ExtractError("run or semantic request changed before revision commit")
                store.save(updated)
            return {"run_id": run_id, "revision": record["id"], "revision_record": record,
                    "maintenance": updated["maintenance"], "reused": False, "started": False}
        finally:
            if temporary.exists():
                for path in temporary.rglob("*"):
                    path.chmod(0o755 if path.is_dir() else 0o644)
                temporary.chmod(0o755)
                shutil.rmtree(temporary)
