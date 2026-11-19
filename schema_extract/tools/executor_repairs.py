"""Explicit executor-only repair of a settled, undelivered current-format run.

Original input locks and historical invocations remain immutable. Preparation
has no execution authority; a sealed binding referenced by state is the commit.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4

from .common import (ExtractError, digest, fsync_directory, json_bytes, read_json, regular_file, rules_root,
                     seal, sha256, sync_tree, write_json)
from .snapshots import freeze_executor_patch, frozen_invocation, verify_executor, verify_snapshot
from .runtime import execution_boundary


CAPABILITY = "executor_patch.v1"


def _id(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ExtractError("invalid executor repair identity")
    return value


def _root(store, run_id):
    return store.run_dir(run_id) / "runtime_repairs"


def _history(state):
    history = state.get("execution_bindings", [])
    if (not isinstance(history, list) or any(not isinstance(value, str) for value in history) or len(set(history)) != len(history) or
            state.get("execution_binding") != (history[-1] if history else None)):
        raise ExtractError("executor binding history is inconsistent")
    return [_id(value) for value in history]


def _changes(previous, target):
    old, new = previous["rules"]["files"], target["rules"]["files"]
    return {name: {"before": old.get(name), "after": new.get(name)} for name in sorted(old.keys() | new.keys())
            if old.get(name) != new.get(name)}


def _target(original, target):
    from .runtime import _executor
    verify_snapshot(target)
    if (target.get("format") != "schema_extract.executor.v1" or
            target.get("original_input_fingerprint") != original["fingerprint"] or
            not set(original["rules"]["files"]) <= set(target["rules"]["files"])):
        raise ExtractError("executor target does not preserve the original input context")
    for name in _changes(original, target):
        if not name.startswith("schema_extract/") or not name.endswith(".py") or "tests" in Path(name).parts:
            raise ExtractError("executor repair changes semantic rules: " + name)
    executor = _executor(target)
    if CAPABILITY not in executor["capabilities"]:
        raise ExtractError("target executor cannot verify its repair binding")
    return executor


def _authorization(store, plan, before, review):
    if (not isinstance(review, dict) or
            set(review) != {"plan_id", "decision", "authority", "scope", "summary", "audit_points"} or
            review["decision"] != "continue_with_audit_marker" or review["plan_id"] != plan["id"] or
            review["authority"] != "user_instruction" or review["scope"] != "extract_tools_only" or
            not isinstance(review["summary"], str) or not review["summary"].strip() or
            not isinstance(review["audit_points"], list) or not review["audit_points"]):
        raise ExtractError("marked continuation requires the user's concrete extract-tool repair instruction")
    attempts = {aid: read_json(store.run_dir(before["run_id"]) / "attempts" / aid / "attempt.json")
                for aid in plan["attempts"]}
    if any(digest(attempt) != plan["attempts"][aid] for aid, attempt in attempts.items()):
        raise ExtractError("continuation audit invocation identity changed")
    for point in review["audit_points"]:
        if (not isinstance(point, dict) or set(point) != {"stage", "attempt_id", "detail"} or
                not isinstance(point["stage"], str) or point["stage"] not in {"P1", "P2", "P3", "P4", "P5", "P6"} or
                point["attempt_id"] is not None and not isinstance(point["attempt_id"], str) or
                not isinstance(point["detail"], str) or not point["detail"].strip()):
            raise ExtractError("continuation audit points must identify the original stage, attempt and concern")
        if point["attempt_id"] is None:
            if point["stage"] != before["stage"]:
                raise ExtractError("unstarted audit point must identify the current stage frontier")
        elif point["attempt_id"] not in attempts or attempts[point["attempt_id"]]["stage"] != point["stage"]:
            raise ExtractError("continuation audit point stage differs from its registered invocation")


def _plan(plan, before, target):
    if (plan.get("format") != "schema_extract.executor_patch_plan.v1" or
            not plan.get("changes") or
            digest({k: v for k, v in plan.items() if k != "id"}) != plan.get("id") or
            digest(before) != plan["before_sha256"] or digest(target) != plan["target_lock_sha256"] or
            plan["run_id"] != before["run_id"] or plan["input_fingerprint"] != before["input_fingerprint"] or
            sorted(plan["attempts"]) != sorted(before["attempts"])):
        raise ExtractError("executor repair plan or original frontier changed")


def verify_bindings(store, state, original, *, evidence=None):
    """Verify the small append-only chain, without relabeling historical evidence."""
    from .runtime import _BOUNDARY, _executor, _key
    boundary = _BOUNDARY.get()
    shared = boundary is not None and boundary["key"] == _key(store, state)
    if evidence is None and shared and "executor_records" in boundary:
        return boundary["executor_records"]
    records, previous = [], original
    history = _history(state)
    for index, binding_id in enumerate(history):
        directory = _root(store, state["run_id"]) / "bindings" / binding_id
        objects = {name: read_json(directory / name) for name in
                   ("binding.json", "plan.json", "before.json", "target.lock.json", "authorization.json")}
        record, plan, before, target, review = (objects[name] for name in objects)
        if (record.get("format") != "schema_extract.executor_binding.v1" or record.get("id") != binding_id or
                digest({k: v for k, v in record.items() if k != "id"}) != binding_id or
                record.get("plan_id") != plan.get("id") or record.get("authorization_sha256") != digest(review)):
            raise ExtractError("executor binding evidence changed")
        _plan(plan, before, target)
        _authorization(store, plan, before, review)
        if (before["run_id"] != state["run_id"] or _history(before) != history[:index] or
                any(before[key] != state[key] for key in ("input_fingerprint", "snapshot", "source_identity", "config")) or
                state["attempts"][:len(before["attempts"])] != before["attempts"] or
                any(state["accepted_receipts"].get(key) != value for key, value in before["accepted_receipts"].items()) or
                before["executor"] != _executor(previous) or plan["changes"] != _changes(previous, target) or
                record["executor"] != _target(original, target)):
            raise ExtractError("executor binding does not extend the original run and prior executor")
        if evidence:
            for name, value in objects.items():
                evidence(directory / name, sha256=sha256(json_bytes(value)))
            evidence(Path(target["snapshot"]) / "executor.lock.json", sha256=sha256(json_bytes(target)))
            for name, checksum in target["rules"]["files"].items():
                evidence(rules_root(target) / name, sha256=checksum)
        records.append({"record": record, "plan": plan, "before": before, "target": target, "authorization": review})
        previous = target
    if evidence is None and shared:
        boundary["executor_records"] = records
    return records


def delivery_audit_markers(state, records):
    """Carry the user's deferred-audit decision in the normal delivery receipt."""
    markers = []
    for item in records:
        authorization = item["authorization"]
        before = item["before"]
        markers.append({"status": "pending_user_decision", "authority": "user_instruction",
            "scope": "extract_tools_only", "operator_id": state["operator_id"], "run_id": state["run_id"],
            "binding": item["record"]["id"], "issue_id": item["plan"]["issue_id"],
            "instruction": authorization["summary"], "audit_points": authorization["audit_points"],
            "retained_progress": {"checkpoint": before["checkpoint"],
                "stage_approvals": {key: value["approval"]["attempt_id"] for key, value in before["stages"].items() if value.get("approval")},
                "node_approvals": {key: value["approval"]["attempt_id"] for key, value in before["nodes"].items() if value.get("approval")}},
            "meaning": "Completed under existing premises after an extract-tool repair; these concerns are not claimed fixed. The user decides any additional independent audit or restart."})
    return markers


def verify_attempt_binding(store, state, original, attempt):
    records = verify_bindings(store, state, original)
    expected = state.get("execution_binding")
    for item in records:
        checksum = item["plan"]["attempts"].get(attempt["attempt_id"])
        if checksum is not None:
            if checksum != digest(attempt):
                raise ExtractError("historical attempt differs from its sealed executor identity")
            expected = item["before"].get("execution_binding")
            break
    if attempt.get("execution_binding") != expected:
        raise ExtractError("attempt belongs to another executor binding")
    return expected


def _control_basis(ledger, run_id):
    return {"hold": ledger.get("active_hold"), "control": ledger["controls"].get(run_id),
            "start": ledger["starts"].get(run_id), "maintenance": ledger["maintenance_requests"].get(run_id),
            "retries": {key: value for key, value in ledger["retries"].items() if value.get("run_id") == run_id},
            "commit": ledger.get("commits", {}).get(run_id)}


def _quiescent(store, state, ledger):
    from .maintenance import active, barrier
    from .monitor import _processes
    from .processes import process_alive
    if (state["published"] or state.get("exports") or state["status"] == "ready" or
            ledger.get("commits", {}).get(state["run_id"])):
        raise ExtractError("executor repair is limited to undelivered unfinished runs")
    if any(state.get(key) for key in ("active_attempts", "unconfirmed_attempts", "unsettled_attempts")):
        raise ExtractError("executor repair requires all invocations and accounting to be settled")
    if (ledger.get("active_hold") or state["control"].get("action") or
            ledger["controls"].get(state["run_id"], {}).get("action") in {"pause", "cancel"} or
            any(request["id"] not in state["control"]["requests"] for request in store.control_requests(state["run_id"])) or
            active(state) or barrier(store, state, ledger) or
            (state.get("maintenance") or {}).get("status") == "waiting" or
            any(value.get("run_id") == state["run_id"] and value.get("status") == "authorized"
                for value in ledger["retries"].values())):
        raise ExtractError("control, recovery or maintenance barrier prevents executor repair")
    launch = ledger["starts"].get(state["run_id"], {}).get("launch", {})
    identities = []
    if launch.get("status") in {"launching", "claimed", "termination_unknown"}:
        identities.append(launch.get("identity") or launch.get("launcher"))
    if state.get("runner"):
        identities.append(state["runner"].get("identity"))
    if identities:
        table, error = _processes()
        if any(process_alive(identity, table) is not False for identity in identities):
            raise ExtractError("old Runner or launch is live or unconfirmed" + (": " + error if error else ""))


def _saved_frontier(store, state):
    """Verify retained bytes against their original save, independent of completion."""
    from .artifacts import (_validate_scope, load_yaml, validate_artifact_sources,
                            SHARED_METADATA, _erase_fields)
    from .receipts import call_identity
    from .runtime import verify_execution
    lock = store.context(state)["rules_lock"]
    parser_root = rules_root(lock)
    template = load_yaml(parser_root / "schema_template", "starter.yaml", parser_root=parser_root)
    attempts = {}
    registered = {state["checkpoint"], state.get("merged_checkpoint"), state.get("recoverable_checkpoint")}
    for owner in [*state["stages"].values(), *state["nodes"].values()]:
        registered.update(owner.get(key) for key in ("draft_checkpoint", "last_draft_checkpoint", "baseline_checkpoint"))
        registered.update(item["checkpoint"] for item in owner.get("candidates", []))
        for key in ("approval", "last_approval"):
            if owner.get(key):
                registered.add(owner[key]["checkpoint"])
    for aid in state["attempts"]:
        attempt = read_json(store.run_dir(state["run_id"]) / "attempts" / aid / "attempt.json")
        verify_execution(store, state, attempt=attempt)
        attempts[aid] = attempt
        registered.add(attempt["base_checkpoint"])
        usage = state.get("attempt_usage", {}).get(aid, {})
        registered.add(usage.get("result_checkpoint"))
        registered.update(usage.get("checkpoints", []))
    registered.discard(None)
    verified, visiting = {}, set()

    def saved(checkpoint_id):
        if checkpoint_id in verified:
            return verified[checkpoint_id]
        if checkpoint_id in visiting:
            raise ExtractError("retained checkpoint provenance contains a cycle")
        visiting.add(checkpoint_id)
        directory, info = store.verify_checkpoint(state, checkpoint_id)
        parent = saved(info["parent"])[1] if info["parent"] else None
        producer = info.get("attempt_id")
        if producer is None:
            if parent is None:
                if (info["stage"], info["node"], info["files"]) != ("P1", "root", {
                        "root.yaml": sha256((parser_root / "schema_template/starter.yaml").read_bytes())}):
                    raise ExtractError("initial checkpoint differs from the frozen starter")
            else:
                if info["stage"] not in {"P1", "P2", "P5"}:
                    raise ExtractError("retained preparation has no supported save provenance")
                if info["stage"] == "P2" and info["files"] != {
                        name: checksum for name, checksum in parent["files"].items() if name in {"root.yaml", "dispatch.md"}}:
                    raise ExtractError("prepared selection checkpoint differs from its parent projection")
                own = info["node"] + ".yaml"
                for name, checksum in info["files"].items():
                    if parent["files"].get(name) == checksum:
                        continue
                    if info["stage"] == "P5" and name == own:
                        value = load_yaml(directory, own, parser_root=parser_root)
                        root = load_yaml(directory, "root.yaml", parser_root=parser_root)
                        if name not in parent["files"]:
                            expected = deepcopy(template)
                            expected.pop("CONTRACT")
                            expected["META_DATA"].update({key: root["META_DATA"][key] for key in SHARED_METADATA})
                            expected["META_DATA"].update(Version=1, Dispatch={"id": info["node"], "when": None})
                            if value != expected:
                                raise ExtractError("prepared node differs from its frozen starter")
                        else:
                            preparation = info.get("preparation", {})
                            old = load_yaml(store.verify_checkpoint(state, info["parent"])[0], own, parser_root=parser_root)
                            if (preparation.get("kind") != "shared_metadata_initialization" or
                                    preparation.get("original_checkpoint") != info["parent"] or
                                    preparation.get("root_sha256") != info["files"].get("root.yaml") or
                                    _erase_fields(old, ["META_DATA." + key for key in SHARED_METADATA]) !=
                                    _erase_fields(value, ["META_DATA." + key for key in SHARED_METADATA]) or
                                    value["META_DATA"]["Version"] != old["META_DATA"]["Version"] + 1 or
                                    any(value["META_DATA"].get(key) != root["META_DATA"].get(key) for key in SHARED_METADATA)):
                                raise ExtractError("prepared node changed beyond shared metadata initialization")
                        continue
                    if any(known["files"].get(name) == checksum for _, known in verified.values()):
                        continue
                    sources = [cid for cid in sorted(registered - visiting)
                               if store.verify_checkpoint(state, cid)[1]["files"].get(name) == checksum]
                    if not sources:
                        raise ExtractError("prepared checkpoint has no registered source for " + name)
                    saved(sources[0])
        else:
            attempt = attempts.get(producer)
            if attempt is None or any(info.get(key) != attempt.get(key) for key in
                    ("stage", "node", "execution_binding", "plan_revision")):
                raise ExtractError("retained checkpoint differs from its registered producer")
            base, baseline = saved(attempt["base_checkpoint"])
            if baseline["files"] != attempt["dependency_digests"]:
                raise ExtractError("retained producer baseline changed")
            usage = state.get("attempt_usage", {}).get(producer, {})
            evidence = usage.get("evidence", {})
            if evidence.get("attempt.json", digest(attempt)) != digest(attempt):
                raise ExtractError("retained producer differs from its recorded invocation")
            attempt_dir = Path(attempt["directory"])
            accepted = producer in state["accepted_receipts"]
            if accepted:
                receipt = read_json(attempt_dir / "receipt.json")
                record = read_json(attempt_dir / "validated.json")
                if (receipt["identity"] != call_identity(attempt) or
                        digest(receipt) != state["accepted_receipts"][producer] or
                        record.get("receipt_sha256") != digest(receipt)):
                    raise ExtractError("retained checkpoint receipt identity changed")
                record_name = "validated.json"
                validation = record
            else:
                if usage.get("result_checkpoint") != checkpoint_id:
                    raise ExtractError("unapproved draft lacks its recorded result checkpoint")
                record_name = next((name for name in ("repair.json", "recovery.json", "validated.json")
                                    if name in evidence), None)
                if record_name is None:
                    raise ExtractError("unapproved draft has no registered save evidence")
                record = read_json(regular_file(attempt_dir, record_name))
                validation = record if record_name == "validated.json" else record.get("validation")
                if record_name == "recovery.json" and (record.get("termination", {}).get("confirmed") is not True or
                                                       record.get("discarded_draft")):
                    raise ExtractError("interrupted draft lacks confirmed safe recovery")
            if ((record_name in evidence and evidence[record_name] != digest(record)) or
                    not accepted and evidence.get(record_name) != digest(record) or
                    not validation or validation.get("artifact_digests") != info["files"]):
                raise ExtractError("retained checkpoint differs from its recorded save evidence")
            edited = attempt["role"] == "worker" or attempt.get("mode") == "adjudicate"
            if edited:
                if info["parent"] != attempt["base_checkpoint"]:
                    raise ExtractError("saved draft has a different parent from its invocation")
                old = {Path(name).stem: load_yaml(base, name, parser_root=parser_root)
                       for name in baseline["files"] if name.endswith(".yaml")}
                new = {Path(name).stem: load_yaml(directory, name, parser_root=parser_root)
                       for name in info["files"] if name.endswith(".yaml")}
                synchronization = validation.get("shared_metadata_sync", {}).get("nodes", [])
                entry = next((item for item in attempt.get("selected_designs") or [] if item["id"] == attempt["node"]), None)
                materials = attempt["materials"]
                parameter_path = regular_file(attempt_dir, "materials/check.json")
                parameters = read_json(parameter_path)
                if (str(parameter_path) != materials["check"] or
                        sha256(parameter_path.read_bytes()) != materials["files"]["check.json"]["sha256"] or
                        parameters.get("attempt_id") != producer or
                        any(parameters.get(key) != attempt[key] for key in ("stage", "role", "node"))):
                    raise ExtractError("saved draft's original scope parameters changed")
                checked = _validate_scope(attempt["stage"], "worker", baseline["files"], info["files"], old, new, template,
                    node_entry=entry, revision_scope=attempt["revision_scope"],
                    first_draft=parameters["first_draft"], maintenance_fields=parameters["maintenance_fields"],
                    synchronized=synchronization)
                if checked["changed_files"] != validation["changed_files"]:
                    raise ExtractError("saved draft changed-file evidence differs from its bytes")
            elif (not accepted or receipt["decision"] != "approve" or validation["check"]["exit_code"] != 0 or
                  validation.get("original_artifact_digests") != attempt["dependency_digests"]):
                raise ExtractError("reviewer checkpoint lacks its read-only approved merge")
            else:
                expected = dict(parent["files"])
                adopted = {attempt["node"] + ".yaml"} if attempt["stage"] == "P5" else {"root.yaml", "dispatch.md"}
                if attempt["stage"] == "P1":
                    adopted.update(baseline["files"])
                expected.update({name: checksum for name, checksum in baseline["files"].items() if name in adopted})
                selected = receipt["plan"] if attempt["stage"] == "P2" else attempt.get("selected_designs")
                if selected is not None:
                    expected = {name: checksum for name, checksum in expected.items()
                                if name == "dispatch.md" or name.removesuffix(".yaml") in {item["id"] for item in selected}}
                if expected != info["files"]:
                    raise ExtractError("reviewer merge differs from the actual reviewed bytes")
            validate_artifact_sources(directory, lock, decision="continue", parser_root=parser_root)
        visiting.remove(checkpoint_id)
        verified[checkpoint_id] = directory, info
        return directory, info

    for checkpoint_id in {state["checkpoint"], state.get("merged_checkpoint"),
                          *(node.get("draft_checkpoint") for node in state["nodes"].values())} - {None}:
        saved(checkpoint_id)
    return verified


@execution_boundary
def candidate_drafts(store, state):
    """Restore every stale node pointer from the verified current P6 save chain."""
    frontier = _saved_frontier(store, state)
    if state["stage"] != "P6":
        return {}
    _, current = frontier[state["checkpoint"]]
    result = {}
    for entry in state["selected_designs"]:
        node_id = entry["id"]
        if node_id == "root":
            continue
        node = state["nodes"][node_id]
        filename = node["file"]
        _, retained = frontier[node["draft_checkpoint"]]
        if current["files"].get(filename) == retained["files"].get(filename):
            continue
        checkpoint = current
        while checkpoint["files"].get(filename) == current["files"].get(filename):
            if not checkpoint.get("parent"):
                raise ExtractError("P6 draft has no original scoped save")
            _, parent = frontier[checkpoint["parent"]]
            if parent["files"].get(filename) != current["files"].get(filename):
                break
            checkpoint = parent
        if checkpoint["stage"] != "P6" or not checkpoint.get("attempt_id"):
            raise ExtractError("P6 draft does not originate in a verified final edit")
        result[node_id] = {"before": node.get("draft_checkpoint"), "after": current["id"],
                           "sha256": current["files"][filename], "producer": checkpoint["attempt_id"]}
    return result


@execution_boundary
def probe(store, state):
    """Read-only target preflight; incomplete material remains the original responsibility."""
    from .artifacts import check_stage_snapshot
    from .runtime import verify_execution
    original = verify_execution(store, state)
    drafts = candidate_drafts(store, state)
    for approval in [*(state["stages"][key].get("approval") for key in ("P1", "P2", "P3", "P4")),
                     *(node.get("approval") for node in state["nodes"].values())]:
        if approval:
            store.verify_approval_origin(state, approval)
    directory, _ = store.verify_checkpoint(state)
    checked = check_stage_snapshot(directory, parser_root=rules_root(original), stage=state["stage"],
                                   selected_designs=state["selected_designs"], node=state["node"])
    if checked["exit_code"] == 2:
        raise ExtractError("target preflight could not run: " + "; ".join(checked["diagnostics"]))
    return {"checkpoint": state["checkpoint"], "drafts": drafts,
            "retained_drafts": {node_id: node["draft_checkpoint"] for node_id, node in state["nodes"].items()
                                if node.get("draft_checkpoint")},
            "checked_nodes": checked["checked_nodes"], "diagnostics": checked["diagnostics"],
            "completion_exit_code": checked["exit_code"],
            "meaning": "saved provenance verified; completion diagnostics retain the original stage responsibility, not approval"}


def _seal_objects(destination, objects):
    temporary = destination.parent / (".building-" + uuid4().hex)
    temporary.mkdir(parents=True)
    try:
        for name, value in objects.items():
            write_json(temporary / name, value)
        seal(temporary)
        sync_tree(temporary)
        os.rename(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            for path in temporary.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)
            temporary.chmod(0o755)
            shutil.rmtree(temporary)


def prepare(store, run_id, *, tree, issue_id, reason):
    from .runtime import executor_lock, verify_execution
    if not all(isinstance(value, str) and value.strip() for value in (issue_id, reason)):
        raise ExtractError("executor repair needs a stable issue ID and concrete reason")
    with store.writer(run_id):
        state = store.load(run_id)
        original = verify_execution(store, state)
        _quiescent(store, state, store.read_supervisor())
        target = freeze_executor_patch(store.root, original, tree)
        _target(original, target)
        changes = _changes(executor_lock(store, state, lock=original), target)
        if not changes:
            raise ExtractError("executor repair has no code changes")
        command, cwd, env = frozen_invocation(target, ["repair-runtime", "probe", run_id, "--state-root", str(store.root)])
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
        if result.returncode:
            raise ExtractError("target executor preflight failed: " + (result.stderr or result.stdout)[-4000:])
        import json
        check = json.loads(result.stdout)
        plan = {"format": "schema_extract.executor_patch_plan.v1", "run_id": run_id,
                "input_fingerprint": state["input_fingerprint"], "before_sha256": digest(state),
                "target_lock_sha256": digest(target), "changes": changes, "issue_id": issue_id.strip(),
                "reason": reason.strip(), "preflight": check,
                "attempts": {aid: digest(read_json(store.run_dir(run_id) / "attempts" / aid / "attempt.json"))
                             for aid in state["attempts"]}}
        plan["id"] = digest(plan)
        destination = _root(store, run_id) / "plans" / plan["id"]
        objects = {"plan.json": plan, "before.json": state, "target.lock.json": target}
        if not destination.exists():
            _seal_objects(destination, objects)
        elif any(read_json(destination / name) != value for name, value in objects.items()):
            raise ExtractError("executor repair preparation identity collision")
        return {"run_id": run_id, "plan": plan, "directory": str(destination), "started": False, "applied": False}


def apply(store, run_id, *, plan_id, review):
    from .runtime import verify_execution
    with store.writer(run_id):
        state = store.load(run_id)
        original = verify_execution(store, state)
        for item in verify_bindings(store, state, original):
            if item["plan"]["id"] == plan_id:
                if item["record"]["authorization_sha256"] != digest(review):
                    raise ExtractError("executor repair is already bound to a different review")
                return {"run_id": run_id, "binding": item["record"], "reused": True, "started": False}
        directory = _root(store, run_id) / "plans" / _id(plan_id)
        plan, before, target = (read_json(directory / name) for name in ("plan.json", "before.json", "target.lock.json"))
        _plan(plan, before, target)
        _authorization(store, plan, before, review)
        verify_executor(target)
        executor = _target(original, target)
        if state != before:
            raise ExtractError("run changed after repair preparation; prepare against the current frontier")
        observed = store.read_supervisor()
        _quiescent(store, state, observed)
        if probe(store, state) != plan["preflight"]:
            raise ExtractError("retained frontier changed after preparation")
        for aid, checksum in plan["attempts"].items():
            if digest(read_json(store.run_dir(run_id) / "attempts" / aid / "attempt.json")) != checksum:
                raise ExtractError("historical invocation changed after repair preparation")
        record = {"format": "schema_extract.executor_binding.v1", "plan_id": plan_id,
                  "authorization_sha256": digest(review), "executor": executor, "registered_at": time.time(),
                  "session_policy": "new executor operation scope; old sessions remain historical"}
        record["id"] = digest(record)
        destination = _root(store, run_id) / "bindings" / record["id"]
        _seal_objects(destination, {"binding.json": record, "plan.json": plan, "before.json": before,
                                    "target.lock.json": target, "authorization.json": review})
        updated = deepcopy(state)
        updated.update(execution_binding=record["id"], execution_bindings=[*_history(state), record["id"]], executor=executor)
        for node_id, entry in plan["preflight"]["drafts"].items():
            node = updated["nodes"][node_id]
            node["last_draft_checkpoint"] = node.get("draft_checkpoint") or node.get("last_draft_checkpoint")
            node["draft_checkpoint"] = entry["after"]
        verify_execution(store, updated)
        with store.control_transaction() as ledger:
            if (_control_basis(ledger, run_id) != _control_basis(observed, run_id) or store.load(run_id) != state):
                raise ExtractError("run or control authority changed before executor repair commit")
            _quiescent(store, state, ledger)
            store.save(updated)
        return {"run_id": run_id, "binding": record, "reused": False, "started": False}
