"""Explicit, same-source successors; old approval and execution authority stay old."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import time
from uuid import uuid4

from .artifacts import _same, load_yaml, validate_artifact_sources, write_yaml
from .budget import RECOVERY as BUDGET_RECOVERY, addition, call_budget, extend
from .common import (ExtractError, digest, read_json, rules_root, seal, source_root,
                     sync_tree, verify_manifest, write_json)
from .inputs import SourceTree
from .monitor import _alive, _processes
from .snapshots import freeze_inputs, verify_run_inputs, verify_snapshot
from .stages import call_view, scope_for


def _quiescent(store, old: dict, ledger: dict) -> None:
    if old.get("active_attempts") or old.get("unconfirmed_attempts") or old.get("unsettled_attempts"):
        raise ExtractError("continue requires all registered calls and accounting to be settled by the old frozen executor")
    launch = ledger["starts"].get(old["run_id"], {}).get("launch", {})
    if launch.get("status") in {"launching", "claimed", "termination_unknown"}:
        table, error = _processes()
        if _alive(launch.get("identity") or launch.get("launcher"), table) is not False:
            raise ExtractError("old launch is live or unconfirmed, including its pre-exec window" + (": " + error if error else ""))


def _scope(previous: dict) -> dict:
    # Copy every field, including dynamically persisted reservation balances and
    # raw one-shot decisions. Only current-generation artifact authority resets.
    value = deepcopy(previous)
    value.update(status="pending", approval=None, baseline_checkpoint=None, candidates=[],
                 generation=value.get("generation", 0) + 1)
    return value


def _inherit(store, state: dict, old: dict, old_lock: dict, new_lock: dict,
             artifacts: Path, checkpoint: dict, relation: dict) -> dict:
    directory = store.run_dir(state["run_id"])
    history = directory / "continued"
    if not history.exists():
        temporary = directory / (".continued-" + uuid4().hex)
        shutil.copytree(artifacts, temporary / "artifacts")
        verify_manifest(temporary / "artifacts", checkpoint["files"])
        write_json(temporary / "state.json", old)
        write_json(temporary / "inputs.lock.json", old_lock)
        write_json(temporary / "checkpoint.json", checkpoint)
        seal(temporary)
        sync_tree(temporary)
        temporary.rename(history)
    else:
        if read_json(history / "state.json") != old or read_json(history / "checkpoint.json") != checkpoint:
            raise ExtractError("successor's inherited evidence differs from its reserved predecessor")
        verify_manifest(history / "artifacts", checkpoint["files"])
    initial = directory / (".draft-" + uuid4().hex)
    initial.mkdir()
    try:
        root = load_yaml(artifacts, "root.yaml", parser_root=rules_root(new_lock))
        starter_path = rules_root(new_lock) / "schema_template"
        starter = load_yaml(starter_path, "starter.yaml", parser_root=rules_root(new_lock))
        old_starter = load_yaml(rules_root(old_lock) / "schema_template", "starter.yaml", parser_root=rules_root(old_lock))
        if _same(root, old_starter):
            shutil.copyfile(starter_path / "starter.yaml", initial / "root.yaml")
        else:
            if not _same(root["contract"].get("hardware_specification"), starter["contract"]["hardware_specification"]):
                root["contract"]["hardware_specification"] = deepcopy(starter["contract"]["hardware_specification"])
                root["meta_data"]["version"] += 1
            write_yaml(initial / "root.yaml", root, source=artifacts / "root.yaml", parser_root=rules_root(new_lock))
            validate_artifact_sources(initial, new_lock, decision="continue")
        state["stages"] = {name: _scope(scope) for name, scope in old["stages"].items()}
        state["nodes"] = deepcopy(old["nodes"])
        for node_id, node in state["nodes"].items():
            node.update(status="draft" if node_id == "root" else "removed", merged=False,
                        draft_checkpoint=None, approved_checkpoints={}, approval=None,
                        approved_sha256=None, approved_generation=None, generation=node.get("generation", 0) + 1,
                        last_draft_checkpoint=None, last_approval=None, authored=False,
                        role="worker", mode="normal", feedback=[], reason=None)
            if "stage" in node:
                node["stage"] = _scope(node["stage"])
        for name in ("deadline", "p6_starts", "reopen_requests", "plan_history", "attempt_usage", "recovered_attempts", "stale_results", "budget_extensions"):
            state[name] = deepcopy(old.get(name, state.get(name)))
        for attempt_id, usage in state["attempt_usage"].items():
            usage.setdefault("origin_run_id", old.get("continuation", {}).get("attempt_origins", {}).get(attempt_id, old["run_id"]))
        state.update(status="queued", stage="P1", node="root", role="worker", mode="normal",
                     plan=None, plan_draft=None, merged_checkpoint=None, plan_revision=old["plan_revision"] + 1,
                     final_review_started=False, revision_scope=[], pending_rechecks=[], reopening=None,
                     reason=None, feedback=[{"detail": "从固定旧稿按本次冻结规则独立重审 P1；旧批准、Design 和索引只作阅读线索。",
                                             "evidence": str(history)}])
        state["continuation"] = {"predecessor": old["run_id"],
                                 "root_run_id": old.get("continuation", {}).get("root_run_id", old["run_id"]),
                                 "checkpoint": checkpoint["id"], "old_input_fingerprint": old["input_fingerprint"],
                                 "old_executor": old.get("executor"), "original_created_at": old.get("continuation", {}).get("original_created_at", old["created_at"]),
                                 "evidence": str(history), "approval_inherited": False,
                                 "reserved_at": relation["created_at"], "predecessor_state_sha256": relation["state_sha256"],
                                 "attempt_origins": {key: value["origin_run_id"] for key, value in state["attempt_usage"].items()}}
        state["checkpoint"] = store.checkpoint(state, initial, attempt_id=None, stage="P1", node="root")
        state["nodes"]["root"]["draft_checkpoint"] = state["checkpoint"]
    finally:
        shutil.rmtree(initial)
    extend(state, relation.get("budget_addition", {}), identity=relation["run_id"], origin="continue:" + old["run_id"])
    # Validate inherited allowances plus the explicit grant. A fresh review
    # never renews one-shot decisions.
    calls = [{"stage": name, "node": "root", "role": "reviewer" if name == "P6" else "worker", "mode": "normal"}
             for name in ("P1", "P2", "P3", "P4", "P6")]
    calls += [{"stage": "P5", "node": item["id"], "role": "worker", "mode": "normal"}
              for item in old.get("plan") or [] if item["id"] != "root"]
    exhausted = []
    for call in calls:
        view = call_view(state, call)
        scope = scope_for(view)
        _, problem = call_budget(view)
        if problem:
            exhausted.append(call["stage"] + (":" + call["node"] if call["stage"] == "P5" else ""))
        previous = old["nodes"][call["node"]] if call["stage"] == "P5" else old if old["stage"] == call["stage"] else {}
        if previous.get("mode") in {"adjudicate", "signoff"} and (scope["adjudications"] or scope["signoffs"]):
            exhausted.append(call["stage"] + ":single_decision_used")
    if exhausted:
        state["status"] = "blocked"
        state["reason"] = {"code": "continuation_budget_exhausted", "scopes": sorted(set(exhausted)),
                           "summary": "截止、调用余额或一次性决定不足以完成必要重审；接续不重置已用计数。",
                           "recovery_condition": BUDGET_RECOVERY.format(run_id=state["run_id"])}
    store.save(state)
    return state


def continue_run(repo: Path, store, old_id: str, checkpoint_id: str, *, add_seconds=0, add_calls=0) -> dict:
    budget_addition = addition(add_seconds, add_calls)
    with store.writer(old_id):
        store.assert_constructed(old_id)
        old = store.load(old_id)
        old_lock = read_json(store.run_dir(old_id) / "inputs.lock.json")
        verify_run_inputs(old, old_lock)
        artifacts, checkpoint = store.verify_checkpoint(old, checkpoint_id)
        state_digest = digest(old)
        # Read process evidence outside the control lock, then reserve against
        # concurrent unified launch intents while already owning the old writer.
        observed_ledger = store.read_supervisor()
        _quiescent(store, old, observed_ledger)
        with store.supervisor() as ledger:
            existing = ledger["continuations"].get(old_id)
            if existing:
                if existing["checkpoint"] != checkpoint_id:
                    raise ExtractError("old run already has a successor for another checkpoint")
                if existing.get("budget_addition", addition()) != budget_addition:
                    raise ExtractError("successor already has a different budget request; use resume " + existing["run_id"] + " --add-seconds N / --add-calls N")
                relation = deepcopy(existing)
            else:
                # A new launch could have been reserved after the observation.
                launch = ledger["starts"].get(old_id, {}).get("launch", {})
                observed = observed_ledger["starts"].get(old_id, {}).get("launch", {})
                if launch != observed:
                    raise ExtractError("old launch changed before continuation reservation")
                relation = {"run_id": str(uuid4()), "predecessor": old_id, "checkpoint": checkpoint_id,
                            "state_sha256": state_digest, "status": "preparing", "created_at": time.time(),
                            "budget_addition": budget_addition}
                ledger["continuations"][old_id] = relation
                relation = deepcopy(relation)
        if relation["status"] == "ready":
            return store.load(relation["run_id"])
        if relation["state_sha256"] != state_digest:
            raise ExtractError("old state changed after its successor reservation; retain both records for diagnosis")
        if relation.get("snapshot"):
            new_lock = read_json(Path(relation["snapshot"]) / "inputs.lock.json")
        else:
            aliases = [*old_lock["source"].get("aliases", []),
                       {key: old_lock["source"][key] for key in ("repository_id", "ref", "files")}]
            aliases = list({digest(alias): alias for alias in aliases}.values())
            new_lock = freeze_inputs(store.root, SourceTree(source_root(old_lock)), old_lock["source"]["bindings"],
                                     sorted(old_lock["source"]["files"]), old["config"],
                                     rules_repo=rules_root(), source_aliases=aliases)
            with store.supervisor() as ledger:
                ledger["continuations"][old_id].update(snapshot=new_lock["snapshot"], input_fingerprint=new_lock["fingerprint"])
        if new_lock["source"]["files"] != old_lock["source"]["files"] or new_lock["config"] != old["config"]:
            raise ExtractError("continuation must preserve source bytes and the full original budget configuration")
        verify_snapshot(new_lock)
        directory = store.run_dir(relation["run_id"])
        directory.mkdir(parents=True, exist_ok=True)
        with store.writer(relation["run_id"]):
            if (directory / "state.json").exists():
                state = store.load(relation["run_id"])
                verify_run_inputs(state, new_lock)
            else:
                state = store.create(new_lock, Path(old["output"]), queued=True, run_id=relation["run_id"])
            if not state.get("continuation"):
                state = _inherit(store, state, old, old_lock, new_lock, artifacts, checkpoint, relation)
            if store.load(old_id) != old:
                raise ExtractError("old state changed while constructing successor")
            with store.supervisor() as ledger:
                ledger["continuations"][old_id].update(status="ready", completed_at=time.time())
                ledger["starts"][state["run_id"]].update(authorized=False, predecessor=old_id)
        return state
