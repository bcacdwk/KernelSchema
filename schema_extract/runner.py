"""Finite runner: staged extraction, independent review, bounded retry and export."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
from uuid import uuid4

from .codex_cli import CodexCLI, terminate_owned, write_json
from .protocol import RECEIPT, prompt_for, validate_receipt
from .records import (Store, StateError, artifact_manifest, clone_artifacts, digest,
                      freeze_inputs, hardware_issue, issue_disposition, read_json, run_check,
                      schema_parts, semantic_digests, tree_manifest)


DEFAULTS = {"model": "gpt-6-astra", "effort": "xhigh", "jobs": 2}


def new_task(store, baseline, output):
    entry = baseline["identity"]["entry"]
    key = entry["id"] + "-" + baseline["fingerprint"][:12]
    directory = store.task_dir(key)
    if (directory / "task.json").exists():
        state = store.load(key)
        if state["output"] != str(output):
            raise StateError("same inputs already have a different explicit output; use export --to")
        return state
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "inputs.lock.json", baseline)
    state = {"format": 1, "key": key, "operator_id": entry["id"],
             "fingerprint": baseline["fingerprint"], "output": str(output), "status": "queued",
             "phase": "worker", "stage": "contract", "node": "root", "review_scope": "contract",
             "checkpoint": None, "history": [], "attempts": [], "active_attempt": None,
             "active_process": None, "dispatches": [], "reviews": {}, "issues": [],
             "check": None, "eligible": False, "result": None, "reason": None,
             "usage": {"calls": 0, "reported_tokens": 0, "elapsed_seconds": 0, "unknown_usage_calls": 0},
             "observed_thread_usage": {}, "usage_observations": {},
             "retry_history": [], "created_at": time.time()}
    store.save(state)
    return state


def seed_inputs(store, old_run, operator):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", old_run):
        raise StateError("invalid seed run ID")
    plan = read_json(store.root / "runs" / old_run / "plan.json")
    matches = [store.load(key) for key in plan["tasks"] if store.load(key)["operator_id"] == operator]
    if len(matches) != 1:
        raise StateError(f"seed run must contain exactly one task for {operator}")
    old = matches[0]
    if old["status"] not in {"ended", "stopped"} or old.get("active_process"):
        raise StateError("seed task is not stopped; confirm its invocation termination first")
    for attempt in old["attempts"]:
        for call in attempt["calls"]:
            result_path = Path(call) / "result.json"
            if not result_path.is_file() or not read_json(result_path).get("termination", {}).get("confirmed"):
                raise StateError("seed invocation termination is not confirmed")
    baseline = store.verify_baseline(old, verify_runtime=False)
    checkpoint = store.verify_checkpoint(old)
    if not checkpoint:
        raise StateError("seed task has no accepted draft checkpoint")
    binding = {"repository_id": baseline["repository_id"], "commit": baseline["snapshot_commit"],
               "paths": sorted(name for name in baseline["files"] if not name.startswith("schema_extract/"))}
    bindings = deepcopy(old.get("source_bindings", []))
    if binding not in bindings:
        bindings.append(binding)
    seed = {"run_id": old_run, "task_key": old["key"], "fingerprint": old["fingerprint"],
            "checkpoint": old["checkpoint"], "files": artifact_manifest(checkpoint / "artifacts"),
            "usage": deepcopy(old["usage"]), "source_bindings": bindings}
    return old, baseline, seed


def verify_seed_inputs(old_baseline, new_baseline):
    if old_baseline["identity"]["entry"] != new_baseline["identity"]["entry"]:
        raise StateError("seed operator selection/metadata changed; old draft cannot be imported")
    for name, expected in old_baseline["files"].items():
        if name.startswith("schema_extract/") and name.endswith(".py"):
            continue  # New execution code/protocol is explicit in the new plan.
        if new_baseline["files"].get(name) != expected:
            raise StateError(f"seed source/spec changed or removed: {name}")
    if old_baseline["identity"]["check_policy"] != new_baseline["identity"]["check_policy"]:
        raise StateError("seed checker policy changed")


def make_plan(store, ids, work_root, output_root, extras, config, run_id=None, *, seed_run=None):
    config = {key: config.get(key, value) for key, value in DEFAULTS.items()}
    if not ids or len(set(ids)) != len(ids):
        raise StateError("select one or more distinct operator IDs explicitly; no default all")
    if not 1 <= config["jobs"] <= 2 or config["effort"] != "xhigh":
        raise StateError("jobs must be 1 or 2 and effort must be xhigh")
    work_root, output_root = work_root.resolve(), output_root.resolve()
    if work_root.is_relative_to(store.repo) or store.root.is_relative_to(work_root):
        raise StateError("--work-root must be outside the repository and recorded snapshots")
    if output_root == store.repo or output_root.is_relative_to(store.root) or output_root.is_relative_to(work_root):
        raise StateError("output root must be separate from repository root, records and work")
    catalog = read_json(store.repo / "kernel_wiki/catalog.json")
    entries = {entry["id"]: entry for entry in catalog["operators"]}
    missing = set(ids) - entries.keys()
    if missing:
        raise StateError(f"unknown operator IDs: {sorted(missing)}")
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise StateError("invalid run ID")
    directory = store.root / "runs" / run_id
    if directory.exists():
        raise StateError("run ID already exists")
    tasks = []
    for operator in ids:
        old, old_baseline, seed = seed_inputs(store, seed_run, operator) if seed_run else (None, None, None)
        baseline = freeze_inputs(store, entries[operator], extras, config, seed=seed)
        if old is not None:
            verify_seed_inputs(old_baseline, baseline)
        state = new_task(store, baseline, output_root / operator)
        if old is not None:
            state = store.seed_checkpoint(state, old, seed)
        tasks.append(state["key"])
    directory.mkdir(parents=True)
    plan = {"format": 1, "run_id": run_id, "tasks": tasks, "config": config,
            "work_root": str(work_root), "output_root": str(output_root), "created_at": time.time()}
    if seed_run:
        plan["seed_run"] = seed_run
    write_json(directory / "plan.json", plan)
    return plan


def merge_issues(state, receipt, *, reviewer=False):
    by_id = {issue["id"]: issue for issue in state["issues"]}
    for issue in receipt["issues"]:
        previous = by_id.get(issue["id"], {})
        disposition = issue_disposition(issue)
        by_id[issue["id"]] = {**previous, **issue, "state": "ignored" if disposition == "ignored" else "open",
                              "disposition": disposition, "checkpoint": state.get("checkpoint"),
                              "first_checkpoint": previous.get("first_checkpoint", state.get("checkpoint"))}
    if reviewer:
        for issue_id in receipt["resolved_issue_ids"]:
            if issue_id in by_id and issue_id not in {item["id"] for item in receipt["issues"]}:
                by_id[issue_id]["state"] = "resolved"
                by_id[issue_id]["closed_checkpoint"] = state.get("checkpoint")
    for issue in by_id.values():
        if hardware_issue(issue):
            issue.update(state="ignored", disposition="ignored")
    state["issues"] = list(by_id.values())


def observe_usage(state, result, call_id):
    """Observe cumulative CLI usage by thread. Never control task progress."""
    observations = state.setdefault("usage_observations", {})
    if call_id in observations:
        return observations[call_id]
    raw = deepcopy(result.get("usage", {}))
    thread = result.get("thread_id")
    totals = state.setdefault("observed_thread_usage", {})
    valid_thread = isinstance(thread, str) and bool(thread)
    previous = totals.get(thread) if valid_thread else None
    problem, delta = None, None
    if not valid_thread or not isinstance(raw, dict) or any(type(raw.get(key)) is not int or raw[key] < 0 for key in ("input_tokens", "output_tokens")):
        problem = "usage_missing_or_invalid"
    elif ((previous is None and result.get("resume_thread_id")) or
          (previous is not None and (not isinstance(previous, dict) or
           any(type(previous.get(key)) is not int for key in ("input_tokens", "output_tokens"))))):
        problem = "resumed_thread_baseline_unknown"
        totals[thread] = raw
    elif previous and any(raw[key] < previous[key] for key in ("input_tokens", "output_tokens")):
        problem = "cumulative_usage_decreased"
    else:
        delta = sum(raw[key] - (previous or {}).get(key, 0) for key in ("input_tokens", "output_tokens"))
        totals[thread] = raw
        state["usage"]["reported_tokens"] += delta
    if problem:
        state["usage"]["unknown_usage_calls"] += 1
    observation = {"thread_id": thread, "raw_usage": raw, "delta_tokens": delta,
                   "anomaly": problem, "scope": "thread_cumulative", "observational_only": True}
    observations[call_id] = observation
    return observation


def finish(store, state, reason, *, stopped=False):
    state["status"] = "stopped" if stopped else "ended"
    state["reason"] = reason
    state["result"] = "draft_with_gaps" if reason == "evidence_gaps" else reason
    state["active_process"] = None
    state["active_attempt"] = None
    if reason != "ready":
        state["eligible"] = False
    store.save(state)
    return state


def bound_review(state, baseline, receipt, checkpoint):
    semantics = read_json(checkpoint / "checkpoint.json")["semantics"]
    return {"fingerprint": state["fingerprint"], "checkpoint": state["checkpoint"],
            "contract": semantics["contract"], "layout": semantics["layout"],
            "nodes": semantics["nodes"], "receipt": receipt,
            "complete": not any(issue_disposition(i) in {"fix", "needs_input"} for i in receipt["issues"]),
            "reviewed_at": time.time()}


def eligible(store, state):
    baseline = store.verify_baseline(state)
    checkpoint = store.verify_checkpoint(state)
    if not checkpoint or state["check"]["exit_code"] != 0:
        return False
    if any(issue["state"] == "open" and issue_disposition(issue) in {"fix", "needs_input"} for issue in state["issues"]):
        return False
    semantics = read_json(checkpoint / "checkpoint.json")["semantics"]
    expected = {"contract", "final"}
    if state["dispatches"]:
        expected |= set(semantics["nodes"])
    for scope in expected:
        review = state["reviews"].get(scope)
        if not review or review["fingerprint"] != state["fingerprint"] or review["contract"] != semantics["contract"]:
            return False
        if not review.get("complete"):
            return False
        if scope == "final" and review["checkpoint"] != state["checkpoint"]:
            return False
        if scope not in {"contract", "final"} and review["nodes"].get(scope) != semantics["nodes"].get(scope):
            return False
    files = artifact_manifest(checkpoint / "artifacts")
    return "dispatch.md" in files and {p[:-5] for p in files if p.endswith(".yaml")} == {"root", *(v["id"] for v in state["dispatches"])}


def validate_source_bindings(baseline, artifacts, bindings=()):
    prefixes = [(baseline["repository_id"] + "@" + baseline["snapshot_commit"] + ":", baseline["files"])]
    prefixes.extend((binding["repository_id"] + "@" + binding["commit"] + ":", binding["paths"]) for binding in bindings)
    for name, part in schema_parts(artifacts, baseline["snapshot"]).items():
        source = part.get("meta_data", {}).get("source", "")
        matches = [(prefix, paths) for prefix, paths in prefixes if isinstance(source, str) and source.startswith(prefix)]
        if not matches:
            raise StateError(f"{name}: meta_data.source is not bound to the fixed snapshot commit")
        if not any(source[len(prefix):].split("::", 1)[0] in paths for prefix, paths in matches):
            raise StateError(f"{name}: meta_data.source does not identify an actual fixed file")


class Runner:
    def __init__(self, store, plan, cli=None):
        self.store, self.plan = store, plan
        self.config = plan["config"]
        self.cli = cli or CodexCLI(model=self.config["model"])
        self.control = store.root / "runs" / plan["run_id"] / "control.json"
        self.cancel_event = threading.Event()
        self.review_slot = threading.Semaphore(1)

    def control_mode(self):
        if self.cancel_event.is_set():
            return "cancel"
        try:
            return read_json(self.control).get("mode")
        except FileNotFoundError:
            return None

    def recover(self, state):
        active = state.get("active_process")
        if not active and state.get("active_attempt"):
            attempt = next(v for v in state["attempts"] if v["id"] == state["active_attempt"])
            if attempt["calls"]:
                records = Path(attempt["calls"][-1])
                result = read_json(records / "result.json") if (records / "result.json").exists() else None
                if not result or not result.get("termination", {}).get("confirmed"):
                    if not (records / "process.json").exists():
                        raise StateError("old invocation identity unavailable; cannot confirm termination or resume")
                    active = read_json(records / "process.json") | {"records": str(records), "started_at": attempt["created_at"]}
        if active:
            process_file = Path(active["records"]) / "process.json"
            latest = read_json(process_file) if process_file.exists() else active
            result = terminate_owned(latest["identity"], latest.get("descendants", []))
            write_json(Path(active["records"]) / "recovery_termination.json", result)
            if not result["confirmed"]:
                raise StateError("old process termination unconfirmed; isolate and do not retry")
            state["usage"]["elapsed_seconds"] += max(0, time.time() - active.get("started_at", time.time()))
            state["usage"]["unknown_usage_calls"] += 1
        self.store.verify_baseline(state)
        self.store.verify_checkpoint(state)
        state["active_process"] = None
        state["active_attempt"] = None
        state["status"] = "queued"
        self.store.save(state)

    def work(self, key):
        state = self.store.load(key)
        if state["status"] != "queued":
            return state
        attempt, work, thread_id = None, None, None
        while state["status"] in {"queued", "running"}:
            mode = self.control_mode()
            if mode:
                return finish(self.store, state, mode, stopped=True)
            baseline = self.store.verify_baseline(state)
            role = state["phase"]
            if attempt is None:
                attempt, work = self.store.new_attempt(state, Path(self.plan["work_root"]), role)
            before = tree_manifest(work)
            records = self.store.task_dir(key) / "attempts" / attempt["id"] / f"call-{len(attempt['calls']) + 1:02d}"
            records.parent.mkdir(parents=True, exist_ok=True)
            state["usage"]["calls"] += 1
            attempt["calls"].append(str(records))
            self.store.save(state)

            def started(process):
                state["active_process"] = {**process, "started_at": time.time()}
                self.store.save(state)

            if role == "reviewer":
                self.review_slot.acquire()
            try:
                if self.control_mode():
                    return finish(self.store, state, self.control_mode(), stopped=True)
                result = self.cli.run(cwd=work, records=records, prompt=prompt_for(state, baseline, work, role),
                                      schema=RECEIPT, role=role, thread_id=thread_id,
                                      cancel=lambda: self.control_mode() == "cancel", on_process=started)
            finally:
                if role == "reviewer":
                    self.review_slot.release()
            state["active_process"] = None
            state["usage"]["elapsed_seconds"] += result["elapsed_seconds"]
            observe_usage(state, result, str(records))
            attempt["outcome"] = result["outcome"]
            attempt["thread_id"] = result.get("thread_id")
            self.store.save(state)
            if result["outcome"] != "completed":
                if result["outcome"] == "termination_unconfirmed":
                    state["active_process"] = read_json(records / "process.json") | {
                        "records": str(records), "started_at": result.get("started_at", time.time() - result["elapsed_seconds"])}
                    state["status"], state["reason"] = "stopped", "termination_unconfirmed"
                    self.store.save(state)
                    return state
                if result["outcome"] == "cancelled":
                    return finish(self.store, state, "cancelled", stopped=True)
                return finish(self.store, state, "engineering_failure")
            self.store.verify_baseline(state)
            try:
                try:
                    receipt = validate_receipt(result["response"], state, baseline, work, role)
                except StateError as error:
                    state["receipt_correction"] = str(error)
                    attempt["receipt_repair_files"] = tree_manifest(work)
                    thread_id = result["thread_id"]
                    write_json(records / "receipt_rejection.json", {"reason": str(error), "retry_same_session": True})
                    self.store.save(state)
                    continue
                if attempt.get("receipt_repair_files") is not None and tree_manifest(work) != attempt["receipt_repair_files"]:
                    raise StateError("receipt-only correction modified artifacts")
                state.pop("receipt_correction", None)
                if role == "reviewer" and tree_manifest(work) != before:
                    raise StateError("read-only reviewer changed artifacts")
                if role == "worker":
                    validate_source_bindings(baseline, work, state.get("source_bindings", []))
                    checkpoint = self.store.verify_checkpoint(state)
                    if checkpoint:
                        previous = read_json(checkpoint / "checkpoint.json")["semantics"]
                        changed_contract = previous["contract"] != semantic_digests(work, baseline["snapshot"])["contract"]
                        if changed_contract and state["stage"] != "contract" and not state.get("allow_contract_correction"):
                            raise StateError("worker changed shared Contract without independent correction review")
                check = run_check(baseline, work, records)
                if check["exit_code"] == 2:
                    return finish(self.store, state, "checker_failure")
                if role == "worker":
                    state = self.store.commit_checkpoint(state, work, receipt, check)
                    attempt = next(v for v in state["attempts"] if v["id"] == attempt["id"])
                    state["dispatches"] = receipt["dispatches"]
                    state["worker_status"] = receipt["status"]
                    merge_issues(state, receipt)
                    if receipt["status"] == "continue":
                        self.store.save(state)
                        thread_id = result["thread_id"]
                        continue
                    self.after_worker(state)
                else:
                    merge_issues(state, receipt, reviewer=True)
                    state["last_review"] = bound_review(state, baseline, receipt, self.store.verify_checkpoint(state))
                    actions = {issue_disposition(i) for i in receipt["issues"]}
                    if "needs_input" in actions:
                        return finish(self.store, state, "needs_input")
                    if "fix" in actions:
                        state["revision_request"] = deepcopy(receipt)
                        state["revision_request"]["issues"] = [i for i in receipt["issues"] if issue_disposition(i) == "fix"]
                        if receipt["impact"] == "contract":
                            if receipt["correction"] != "transcription" or not any(i["kind"] == "transcription" and i["evidence"] for i in receipt["issues"]):
                                raise StateError("Contract revision must be a source-backed transcription correction; preserve actual requirements")
                            state["stage"], state["node"] = "contract", "root"
                            state["allow_contract_correction"] = True
                            state["reviews"].clear()
                        elif receipt["impact"] in {"root", "layout"}:
                            state["stage"], state["node"] = "node", "root"
                            state["reviews"] = {k: v for k, v in state["reviews"].items() if k == "contract"}
                        elif state["stage"] != "contract":
                            state["stage"] = "node"
                            requested = receipt["revision_node"] or state["node"]
                            if requested not in {"root", *(v["id"] for v in state["dispatches"])}:
                                raise StateError("review revision_node is not a declared node")
                            state["node"] = requested
                        state["phase"] = "worker"
                    else:
                        checkpoint = self.store.verify_checkpoint(state)
                        state["reviews"][state["review_scope"]] = bound_review(state, baseline, receipt, checkpoint)
                        state.pop("revision_request", None)
                        state.pop("allow_contract_correction", None)
                        if state["review_scope"] == "final":
                            if check["exit_code"] != 0:
                                state["phase"] = "worker"
                                state["revision_request"] = {"summary": "Fix the reported non-hardware checker errors without changing requirements.", "check": check}
                                self.store.save(state)
                                attempt, work, thread_id = None, None, None
                                continue
                            state["eligible"] = eligible(self.store, state)
                            return finish(self.store, state, "ready" if state["eligible"] else "evidence_gaps")
                        self.after_review(state)
                self.store.save(state)
                attempt, work, thread_id = None, None, None
            except (StateError, ValueError, OSError) as error:
                state["last_error"] = str(error)
                write_json(records / "rejection.json", {"reason": str(error), "checkpoint_accepted": False})
                return finish(self.store, state, "protocol_failure")
        return state

    def after_worker(self, state):
        stage = state["stage"]
        state["phase"] = "reviewer"
        state["review_scope"] = "contract" if stage == "contract" else (
            "final" if stage == "index" or (state["node"] == "root" and not state["dispatches"]) else state["node"])

    def after_review(self, state):
        state["phase"] = "worker"
        if state["review_scope"] == "contract":
            state["stage"], state["node"] = "node", "root"
            return
        order = ["root", *(item["id"] for item in state["dispatches"])]
        checkpoint = self.store.verify_checkpoint(state)
        files = artifact_manifest(checkpoint / "artifacts")
        for node in order:
            if node not in state["reviews"]:
                state["stage"], state["node"] = "node", node
                if node + ".yaml" in files:
                    state["phase"], state["review_scope"] = "reviewer", node
                return
        state["stage"], state["node"] = "index", "root"

    def run(self):
        with ThreadPoolExecutor(max_workers=self.config["jobs"]) as pool:
            futures = {pool.submit(self.work, key): key for key in self.plan["tasks"]}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    state = future.result()
                    print(json.dumps({"task": key, "status": state["status"], "result": state["result"],
                                      "checkpoint": state["checkpoint"], "usage": state["usage"]}), flush=True)
                except BaseException as error:
                    state = self.store.load(key)
                    state["status"], state["reason"], state["last_error"] = "stopped", "runner_failure", str(error)
                    self.store.save(state)
                    print(json.dumps({"task": key, "error": str(error)}), flush=True)


def publish(store, state, target=None):
    if state["status"] != "ended" or state["result"] != "ready" or not eligible(store, state):
        raise StateError("task is not eligible: drafts/gaps/check/review failures cannot be exported")
    checkpoint = store.verify_checkpoint(state)
    baseline = store.verify_baseline(state)
    target = Path(target or state["output"]).resolve()
    if target.is_relative_to(store.root):
        raise StateError("publication target may not be inside runtime records")
    receipt = {"task_key": state["key"], "fingerprint": state["fingerprint"], "checkpoint": state["checkpoint"],
               "source_repository": baseline["repository_id"], "source_commit": baseline["snapshot_commit"],
               "source_files": baseline["files"], "origin_head": baseline["origin_head"],
               "seed": state.get("seed"), "source_bindings": state.get("source_bindings", []),
               "check_policy": "skip-hardware", "reviews": state["reviews"], "usage": state["usage"],
               "notes": [issue for issue in state["issues"] if issue_disposition(issue) == "note"],
               "model": baseline["identity"]["config"]["model"], "effort": "xhigh"}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + ".publish-" + uuid4().hex[:8])
    clone_artifacts(checkpoint / "artifacts", temporary)
    (temporary / ".extraction").mkdir()
    write_json(temporary / ".extraction/receipt.json", receipt)
    check = run_check(baseline, temporary, temporary / ".extraction")
    if check["exit_code"] != 0:
        shutil.rmtree(temporary)
        raise StateError("non-hardware check failed; publication refused")
    # Diagnostics include the staging path and are kept only in runtime records.
    (temporary / ".extraction/check.json").unlink()
    if target.exists():
        same = tree_manifest(target) == tree_manifest(temporary)
        shutil.rmtree(temporary)
        if not same:
            state["publication_conflict"] = str(target)
            store.save(state)
            raise StateError("publication conflict: existing target left unchanged")
    else:
        # Reserve an absent destination before rename. A concurrent creator is
        # never overwritten, even when it created an empty directory.
        try:
            target.mkdir()
        except FileExistsError as error:
            shutil.rmtree(temporary)
            raise StateError("publication destination was concurrently created; left unchanged") from error
        os.rename(temporary, target)
    state["published"] = {"path": str(target), "checkpoint": state["checkpoint"]}
    store.save(state)
    return str(target)


def sync_issues(store, plan):
    """Prepare a reviewable patch; the supervisor applies repository text edits."""
    path = store.repo / "schema_extract/ISSUES.md"
    text = path.read_text()
    for key in plan["tasks"]:
        state = store.load(key)
        for issue in state["issues"]:
            disposition = issue_disposition(issue)
            issue_state = "ignored" if disposition == "ignored" else issue["state"]
            ledger_key = issue.get("ledger_task_key", key)
            marker = f"<!-- extraction:{ledger_key}:{issue['id']} -->"
            end = "<!-- /extraction -->"
            block = (f"{marker}\n\n## SE-{ledger_key}-{issue['id']}\n\n"
                     f"- **状态：** `{issue_state}`；类型 `{issue['kind']}`；处理 `{disposition}`。\n"
                     f"- **任务／阶段：** `{state['operator_id']}` / {issue['scope']}；run `{plan['run_id']}`。\n"
                     f"- **现象与依据：** {issue['detail']}\n"
                     f"- **证据：** {'; '.join(issue['evidence'])}\n"
                     f"- **基线／检查点：** `{state['fingerprint']}` / `{issue.get('checkpoint')}`。\n"
                     + (f"- **问题来源：** `{ledger_key}` / `{issue.get('first_checkpoint')}`；保留原问题身份。\n" if ledger_key != key else "") +
                     f"- **影响与重启条件：** {', '.join(issue['blocks'])}；{issue['restart_condition']}\n\n{end}")
            start = text.find(marker)
            if start < 0:
                text = text.rstrip() + "\n\n" + block + "\n"
            else:
                stop = text.index(end, start) + len(end)
                text = text[:start] + block + text[stop:]
    old = path.read_text()
    if text != old:
        patch = "*** Begin Patch\n*** Update File: " + str(path) + "\n@@\n" + "".join("-" + line + "\n" for line in old.splitlines()) + "".join("+" + line + "\n" for line in text.splitlines()) + "*** End Patch\n"
        output = store.root / "runs" / plan["run_id"] / "issues.patch"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(patch)
        return str(output)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--state-root", type=Path, help="default: <repo>/.schema_extract")
    parser.add_argument("--codex", default="codex", help="local Codex executable")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("discover")
    plan_cmd = sub.add_parser("plan")
    plan_cmd.add_argument("--id", action="append", required=True)
    plan_cmd.add_argument("--run-id")
    plan_cmd.add_argument("--seed-run", help="reuse verified accepted drafts from this stopped run; preserve usage and re-review Contract")
    plan_cmd.add_argument("--work-root", required=True, type=Path)
    plan_cmd.add_argument("--output-root", required=True, type=Path)
    plan_cmd.add_argument("--source", action="append", default=[], help="extra fixed repository-relative material")
    plan_cmd.add_argument("--model", default=DEFAULTS["model"])
    plan_cmd.add_argument("--jobs", type=int, choices=(1, 2), default=DEFAULTS["jobs"])
    for name in ("run", "status", "pause", "cancel", "resume", "retry", "export"):
        command = sub.add_parser(name)
        command.add_argument("run_id")
        if name in {"resume", "retry", "export"}:
            command.add_argument("--id", action="append", required=True)
        if name in {"resume", "retry"}:
            command.add_argument("--reason")
        if name == "export":
            command.add_argument("--to", type=Path, help="only with one selected task; must be a new directory")
    args = parser.parse_args(argv)
    store = Store(args.repo, args.state_root or args.repo / ".schema_extract")
    if args.command == "discover":
        entries = read_json(store.repo / "kernel_wiki/catalog.json")["operators"]
        print(json.dumps([{k: entry.get(k) for k in ("id", "title", "category")} for entry in entries], ensure_ascii=False, indent=2))
        return 0
    try:
        if args.command == "plan":
            config = {**DEFAULTS, "model": args.model, "jobs": args.jobs}
            with store.lock():
                plan = make_plan(store, args.id, args.work_root, args.output_root, args.source, config, args.run_id, seed_run=args.seed_run)
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        plan = read_json(store.root / "runs" / args.run_id / "plan.json")
        if args.command == "status":
            print(json.dumps([store.load(key) for key in plan["tasks"]], ensure_ascii=False, indent=2))
            return 0
        if args.command in {"pause", "cancel"}:
            write_json(store.root / "runs" / args.run_id / "control.json", {"mode": args.command, "requested_at": time.time()})
            print("Control requested; task status changes only after runner confirms the boundary/termination.")
            return 0
        with store.lock():
            runner = Runner(store, plan, cli=CodexCLI(executable=args.codex, model=plan["config"]["model"]))
            if args.command in {"resume", "retry", "export"}:
                selected = [key for key in plan["tasks"] if store.load(key)["operator_id"] in args.id]
                if len(selected) != len(set(args.id)):
                    raise StateError("selected IDs are not distinct tasks in this plan")
                if args.command == "export":
                    if args.to and len(selected) != 1:
                        raise StateError("--to requires exactly one selected task")
                    for key in selected:
                        print(publish(store, store.load(key), args.to))
                    return 0
                for key in selected:
                    state = store.load(key)
                    if state["status"] == "ended" and not args.reason:
                        raise StateError("ended task with same inputs requires an explicit --reason; new evidence requires a new plan")
                    if state["result"] == "ready":
                        continue
                    runner.recover(state)
                    state["retry_history"].append({"reason": args.reason or "resume interrupted/paused task", "at": time.time()})
                    store.save(state)
                if runner.control.exists():
                    runner.control.unlink()
                runner.plan = {**plan, "tasks": selected}
            old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            for sig in old_handlers:
                signal.signal(sig, lambda *_: runner.cancel_event.set())
            try:
                runner.run()
                print(json.dumps({"issues_patch": sync_issues(store, plan)}, ensure_ascii=False), flush=True)
            finally:
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
        return 0
    except (StateError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"schema_extract: {error}", file=__import__("sys").stderr)
        return 2
