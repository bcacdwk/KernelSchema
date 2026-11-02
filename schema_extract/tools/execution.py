"""Run extraction stages, independent reviews, recovery and export."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import threading
import time
from uuid import uuid4

from .codex_cli import CodexCLI, terminate_owned, write_json
from .prompts import prompt_for
from .protocol import RECEIPT, validate_receipt
from .records import (StateError, artifact_manifest, clone_artifacts, hardware_issue, issue_disposition,
                      optional_json, read_json, review_current, run_check, schema_parts, semantic_digests, tree_manifest)
from .review_context import build_review_context


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


def task_status(store, state, now=None):
    """Small, read-only progress view; in-flight usage is deliberately unknown."""
    now = time.time() if now is None else now
    summary = {name: state.get(name) for name in (
        "key", "operator_id", "status", "phase", "stage", "node", "review_scope",
        "checkpoint", "result", "reason", "last_error")}
    usage = state.get("usage", {})
    summary["usage"] = {"calls_started": usage.get("calls", 0),
                        "completed_call_seconds": usage.get("elapsed_seconds", 0),
                        "reported_tokens": usage.get("reported_tokens", 0),
                        "unknown_usage_calls": usage.get("unknown_usage_calls", 0),
                        "tokens_include_active_call": False}
    summary["check"] = {name: state["check"].get(name) for name in ("exit_code", "policy")} if state.get("check") else None
    summary["reviews"] = list(state.get("reviews", {}))
    summary["open_issues"] = {kind: sum(i.get("state") == "open" and issue_disposition(i) == kind
                                      for i in state.get("issues", []))
                              for kind in ("fix", "note", "needs_input")}
    active = state.get("active_process")
    summary["active_call"] = None
    if active:
        records = Path(active["records"]).resolve()
        if not records.is_relative_to((store.task_dir(state["key"]) / "attempts").resolve()):
            raise StateError("invocation path leaves the selected task")
        call = {"records": str(records), "pid": active.get("identity", {}).get("pid"),
                "elapsed_seconds": round(max(0, now - active.get("started_at", now)), 1),
                "thread_id": None, "events_bytes": 0, "event_age_seconds": None}
        events = records / "events.jsonl"
        try:
            stat = events.stat()
            call.update(events_bytes=stat.st_size, event_age_seconds=round(max(0, now - stat.st_mtime), 1))
            with events.open("rb") as handle:
                first = json.loads(handle.readline(4096))
            if first.get("type") == "thread.started":
                call["thread_id"] = first.get("thread_id")
        except (FileNotFoundError, ValueError, AttributeError):
            pass  # The first event may not yet exist or may still be being written.
        summary["active_call"] = call
    return summary


def bound_review(state, receipt, checkpoint):
    semantics = read_json(checkpoint / "checkpoint.json")["semantics"]
    return {"fingerprint": state["fingerprint"], "checkpoint": state["checkpoint"],
            "contract": semantics["contract"], "layout": semantics["layout"],
            "conditions": semantics["conditions"], "dispatches": semantics["dispatches"],
            "selection_revision": semantics["selection_revision"],
            "nodes": semantics["nodes"], "receipt": receipt,
            "complete": not any(issue_disposition(i) in {"fix", "needs_input"} for i in receipt["issues"]),
            "reviewed_at": time.time()}


def request_revision(state, receipt):
    """Return only the affected scope and concrete fixes to the author."""
    state["revision_request"] = deepcopy(receipt)
    state["revision_request"]["issues"] = [i for i in receipt["issues"] if issue_disposition(i) == "fix"]
    impact = receipt["impact"]
    if impact == "contract":
        if receipt["correction"] != "transcription" or not any(i["kind"] == "transcription" and i["evidence"] for i in receipt["issues"]):
            raise StateError("Contract revision must be a source-backed transcription correction; preserve actual requirements")
        state.update(stage="contract", node="root", allow_contract_correction=True)
        state["reviews"].clear()
    elif impact == "index":
        state.update(stage="index", node="root")
        state["reviews"].pop("final", None)
    elif impact in {"root", "layout"}:
        state.update(stage="node", node="root")
        state["reviews"] = {k: v for k, v in state["reviews"].items() if k == "contract"}
    elif state["stage"] != "contract":
        requested = receipt["revision_node"] or state["node"]
        if requested not in {"root", *(v["id"] for v in state["dispatches"])}:
            raise StateError("review revision_node is not a declared node")
        state.update(stage="node", node=requested)
    state["phase"] = "worker"


def eligible(store, state):
    store.verify_baseline(state)
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
        if not review_current(state, scope, semantics):
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
        return (optional_json(self.control) or {}).get("mode")

    def recover(self, state):
        active = state.get("active_process")
        if not active and state.get("active_attempt"):
            attempt = next(v for v in state["attempts"] if v["id"] == state["active_attempt"])
            if attempt["calls"]:
                records = Path(attempt["calls"][-1])
                result = optional_json(records / "result.json")
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
        author_session = None  # In-memory only: recovery always starts a fresh context.
        while state["status"] in {"queued", "running"}:
            mode = self.control_mode()
            if mode:
                return finish(self.store, state, mode, stopped=True)
            baseline = self.store.verify_baseline(state)
            if (attempt is None and state["dispatches"]
                    and ((state["phase"] == "worker" and state["stage"] == "index"
                          and not state.get("revision_request"))
                         or (state["phase"] == "reviewer" and state["review_scope"] == "final"))
                    and self.pending_review(state)):
                self.store.save(state)
            role = state["phase"]
            if attempt is None:
                if role == "worker" and author_session:
                    attempt_id, work, thread_id = author_session
                    checkpoint = self.store.verify_checkpoint(state)
                    if tree_manifest(work) != artifact_manifest(checkpoint / "artifacts"):
                        raise StateError("author workspace changed after its accepted checkpoint; recover into a new attempt")
                    attempt = next(item for item in state["attempts"] if item["id"] == attempt_id)
                    state.update(active_attempt=attempt_id, status="running")
                    self.store.save(state)
                else:
                    attempt, work = self.store.new_attempt(state, Path(self.plan["work_root"]), role)
            before = tree_manifest(work)
            records = self.store.task_dir(key) / "attempts" / attempt["id"] / f"call-{len(attempt['calls']) + 1:02d}"
            records.parent.mkdir(parents=True, exist_ok=True)

            def started(process):
                state["active_process"] = {**process, "started_at": time.time()}
                self.store.save(state)

            if role == "reviewer":
                self.review_slot.acquire()
            try:
                mode = self.control_mode()
                if mode:
                    return finish(self.store, state, mode, stopped=True)
                review_context = build_review_context(self.store, state, baseline, work, role)
                prompt = prompt_for(state, baseline, work, role, resumed=bool(thread_id), review_context=review_context)
                # Only registered invocations need termination evidence on
                # recovery. Preparation may fail or receive a control request
                # while constructing the checkpoint-bound prompt.
                mode = self.control_mode()
                if mode:
                    return finish(self.store, state, mode, stopped=True)
                state["usage"]["calls"] += 1
                attempt["calls"].append(str(records))
                self.store.save(state)
                result = self.cli.run(cwd=work, records=records, prompt=prompt,
                                      schema=RECEIPT, role=role, thread_id=thread_id,
                                      cancel=lambda: self.control_mode() == "cancel", on_process=started)
                if review_context is not None:
                    write_json(records / "review_context.json", review_context)
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
            review_report = None
            try:
                if role == "reviewer" and tree_manifest(work) != before:
                    raise StateError("read-only reviewer changed artifacts")
                if attempt.get("receipt_repair_files") is not None and tree_manifest(work) != attempt["receipt_repair_files"]:
                    raise StateError("receipt-only correction modified artifacts")
                try:
                    receipt = validate_receipt(result["response"], state, baseline, work, role)
                except StateError as error:
                    state["receipt_correction"] = str(error)
                    attempt["receipt_repair_files"] = tree_manifest(work)
                    thread_id = result["thread_id"]
                    write_json(records / "receipt_rejection.json", {"reason": str(error), "retry_same_session": True})
                    self.store.save(state)
                    continue
                state.pop("receipt_correction", None)
                attempt.pop("receipt_repair_files", None)
                normalizations = [{"field": key, "index": index, "before": before["path"], "after": after["path"]}
                                  for key in ("source_evidence", "artifact_evidence")
                                  for index, (before, after) in enumerate(zip(result["response"][key], receipt[key]))
                                  if before["path"] != after["path"]]
                if normalizations:
                    write_json(records / "receipt_normalization.json", {"paths": normalizations, "anchors_validated": True})
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
                    state["worker_status"] = receipt["status"]
                    author_session = (attempt["id"], work, result["thread_id"])
                    merge_issues(state, receipt)
                    if receipt["status"] == "continue":
                        self.store.save(state)
                        thread_id = result["thread_id"]
                        continue
                    self.after_worker(state)
                else:
                    proposals = receipt.pop("framework_candidates", [])
                    merge_issues(state, receipt, reviewer=True)
                    state["last_review"] = bound_review(state, receipt, self.store.verify_checkpoint(state))
                    actions = {issue_disposition(i) for i in receipt["issues"]}
                    if proposals:
                        source = {"role": "reviewer", "checkpoint": state["checkpoint"],
                                  "record": str((records / "result.json").relative_to(self.store.root))}
                        review_report = {"source": source, "candidates": proposals}
                        state.setdefault("review_candidate_reports", []).append(review_report)
                    if "needs_input" in actions:
                        return finish(self.store, state, "needs_input")
                    if "fix" in actions:
                        request_revision(state, receipt)
                    else:
                        state["reviews"][state["review_scope"]] = state["last_review"]
                        state.pop("revision_request", None)
                        state.pop("allow_contract_correction", None)
                        if state["review_scope"] != "final":
                            self.after_review(state)
                        elif check["exit_code"] != 0:
                            state["phase"] = "worker"
                            state["revision_request"] = {"summary": "Fix the reported non-hardware checker errors without changing requirements.", "check": check}
                        else:
                            state["eligible"] = eligible(self.store, state)
                            mode = self.control_mode()
                            if mode:
                                return finish(self.store, state, mode, stopped=True)
                            return finish(self.store, state, "ready" if state["eligible"] else "evidence_gaps")
                self.store.save(state)
                attempt, work, thread_id = None, None, None
            except (StateError, ValueError, OSError) as error:
                if review_report:
                    state["review_candidate_reports"].pop()  # This review transition was rejected.
                state["last_error"] = str(error)
                write_json(records / "rejection.json", {"reason": str(error), "checkpoint_accepted": False})
                return finish(self.store, state, "protocol_failure")
        return state

    def after_worker(self, state):
        stage = state["stage"]
        if stage == "index" and state["dispatches"] and self.pending_review(state):
            return
        state["phase"] = "reviewer"
        state["review_scope"] = "contract" if stage == "contract" else (
            "final" if stage == "index" or (state["node"] == "root" and not state["dispatches"]) else state["node"])

    def after_review(self, state):
        state["phase"] = "worker"
        if state["review_scope"] == "contract":
            state["stage"], state["node"] = "node", "root"
            checkpoint = self.store.verify_checkpoint(state)
            parser_root = self.store.baseline(state["key"])["snapshot"]
            if state.get("seed") and schema_parts(checkpoint / "artifacts", parser_root)["root"].get("design"):
                state["phase"] = "reviewer"
                state["review_scope"] = "root" if state["dispatches"] else "final"
            return
        if self.pending_review(state):
            return
        state["stage"], state["node"] = "index", "root"
        checkpoint = self.store.verify_checkpoint(state)
        receipt = read_json(checkpoint / "checkpoint.json").get("worker_receipt", {})
        if receipt.get("stage") == "index" and receipt.get("status") == "complete":
            # An index author's actual YAML edit may have required a node
            # review. Once that review is complete, this same checkpoint's
            # finished index can go straight to independent final review.
            state["phase"], state["review_scope"] = "reviewer", "final"

    def pending_review(self, state):
        """Schedule missing or stale independent reviews before index/final."""
        checkpoint = self.store.verify_checkpoint(state)
        if checkpoint is None:
            return False
        record = read_json(checkpoint / "checkpoint.json")
        semantics = record["semantics"]
        for scope in ["contract", "root", *(item["id"] for item in state["dispatches"])]:
            if review_current(state, scope, semantics):
                continue
            state["reviews"].pop(scope, None)
            state["stage"] = "contract" if scope == "contract" else "node"
            state["node"] = "root" if scope == "contract" else scope
            state["phase"] = "reviewer" if state["node"] + ".yaml" in record["files"] else "worker"
            state["review_scope"] = scope
            return True
        return False

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
