"""Durable invocation observations, scoped process fences and session handoffs."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

from .audit import usage_summary
from .common import ExtractError, digest, read_json, write_json
from .processes import terminate_owned
from .stages import scope_for


def confirm_termination(attempt: dict, result: dict | None, *, launch_state=None) -> dict:
    """Missing ownership after launch is an explicit fence, never proof of exit."""
    directory = Path(attempt["directory"])
    try:
        if (directory / "process.json").exists():
            process = read_json(directory / "process.json")
            identity, known = process["identity"], process.get("descendants", [])
            if (type(identity.get("pid")) is not int or identity["pid"] <= 1 or
                    identity["pid"] == os.getpid() or not isinstance(identity.get("started"), str)):
                raise ExtractError("invalid invocation process identity")
            confirmation = terminate_owned(identity, known)
        elif result and result.get("termination", {}).get("confirmed"):
            confirmation = deepcopy(result["termination"])
        elif (launch_state == "prepared" and not (directory / "launch.json").exists() and
              not (directory / "events.jsonl").exists() and not (directory / "result.json").exists() and
              result is None):
            confirmation = {"confirmed": True, "actions": [], "survivors": [],
                            "reason": "Runner state records preparation without dispatch authorization"}
        else:
            raise ExtractError("invocation termination is unknown: no process identity or positive pre-dispatch record")
    except (ExtractError, KeyError, TypeError, AttributeError, OSError, subprocess.SubprocessError) as exc:
        confirmation = {"confirmed": False, "actions": [], "survivors": [], "reason": str(exc)}
    confirmation["observed_at"] = time.time()
    write_json(directory / "termination.json", confirmation)
    return confirmation


def account(state: dict, attempt: dict, elapsed: float, *, result=None, processing=None, final=False) -> None:
    """Charge an attempt's monotonic high-water mark once, including after replay."""
    elapsed = max(0.0, float(elapsed))
    usage = state.setdefault("attempt_usage", {})
    entry = usage.setdefault(attempt["attempt_id"], {})
    previous = entry.get("elapsed_seconds", 0.0)
    if elapsed > previous:
        scope_for(state, attempt)["elapsed_seconds"] += elapsed - previous
    entry.update(stage=attempt["stage"], node=attempt["node"], elapsed_seconds=max(previous, elapsed))
    entry.setdefault("origin_run_id", attempt.get("run_id", state.get("run_id")))
    for key in ("role", "mode", "session_group", "base_checkpoint", "input_fingerprint", "created_at"):
        if key in attempt:
            entry.setdefault(key, attempt[key])
    entry.setdefault("requested_session_id", attempt.get("session_id"))
    entry.setdefault("usage", usage_summary(None))
    evidence = entry.setdefault("evidence", {})
    evidence.setdefault("attempt.json", digest(attempt))
    timing = entry.setdefault("timing", {"cli_seconds": None, "processing_seconds": None,
                                         "processing_estimated_seconds": None})
    if result is not None:
        # Preserve the first persistent result binding. Reconciliation may add
        # an in-memory termination confirmation; it does not rewrite that file.
        evidence.setdefault("result.json", digest(result))
        if result.get("usage") is not None and "recorded_usage" not in entry:
            entry["usage"] = usage_summary(result["usage"])
            entry["recorded_usage"] = True
        for key in ("session_id", "outcome", "cli_version", "started_at", "completed_at"):
            if result.get(key) is not None:
                entry.setdefault(key, result[key])
        if result.get("elapsed_seconds") is not None:
            timing["cli_seconds"] = max(timing.get("cli_seconds") or 0, result["elapsed_seconds"])
        if result.get("trace"):
            entry.setdefault("trace", {key: result["trace"].get(key) for key in
                                       ("sha256", "bytes", "complete_bytes", "stable", "truncated_line")})
    if processing is not None:
        timing["processing_seconds"] = max(timing.get("processing_seconds") or 0, processing.get("elapsed_seconds", 0))
        timing["processing_estimated_seconds"] = max(timing.get("processing_estimated_seconds") or 0,
                                                     processing.get("estimated_seconds", 0))
        timing["processing_status"] = "partial" if processing.get("started_at") is not None else "conservative" if timing["processing_estimated_seconds"] else "reported"
    if final:
        directory = Path(attempt["directory"])
        # Only bounded, named records at this call's close; never historical
        # directory walks or a second events.jsonl pass.
        for name in ("invocation.json", "response.schema.json", "response.json", "receipt.json", "validated.json",
                     "check.json", "merge_check.json", "event.json",
                     "repair.json", "recovery.json", "rejected.json", "stale.json", "handoff.json"):
            path = directory / name
            if path.is_file():
                try:
                    observed = digest(read_json(path))
                    if name in evidence and evidence[name] != observed:
                        issue = "recorded evidence changed: " + name
                        entry["audit_issues"] = sorted({*entry.get("audit_issues", []), issue})
                    else:
                        evidence.setdefault(name, observed)
                except ExtractError:
                    entry["audit_issues"] = sorted({*entry.get("audit_issues", []), "unreadable " + name})
        pending = {*state.get("active_attempts", {}).values(), *state.get("unconfirmed_attempts", {}),
                   *state.get("unsettled_attempts", {})}
        entry["settled"] = attempt["attempt_id"] not in pending
        entry["receipt_sha256"] = state.get("accepted_receipts", {}).get(attempt["attempt_id"])
        entry["checkpoints"] = sorted({value for value in (*entry.get("checkpoints", []),
                                                          attempt.get("base_checkpoint"), entry.get("result_checkpoint")) if value})


def elapsed_on_disk(attempt: dict, result: dict | None) -> float:
    directory = Path(attempt["directory"])
    if (directory / "finished.json").is_file():
        return read_json(directory / "finished.json")["elapsed_seconds"]
    event = read_json(directory / "event.json") if (directory / "event.json").is_file() else {}
    elapsed = event.get("elapsed_seconds", (result or {}).get("elapsed_seconds"))
    if elapsed is None:
        elapsed = max(0.0, time.time() - attempt["created_at"])
    process = read_json(directory / "processing.json") if (directory / "processing.json").is_file() else {}
    # A crash during checking has no final timer sample. Charge its last open
    # interval conservatively; the original wall-clock deadline also stays fixed.
    elapsed += process.get("elapsed_seconds", 0.0)
    if process.get("started_at") is not None:
        elapsed += max(0.0, time.time() - process["started_at"])
    return elapsed


def handoff_session(state: dict, attempt: dict, result: dict | None, reason: str) -> dict:
    existing = next((item for item in state.setdefault("session_handoffs", [])
                     if item["attempt_id"] == attempt["attempt_id"]), None)
    if existing:
        return existing
    old = (result or {}).get("session_id") or attempt.get("session_id")
    events = Path(attempt["directory"]) / "events.jsonl"
    if not old and events.is_file():
        with events.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                    if event.get("type") == "thread.started":
                        old = event.get("thread_id")
                        break
                except (ValueError, AttributeError):
                    pass
    roles = state["sessions"].setdefault(attempt["session_group"], {"worker": None, "reviewer": None})
    if roles[attempt["role"]] in {None, old, attempt.get("session_id")}:
        roles[attempt["role"]] = None
    checkpoint = (state["nodes"][attempt["node"]]["draft_checkpoint"]
                  if attempt["stage"] == "P5" else state["checkpoint"])
    record = {"attempt_id": attempt["attempt_id"], "session_group": attempt["session_group"],
              "stage": attempt["stage"], "node": attempt["node"], "role": attempt["role"],
              "previous_session_id": old, "next_attempt_id": None, "next_session_id": None,
              "checkpoint": checkpoint, "input_fingerprint": state["input_fingerprint"],
              "reason": reason, "at": time.time()}
    state["session_handoffs"].append(record)
    write_json(Path(attempt["directory"]) / "handoff.json", record)
    return record


def consume_controls(store, state: dict) -> bool:
    control = state.setdefault("control", {"action": None, "requests": {}, "resume_status": None})
    changed = False
    for request in store.control_requests(state["run_id"]):
        if request["id"] in control["requests"]:
            continue
        control["requests"][request["id"]] = {**request, "observed_at": time.time()}
        # Cancellation cannot be weakened by a later pause request.
        if control["action"] != "cancel":
            control["action"] = request["action"]
            control["hold_id"] = request.get("hold_id")
            control["last_request_id"] = request["id"]
            control["revision"] = request.get("control_revision")
        changed = True
    if changed:
        store.save(state)
    return changed
