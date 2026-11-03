"""Durable invocation observations, scoped process fences and session handoffs."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

from .common import ExtractError, read_json, write_json
from .processes import terminate_owned
from .stages import scope_for


def confirm_termination(attempt: dict, result: dict | None) -> dict:
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
        elif not (directory / "launch.json").exists() and not (directory / "events.jsonl").exists():
            confirmation = {"confirmed": True, "actions": [], "survivors": [],
                            "reason": "no durable launch intent; invocation was not started"}
        else:
            raise ExtractError("invocation launched but its process identity is unavailable")
    except (ExtractError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
        confirmation = {"confirmed": False, "actions": [], "survivors": [], "reason": str(exc)}
    confirmation["observed_at"] = time.time()
    write_json(directory / "termination.json", confirmation)
    return confirmation


def account(state: dict, attempt: dict, elapsed: float) -> None:
    """Charge an attempt's monotonic high-water mark once, including after replay."""
    elapsed = max(0.0, float(elapsed))
    usage = state.setdefault("attempt_usage", {})
    previous = usage.get(attempt["attempt_id"], {}).get("elapsed_seconds", 0.0)
    if elapsed > previous:
        scope_for(state, attempt)["elapsed_seconds"] += elapsed - previous
    usage[attempt["attempt_id"]] = {"stage": attempt["stage"], "node": attempt["node"],
                                    "elapsed_seconds": max(previous, elapsed)}


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
        for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
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


def consume_controls(store, state: dict, *, verify=None) -> bool:
    control = state.setdefault("control", {"action": None, "requests": {}, "resume_status": None})
    changed = False
    for request in store.control_requests(state["run_id"]):
        if request["id"] in control["requests"]:
            continue
        control["requests"][request["id"]] = {**request, "observed_at": time.time()}
        # Cancellation cannot be weakened by a later pause request.
        if control["action"] != "cancel":
            control["action"] = request["action"]
        changed = True
    if changed:
        if verify is not None:
            verify()
        store.save(state)
    return changed
