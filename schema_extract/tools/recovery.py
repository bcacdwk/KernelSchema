"""Durable invocation observations, scoped process fences and session handoffs."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import time
from uuid import UUID

from .audit import parse_events, usage_summary
from .common import ExtractError, digest, read_json, write_json
from .processes import GATE_PROTOCOL, gate_reaped, launch_not_started, process_alive, process_table, terminate_owned
from .stages import NODE_STOPS, call_view, node_signature, owner_for, scope_for, scope_key


HARD_RECOVERY_STOPS = {"external_confirmation", "adjudication_already_used", "interrupted_single_decision",
                       "reopen_already_used", "invalid_root_prerequisite",
                       "missing_input", "blocked_nodes", "adjudication_blocked", "checker_input_error"}


def retry_classification(kind: str, result: dict) -> str:
    """Classify existing repairs; only EX_TEMPFAIL is proven transient here."""
    if kind == "receipt_repair":
        return "protocol_repair"
    if result.get("outcome") == "cli_error" and result.get("exit_code") == 75:
        return "transient_execution"
    return "execution_failure"


def retry_delay(failures: int) -> float:
    return min(60.0, 5.0 * 2 ** min(4, max(0, failures - 1)))


def _invocation(attempt):
    value = read_json(Path(attempt["directory"]) / "invocation.json")
    if (value.get("attempt_sha256") != digest(attempt) or
            any(value.get(key) != attempt.get(key) for key in
                ("run_id", "attempt_id", "backend", "model", "effort", "session_scope"))):
        raise ExtractError("recovery invocation differs from its registered attempt")
    if (attempt.get("executable") is not None and value.get("executable") != attempt["executable"] or
            attempt.get("cli_version") is not None and value.get("cli_version") != attempt["cli_version"]):
        raise ExtractError("recovery executable/version differs from its registered attempt")
    return value


def _launch(attempt):
    invocation = _invocation(attempt)
    value = read_json(Path(attempt["directory"]) / "launch.json")
    if (value.get("protocol") != GATE_PROTOCOL or
            any(value.get(key) != invocation.get(key) for key in
                ("run_id", "attempt_id", "attempt_sha256", "launch_id"))):
        raise ExtractError("recovery launch differs from its registered invocation")
    return value


def _exit(attempt):
    launch = _launch(attempt)
    value = read_json(Path(attempt["directory"]) / "exit.json")
    if (value.get("attempt_sha256") != digest(attempt) or value.get("launch_id") != launch["launch_id"] or
            (value.get("exit_code") is not None and type(value["exit_code"]) is not int)):
        raise ExtractError("exit record differs from its registered invocation")
    return value


def never_launched(attempt, result):
    """Invocation-bound setup failure evidence shared by recovery and audit."""
    directory = Path(attempt["directory"])
    if (not result or result.get("exit_code") is not None or
            result.get("outcome") not in {"cli_error", "cancelled"} or
            result.get("termination", {}).get("confirmed") is not True or
            result.get("termination", {}).get("survivors") != [] or
            result.get("termination", {}).get("reason") != "not_started" or
            any(os.path.lexists(directory / name) for name in ("launch.json", "process.json", "exit.json", "cli-exit.json", "events.jsonl"))):
        return False
    try:
        invocation = _invocation(attempt)
        return bool(invocation.get("error") and result.get("attempt_sha256") == digest(attempt) and
                    result.get("launch_id") == invocation["launch_id"])
    except (ExtractError, KeyError):
        return False


def verified_result(attempt: dict, result: dict, termination: dict) -> dict:
    """Read native evidence and the actual exit record without changing a session."""
    from .agent import read_result
    if termination.get("confirmed") is not True:
        raise ExtractError("recovery result requires positively confirmed termination")
    if result.get("attempt_sha256") != digest(attempt):
        raise ExtractError("saved result does not bind this attempt")
    if never_launched(attempt, result):
        exit_code = None
    else:
        exited = _exit(attempt)
        exit_code = exited["exit_code"]
        if result.get("exit_code") != exit_code:
            raise ExtractError("saved result exit code differs from the actual exit record")
    outcome = result.get("outcome")
    if not isinstance(outcome, str) or not outcome:
        raise ExtractError("saved result has no explicit outcome")
    # Once the original terminal intent is unknown, later death cannot prove
    # success or undo a possible timeout/cancellation.
    if outcome == "termination_unconfirmed":
        outcome = "interrupted"
    recovered = read_result(attempt, exit_code=exit_code, termination=termination, outcome=outcome)
    if recovered["termination"].get("confirmed") is not True:
        raise ExtractError("recovery lacks confirmed descendant closure")
    for field in ("run_id", "attempt_id", "backend", "model", "effort", "session_scope", "session_directory",
                  "executable", "cli_version", "launch_id", "attempt_sha256", "response_schema_digest", "prompt_sha256",
                  "cwd", "workspace", "cwd_before", "cwd_after"):
        if result.get(field) != recovered.get(field):
            raise ExtractError("saved result invocation identity changed: " + field)
    if (result.get("response") != recovered["response"] or
            result.get("trace") != recovered["trace"] or
            result.get("session_id") != recovered["session_id"]):
        raise ExtractError("saved result differs from the original native events")
    return {**result, **recovered}


def recover_result(attempt: dict, result: dict, termination: dict) -> dict:
    """Replay native result settlement; missing results use interruption recovery."""
    from .agent import settle_session
    recovered = verified_result(attempt, result, termination)
    if never_launched(attempt, result):
        return {**recovered, "session_status": "not_started"}
    session = settle_session(attempt, result=recovered)
    return {**recovered, "session_status": session["status"]}


def _closed_ownership(attempt):
    return gate_reaped(attempt) or launch_not_started(attempt)


def termination_observed(attempt: dict, result: dict | None, *, table=None, launch_state=None) -> bool:
    """Read-only exit evidence; never stop a live scope or treat silence as exit."""
    directory = Path(attempt["directory"])
    table = process_table(cwd=directory) if table is None else table
    process_path = directory / "process.json"
    if process_path.exists():
        try:
            process = read_json(process_path)
            if process.get("launch_id") != _launch(attempt)["launch_id"]:
                return False
            identity = process["identity"]
            if any(process_alive(item, table) is not False for item in (identity, *process.get("descendants", []))):
                return False
            if any(item["pgid"] == identity["pid"] and not item["stat"].startswith("Z") for item in table.values()):
                return False
            return _closed_ownership(attempt)
        except (ExtractError, OSError, KeyError, TypeError, AttributeError):
            return False
    if launch_not_started(attempt, table=table):
        return True
    if never_launched(attempt, result):
        return True
    if result and result.get("termination", {}).get("confirmed"):
        try:
            return _closed_ownership(attempt) and _exit(attempt).get("termination", {}).get("confirmed") is True
        except ExtractError:
            return False
    return bool(launch_state == "prepared" and not any((directory / name).exists() for name in
                ("invocation.json", "launch.json", "process.json", "events.jsonl", "exit.json", "cli-exit.json", "result.json")) and result is None)


def scoped_restart_evidence(store, state: dict, scope: str, *, table=None, automatic=True) -> dict:
    """Bind one failed scope; sibling progress is not a change of authority."""
    from .runner import _budget
    from .runtime import verify_execution
    verify_execution(store, state)
    if "scoped_recovery.v2" not in state.get("executor", {}).get("capabilities", []):
        raise ExtractError("frozen executor lacks current scoped recovery capability")
    if state["status"] in {"paused", "cancelled", "queued", "ready"} or state["control"]["action"]:
        raise ExtractError("scope recovery is blocked by stopped control or completed work")
    stage, _, node = scope.partition(":")
    node = node or "root"
    if (stage != state["stage"] or scope_key(stage, node) != scope or
            (stage == "P5" and node not in {item["id"] for item in state.get("selected_designs") or []})):
        raise ExtractError("retry scope is no longer current")
    call = {"stage": stage, "node": node, "role": None, "mode": "normal"}
    owner, budget = owner_for(state, call), scope_for(state, call)
    reason = owner.get("reason") or {}
    if (owner.get("mode", "normal") != "normal" or owner["status"] not in NODE_STOPS - {"blocked"} or
            reason.get("code") in HARD_RECOVERY_STOPS or
            any(item.endswith(":single_decision_used") for item in reason.get("scopes", []))):
        raise ExtractError("scope is not an ordinary recoverable failure")
    if automatic and (reason.get("classification") != "transient_execution" or
                      reason.get("code") != "cli_retry" or reason.get("scope") != scope or
                      budget["cli_failures"] > state["config"]["retry_limit"]):
        raise ExtractError("scope has no explicit transient failure within its original retry limit")
    call["role"] = owner["role"]
    _, exhausted = _budget(call_view(state, call))
    if exhausted:
        raise ExtractError(exhausted)
    attempt_id = reason.get("attempt_id") or state.get("active_attempts", {}).get(scope)
    if not attempt_id and not automatic:
        usage = state.get("attempt_usage", {})
        attempt_id = next((key for key in reversed(state.get("attempts", []))
                           if usage.get(key, {}).get("stage") == stage and usage.get(key, {}).get("node") == node), None)
    if not attempt_id or str(UUID(attempt_id)) != attempt_id:
        raise ExtractError("scope failure has no canonical attempt identity")
    directory = store.run_dir(state["run_id"]) / "attempts" / attempt_id
    attempt = read_json(directory / "attempt.json")
    result = read_json(directory / "result.json") if (directory / "result.json").is_file() else None
    verify_execution(store, state, attempt=attempt)
    if (Path(attempt["directory"]) != directory or Path(attempt["workspace"]) != directory / "workspace" or
            attempt["run_id"] != state["run_id"] or attempt["input_fingerprint"] != state["input_fingerprint"] or
            attempt["attempt_id"] != attempt_id or scope_key(attempt["stage"], attempt["node"]) != scope or
            attempt["generation"] != (owner["generation"] if stage == "P5" else budget["generation"]) or
            attempt["mode"] != "normal" or attempt["role"] != owner["role"]):
        raise ExtractError("retry no longer matches its failed attempt identity")
    active = state.get("active_attempts", {}).get(scope)
    pending = attempt_id in {*state.get("unconfirmed_attempts", {}), *state.get("unsettled_attempts", {})}
    if active not in {None, attempt_id} or (active and attempt_id not in state.get("unconfirmed_attempts", {})):
        raise ExtractError("target scope still owns an invocation being processed")
    if automatic and (pending or active or not result or digest(result) != reason.get("result_sha256") or
                      retry_classification("cli_retry", result) != "transient_execution" or
                      not result.get("termination", {}).get("confirmed")):
        raise ExtractError("automatic retry requires a confirmed, settled transient failure")
    if not termination_observed(attempt, result, table=table,
                                launch_state=state.get("unsettled_attempts", {}).get(attempt_id)):
        raise ExtractError("target failure process or descendant is live or unknown")
    if automatic:
        observed = verified_result(attempt, result, result["termination"])
        if retry_classification("cli_retry", observed) != "transient_execution":
            raise ExtractError("native recovery evidence does not confirm a transient execution failure")
    dependencies, sources = {}, {}
    if stage == "P5":
        entry = next(item for item in state["selected_designs"] if item["id"] == node)
        if attempt.get("node_signature") != node_signature(entry) or state.get("reopening"):
            raise ExtractError("retry selected instance or upstream review changed")
        if not all(state["stages"][key]["status"] == "approved" for key in ("P1", "P2", "P3", "P4")):
            raise ExtractError("retry upstream approvals are not current")
        original, _ = store.verify_checkpoint(state, attempt["base_checkpoint"])
        current = store.validate_sources(state, attempt, original)
        dependencies = current["dependency_digests"]
        sources = current["adopted_source_identities"]
    elif attempt.get("plan_revision") != state["plan_revision"]:
        raise ExtractError("retry stage revision changed")
    target = {"generation": attempt["generation"], "checkpoint": call_view(state, call)["checkpoint"],
              "role": owner["role"], "mode": owner["mode"], "dependencies": dependencies,
              "source_identities": sources, "input_fingerprint": state["input_fingerprint"],
              "executor_digest": state["executor"]["digest"]}
    return {"run_id": state["run_id"], "scope": scope, "attempt_id": attempt_id,
            "failure_id": digest({"attempt": attempt, "result_sha256": digest(result), "reason": reason, "target": target}),
            "reason_sha256": digest(reason), "state_revision": state.get("state_revision"),
            "target": target, "settlement_required": bool(active or pending),
            "next_due_at": reason.get("next_due_at", 0),
            "failure_observed_at": reason.get("observed_at") or (result or {}).get("completed_at")}


def confirm_termination(attempt: dict, result: dict | None, *, launch_state=None) -> dict:
    """Missing ownership after launch is an explicit fence, never proof of exit."""
    directory = Path(attempt["directory"])
    try:
        if (directory / "process.json").exists():
            process = read_json(directory / "process.json")
            if process.get("launch_id") != _launch(attempt)["launch_id"]:
                raise ExtractError("process identity does not belong to this invocation launch")
            identity, known = process["identity"], process.get("descendants", [])
            if (type(identity.get("pid")) is not int or identity["pid"] <= 1 or
                    identity["pid"] == os.getpid() or not isinstance(identity.get("started"), str)):
                raise ExtractError("invalid invocation process identity")
            confirmation = terminate_owned(identity, known, cwd=directory)
            if confirmation["confirmed"] and not _closed_ownership(attempt):
                confirmation.update(confirmed=False, reason="invocation anchor exited without confirmed descendant closure")
        elif launch_not_started(attempt):
            confirmation = {"confirmed": True, "actions": [], "survivors": [],
                            "reason": "invocation-bound launch gate confirms exec did not start"}
        elif never_launched(attempt, result):
            confirmation = deepcopy(result["termination"])
        elif (result and result.get("termination", {}).get("confirmed") and _closed_ownership(attempt) and
              _exit(attempt).get("termination", {}).get("confirmed")):
            confirmation = deepcopy(result["termination"])
        elif (launch_state == "prepared" and result is None and
              not any((directory / name).exists() for name in
                      ("invocation.json", "launch.json", "process.json", "events.jsonl", "exit.json", "cli-exit.json", "result.json"))):
            confirmation = {"confirmed": True, "actions": [], "survivors": [],
                            "not_dispatched": True,
                            "reason": "Runner state records preparation without dispatch authorization"}
        else:
            raise ExtractError("invocation termination is unknown: no process identity or positive pre-dispatch record")
    except (ExtractError, KeyError, TypeError, AttributeError, OSError, subprocess.SubprocessError) as exc:
        confirmation = {"confirmed": False, "actions": [], "survivors": [], "reason": str(exc)}
    confirmation.update(observed_at=time.time(), attempt_id=attempt["attempt_id"], attempt_sha256=digest(attempt))
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
    for key in ("role", "mode", "session_group", "session_scope", "backend", "model", "effort", "executable",
                "base_checkpoint", "input_fingerprint", "created_at"):
        if key in attempt:
            entry.setdefault(key, attempt[key])
    entry.setdefault("requested_session_id", attempt.get("session_id"))
    entry.setdefault("usage", usage_summary({"backend": attempt.get("backend")}))
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
        if attempt.get("backend") == "codex" and result.get("usage_observation_sha256") and "usage_observation" not in entry:
            try:
                from .observation import FORMAT, observation_summary
                observed = read_json(Path(attempt["directory"]) / "usage-observation.json")
                if (digest(observed) != result["usage_observation_sha256"] or observed.get("format") != FORMAT or
                        any(observed.get("identity", {}).get(key) != attempt.get(key)
                            for key in ("run_id", "attempt_id")) or
                        observed.get("identity", {}).get("launch_id") != result.get("launch_id") or
                        observed.get("identity", {}).get("attempt_sha256") != digest(attempt) or
                        observed.get("session_id") != result.get("session_id")):
                    raise ExtractError("usage observation binding differs from its invocation")
                entry["usage_observation"] = observation_summary(observed)
                evidence.setdefault("usage-observation.json", digest(observed))
            except (ExtractError, OSError, ValueError, KeyError, TypeError) as exc:
                entry["audit_issues"] = sorted({*entry.get("audit_issues", []), "usage observation unavailable: " + str(exc)})
        for key in ("session_id", "session_status", "outcome", "cli_version", "started_at", "completed_at",
                    "requested_model", "requested_effort", "reported_model", "reported_models", "reported_effort", "native_status", "raw_logs"):
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
        for name in ("invocation.json", "response.schema.json", "response.json", "exit.json", "cli-exit.json",
                     "launch.json", "process.json", "termination.json", "receipt.json", "validated.json",
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


def handoff_session(state: dict, attempt: dict, result: dict | None, reason: str, *, termination=None) -> dict:
    from .agent import retire_session
    existing = next((item for item in state.setdefault("session_handoffs", [])
                     if item["attempt_id"] == attempt["attempt_id"]), None)
    if existing:
        return existing
    directory = Path(attempt["directory"])
    termination = termination or (read_json(directory / "termination.json")
        if (directory / "termination.json").is_file() else (result or {}).get("termination"))
    if not termination or termination.get("confirmed") is not True:
        raise ExtractError("cannot hand off a session before termination is confirmed")
    prepared = (termination.get("not_dispatched") is True and termination.get("attempt_sha256") == digest(attempt) and
                not any((directory / name).exists() for name in ("invocation.json", "launch.json", "process.json", "events.jsonl", "exit.json", "cli-exit.json")))
    if never_launched(attempt, result) or prepared:
        # Agent never acquired this attempt's session. The prior owner and
        # exact UUID remain usable; there is no interrupted session to retire.
        return {"attempt_id": attempt["attempt_id"], "status": "not_started"}
    old = (result or {}).get("session_id") or attempt.get("session_id")
    if not old and (directory / "events.jsonl").is_file():
        parsed = parse_events(directory / "events.jsonl", backend=attempt["backend"])
        if len(parsed["trace"]["session_ids"]) == 1:
            old = parsed["trace"]["session_ids"][0]
    retire_session(attempt, termination=termination, reason=reason)
    roles = state["sessions"].setdefault(attempt["session_group"], {"worker": None, "reviewer": None})
    if roles[attempt["role"]] in {None, old, attempt.get("session_id")}:
        roles[attempt["role"]] = None
    checkpoint = (state["nodes"][attempt["node"]]["draft_checkpoint"]
                  if attempt["stage"] == "P5" else state["checkpoint"])
    record = {"attempt_id": attempt["attempt_id"], "session_group": attempt["session_group"],
              "backend": attempt["backend"], "session_scope": attempt["session_scope"],
              "stage": attempt["stage"], "node": attempt["node"], "role": attempt["role"],
              "previous_session_id": old, "next_attempt_id": None, "next_session_id": None,
              "checkpoint": checkpoint, "input_fingerprint": state["input_fingerprint"],
              "termination": termination, "reason": reason, "at": time.time()}
    state["session_handoffs"].append(record)
    write_json(directory / "handoff.json", record)
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
