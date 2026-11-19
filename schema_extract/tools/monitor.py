"""Small observations and attributable controls over the existing run records."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import UUID, uuid4

from .common import ExtractError, RunnerBusy, SupervisorBusy, digest, read_json, relative_name, rules_root
from .processes import (launch_not_started, process_alive, process_table,
                        process_identity_matches, identity_key, owned_processes, signal_process,
                        terminate_owned, observe_owned, gate_reaped, GATE_PROTOCOL)
from .recovery import scoped_restart_evidence, termination_observed
from .prompts import render_operation


REASON_KINDS = {
    **dict.fromkeys(("missing_input", "blocked_nodes", "external_confirmation", "adjudication_blocked",
                     "checker_input_error", "invalid_root_prerequisite"), "operator_blocked"),
    **dict.fromkeys(("call_limit_reached", "adjudication_already_used", "interrupted_single_decision",
                     "reopen_already_used", "cli_retry_limit_reached", "receipt_repair_limit_reached"),
                   "call_limit_or_single_decision"),
    **dict.fromkeys(("checker_tool_error", "authority_changed", "proposal_rejected", "unreadable_result",
                     "invalid_final_prerequisites"), "framework_or_protocol"),
    "termination_unconfirmed": "unknown_termination", "cli_retry": "execution_failure", "receipt_repair": "protocol_repair",
}


def _processes():
    try:
        return process_table(), None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return None, str(exc)


def _optional(path: Path) -> dict | None:
    if not path.exists():
        return None
    value = read_json(path)
    if not isinstance(value, dict):
        raise ExtractError("expected an object: " + str(path))
    return value


def _attempt_observation(store, run: dict, attempt_id: str, table, now: float) -> dict:
    if str(UUID(attempt_id)) != attempt_id:
        raise ExtractError("invalid active attempt UUID")
    directory = store.run_dir(run["run_id"]) / "attempts" / attempt_id
    attempt = _optional(directory / "attempt.json")
    if not attempt or attempt.get("run_id") != run["run_id"] or attempt.get("attempt_id") != attempt_id:
        raise ExtractError("active attempt identity is missing or mismatched: " + str(directory))
    process, launch, result = (_optional(directory / name) for name in ("process.json", "launch.json", "result.json"))
    logs = {}
    for name in ("events.jsonl", "stderr.txt"):
        try:
            info = (directory / name).stat()
            logs[name] = {"size": info.st_size, "mtime": info.st_mtime}
        except FileNotFoundError:
            pass
    living = []
    known = [process["identity"], *process.get("descendants", [])] if process else []
    if table is not None:
        living = sorted({item["pid"] for item in known if process_alive(item, table)})
        while living:
            children = {pid for pid, item in table.items() if item["ppid"] in living and not item["stat"].startswith("Z")}
            if children <= set(living):
                break
            living = sorted(set(living) | children)
    confirmed = bool(result and result.get("termination", {}).get("confirmed"))
    if living:
        status = "running"
    elif table is None:
        status = "unknown"
    elif confirmed and termination_observed(attempt, result, table=table):
        status = "terminated"
    elif launch_not_started(attempt, table=table):
        status = "not_dispatched"
    elif (run.get("unsettled_attempts", {}).get(attempt_id) == "prepared" and
          not launch and not logs and not result and not process):
        status = "not_dispatched"
    else:
        status = "unknown"
    scope = attempt["stage"] + (":" + attempt["node"] if attempt["stage"] == "P5" else "")
    from .audit import usage_summary
    return {"attempt_id": attempt_id, "scope": scope, "role": attempt["role"], "status": status,
            "backend": attempt.get("backend"), "requested_model": attempt.get("model"),
            "requested_effort": attempt.get("effort"), "session_scope": attempt.get("session_scope"),
            "session_id": (result or {}).get("session_id"),
            "reported_model": (result or {}).get("reported_model"),
            "reported_effort": (result or {}).get("reported_effort"),
            "usage": usage_summary((result or {}).get("usage") or {"backend": attempt.get("backend")}),
            "process_ids": living, "termination_recorded": confirmed,
            "logs": logs, "outcome": result.get("outcome") if result else None, "evidence": str(directory)}


def classify(run: dict) -> list[dict]:
    """Only explicit failure evidence is transient; silence is ordinary waiting."""
    anomalies = []
    runner = run.get("runner_observation", {})
    launch = run.get("launch") or {}
    if run.get("status") == "queued" and (launch.get("status") == "launch_failed" or
            launch.get("alive") is False and launch.get("status") in {"launching", "claimed"}):
        anomalies.append({"run_id": run["run_id"], "scope": "startup", "classification": "execution_failure",
                          "summary": "已登记的启动进程在任务仍排队时退出；保留原运行身份。",
                          "evidence": run["state_directory"] + "/runner.log",
                          "recovery_condition": "核对启动日志；重复原命令对账，保留已用计数。"})
    if run.get("status") == "running" and runner.get("lock") != "held":
        if launch.get("alive") is True and launch.get("status") in {"launching", "claimed"}:
            pass  # An owned startup window is waiting, not a deadlock.
        else:
            confirmed = runner.get("alive") is False and runner.get("lock") in {"free", "absent"}
            anomalies.append({"run_id": run["run_id"], "scope": run.get("stage"),
                              "classification": "runner_exited" if confirmed else "unknown",
                              "summary": "Runner 已退出且未持写锁。" if confirmed else "Runner 身份与写锁证据不足或不一致。",
                              "evidence": run["state_directory"] + "/runner.lock",
                              "recovery_condition": "核对持久启动身份和未结调用；确认旧进程终止后沿原记录显式恢复。"})
    reasons = [(run.get("stage"), run.get("reason"))]
    reasons += [("P5:" + node, detail.get("reason")) for node, detail in run.get("node_details", {}).items()
                if detail.get("reason")]
    for scope, reason in reasons:
        if not reason:
            continue
        code = reason.get("code", "unknown")
        kind = ("transient_execution" if reason.get("classification") == "transient_execution" else
                REASON_KINDS.get(code, "unknown"))
        anomaly = {"run_id": run["run_id"], "scope": scope, "classification": kind,
                          "code": code, "summary": reason.get("summary", code),
                          "evidence": run["state_directory"] + "/state.json",
                          "recovery_condition": reason.get("recovery_condition", "核对固定证据和已用计数后显式控制。"),
                          "suggested_action": "核对工具与协议证据及本次冻结身份" if kind == "framework_or_protocol" else "保留其他任务，核实本范围恢复条件"}
        if kind == "framework_or_protocol":
            attempt_id = reason.get("attempt_id")
            if not attempt_id and scope != "P5" and not str(scope).startswith("P5:"):
                attempt_id = run.get("last_attempt_id")
            evidence = [anomaly["evidence"]]
            if attempt_id:
                evidence.append(run["state_directory"] + "/attempts/" + str(attempt_id))
            anomaly["engineering_handoff"] = {"problem": anomaly["summary"],
                "scope": scope, "executor": run.get("executor"), "instructions": render_operation("monitor_handoff"),
                "read_only_evidence": evidence,
                "reproduce_entry": [sys.executable, "schema_extract/extract_yaml.py", "inspect", "--run", run["run_id"],
                                    "--state-root", str(Path(run["state_directory"]).parent.parent), "--json"],
                "recovery_condition": anomaly["recovery_condition"]}
        anomalies.append(anomaly)
    for attempt in run.get("invocations", []):
        if attempt.get("status") == "unknown" or attempt.get("error"):
            anomalies.append({"run_id": run["run_id"], "scope": attempt.get("scope"),
                              "classification": "unknown",
                              "evidence": attempt["evidence"], "summary": attempt.get("error", "核对本次调用的进程身份、归属及结果记录。"),
                              "recovery_condition": "确认旧调用及其子进程终止；不能按日志沉默判断死锁。"})
    return anomalies


def inspect_runs(store, run_ids=None) -> list[dict]:
    table, process_error = _processes()  # Exactly one process-table observation per scan.
    now, rows = time.time(), []
    try:
        ledger, ledger_error = store.read_supervisor(), None
    except Exception as exc:
        ledger, ledger_error = {}, str(exc)
    for run_id in sorted(set(run_ids if run_ids is not None else store.run_ids())):
        try:
            view = store.read_summary(run_id)
            if ("frozen_execution.v1" not in view.get("executor", {}).get("capabilities", []) or
                    view.get("execution", {}).get("backend") not in {"claude", "codex"}):
                raise ExtractError("observation requires the current frozen run and explicit backend identity")
            active = dict.fromkeys([*view.get("active_attempts", {}).values(),
                                   *view.get("unconfirmed_attempts", {}), *view.get("unsettled_attempts", {})])
            invocations = []
            for attempt_id in active:
                try:
                    invocations.append(_attempt_observation(store, view, attempt_id, table, now))
                except Exception as exc:
                    invocations.append({"attempt_id": attempt_id, "status": "unknown", "error": str(exc),
                                        "evidence": str(store.run_dir(run_id) / "attempts" / str(attempt_id))})
            launch = deepcopy(ledger.get("starts", {}).get(run_id, {}).get("launch", {}))
            launch["alive"] = process_alive(launch.get("identity") or launch.get("launcher"), table)
            identity = (view.get("runner") or {}).get("identity") or launch.get("identity")
            lock_status = store.writer_status(run_id)
            view.update(invocations=invocations,
                        runner_observation={"lock": lock_status, "alive": process_alive(identity, table), "identity": identity},
                        global_hold=ledger.get("active_hold"),
                        launch=launch,
                        gate_capable="hold.v1" in view.get("executor", {}).get("capabilities", []))
            view["maintenance_request"] = deepcopy(ledger.get("maintenance_requests", {}).get(run_id))
            view["maintenance_gate_capable"] = "current_maintenance.v1" in view.get("executor", {}).get("capabilities", [])
            if process_error or ledger_error:
                view["observer_error"] = process_error or ledger_error
            view["anomalies"] = classify(view)
            rows.append(view)
        except Exception as exc:
            rows.append({"run_id": run_id, "error": str(exc), "observation": {"status": "unknown", "observed_at": now}})
    return rows


def scan(store, run_ids=None) -> dict:
    rows = inspect_runs(store, run_ids)
    now = time.time()
    with store.supervisor() as ledger:
        return record_scan(ledger, rows, now)


def record_scan(ledger: dict, rows: list[dict], observed_at: float) -> dict:
    """Commit an already collected observation; the caller holds the control lock.

    A bound clock scan uses the same transaction for this record and its next
    reminder time. No process observation, waiting or task control happens here.
    """
    observations = ledger["observations"]
    latest = observations.setdefault("latest", {})
    changes, superseded = [], []
    for row in rows:
        run_id = row["run_id"]
        prior = latest.get(run_id, {})
        sampled_at = row.get("observation", {}).get("observed_at", observed_at)
        if prior.get("observed_at") is not None and sampled_at < prior["observed_at"]:
            superseded.append(run_id)
            continue
        stable = prior.get("state_revision") is not None and prior.get("state_revision") == row.get("state_revision")
        row["unchanged_since"] = prior.get("unchanged_since", sampled_at) if stable else sampled_at
        if (stable and sampled_at - row["unchanged_since"] >= 300 and row.get("status") == "running" and
                not row.get("active_attempts") and not row.get("unsettled_attempts") and
                row.get("runner_observation", {}).get("lock") == "held" and row["runner_observation"]["alive"] is True):
            row["anomalies"].append({"run_id": row["run_id"], "scope": row.get("stage"), "classification": "possible_stall",
                                     "summary": "两次观察间活 Runner 无在途调用、state 修订长期不变；需诊断。",
                                     "evidence": row["state_directory"] + "/state.json",
                                     "recovery_condition": "核对准备/处理边界；此观察不授权终止或重试。"})
        compact = {key: row.get(key) for key in
                   ("status", "stage", "checkpoint", "published", "state_revision", "unchanged_since",
                    "runner_observation", "error", "anomalies")}
        comparable = {key: value for key, value in prior.items() if key != "observed_at"}
        if comparable != compact:
            changes.append({"run_id": run_id, "before": deepcopy(latest.get(run_id)),
                            "after": {**compact, "observed_at": sampled_at}})
        latest[run_id] = {**compact, "observed_at": sampled_at}
    observations.update(observed_at=max(observations.get("observed_at", observed_at), observed_at),
                        changes=(observations.get("changes", []) +
                                 [{"at": item["after"]["observed_at"], **item} for item in changes])[-64:])
    return {"observed_at": observed_at, "runs": rows, "changes": changes, "superseded_run_ids": superseded,
            "anomalies": [item for row in rows for item in row.get("anomalies", [])]}


def _cancel_binding(state):
    return {"input_fingerprint": state["input_fingerprint"], "executor_digest": state["executor"]["digest"]}


def _cancel_attempts(store, state, attempt_ids=()):
    """Read only the already registered unfinished attempts and their gate bindings."""
    records, unknown = {}, []
    ids = set(state.get("active_attempts", {}).values()) | set(state.get("unconfirmed_attempts", {})) | set(state.get("unsettled_attempts", {}))
    for attempt_id in sorted(ids | set(attempt_ids)):
        try:
            if str(UUID(attempt_id)) != attempt_id or attempt_id not in state["attempts"]:
                raise ExtractError("unregistered attempt identity")
            directory = store.run_dir(state["run_id"]) / "attempts" / attempt_id
            attempt = read_json(directory / "attempt.json")
            if (attempt.get("attempt_id") != attempt_id or attempt.get("run_id") != state["run_id"] or
                    attempt.get("input_fingerprint") != state["input_fingerprint"] or
                    attempt.get("directory") != str(directory)):
                raise ExtractError("attempt differs from the requested frozen run")
            launch = _optional(directory / "launch.json")
            process = _optional(directory / "process.json")
            invocation = _optional(directory / "invocation.json")
            if launch is None and process is None:
                if state.get("unsettled_attempts", {}).get(attempt_id) == "prepared":
                    records[attempt_id] = {"attempt": attempt, "not_dispatched": True}
                    continue
                raise ExtractError("unfinished attempt has no gate record")
            if launch_not_started(attempt):
                records[attempt_id] = {"attempt": attempt, "not_dispatched": True}
                continue
            if (not invocation or invocation.get("attempt_sha256") != digest(attempt) or
                    any(invocation.get(key) != attempt.get(key) for key in
                        ("run_id", "attempt_id", "backend", "model", "effort", "session_scope")) or
                    not launch or launch.get("protocol") != GATE_PROTOCOL or
                    launch.get("launch_id") != invocation.get("launch_id") or
                    launch.get("attempt_id") != attempt_id or launch.get("run_id") != state["run_id"] or
                    launch.get("attempt_sha256") != digest(attempt) or
                    not process or process.get("launch_id") != launch.get("launch_id") or
                    not process.get("identity") or
                    (launch.get("identity") is not None and
                     process_identity_matches(process["identity"], launch["identity"]) is not True)):
                raise ExtractError("unfinished attempt gate identity is unconfirmed")
            records[attempt_id] = {"attempt": attempt, "identity": process["identity"],
                                   "descendants": process.get("descendants", []), "launch_id": launch["launch_id"]}
        except (ExtractError, OSError, KeyError, TypeError, ValueError) as exc:
            unknown.append({"attempt_id": attempt_id, "reason": str(exc)})
    return records, unknown


def deliver_cancel(store, request: dict) -> dict:
    """Persist delivery ownership before one KILL pass; never acquire runner.lock."""
    run_id, request_id = request["run_id"], request["id"]
    state = store.load(run_id)
    ledger = store.read_supervisor()
    control = ledger["controls"].get(run_id, {})
    previous = control.get("deliveries", {}).get(request_id, {})
    binding = _cancel_binding(state)
    if previous.get("confirmed") is True:
        return deepcopy(previous)

    def authorized(current):
        value = current["controls"].get(run_id, {})
        launch = current["starts"].get(run_id, {}).get("launch", {})
        return (request.get("action") == "cancel" and request.get("binding") == binding and
                request["control_revision"] > launch.get("control_revision", 0) and
                value.get("action") in {"cancel", "pause"} and
                value.get("revision", 0) >= request["control_revision"])
    if not authorized(ledger):
        return {"status": "superseded", "confirmed": False}
    runner = previous.get("runner") or (state.get("runner") or {}).get("identity")
    launch = ledger["starts"].get(run_id, {}).get("launch", {})
    table, process_error = _processes()
    startup_unknown = False
    if launch.get("status") in {"launching", "claimed"}:
        launch_identity = launch.get("identity")
        if (launch.get("input_fingerprint") != state["input_fingerprint"] or
                launch.get("executor_digest") != state["executor"]["digest"] or not launch_identity):
            startup_unknown = True
        elif not runner or process_alive(runner, table) is False:
            runner = launch_identity
        elif process_identity_matches(runner, launch_identity) is not True:
            startup_unknown = True
    known = previous.get("runner_descendants", [])
    if runner and table is not None:
        known = list({identity_key(value): value for value in [*known, *owned_processes(runner, known, table=table)]}.values())
    records, unknown = _cancel_attempts(store, state, previous.get("targets", {}))
    if startup_unknown:
        unknown.append({"scope": "startup", "reason": "Runner launch ownership is unconfirmed"})
    delivery = {**previous, "request_id": request_id, "binding": binding, "status": "pending", "confirmed": False,
                "runner": runner, "runner_descendants": known, "requested_at": previous.get("requested_at", time.time()),
                "attempts": previous.get("attempts", {}), "actions": previous.get("actions", []), "unknown": unknown,
                "targets": {key: {name: value for name, value in record.items() if name != "attempt"}
                            for key, record in records.items()}}
    if process_error:
        delivery["observer_error"] = process_error
    with store.supervisor() as ledger:
        latest = ledger["controls"].get(run_id, {})
        if not authorized(ledger):
            return {"status": "superseded", "confirmed": False}
        latest.setdefault("deliveries", {})[request_id] = deepcopy(delivery)
    if table is None:
        return delivery

    def current_cancel():
        return authorized(store.read_supervisor())

    def signal_target(identity):
        if not current_cancel():
            return
        if identity.get("pid") == os.getpid():
            raise ExtractError("cancel delivery cannot terminate its own control process")
        if signal_process(identity):
            delivery["actions"].append({"pid": identity["pid"], "start_token": identity["start_token"], "signal": "SIGKILL"})

    try:
        # Stop synchronous Python/check work independently of its writer lock.
        if runner and process_alive(runner, table) is True:
            signal_target(runner)
        # Re-read committed gates after the signal. Only a confirmed exited
        # Runner permits auxiliary selection from this captured ownership tree.
        current = store.load(run_id)
        if _cancel_binding(current) != binding:
            raise ExtractError("frozen run changed during cancel delivery")
        records, unknown = _cancel_attempts(store, current, delivery["targets"])
        if startup_unknown:
            unknown.append({"scope": "startup", "reason": "Runner launch ownership is unconfirmed"})
        delivery["unknown"] = unknown
        delivery["targets"] = {key: {name: value for name, value in record.items() if name != "attempt"}
                               for key, record in records.items()}
        with store.supervisor() as ledger:
            latest = ledger["controls"].get(run_id, {})
            if not authorized(ledger):
                return {"status": "superseded", "confirmed": False}
            latest.setdefault("deliveries", {})[request_id] = deepcopy(delivery)
        runner_exited = not runner or process_alive(runner, process_table()) is False
        gates = {identity_key(record["identity"]) for record in records.values() if record.get("identity")}
        for attempt_id, record in records.items():
            if record.get("not_dispatched"):
                termination = {"confirmed": True, "actions": [], "survivors": [], "reason": "prepared; no gate dispatched"}
            else:
                termination = (terminate_owned(record["identity"], record["descendants"]) if current_cancel()
                               else observe_owned(record["identity"], record["descendants"]))
                if termination["confirmed"] and not (gate_reaped(record["attempt"]) or launch_not_started(record["attempt"])):
                    termination.update(confirmed=False, reason="gate descendant closure is not recorded")
            delivery["attempts"][attempt_id] = termination
        # Unknown attempt gates may be among these children; do not guess which
        # child is an auxiliary process when any gate identity is unavailable.
        if not unknown and runner_exited:
            for identity in reversed(known):
                if identity_key(identity) not in gates and (not runner or identity_key(identity) != identity_key(runner)):
                    signal_target(identity)
        observed = (observe_owned(runner, known) if runner else
                    {"confirmed": store.writer_status(run_id) in {"free", "absent"}, "survivors": [], "actions": []})
        delivery["runner_termination"] = observed
        delivery["runner_descendants"] = observed.get("descendants", known)
        confirmed = (observed["confirmed"] and store.writer_status(run_id) in {"free", "absent"} and
                     not unknown and all(value["confirmed"] for value in delivery["attempts"].values()))
        delivery.update(confirmed=confirmed, status="confirmed" if confirmed else "pending", observed_at=time.time())
    except (ExtractError, OSError) as exc:
        delivery.update(status="pending", confirmed=False, observer_error=str(exc))
    try:
        with store.supervisor() as ledger:
            latest = ledger["controls"].get(run_id, {})
            if not authorized(ledger):
                return {"status": "superseded", "confirmed": False}
            latest.setdefault("deliveries", {})[request_id] = deepcopy(delivery)
    except SupervisorBusy:
        delivery.update(status="pending", confirmed=False, evidence_committed=False,
                        reason="cancel request is committed; delivery observation could not acquire supervisor.lock")
    return delivery


def hold_all(store) -> dict:
    with store.supervisor() as ledger:
        if not ledger["active_hold"]:
            hold_id = str(uuid4())
            ledger["active_hold"] = hold_id
            ledger["holds"][hold_id] = {"id": hold_id, "created_at": time.time(),
                                          "revision": ledger["revision"] + 1, "status": "active", "members": {}}
        hold_id = ledger["active_hold"]
    # The gate is durable before any run is inspected or asked to pause.
    for run_id in store.run_ids():
        try:
            state = store.load(run_id)
            with store.supervisor() as ledger:
                if ledger["active_hold"] != hold_id:
                    break
                member = store.hold_member(ledger, state)
                if (member.get("request_id") and state["status"] == "paused" and
                        state.get("control", {}).get("hold_id") == hold_id and not state["active_attempts"] and
                        not state.get("unconfirmed_attempts") and not state.get("unsettled_attempts")):
                    member["status"] = "paused_by_hold"
        except Exception as exc:
            with store.supervisor() as ledger:
                ledger["holds"][hold_id]["members"][run_id] = {"status": "unknown", "error": str(exc)}
    return deepcopy(store.read_supervisor()["holds"][hold_id])


def release_hold(store, hold_id: str, repo: Path) -> dict:
    with store.supervisor() as ledger:
        if hold_id not in ledger["holds"] or ledger["active_hold"] not in {None, hold_id}:
            raise ExtractError("hold ID is not the active or previously released hold")
        hold = ledger["holds"][hold_id]
        ledger["active_hold"] = None
        hold.update(status="released", released_at=hold.get("released_at", time.time()))
        members = deepcopy(hold["members"])
    for run_id, member in members.items():
        try:
            state = store.load(run_id)
            ledger = store.read_supervisor()
            latest = ledger["controls"].get(run_id, {})
            start = ledger["starts"].get(run_id, {})
            requests = store.control_requests(run_id)
            manual_after = any(not request.get("hold_id") and request["created_at"] >= ledger["holds"][hold_id]["created_at"]
                               for request in requests)
            if manual_after:
                result = {"status": "preserved", "reason": "实际请求目录有后续手动控制。"}
            elif member["status"] == "queued_blocked":
                if state["status"] == "queued" and start.get("authorized"):
                    result = start_background(store, run_id, repo, mode="queued")
                else:
                    result = {"status": "queued" if state["status"] == "queued" else "preserved"}
            elif (member.get("request_id") != latest.get("id") or latest.get("hold_id") != hold_id or
                  latest.get("revision") != member.get("control_revision")):
                result = {"status": "preserved", "reason": "后续手动控制覆盖了本 hold。"}
            elif (state["status"] != "paused" or state["control"].get("hold_id") != hold_id or
                  state["active_attempts"] or state.get("unconfirmed_attempts") or state.get("unsettled_attempts")):
                result = {"status": "pending", "reason": "尚未确认由本 hold 暂停；保留原状态，再次释放可重新核对。"}
            else:
                result = start_background(store, run_id, repo, mode="hold", hold_id=hold_id)
            member["release"] = result
        except Exception as exc:
            member["release"] = {"status": "unknown", "error": str(exc)}
    with store.supervisor() as ledger:
        ledger["holds"][hold_id]["members"].update(members)
    return deepcopy(store.read_supervisor()["holds"][hold_id])


def _identity(table=None) -> dict:
    table = process_table() if table is None else table
    identity = table.get(os.getpid()) if table is not None else None
    if not identity:
        raise ExtractError("cannot establish launcher process identity")
    return identity


def reserve_launch(store, state: dict, *, mode="resume", hold_id=None, retry_id=None, background=False,
                   add_calls=0) -> dict:
    from .maintenance import barrier
    from .runtime import verify_execution
    verify_execution(store, state)
    if mode not in {"resume", "queued", "retry", "hold", "maintenance", "export"}:
        raise ExtractError("unknown current-run launch operation")
    table = process_table()
    identity = _identity(table)
    requests = store.control_requests(state["run_id"])
    with store.supervisor() as ledger:
        store.assert_startable(state["run_id"], ledger)
        current = store.load(state["run_id"])
        if (current["input_fingerprint"], current["executor"]) != (state["input_fingerprint"], state["executor"]):
            raise ExtractError("frozen run identity changed before launch reservation")
        if mode == "maintenance" and not (barrier(store, current, ledger) or current.get("maintenance")):
            raise ExtractError("maintenance launch requires a current submitted request")
        if mode == "retry":
            request = ledger["retries"].get(retry_id)
            if not request or request["status"] != "authorized":
                raise ExtractError("scope recovery authorization was consumed or replaced")
            check_restart_authority(ledger, request, state)
        start = ledger["starts"].setdefault(state["run_id"], {"output": state["output"], "authorized": False,
                                                             "created_at": state["created_at"]})
        previous = start.get("launch", {})
        if previous.get("status") in {"launching", "claimed"}:
            owner = previous.get("identity") or previous.get("launcher")
            alive = process_alive(owner, table)
            if alive is not False:
                return {**previous, "reused": True}
        if mode == "queued" and state["status"] != "queued":
            return {"status": "preserved", "run_id": state["run_id"], "run_status": state["status"]}
        latest = ledger["controls"].get(state["run_id"], {})
        if any(not value.get("confirmed") for value in latest.get("deliveries", {}).values()):
            raise ExtractError("cancel termination is pending; reconcile its owned processes before another launch")
        if mode == "maintenance" and (state["status"] in {"paused", "cancelled"} or
                state["control"]["action"] or latest.get("action") in {"pause", "cancel"} or
                any(item["id"] not in state["control"]["requests"] for item in requests)):
            return {"status": "preserved", "run_id": state["run_id"], "reason": "manual control preserves the pending maintenance request"}
        if mode == "queued" and latest.get("action") in {"pause", "cancel"} and not latest.get("hold_id"):
            return {"status": "preserved", "run_id": state["run_id"], "reason": "pending manual control"}
        if mode == "queued" and any(not request.get("hold_id") and request["id"] not in state["control"]["requests"]
                                    for request in requests):
            return {"status": "preserved", "run_id": state["run_id"], "reason": "unconsumed manual control request"}
        if mode == "resume":
            ledger["controls"][state["run_id"]] = {"revision": latest.get("revision", 0) + 1,
                "action": "resume", "id": str(uuid4()), "hold_id": None, "at": time.time(),
                "deliveries": deepcopy(latest.get("deliveries", {}))}
        if mode in {"resume", "queued"}:
            start["authorized"] = True
        launch = {"id": str(uuid4()), "run_id": state["run_id"], "mode": mode,
                  "hold_id": hold_id, "retry_id": retry_id, "status": "launching", "background": background,
                  "input_fingerprint": state["input_fingerprint"], "executor_digest": state["executor"]["digest"],
                  "launcher": identity, "created_at": time.time(),
                  "request_ids": [request["id"] for request in requests],
                  "control_revision": ledger["controls"].get(state["run_id"], {}).get("revision", 0)}
        if add_calls:
            from .stages import validate_add_calls
            launch["add_calls"] = validate_add_calls(add_calls)
        if not background:
            launch.update(status="claimed", identity=identity)
        start["launch"] = launch
        return deepcopy(launch)


def claim_launch(store, run_id: str, launch_id: str) -> dict:
    identity = _identity()
    with store.supervisor() as ledger:
        store.assert_startable(run_id, ledger)
        launch = ledger["starts"].get(run_id, {}).get("launch")
        if not launch or launch["id"] != launch_id or launch["status"] != "launching":
            raise ExtractError("background launch was replaced or already claimed")
        state = store.load(run_id)
        if (state["input_fingerprint"] != launch.get("input_fingerprint") or
                state["executor"]["digest"] != launch.get("executor_digest")):
            raise ExtractError("frozen run identity changed after launch reservation")
        if launch["mode"] == "retry":
            request = ledger["retries"].get(launch.get("retry_id"))
            if not request or request["status"] != "authorized":
                raise ExtractError("scope recovery authorization was consumed or replaced")
            check_restart_authority(ledger, request, state)
        launch.update(status="claimed", identity=identity, claimed_at=time.time())
        return deepcopy(launch)


def start_background(store, run_id: str, repo: Path, *, mode="queued", hold_id=None, retry_id=None) -> dict:
    state = store.load(run_id)
    if store.read_supervisor()["active_hold"]:
        with store.supervisor() as ledger:
            store.hold_member(ledger, state)
        return {"run_id": run_id, "status": "held", "hold_id": store.read_supervisor()["active_hold"]}
    # Snapshot verification is deliberately outside supervisor.lock.
    from .runtime import frozen_run_invocation
    frozen_run_invocation(store, state, [])
    launch = reserve_launch(store, state, mode=mode, hold_id=hold_id, retry_id=retry_id, background=True)
    if launch.get("reused") or launch["status"] != "launching":
        return launch
    command = [sys.executable, "-B", str(rules_root() / "schema_extract/extract_yaml.py"),
               "_background", run_id, "--launch-id", launch["id"], "--repo", str(repo), "--state-root", str(store.root)]
    try:
        log_path = store.run_dir(run_id) / "runner.log"
        with log_path.open("ab") as stream:
            process = subprocess.Popen(command, cwd=store.run_dir(run_id), stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=stream, start_new_session=True, close_fds=True)
        table, _ = _processes()
        identity = table.get(process.pid) if table is not None else None
        exited = process.poll()
        with store.supervisor() as ledger:
            current = ledger["starts"][run_id]["launch"]
            if current["id"] == launch["id"]:
                current.update(pid=process.pid, log=str(log_path))
                if current["status"] == "launching":
                    if exited is not None:
                        current.update(status="launch_failed", exit_code=exited,
                                       error="background Runner exited before claiming its launch")
                    elif identity:
                        current["identity"] = identity
                launch = deepcopy(current)
        return launch
    except OSError as exc:
        with store.supervisor() as ledger:
            current = ledger["starts"][run_id]["launch"]
            if current["id"] == launch["id"]:
                current.update(status="launch_failed", error=str(exc))
        return {"run_id": run_id, "status": "launch_failed", "error": str(exc), "id": launch["id"]}


def catalog_targets(repo: Path, ids: list[str], output_root: Path) -> list[dict]:
    catalog = read_json(repo / "kernel_wiki/catalog.json")
    rows = []
    root = output_root.absolute()
    if root.resolve() != root:
        raise ExtractError("output-root must not traverse symlinks")
    for operator_id in ids:
        try:
            entries = [entry for entry in catalog["operators"] if entry["id"] == operator_id]
            if len(entries) != 1 or ids.count(operator_id) != 1:
                raise ExtractError("unknown or duplicate catalog ID: " + operator_id)
            relative = Path(relative_name(entries[0]["paths"]["readme"])).parent
            output = root / relative
            if relative == Path(".") or output.resolve() != output or not output.is_relative_to(root):
                raise ExtractError("catalog destination leaves output-root or traverses symlinks")
            rows.append({"operator_id": operator_id, "output": output})
        except (ExtractError, KeyError, TypeError) as exc:
            rows.append({"operator_id": operator_id, "error": str(exc)})
    for row in rows:
        if "output" in row and sum(other.get("output") == row["output"] for other in rows) > 1:
            row["error"] = "duplicate catalog destination"
    return rows


def registered_extract(repo: Path, store, output: Path, *, operator_id=None, source=None, bindings=None, config=None, queued=False):
    """One output reservation survives parent exit, duplicate CLI and creation crashes."""
    output = output.absolute()
    if output.resolve() != output:
        raise ExtractError("output must not traverse symlinks")
    source = str(Path(source).expanduser().resolve()) if source and not source.startswith("git:") else source
    bindings = bindings.resolve() if bindings else None
    key = digest({"repo": str(repo), "operator_id": operator_id, "source": source,
                  "bindings": str(bindings) if bindings else None, "config": config, "output": str(output)})
    if not store.read_supervisor().get("runs_indexed"):
        existing, errors = [], {}
        indexed = store.read_supervisor()["starts"]
        for run_id in store.run_ids():
            # Registration adopts only run/output ownership, never approvals.
            if run_id in indexed:
                continue
            try:
                state = store.load(run_id)
                existing.append((run_id, {"run_id": run_id, "output": state["output"], "operator_id": state["operator_id"],
                                         "created_at": state["created_at"], "authorized": False, "status": "preexisting"}))
            except Exception as exc:
                errors[run_id] = str(exc)
                try:
                    hint = _optional(store.run_dir(run_id) / "summary.json") or {}
                    if isinstance(hint.get("output"), str) and Path(hint["output"]).is_absolute():
                        # A stale hint can conservatively reserve an output;
                        # it cannot authorize execution of the damaged run.
                        existing.append((run_id, {"run_id": run_id, "output": hint["output"], "authorized": False,
                                                  "created_at": hint.get("created_at"), "status": "unknown"}))
                except Exception:
                    pass
        with store.supervisor() as ledger:
            for run_id, record in existing:
                ledger["starts"].setdefault(run_id, record)
            ledger["runs_indexed"] = True
            ledger["index_errors"] = errors
    with store.supervisor() as ledger:
        found = next(((run_id, value) for run_id, value in ledger["starts"].items() if value["output"] == str(output)), None)
        preexisting = False
        if found:
            run_id, registration = found
            if not registration.get("request_key"):
                if operator_id and registration.get("operator_id") not in {None, operator_id}:
                    raise ExtractError("output belongs to another operator run " + run_id)
                preexisting = True
            elif registration["request_key"] != key:
                raise ExtractError("output is already registered to run " + run_id +
                                   " with different request parameters; its frozen inputs are unchanged. "
                                   "Use status " + run_id + " --full, resume that run, or extract to a different --output for new inputs/parameters")
        else:
            if output.exists() or output.is_symlink():
                raise ExtractError("output must not exist")
            from .runner import normalize_config
            normalize_config(config)
            run_id = str(uuid4())
            registration = {"run_id": run_id, "output": str(output), "request_key": key,
                            "created_at": time.time(), "authorized": not queued, "status": "registered"}
            ledger["starts"][run_id] = registration
        registration = deepcopy(registration)
    if preexisting:
        return {"run_id": run_id, "status": "preexisting", "output": str(output), "reused": True,
                "authorized": False, "reason": "已存在的运行保留原身份；用显式 resume 控制。"}, store.load(run_id)
    directory = store.run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with store.writer(run_id):
            if (directory / "state.json").exists():
                state = store.load(run_id)
            else:
                from .runner import create_run
                _, state, _ = create_run(repo, store.root, output, operator_id=operator_id, source=source,
                                         bindings=bindings, config=config, queued=True, run_id=run_id,
                                         created_at=registration["created_at"])
    except RunnerBusy:
        return {"run_id": run_id, "status": "creation_or_runner_active", "output": str(output),
                "reused": True, "authorized": registration["authorized"],
                "background": store.read_supervisor()["starts"][run_id].get("launch")}, None
    with store.supervisor() as ledger:
        ledger["starts"][run_id]["status"] = "created"
        if ledger["active_hold"]:
            store.hold_member(ledger, state)
    return {"run_id": run_id, "status": state["status"], "output": str(output), "reused": found is not None,
            "authorized": registration["authorized"]}, state


def restart_evidence(store, state: dict, scope: str, *, table=None) -> dict:
    return scoped_restart_evidence(store, state, scope, table=table)


def check_restart_authority(ledger: dict, request: dict, state: dict) -> None:
    """Recheck controls at consumption; newer observations do not revoke work."""
    run_id = state["run_id"]
    from .maintenance import publication_barrier
    if publication_barrier(None, state, ledger):
        raise ExtractError("scope recovery is fenced by current maintenance")
    if ledger["active_hold"]:
        raise ExtractError("scope recovery was superseded by hold")
    latest = ledger["controls"].get(run_id, {})
    if (state["control"]["action"] or latest.get("action") in {"pause", "cancel"} or
            request["control_revision"] != latest.get("revision", 0) or
            set(state["control"]["requests"]) - set(request.get("request_ids", []))):
        raise ExtractError("scope recovery was superseded by manual control")


def request_restart(store, run_id: str, scope: str, repo: Path) -> dict:
    state = store.load(run_id)
    evidence = restart_evidence(store, state, scope)
    requests = store.control_requests(run_id)
    with store.supervisor() as ledger:
        store.assert_startable(run_id, ledger)
        from .maintenance import publication_barrier
        if publication_barrier(store, state, ledger):
            raise ExtractError("scope recovery is fenced by current maintenance")
        latest = ledger["controls"].get(run_id, {})
        if (latest.get("action") in {"pause", "cancel"} or
                any(item["id"] not in state["control"]["requests"] for item in requests)):
            raise ExtractError("scope recovery cannot override a pending pause, cancel or hold")
        previous = next((value for value in reversed(ledger["retries"].values())
                         if value["failure_id"] == evidence["failure_id"]), None)
        if previous and previous["status"] == "authorized":
            try:
                check_restart_authority(ledger, previous, state)
            except ExtractError as exc:
                previous.update(status="superseded", completed_at=time.time(), reason=str(exc))
                previous = None
        elif previous and previous["status"] == "superseded":
            previous = None
        if previous:
            request = deepcopy(previous)
        else:
            request = {**evidence, "id": str(uuid4()), "status": "authorized", "created_at": time.time(),
                       "control_revision": latest.get("revision", 0),
                       "request_ids": [item["id"] for item in requests]}
            ledger["retries"][request["id"]] = request
            request = deepcopy(request)
    if request["status"] != "authorized":
        return request
    writer = store.writer_status(run_id)
    if writer == "held":
        request["delivery"] = "active_runner"
    elif writer in {"free", "absent"}:
        request["launch"] = start_background(store, run_id, repo, mode="retry", retry_id=request["id"])
    else:
        request["delivery"] = "waiting" if writer in {"free", "absent"} else "writer_unknown"
    return request
