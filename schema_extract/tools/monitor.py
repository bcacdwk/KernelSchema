"""Small observations and attributable controls over the existing run records."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import UUID, uuid4

from .common import ExtractError, RunnerBusy, digest, read_json, relative_name, rules_root
from .processes import process_table


REASON_KINDS = {
    **dict.fromkeys(("missing_input", "blocked_nodes", "external_confirmation", "adjudication_blocked",
                     "checker_input_error", "invalid_root_prerequisite", "reopen_requires_new_evidence"), "operator_blocked"),
    **dict.fromkeys(("budget_exhausted", "checker_budget_exhausted", "continuation_budget_exhausted",
                     "adjudication_already_used", "interrupted_single_decision", "reopen_already_used",
                     "cli_retry_budget_exhausted", "receipt_repair_budget_exhausted"), "budget_or_single_decision"),
    **dict.fromkeys(("checker_tool_error", "authority_changed", "proposal_rejected", "unreadable_result",
                     "invalid_final_prerequisites"), "framework_or_protocol"),
    "termination_unconfirmed": "unknown_termination", "cli_retry": "execution_failure", "receipt_repair": "protocol_repair",
}


def _alive(identity, table) -> bool | None:
    if table is None or not isinstance(identity, dict):
        return None
    if type(identity.get("pid")) is not int or not isinstance(identity.get("started"), str):
        return None
    if identity["started"] in {"unobserved", "already_exited"}:
        return None
    current = table.get(identity["pid"])
    return bool(current and current["started"] == identity["started"] and not current["stat"].startswith("Z"))


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
    invocation = _optional(directory / "invocation.json") or {}
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
        living = sorted({item["pid"] for item in known if _alive(item, table)})
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
    elif confirmed:
        status = "terminated"
    elif (run.get("unsettled_attempts", {}).get(attempt_id) == "prepared" and
          not launch and not logs and not result and not process):
        status = "not_dispatched"
    else:
        status = "unknown"
    scope = attempt["stage"] + (":" + attempt["node"] if attempt["stage"] == "P5" else "")
    budget = run.get("budget", {})
    started = invocation.get("started_at", attempt.get("created_at"))
    limits = [value for value in (budget.get("deadline"), budget.get("scopes", {}).get(scope, {}).get("deadline"),
              started + attempt.get("timeout_seconds", budget.get("limits", {}).get("call_seconds", 0)) if started else None)
              if isinstance(value, (int, float))]
    deadline = min(limits) if limits else None
    return {"attempt_id": attempt_id, "scope": scope, "role": attempt["role"], "status": status,
            "process_ids": living, "termination_recorded": confirmed, "deadline": deadline,
            "overdue": bool(deadline and now >= deadline and status != "terminated"),
            "logs": logs, "outcome": result.get("outcome") if result else None, "evidence": str(directory)}


def classify(run: dict) -> list[dict]:
    """Only explicit failure evidence is transient; silence is ordinary waiting."""
    if run.get("successor"):
        return []  # Its immutable predecessor state remains historical evidence.
    anomalies = []
    runner = run.get("runner_observation", {})
    launch = run.get("launch") or {}
    if run.get("status") == "queued" and launch.get("alive") is False and launch.get("status") in {"launching", "claimed", "launch_failed"}:
        anomalies.append({"run_id": run["run_id"], "scope": "startup", "classification": "execution_failure",
                          "summary": "已登记的启动进程在任务仍排队时退出；保留原运行身份。",
                          "evidence": run["state_directory"] + "/runner.log",
                          "recovery_condition": "核对启动日志；重复原命令对账，不创建新预算。"})
    now = run.get("observation", {}).get("observed_at", time.time())
    if run.get("status") == "running" and not run.get("active_attempts") and not run.get("unsettled_attempts"):
        budget = run.get("budget", {})
        for scope, deadline in (("operator", budget.get("deadline")),
                                (run.get("stage"), budget.get("scopes", {}).get(run.get("stage"), {}).get("deadline"))):
            if isinstance(deadline, (int, float)) and now >= deadline:
                anomalies.append({"run_id": run["run_id"], "scope": scope, "classification": "deadline_exceeded",
                                  "summary": "当前截止已过且没有在途调用；需诊断 Runner 边界进展。",
                                  "evidence": run["state_directory"] + "/state.json",
                                  "recovery_condition": "保留原截止和计数；巡检不终止进程或自动重试。"})
    if run.get("status") == "running" and runner.get("lock") != "held":
        if launch.get("alive") is True and launch.get("status") in {"launching", "claimed"}:
            pass  # An owned startup window is waiting, not a deadlock.
        else:
            confirmed = runner.get("alive") is False and runner.get("lock") in {"free", "absent"}
            anomalies.append({"run_id": run["run_id"], "scope": run.get("stage"),
                              "classification": "runner_exited" if confirmed else "unknown",
                              "summary": "Runner 已退出且未持写锁。" if confirmed else "Runner 身份与写锁证据不足或不一致。",
                              "evidence": run["state_directory"] + "/runner.lock",
                              "recovery_condition": "核对持久启动身份和未结调用；确认旧进程终止后按原预算显式恢复。"})
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
                          "recovery_condition": reason.get("recovery_condition", "核对固定证据和原预算后显式控制。"),
                          "suggested_action": "定位工具/协议证据；确需换版时显式 continue" if kind == "framework_or_protocol" else "保留其他任务，核实本范围恢复条件"}
        if kind == "framework_or_protocol":
            anomaly["engineering_handoff"] = {"problem": anomaly["summary"],
                "scope": "schema_extract/tools 的对应调用、校验或恢复边界", "executor": run.get("executor"),
                "read_only_evidence": [anomaly["evidence"], run["state_directory"] + "/attempts/" + str(run.get("last_attempt_id") or "")],
                "reproduce_entry": [sys.executable, "schema_extract/extract.py", "inspect", "--run", run["run_id"],
                                    "--state-root", str(Path(run["state_directory"]).parent.parent), "--json"],
                "recovery_condition": anomaly["recovery_condition"]}
        anomalies.append(anomaly)
    for attempt in run.get("invocations", []):
        if attempt.get("status") == "unknown" or attempt.get("overdue") or attempt.get("error"):
            anomalies.append({"run_id": run["run_id"], "scope": attempt.get("scope"),
                              "classification": "unknown" if attempt.get("status") == "unknown" else "deadline_exceeded",
                              "evidence": attempt["evidence"], "summary": attempt.get("error", "核对本次调用的进程身份、原截止条件及结果记录。"),
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
            launch["alive"] = _alive(launch.get("identity") or launch.get("launcher"), table)
            identity = (view.get("runner") or {}).get("identity") or launch.get("identity")
            lock_status = store.writer_status(run_id)
            view.update(invocations=invocations,
                        runner_observation={"lock": lock_status, "alive": _alive(identity, table), "identity": identity},
                        global_hold=ledger.get("active_hold"),
                        launch=launch,
                        successor=ledger.get("continuations", {}).get(run_id),
                        gate_capable="hold.v1" in view.get("executor", {}).get("capabilities", []))
            if view["successor"]:
                view["disposition"] = "superseded" if view["successor"]["status"] == "ready" else "continuing"
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


def record_scan(ledger: dict, rows: list[dict], now: float) -> dict:
    """Commit an already collected observation; the caller holds the control lock.

    A bound clock scan uses the same transaction for this record and its next
    deadline. No process observation, waiting or task control happens here.
    """
    observations = ledger["observations"]
    previous = observations.get("latest", {})
    for row in rows:
        prior = previous.get(row["run_id"], {})
        stable = prior.get("state_revision") is not None and prior.get("state_revision") == row.get("state_revision")
        row["unchanged_since"] = prior.get("unchanged_since", now) if stable else now
        limit = min(300, row.get("budget", {}).get("limits", {}).get("call_seconds") or 300)
        if (not row.get("successor") and stable and now - row["unchanged_since"] >= limit and row.get("status") == "running" and
                not row.get("active_attempts") and not row.get("unsettled_attempts") and
                row.get("runner_observation", {}).get("lock") == "held" and row["runner_observation"]["alive"] is True):
            row["anomalies"].append({"run_id": row["run_id"], "scope": row.get("stage"), "classification": "possible_stall",
                                     "summary": "两次观察间活 Runner 无在途调用、state 修订长期不变；需诊断。",
                                     "evidence": row["state_directory"] + "/state.json",
                                     "recovery_condition": "核对准备/处理边界；此观察不授权终止或重试。"})
    compact = {row["run_id"]: {key: row.get(key) for key in
               ("status", "stage", "checkpoint", "published", "state_revision", "unchanged_since", "error", "anomalies")}
               for row in rows}
    changes = [{"run_id": run_id, "before": previous.get(run_id), "after": value}
               for run_id, value in compact.items() if previous.get(run_id) != value]
    observations.update(latest={**previous, **compact}, observed_at=now,
                        changes=(observations.get("changes", []) + [{"at": now, **item} for item in changes])[-64:])
    return {"observed_at": now, "runs": rows, "changes": changes,
            "anomalies": [item for row in rows for item in row.get("anomalies", [])]}


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
            elif not member.get("gate_capable"):
                result = {"status": "manual_resume_required", "reason": "旧冻结执行器没有 hold.v1；不声称发布已被闸门拦住。"}
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


def _check_clock_binding(ledger: dict, binding: dict | None, *, require_completed_scan=True) -> None:
    if binding is None:
        return
    clock = ledger.get("clock", {})
    if (not clock.get("enabled") or clock.get("generation") != binding.get("generation") or
            clock.get("thread_id") != binding.get("thread_id") or
            (require_completed_scan and clock.get("last_scan", {}).get("id") != binding.get("check_id"))):
        raise ExtractError("clock scan authorization was stopped or superseded")


def reserve_launch(store, state: dict, *, mode="resume", hold_id=None, retry_id=None, background=False,
                   budget_addition=None) -> dict:
    table = process_table()
    identity = _identity(table)
    requests = store.control_requests(state["run_id"])
    with store.supervisor() as ledger:
        store.assert_startable(state["run_id"], ledger)
        if mode == "retry":
            _check_clock_binding(ledger, ledger["retries"].get(retry_id, {}).get("clock_binding"))
        start = ledger["starts"].setdefault(state["run_id"], {"output": state["output"], "authorized": False,
                                                             "created_at": state["created_at"]})
        previous = start.get("launch", {})
        if previous.get("status") in {"launching", "claimed"}:
            owner = previous.get("identity") or previous.get("launcher")
            alive = _alive(owner, table)
            if alive is not False:
                return {**previous, "reused": True}
        if mode == "queued" and state["status"] != "queued":
            return {"status": "preserved", "run_id": state["run_id"], "run_status": state["status"]}
        latest = ledger["controls"].get(state["run_id"], {})
        if mode == "queued" and latest.get("action") in {"pause", "cancel"} and not latest.get("hold_id"):
            return {"status": "preserved", "run_id": state["run_id"], "reason": "pending manual control"}
        if mode == "queued" and any(not request.get("hold_id") and request["id"] not in state["control"]["requests"]
                                    for request in requests):
            return {"status": "preserved", "run_id": state["run_id"], "reason": "unconsumed manual control request"}
        if mode == "resume":
            ledger["controls"][state["run_id"]] = {"revision": latest.get("revision", 0) + 1,
                "action": "resume", "id": str(uuid4()), "hold_id": None, "at": time.time()}
        if mode in {"resume", "queued"}:
            start["authorized"] = True
        launch = {"id": str(uuid4()), "run_id": state["run_id"], "mode": mode,
                  "hold_id": hold_id, "retry_id": retry_id, "status": "launching", "background": background,
                  "launcher": identity, "created_at": time.time(),
                  "request_ids": [request["id"] for request in requests],
                  "control_revision": ledger["controls"].get(state["run_id"], {}).get("revision", 0)}
        if budget_addition and any(budget_addition.values()):
            launch["budget_addition"] = budget_addition
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
        if launch["mode"] == "retry":
            _check_clock_binding(ledger, ledger["retries"].get(launch.get("retry_id"), {}).get("clock_binding"))
        launch.update(status="claimed", identity=identity, claimed_at=time.time())
        return deepcopy(launch)


def start_background(store, run_id: str, repo: Path, *, mode="queued", hold_id=None, retry_id=None) -> dict:
    state = store.load(run_id)
    if store.read_supervisor()["active_hold"]:
        with store.supervisor() as ledger:
            store.hold_member(ledger, state)
        return {"run_id": run_id, "status": "held", "hold_id": store.read_supervisor()["active_hold"]}
    # Snapshot verification is deliberately outside supervisor.lock.
    from .snapshots import frozen_invocation
    frozen_invocation(read_json(store.run_dir(run_id) / "inputs.lock.json"), [])
    launch = reserve_launch(store, state, mode=mode, hold_id=hold_id, retry_id=retry_id, background=True)
    if launch.get("reused") or launch["status"] != "launching":
        return launch
    command = [sys.executable, "-B", str(rules_root() / "schema_extract/extract.py"),
               "_background", run_id, "--launch-id", launch["id"], "--repo", str(repo), "--state-root", str(store.root)]
    try:
        log_path = store.run_dir(run_id) / "runner.log"
        with log_path.open("ab") as stream:
            process = subprocess.Popen(command, cwd=rules_root(), stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=stream, start_new_session=True, close_fds=True)
        table, _ = _processes()
        identity = table.get(process.pid) if table is not None else None
        with store.supervisor() as ledger:
            current = ledger["starts"][run_id]["launch"]
            if current["id"] == launch["id"]:
                current.update(pid=process.pid, log=str(log_path))
                if identity and current["status"] == "launching":
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
            if output.exists():
                entries = list(output.iterdir()) if output.is_dir() else []
                if (len(entries) != 1 or entries[0].name != ".gitkeep" or entries[0].is_symlink() or
                        not entries[0].is_file() or entries[0].stat().st_size):
                    raise ExtractError("output must be absent or contain only an empty .gitkeep")
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


def configure_auto_restart(store, run_ids: list[str], limit: int, *, clock_binding=None) -> dict:
    """The explicit clock-start option authorizes retries without refunding use."""
    if type(limit) is not int or limit < 0:
        raise ExtractError("auto-restart limit must be a nonnegative integer")
    for run_id in run_ids:
        store.run_dir(run_id)
    with store.supervisor() as ledger:
        _check_clock_binding(ledger, clock_binding, require_completed_scan=False)
        policy = ledger["auto_restart"]
        policy.update(enabled=limit > 0, run_ids=sorted(set(run_ids)), limit=limit,
                      used=policy.get("used", {}), configured_at=time.time())
        return deepcopy(policy)


def restart_evidence(store, state: dict, scope: str, *, table=None) -> dict:
    """Re-read one explicit failure from authoritative state and its own files."""
    from .runner import _budget
    from .stages import call_view, owner_for, scope_for, scope_key
    if "scoped_restart.v1" not in state.get("executor", {}).get("capabilities", []):
        raise ExtractError("frozen executor does not support scope-bound restart authorization")
    if (state["status"] in {"paused", "cancelled", "queued"} or state["control"]["action"] or
            state["active_attempts"] or state.get("unconfirmed_attempts") or state.get("unsettled_attempts")):
        raise ExtractError("automatic retry requires a stopped Runner with no unsettled invocation or manual control")
    stage, _, node = scope.partition(":")
    node = node or "root"
    if stage != state["stage"] or (stage == "P5" and node not in {item["id"] for item in state.get("plan") or []}):
        raise ExtractError("retry scope is no longer current")
    call = {"stage": stage, "node": node, "role": None, "mode": "normal"}
    owner, budget = owner_for(state, call), scope_for(state, call)
    reason = owner.get("reason") or {}
    if (owner.get("mode", "normal") != "normal" or reason.get("classification") != "transient_execution" or
            reason.get("code") != "cli_retry" or reason.get("scope") != scope or
            budget["cli_failures"] > state["config"]["retry_limit"]):
        raise ExtractError("scope has no explicitly classified transient failure within its original retry limit")
    call["role"] = owner["role"]
    _, exhausted = _budget(call_view(state, call))
    if exhausted:
        raise ExtractError(exhausted)
    attempt_id = reason["attempt_id"]
    if str(UUID(attempt_id)) != attempt_id:
        raise ExtractError("invalid failure attempt UUID")
    directory = store.run_dir(state["run_id"]) / "attempts" / attempt_id
    attempt, result = read_json(directory / "attempt.json"), read_json(directory / "result.json")
    if (attempt["run_id"] != state["run_id"] or attempt["input_fingerprint"] != state["input_fingerprint"] or
            attempt["attempt_id"] != attempt_id or scope_key(attempt["stage"], attempt["node"]) != scope or
            attempt["generation"] != (owner["generation"] if stage == "P5" else budget["generation"]) or
            attempt["mode"] != "normal" or attempt["role"] != owner["role"] or
            digest(result) != reason["result_sha256"] or result.get("outcome") != "cli_error" or result.get("exit_code") != 75 or
            not result.get("termination", {}).get("confirmed")):
        raise ExtractError("transient retry no longer matches its terminated failure identity")
    table = process_table() if table is None else table
    process = _optional(directory / "process.json")
    if process:
        identities = [process["identity"], *process.get("descendants", [])]
        if any(_alive(identity, table) is not False for identity in identities):
            raise ExtractError("failure process or descendant is live or unknown")
        if any(item["pgid"] == process["identity"]["pid"] and not item["stat"].startswith("Z") for item in table.values()):
            raise ExtractError("failure process group is not confirmed empty")
    return {"run_id": state["run_id"], "scope": scope, "attempt_id": attempt_id,
            "failure_id": digest({"attempt": attempt, "result_sha256": digest(result), "reason": reason}),
            "reason_sha256": digest(reason), "state_revision": state["state_revision"],
            "lineage": state.get("continuation", {}).get("root_run_id", state["run_id"])}


def request_restart(store, run_id: str, scope: str, repo: Path, *, clock_binding=None) -> dict:
    state = store.load(run_id)
    if store.writer_status(run_id) not in {"free", "absent"}:
        raise ExtractError("Runner is live or its lock is unknown")
    evidence = restart_evidence(store, state, scope)
    with store.supervisor() as ledger:
        store.assert_startable(run_id, ledger)
        _check_clock_binding(ledger, clock_binding)
        policy = ledger["auto_restart"]
        if not policy.get("enabled") or run_id not in policy.get("run_ids", []):
            raise ExtractError("no automatic restart authorization for this run")
        previous = next((value for value in ledger["retries"].values() if value["failure_id"] == evidence["failure_id"]), None)
        if previous:
            return deepcopy(previous)
        used = policy.setdefault("used", {})
        if used.get(evidence["lineage"], 0) >= policy["limit"]:
            raise ExtractError("persistent lineage auto-restart limit exhausted")
        request = {**evidence, "id": str(uuid4()), "status": "authorized", "created_at": time.time(),
                   "control_revision": ledger["controls"].get(run_id, {}).get("revision", 0)}
        if clock_binding is not None:
            request["clock_binding"] = deepcopy(clock_binding)
        ledger["retries"][request["id"]] = request
        used[evidence["lineage"]] = used.get(evidence["lineage"], 0) + 1
        request = deepcopy(request)
    request["launch"] = start_background(store, run_id, repo, mode="retry", retry_id=request["id"])
    return request


def authorized_restarts(store, rows: list[dict], repo: Path, *, clock_binding=None) -> list[dict]:
    """Apply authorized retries after a clock scan; a plain scan does not retry."""
    if not store.read_supervisor()["auto_restart"].get("enabled"):
        return []
    results = []
    for row in rows:
        for anomaly in row.get("anomalies", []):
            if anomaly.get("classification") != "transient_execution":
                continue
            try:
                results.append(request_restart(store, row["run_id"], anomaly["scope"], repo, clock_binding=clock_binding))
            except (ExtractError, OSError) as exc:
                results.append({"run_id": row["run_id"], "scope": anomaly["scope"], "status": "preserved", "reason": str(exc)})
    return results
