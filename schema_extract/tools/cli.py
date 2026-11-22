"""Supervisor controls route to each run's own immutable executor."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys

from . import __version__
from .common import ExtractError, RunnerBusy, SupervisorBusy, digest, read_json, rules_root
from .runner import DEFAULT_CONFIG, export_checkpoint, run, summary
from .runtime import frozen_run_invocation
from .store import Store, control_error
from . import monitor
from .stages import validate_add_calls


CATEGORIES = {"schema_docs", "schema_template", "kernel_wiki", "extract_workflow", "extract_tools"}
OUTCOMES = {"declined", "resolved", "partially_adopted"}
STABLE_ID = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def registry(repo: Path) -> dict:
    """Read stable framework decisions from the current issue registry."""
    records = {}
    root = repo / "schema_extract/issues"
    for path in sorted(root.rglob("*.json")):
        value = read_json(path)
        identity, category = value.get("id"), value.get("category")
        if (not isinstance(identity, str) or not STABLE_ID.fullmatch(identity) or
                identity != path.stem or category not in CATEGORIES or identity in records):
            raise ExtractError("invalid or duplicate stable maintenance ID: " + str(path))
        for name in ("problem", "proposal", "necessity"):
            if not isinstance(value.get(name), str) or not value[name].strip():
                raise ExtractError("maintenance record is missing " + name + ": " + str(path))
        if not isinstance(value.get("evidence"), list) or not value["evidence"] or not all(
                isinstance(item, str) and item.strip() for item in value["evidence"]):
            raise ExtractError("maintenance record needs reproducible evidence: " + str(path))
        archived = path.parent.relative_to(root).parts[:1] == ("archive",)
        expected = ("archive", category) if archived else (category,)
        if path.parent.relative_to(root).parts != expected:
            raise ExtractError("maintenance record is in the wrong category directory: " + str(path))
        decision = value.get("decision")
        if archived:
            if (not isinstance(decision, dict) or decision.get("outcome") not in OUTCOMES or
                    not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
                raise ExtractError("archived decision needs its outcome and reason: " + str(path))
            if decision["outcome"] == "partially_adopted" and not all(
                    isinstance(decision.get(key), str) and decision[key].strip() for key in ("adopted", "remaining")):
                raise ExtractError("partial decision needs adopted and remaining scope: " + str(path))
        elif decision is not None:
            raise ExtractError("decided issues belong in archive: " + str(path))
        records[identity] = {**value, "path": str(path.relative_to(repo)),
                             "status": decision["outcome"] if archived else "pending"}
    return {"records": records, "pending": sum(item["status"] == "pending" for item in records.values()),
            "archived": sum(item["status"] != "pending" for item in records.values())}


def candidate_report(store, state: dict, repo: Path) -> list[dict]:
    registered, groups = registry(repo)["records"], {}
    for attempt_id, receipt_hash in state["accepted_receipts"].items():
        path = store.run_dir(state["run_id"]) / "attempts" / attempt_id / "receipt.json"
        receipt = read_json(path)
        if digest(receipt) != receipt_hash:
            raise ExtractError("accepted framework proposal receipt has changed: " + str(path))
        for item in receipt.get("framework_candidates", []):
            valid = bool(STABLE_ID.fullmatch(item["id"]))
            group = groups.setdefault(item["id"], {"id": item["id"],
                "registered": registered.get(item["id"]) if valid else None, "reports": []})
            if not valid:
                group["invalid_reason"] = "候选需要小写字母数字及-或_组成的稳定ID；未登记，原回执仍可查看。"
            source = {"run_id": state["run_id"], "attempt_id": attempt_id,
                      "stage": receipt["identity"]["stage"], "node": receipt["identity"]["node"],
                      "role": receipt["identity"]["role"], "receipt": str(path),
                      "evidence": [receipt["source_evidence"][index] for index in item["evidence"]]}
            group["reports"].append({"candidate": item, "source": source})
    for group in groups.values():
        confirmed = [report["source"] for report in group["reports"] if report["source"]["role"] == "reviewer"]
        group["confirmed_by"] = confirmed
        group["status"] = "confirmed" if confirmed and "invalid_reason" not in group else "candidate"
    return [groups[key] for key in sorted(groups)]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Schema Extract v2: P1-P6 extraction with concurrent ready nodes and an independently approved checkpoint.")
    result.add_argument("--version", action="version", version=f"Schema Extract {__version__}")
    sub = result.add_subparsers(dest="command", required=True, metavar="COMMAND")
    repair = sub.add_parser("repair-runtime", help="prepare or apply a reviewed executor-only patch without resetting progress")
    repair.add_argument("operation", choices=("prepare", "apply", "probe"))
    repair.add_argument("run_id")
    repair.add_argument("--executor-tree", type=Path)
    repair.add_argument("--issue-id")
    repair.add_argument("--reason")
    repair.add_argument("--plan")
    repair.add_argument("--authorization", type=Path,
                        help="user instruction to continue this exact plan with delivery audit markers")
    repair.add_argument("--repo", type=Path, default=Path.cwd())
    repair.add_argument("--state-root", type=Path)
    extract = sub.add_parser("extract", help="freeze inputs and independently approve all six stages")
    select = extract.add_mutually_exclusive_group(required=True)
    select.add_argument("--id", dest="operator_id")
    select.add_argument("--ids", nargs="+", help="catalog IDs, mapped beneath --output-root")
    select.add_argument("--source", help="catalog directory or schema_extract.source.v2 JSON descriptor")
    extract.add_argument("--bindings", type=Path, help="descriptor for git:<ref>:<directory> shorthand")
    extract.add_argument("--output", type=Path)
    extract.add_argument("--output-root", type=Path)
    launch = extract.add_mutually_exclusive_group()
    launch.add_argument("--queue", action="store_true", help="freeze and queue only; resume starts the frozen run")
    launch.add_argument("--background", action="store_true", help="durably register and start a detached frozen Runner")
    extract.add_argument("--repo", type=Path, default=Path.cwd(), help="catalog/Git source repository")
    extract.add_argument("--state-root", type=Path, help="separate v2 store (default: REPO/.schema_extract/v2)")
    extract.add_argument("--worker-effort", choices=("low", "medium", "high", "xhigh", "max"), default=None,
                         help="worker reasoning effort (Claude default: max)")
    extract.add_argument("--reviewer-effort", choices=("low", "medium", "high", "xhigh", "max"), default=None,
                         help="reviewer reasoning effort (Claude default: max)")
    extract.add_argument("--backend", choices=("claude", "codex"), default=DEFAULT_CONFIG["backend"],
                         help="stage backend (default: claude); Codex requires explicit model and role efforts")
    extract.add_argument("--model", help="stage model (Claude default: qwen3.8-max[1m])")
    extract.add_argument("--agent-binary", help="executable for the selected stage backend")
    for option in ("revision_limit", "max_calls"):
        extract.add_argument("--" + option.replace("_", "-"), type=int, default=DEFAULT_CONFIG[option],
                             help="finite call limit per stage, and per P5 node; includes reserved adjudication calls" if option == "max_calls" else None)
    for name, help_text in (("status", "read a saved run without advancing it"),
                            ("pause", "request a pause at invocation boundaries"),
                            ("cancel", "request termination of all started invocations"),
                            ("resume", "resume the frozen run; add calls only with an explicit option"),
                            ("export", "export the final reviewed checkpoint to another destination"),
                            ("candidates", "read confirmed framework issues and their original candidates"),
                            ("maintain", "request bounded review or correction within the current frozen run")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("run_id", nargs="?" if name in {"pause", "resume"} else None)
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-root", type=Path)
        if name == "status":
            command.add_argument("--full", action="store_true", help="print the entire authoritative state")
        if name == "maintain":
            command.add_argument("--issue-id", required=True, help="stable issue identity; repetitions never refund used calls")
            command.add_argument("--tasks", type=Path, required=True, help="JSON array of stage/node/action/fields/reason tasks")
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
        if name in {"pause", "resume"}:
            command.add_argument("--all", action="store_true")
        if name == "resume":
            command.add_argument("--hold", help="release only the named global hold")
            command.add_argument("--scope", help="recover one failed scope, such as P5:node, through its existing Runner")
            _call_options(command)
    for name in ("issues", "inspect"):
        command = sub.add_parser(name, help="read-only framework registry" if name == "issues" else "read-only run summaries")
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-root", type=Path)
        if name == "inspect":
            command.add_argument("--run", nargs="+", dest="run_ids")
            command.add_argument("--json", action="store_true")
    standalone = sub.add_parser("standalone", help="materialize one node with effective Design and its shared Contract as an independent reading view")
    standalone.add_argument("source", type=Path)
    standalone.add_argument("--output", type=Path, required=True)
    standalone.add_argument("--repo", type=Path, default=Path.cwd())
    standalone.add_argument("--state-root", type=Path)
    evaluation = sub.add_parser("review-eval", help="evaluate current document cases with independent bounded reviewers")
    evaluation.add_argument("--suite", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--backend", choices=("claude", "codex"), default=DEFAULT_CONFIG["backend"])
    evaluation.add_argument("--agent-binary")
    evaluation.add_argument("--model")
    evaluation.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"))
    evaluation.add_argument("--case", action="append", dest="cases")
    evaluation.add_argument("--repo", type=Path, default=rules_root(), help="rules to freeze for this evaluation")
    audit = sub.add_parser("audit", help="read recorded usage/session evidence; explicitly verify or archive selected evidence")
    audit.add_argument("run_id")
    selection = audit.add_mutually_exclusive_group()
    selection.add_argument("--attempt", metavar="UUID")
    selection.add_argument("--session", metavar="UUID")
    audit.add_argument("--verify", action="store_true")
    audit.add_argument("--archive", type=Path, metavar="PATH")
    audit.add_argument("--json", action="store_true")
    audit.add_argument("--repo", type=Path, default=Path.cwd())
    audit.add_argument("--state-root", type=Path)
    watches = sub.add_parser("monitor", help="observe and control registered extraction runs")
    watch = watches.add_subparsers(dest="monitor_command", required=True)
    scan = watch.add_parser("scan", help="one lightweight observation; no task is resumed")
    scan.add_argument("--run", nargs="+", dest="run_ids")
    scan.add_argument("--wake", metavar="ID", help="finish this accepted clock reminder")
    scan.add_argument("--thread", metavar="UUID", help="explicitly finish this thread's accepted unfinished reminder")
    scan.add_argument("--json", action="store_true")
    scan.add_argument("--repo", type=Path, default=Path.cwd())
    scan.add_argument("--state-root", type=Path)
    for name in ("start", "status", "ack", "stop"):
        command = watch.add_parser(name, help={"start": "bind and start one persistent same-thread clock",
                                               "status": "observe the real clock process, delivery and progress",
                                               "ack": "accept a due reminder once in its bound thread",
                                               "stop": "invalidate reminders and stop only the clock"}[name])
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-root", type=Path)
        if name in {"start", "ack"}:
            command.add_argument("--thread", required=True, metavar="UUID")
        if name == "start":
            command.add_argument("--run", nargs="+", dest="run_ids")
            command.add_argument("--codex-bin", help="Codex queue executable, or executable test substitute")
        elif name == "ack":
            command.add_argument("wake_id", metavar="ID")
    background = sub.add_parser("_background")
    background.add_argument("run_id")
    background.add_argument("--launch-id", required=True)
    background.add_argument("--repo", type=Path, default=Path.cwd())
    background.add_argument("--state-root", type=Path)
    clock = sub.add_parser("_clock")
    clock.add_argument("--launch-id", required=True)
    clock.add_argument("--lock-fd", type=int)
    clock.add_argument("--repo", type=Path, default=Path.cwd())
    clock.add_argument("--state-root", type=Path)
    return result


def _call_options(command):
    command.add_argument("--add-calls", type=int, default=0,
                         help="explicitly add this many calls to each stage/node limit; keep used calls and decisions")


def _frozen_entry(store, state: dict, command: list[str]) -> None:
    arguments, directory, environment = frozen_run_invocation(store, store.load(state["run_id"]), command)
    os.chdir(directory)
    if rules_root() == Path(environment["PYTHONPATH"]).resolve():
        return
    # The absolute entry and import path select frozen code; diagnostics belong
    # in the run directory. -B keeps bytecode out of the immutable snapshot.
    os.execve(sys.executable, arguments, environment)


def _print_observations(rows, *, as_json=False):
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    from .wakeup import format_beijing_time
    for row in rows:
        if row.get("error"):
            print(f"{row['run_id']} unknown: {row['error']}")
            continue
        executor = row.get("executor", {})
        print(f"{row['run_id']} {row.get('operator_id')} {row.get('disposition', row.get('status'))} {row.get('stage')} "
              f"nodes={row.get('completed_node_count')}/{row.get('planned_node_count')} "
              f"active={len(row.get('active_attempts', {}))} "
              f"observed={row['observation']['status']} at={format_beijing_time(row['observation']['observed_at'])} "
              f"executor={executor.get('version')}/{str(executor.get('digest') or 'unknown')[:12]} "
              f"gate={'v1' if row.get('gate_capable') else 'unsupported'} "
              f"last={format_beijing_time(row.get('updated_at'))} "
              f"published={row.get('published')} checkpoint={row.get('checkpoint')}")
        for item in row.get("anomalies", []):
            print(f"  {item.get('classification')} {item.get('scope')}: {item.get('summary')} [{item.get('evidence')}]")
            if item.get("recovery_condition"):
                print("  next: " + item["recovery_condition"])


def _launch_context(store, state: dict, *, mode="resume", add_calls=0) -> dict:
    token = os.environ.get("SCHEMA_EXTRACT_ENTRY_ID")
    entry = store.read_supervisor()["starts"].get(state["run_id"], {}).get("launch", {})
    if token and entry.get("id") == token and entry.get("status") == "claimed" and entry.get("identity", {}).get("pid") == os.getpid():
        from .processes import process_identity_matches
        if process_identity_matches(entry.get("identity"), monitor._identity()) is True:
            return entry
        raise ExtractError("claimed launcher process identity changed")
    if mode == "resume":
        control = store.read_supervisor()["controls"].get(state["run_id"], {})
        requests = {request["id"]: request for request in store.control_requests(state["run_id"])}
        pending = {request_id for request_id, delivery in control.get("deliveries", {}).items()
                   if not delivery.get("confirmed")}
        pending.update(request_id for request_id, request in requests.items()
                       if request["action"] == "cancel" and
                       request["control_revision"] > entry.get("control_revision", 0) and
                       not control.get("deliveries", {}).get(request_id, {}).get("confirmed"))
        for request_id in sorted(pending):
            request = requests.get(request_id)
            if not request or request["action"] != "cancel":
                raise ExtractError("pending cancel delivery has no matching control request")
            delivery = monitor.deliver_cancel(store, request)
            if not delivery.get("confirmed"):
                raise ExtractError("cancel termination is pending; resume after owned processes have exited")
    entry = monitor.reserve_launch(store, state, mode=mode, add_calls=add_calls)
    if not entry.get("reused") and entry.get("id"):
        os.environ["SCHEMA_EXTRACT_ENTRY_ID"] = entry["id"]
    return entry


def _registration_result(store, registration: dict, state: dict | None, *, queued=False, background=False):
    """Report registration separately from execution and delivery, including fixed inputs."""
    run_id = registration["run_id"]
    directory = store.run_dir(run_id)
    observed = state or (store.read_summary(run_id) if (directory / "state.json").is_file() else {})
    result = {**registration, "run_status": observed.get("status"), "published": observed.get("published", False),
              "stage": observed.get("stage"), "reason": observed.get("reason"), "publication": observed.get("publication")}
    lock_path = directory / "inputs.lock.json"
    if lock_path.is_file():
        lock = read_json(lock_path)
        if observed and observed["input_fingerprint"] != lock["fingerprint"]:
            raise ExtractError("registered run differs from its input lock")
        result["frozen"] = {"inputs_lock": str(lock_path), "input_fingerprint": lock["fingerprint"],
                            "source": {key: lock["source"][key] for key in ("id", "repository_id", "ref", "ref_kind")},
                            "rules_id": lock["rules"]["id"], "executor": observed.get("executor"), "config": lock["config"]}
    if registration.get("reused"):
        result["input_policy"] = "沿用已登记的固定源码、规则和参数。来源、规则或参数改变时，用新输出目录创建运行。"
    launch = registration.get("background") or {}
    if result["run_status"] == "ready" and result["published"]:
        outcome, next_step = "published", "交付回执：" + str(Path(registration["output"]) / ".extraction/receipt.json")
    elif background and launch.get("error"):
        outcome, next_step = "delivery_pending", "inspect --run " + run_id + " --json 核对原登记与 launch；未知投递不能作为重放依据。"
    elif background and launch.get("status") in {"launching", "claimed"}:
        outcome = "already_running" if launch.get("reused") or state is None else "start_requested"
        next_step = "status " + run_id + " 查看阶段；后台启动成功不表示已完成发布。"
    elif background and launch.get("status") == "held":
        outcome, next_step = "held", "resume --all --hold " + launch["hold_id"] + " 释放该次暂停。"
    elif result["run_status"] == "queued":
        outcome, next_step = "queued", "resume " + run_id + " 显式启动已登记的固定任务。"
    elif result["run_status"] == "running" and store.writer_status(run_id) == "held":
        outcome, next_step = "already_running", "status " + run_id + " 查看进度；本命令仅返回原运行身份。"
    elif state is None:
        outcome, next_step = "creation_or_runner_active", "status " + run_id + " 查看原任务；启动窗口中断后可重复原命令对账。"
    elif result["run_status"] == "ready":
        outcome, next_step = "publication_pending", "status " + run_id + " 查看发布原因；目标冲突时用 export " + run_id + " --output NEW_DIRECTORY 另行交付。"
    else:
        outcome = "stopped"
        next_step = "status " + run_id + " 查看原因；resume 使用本次冻结执行器和原预算。"
    result.update(result=outcome, next_step=next_step)
    success = (outcome == "published" and state is not None or queued and outcome == "queued" or
               background and outcome in {"start_requested", "already_running", "held"})
    return result, 0 if success else 1


def _busy_result(state: dict, entry: dict) -> int:
    print(json.dumps({"run_id": state["run_id"], "result": "already_running" if entry.get("reused") else "preserved",
                      "status": state["status"], "published": state["published"], "launch": entry, "committed": entry.get("committed", bool(entry.get("id"))),
                      "next_step": "status " + state["run_id"] + " 查看原任务；本命令未执行到发布，也未追加调用次数。"}, ensure_ascii=False, indent=2))
    return 1


def _monitor_wait():
    last = {}
    def waiting(phase, elapsed):
        if elapsed >= 1 and (phase not in last or elapsed - last[phase] >= 30):
            print(f"monitor {phase}: 正在等待控制锁（{elapsed:.0f} 秒）；可中断，随后用同一提醒 ID 和 monitor status 核对并续接。",
                  file=sys.stderr, flush=True)
            last[phase] = elapsed
    return waiting


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    operation = {}
    try:
        args.repo = args.repo.resolve()
        state_root = (getattr(args, "state_root", None) or args.repo / ".schema_extract/v2").resolve()
        if args.command == "repair-runtime":
            from . import executor_repairs
            store = Store(state_root)
            if args.operation == "prepare":
                if not args.executor_tree or not args.issue_id or not args.reason:
                    raise ExtractError("repair preparation requires --executor-tree, --issue-id and --reason")
                result = executor_repairs.prepare(store, args.run_id, tree=args.executor_tree.resolve(),
                                                   issue_id=args.issue_id, reason=args.reason)
            elif args.operation == "probe":
                result = executor_repairs.probe(store, store.load(args.run_id))
            else:
                if not args.plan or not args.authorization:
                    raise ExtractError("repair application requires --plan and --authorization")
                from .snapshots import frozen_invocation
                target = read_json(store.run_dir(args.run_id) / "runtime_repairs" / "plans" /
                                   executor_repairs._id(args.plan) / "target.lock.json")
                review = args.authorization.expanduser().resolve()
                arguments, directory, environment = frozen_invocation(target, ["repair-runtime", "apply", args.run_id,
                    "--plan", args.plan, "--authorization", str(review), "--state-root", str(state_root)])
                if rules_root() != directory:
                    os.chdir(directory)
                    os.execve(sys.executable, arguments, environment)
                result = executor_repairs.apply(store, args.run_id, plan_id=args.plan, review=read_json(review))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "standalone":
            from .standalone import create_view
            print(json.dumps(create_view(args.source, args.output, args.repo), ensure_ascii=False, indent=2))
            return 0
        if args.command == "review-eval":
            from .review_eval import evaluate
            report = evaluate(args.suite.resolve(), args.output.resolve(), args.repo, backend=args.backend,
                              agent_binary=args.agent_binary, model=args.model, effort=args.effort,
                              selected=args.cases)
            return 0 if report["passed"] else 1
        if args.command == "audit":
            from .audit import _parent_fd, audit_run, format_audit
            with _parent_fd((args.state_root or args.repo / ".schema_extract/v2").absolute() / "state-root-check"):
                pass
            result = audit_run(Store(state_root), args.run_id, attempt_id=args.attempt, session_id=args.session,
                               verify=args.verify, archive=args.archive)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else format_audit(result))
            return 1 if result["verification"]["status"] == "incomplete" or (result["archive"] and result["archive"]["status"] != "archived") else 0
        if args.command == "monitor":
            from .wakeup import Clock, clock_display
            store = Store(state_root)
            clock = Clock(store, args.repo, on_wait=_monitor_wait())
            if args.monitor_command == "scan":
                if args.wake or args.thread:
                    if args.run_ids:
                        raise ExtractError("a bound clock scan uses its registered scope; --run belongs to a manual scan")
                    result = clock.scan(args.wake, args.thread)
                else:
                    result = monitor.scan(store, args.run_ids)
            elif args.monitor_command == "start":
                result = clock.start(args.thread, args.run_ids, codex_binary=args.codex_bin)
            elif args.monitor_command == "ack":
                result = clock.ack(args.wake_id, args.thread)
            elif args.monitor_command == "stop":
                result = clock.stop()
            else:
                result = clock.status()
            shown = clock_display(result)
            if args.monitor_command != "scan" or args.json:
                print(json.dumps(shown, ensure_ascii=False, indent=2))
            else:
                _print_observations(result["runs"])
                print(f"changes={len(result['changes'])} anomalies={len(result['anomalies'])}")
                if result.get("superseded_run_ids"):
                    print("以下 run 的本次采样已被更新观察超过：" + ", ".join(result["superseded_run_ids"]))
                if result.get("clock"):
                    print(json.dumps(shown["clock"], ensure_ascii=False))
            return 0
        if args.command == "_clock":
            from .wakeup import Clock
            return Clock(Store(state_root), args.repo).run(args.launch_id, args.lock_fd)
        if args.command in {"pause", "resume"} and args.all:
            if getattr(args, "scope", None):
                raise ExtractError("--scope requires a single run UUID")
            if args.command == "resume" and args.add_calls:
                raise ExtractError("call additions require a single run UUID; releasing a hold does not add calls")
            if args.run_id or (args.command == "resume" and not args.hold):
                raise ExtractError("--all takes no run UUID; resume --all requires --hold ID")
            store = Store(state_root)
            result = monitor.hold_all(store) if args.command == "pause" else monitor.release_hold(store, args.hold, args.repo)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "maintain":
            args.tasks = args.tasks.expanduser().resolve()
        if args.command in {"pause", "resume"} and (not args.run_id or getattr(args, "hold", None)):
            raise ExtractError("single-run control requires a UUID; --hold requires --all")
        if args.command == "resume" and args.scope:
            if args.add_calls:
                raise ExtractError("scope recovery preserves used calls; add calls with an explicit whole-run resume first")
            store = Store(state_root)
            state = store.load(args.run_id)
            if "scoped_recovery.v2" not in state.get("executor", {}).get("capabilities", []):
                raise ExtractError("frozen executor lacks scoped_recovery.v2")
            command = ["resume", args.run_id, "--scope", args.scope, "--repo", str(args.repo), "--state-root", str(state_root)]
            _frozen_entry(store, state, command)
            request = monitor.request_restart(store, args.run_id, args.scope, args.repo)
            print(json.dumps(request, ensure_ascii=False, indent=2))
            return 0 if (request["status"] in {"authorized", "consumed"} and
                         request.get("launch", {}).get("status") != "launch_failed" and
                         not request.get("launch", {}).get("error")) else 1
        if args.command == "_background":
            store = Store(state_root)
            if os.environ.get("SCHEMA_EXTRACT_ENTRY_ID") == args.launch_id:
                entry = store.read_supervisor()["starts"].get(args.run_id, {}).get("launch", {})
                from .processes import process_identity_matches
                if (entry.get("id") != args.launch_id or entry.get("status") != "claimed" or
                        process_identity_matches(entry.get("identity"), monitor._identity()) is not True):
                    raise ExtractError("background launch claim changed before frozen entry")
            else:
                entry = monitor.claim_launch(store, args.run_id, args.launch_id)
            os.environ["SCHEMA_EXTRACT_ENTRY_ID"] = entry["id"]
            state = store.load(args.run_id)
            command = ["_background", args.run_id, "--launch-id", entry["id"],
                       "--repo", str(args.repo), "--state-root", str(state_root)]
            _frozen_entry(store, state, command)
            state = run(store, args.run_id, resume=entry["mode"] != "maintenance",
                        queued_only=entry["mode"] == "queued", launch_id=entry["id"])
            print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
            return 0 if state["status"] == "ready" and state["published"] else 1
        if args.command in {"issues", "inspect"}:
            result = registry(args.repo) if args.command == "issues" else monitor.inspect_runs(Store(state_root), args.run_ids)
            if args.command == "inspect":
                _print_observations(result, as_json=args.json)
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command != "extract":
            store = Store(state_root)
            state = store.load(args.run_id)
            command = [args.command, args.run_id, "--repo", str(args.repo), "--state-root", str(state_root)]
            if getattr(args, "full", False):
                command.append("--full")
            if args.command == "maintain":
                command.extend(["--issue-id", args.issue_id, "--tasks", str(args.tasks.absolute())])
            if args.command == "export":
                command.extend(["--output", str(args.output.absolute())])
            lock = read_json(store.run_dir(args.run_id) / "inputs.lock.json")
            if lock["fingerprint"] != state["input_fingerprint"]:
                raise ExtractError("run input fingerprint differs from its lock")
            entry, request = None, None
            if args.command in {"resume", "export"}:
                add_calls = validate_add_calls(args.add_calls) if args.command == "resume" else 0
                if add_calls:
                    if "call_extension.v1" not in state.get("executor", {}).get("capabilities", []):
                        raise ExtractError("this frozen executor cannot add calls")
                    command.extend(["--add-calls", str(add_calls)])
                store.assert_startable(args.run_id)
                entry = _launch_context(store, state, mode=args.command, add_calls=add_calls)
                operation = {"committed": True, "run_id": args.run_id, "launch": entry}
                if entry.get("reused"):
                    return _busy_result(state, entry)
            elif args.command in {"pause", "cancel"}:
                request_id = os.environ.get("SCHEMA_EXTRACT_CONTROL_REQUEST")
                if request_id:
                    request = next((value for value in store.control_requests(args.run_id)
                                    if value["id"] == request_id and value["action"] == args.command), None)
                if request is None:
                    request = store.request_control(args.run_id, args.command)
                    os.environ["SCHEMA_EXTRACT_CONTROL_REQUEST"] = request["id"]
                operation = {"committed": True, "run_id": args.run_id, "request": request}
            if args.command == "cancel":
                try:
                    delivery = monitor.deliver_cancel(store, request)
                except SupervisorBusy as exc:
                    delivery = {"status": "pending", "confirmed": False, "evidence_committed": False, "reason": str(exc)}
                print(json.dumps({"committed": True, "request": request, "delivery": delivery}, ensure_ascii=False, indent=2))
                return 0 if delivery.get("confirmed") else 1
            _frozen_entry(store, state, command)
            if args.command == "maintain":
                from .maintenance import submit_request
                request = submit_request(store, args.run_id, issue_id=args.issue_id, tasks=read_json(args.tasks.absolute()))
                operation = {"committed": True, "run_id": args.run_id, "request": request}
                writer = store.writer_status(args.run_id)
                delivery = ({"status": "active_runner"} if writer == "held" else
                            monitor.start_background(store, args.run_id, args.repo, mode="maintenance")
                            if writer in {"free", "absent"} else {"status": "writer_unknown"})
                result, code = ({"committed": True, "request": request, "delivery": delivery},
                                0 if delivery["status"] != "launch_failed" and not delivery.get("error") else 1)
            elif args.command == "status":
                result = state if args.full else summary(store, state)
                code = 0
            elif args.command in {"pause", "cancel"}:
                try:
                    state = run(store, args.run_id)
                except RunnerBusy:
                    state = store.load(args.run_id)
                result, code = {"committed": True, "request": request, "run": summary(store, state)}, 0
            elif args.command == "export":
                result = export_checkpoint(store, args.run_id, args.output)
                code = 0 if result["status"] == "published" else 1
            elif args.command == "candidates":
                result, code = candidate_report(store, state, args.repo), 0
            else:
                state = run(store, args.run_id, resume=entry.get("mode") != "maintenance",
                            queued_only=entry.get("mode") == "queued", launch_id=entry.get("id"))
                result = summary(store, state)
                code = 0 if state["status"] == "ready" and state["published"] else 1
            if entry and isinstance(result, dict):
                result.update(committed=True, launch=entry)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return code
        config = {key: getattr(args, key) for key in ("backend", "agent_binary", "worker_effort", "reviewer_effort", "model", "revision_limit", "max_calls")
                  if getattr(args, key) is not None}
        if args.ids:
            if args.output or not args.output_root or not (args.background or args.queue):
                raise ExtractError("--ids requires --output-root and --background or --queue")
            store, results = Store(state_root), []
            targets = monitor.catalog_targets(args.repo, args.ids, args.output_root)
            for target in targets:
                if target.get("error"):
                    results.append(target)
                    continue
                operation = {}
                try:
                    registered, state = monitor.registered_extract(args.repo, store, target["output"], operator_id=target["operator_id"], config=config, queued=args.queue)
                    operation = {"committed": True, "run_id": registered["run_id"], "registration": registered}
                    if state is not None and args.background and registered["authorized"]:
                        registered["background"] = monitor.start_background(store, state["run_id"], args.repo)
                    result, code = _registration_result(store, registered, state, queued=args.queue, background=args.background)
                    results.append({"operator_id": target["operator_id"], **result, "exit_code": code})
                except (ExtractError, OSError, KeyError, TypeError, AttributeError) as exc:
                    results.append({"operator_id": target["operator_id"], **control_error(exc, operation)})
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
            return 1 if any(value.get("error") or value.get("exit_code") for value in results) else 0
        if not args.output or args.output_root:
            raise ExtractError("single-source extraction requires --output and does not take --output-root")
        store = Store(state_root)
        registered, state = monitor.registered_extract(args.repo, store, args.output, operator_id=args.operator_id,
                                                      source=args.source, bindings=args.bindings, config=config, queued=args.queue)
        operation = {"committed": True, "run_id": registered["run_id"], "registration": registered}
        if state is None or args.background or (registered.get("reused") and (not registered["authorized"] or state["status"] != "queued")):
            if state is not None and args.background and registered["authorized"]:
                registered["background"] = monitor.start_background(store, state["run_id"], args.repo)
            result, code = _registration_result(store, registered, state, queued=args.queue, background=args.background)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return code
        print(json.dumps({"run_id": state["run_id"], "state_directory": str(store.run_dir(state["run_id"])),
                          "status": state["status"]}, ensure_ascii=False), file=sys.stderr, flush=True)
        if args.queue:
            print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
            return 0
        entry = _launch_context(store, state, mode="queued" if state["status"] == "queued" else "resume")
        operation["launch"] = entry
        if entry.get("reused") or entry.get("status") == "preserved":
            return _busy_result(state, entry)
        _frozen_entry(store, state, ["resume", state["run_id"], "--repo", str(args.repo), "--state-root", str(state_root)])
        state = run(store, state["run_id"], resume=True, queued_only=entry.get("mode") == "queued", launch_id=entry.get("id"))
        print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
        return 0 if state["status"] == "ready" and state["published"] else 1
    except SupervisorBusy as exc:
        print(json.dumps(control_error(exc, operation), ensure_ascii=False), file=sys.stderr)
        return 1
    except (ExtractError, OSError) as exc:
        print(json.dumps(control_error(exc, operation), ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
