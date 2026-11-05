"""Supervisor controls route to each run's own immutable executor."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys

from . import __version__
from .common import ExtractError, RunnerBusy, digest, read_json, rules_root
from .runner import DEFAULT_CONFIG, export_checkpoint, run, summary
from .snapshots import frozen_invocation
from .runtime import frozen_run_invocation
from .store import Store
from . import monitor
from .budget import CAPABILITY as BUDGET_CAPABILITY, addition


CATEGORIES = {"schema_docs", "schema_template", "kernel_wiki", "extract_workflow", "extract_tools"}
OUTCOMES = {"declined", "resolved", "partially_adopted"}
STABLE_ID = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def registry(repo: Path) -> dict:
    """Legacy files remain historical data, never an imported runtime package."""
    records = {}
    for root in (repo / "schema_extract_old/issues", repo / "schema_extract/issues"):
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
    return [groups[key] for key in sorted(groups)]


def inspect_runs(store, repo: Path | None = None, run_ids=None) -> list[dict]:
    return monitor.inspect_runs(store, run_ids)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Schema Extract v2: P1-P6 extraction with concurrent ready nodes and an independently approved checkpoint.")
    result.add_argument("--version", action="version", version=f"Schema Extract {__version__}")
    sub = result.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="freeze inputs and independently approve all six stages")
    select = extract.add_mutually_exclusive_group(required=True)
    select.add_argument("--id", dest="operator_id")
    select.add_argument("--ids", nargs="+", help="catalog IDs, mapped beneath --output-root")
    select.add_argument("--source", help="catalog directory or schema_extract.source.v2 JSON descriptor")
    extract.add_argument("--bindings", type=Path, help="descriptor for git:<ref>:<directory> shorthand")
    extract.add_argument("--output", type=Path)
    extract.add_argument("--output-root", type=Path)
    launch = extract.add_mutually_exclusive_group()
    launch.add_argument("--queue", action="store_true", help="freeze and queue only; resume starts the fixed-budget run")
    launch.add_argument("--background", action="store_true", help="durably register and start a detached frozen Runner")
    extract.add_argument("--repo", type=Path, default=Path.cwd(), help="catalog/Git source repository")
    extract.add_argument("--state-root", type=Path, help="separate v2 store (default: REPO/.schema_extract/v2)")
    extract.add_argument("--worker-effort", choices=("low", "medium", "high", "xhigh", "max"), default=DEFAULT_CONFIG["worker_effort"])
    extract.add_argument("--reviewer-effort", choices=("low", "medium", "high", "xhigh", "max"), default=DEFAULT_CONFIG["reviewer_effort"])
    extract.add_argument("--model", help="optional model override; provider settings remain loaded")
    extract.add_argument("--codex-bin", default=DEFAULT_CONFIG["codex_binary"], help="Codex CLI executable, or executable test substitute")
    for option in ("revision_limit", "operator_seconds", "stage_seconds", "call_seconds", "max_calls"):
        extract.add_argument("--" + option.replace("_", "-"), type=int, default=DEFAULT_CONFIG[option],
                             help="finite call budget per stage, and per P5 node; includes reserved adjudication calls" if option == "max_calls" else None)
    imported = sub.add_parser("import-legacy", parents=[extract], add_help=False,
                              help="import a terminated v1 checkpoint and independently re-review from P1")
    imported.add_argument("--from-task", dest="source_task", type=Path, required=True)
    imported.add_argument("--checkpoint", dest="checkpoint_id", required=True)
    for name, help_text in (("status", "read a saved run without advancing it"),
                            ("pause", "request a pause at invocation boundaries"),
                            ("cancel", "request termination of all started invocations"),
                            ("resume", "resume the frozen run; add finite budget only with explicit options"),
                            ("export", "export the final reviewed checkpoint to another destination"),
                            ("candidates", "read accepted framework proposals and prior stable-ID decisions")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("run_id", nargs="?" if name in {"pause", "resume"} else None)
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-root", type=Path)
        if name == "status":
            command.add_argument("--full", action="store_true", help="print the entire authoritative state")
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
        if name in {"pause", "resume"}:
            command.add_argument("--all", action="store_true")
        if name == "resume":
            command.add_argument("--hold", help="release only the named global hold")
            command.add_argument("--scope", help="recover one failed scope, such as P5:node, through its existing Runner")
            _budget_options(command)
    for name in ("issues", "inspect"):
        command = sub.add_parser(name, help="read-only framework registry" if name == "issues" else "read-only run summaries; does not install a scheduler")
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-root", type=Path)
        if name == "inspect":
            command.add_argument("--start", metavar="RUN_UUID", help="request execution of one explicitly selected queued run")
            command.add_argument("--run", nargs="+", dest="run_ids")
            command.add_argument("--json", action="store_true")
    continued = sub.add_parser("continue", help="prepare or adopt an in-run semantic revision; drafts and budgets remain")
    continued.add_argument("run_id")
    continued.add_argument("--checkpoint", help="required only for an explicit full restart")
    continued.add_argument("--full-restart", action="store_true", help="explicitly create a successor that re-reviews from P1")
    continued.add_argument("--reason", help="revision reason; a full restart requires a nonempty reason")
    continued.add_argument("--plan", type=Path, help="independent revision proposal or prepared plan JSON")
    continued.add_argument("--target-repo", type=Path, help="target rule worktree; source and configuration remain fixed")
    continued.add_argument("--request-id", help="stable semantic request UUID")
    continued.add_argument("--repo", type=Path, default=Path.cwd())
    continued.add_argument("--state-root", type=Path)
    _budget_options(continued)
    revised = sub.add_parser("revise", help="request, prepare, apply or inspect a continuous semantic revision")
    revision_commands = revised.add_subparsers(dest="revision_command", required=True)
    for name in ("request", "prepare", "apply", "status"):
        command = revision_commands.add_parser(name)
        command.add_argument("run_id")
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-root", type=Path)
        if name != "status":
            command.add_argument("--request-id", help="stable request UUID; omitted selects the current request")
        if name == "request":
            command.add_argument("--reason", required=True)
            command.add_argument("--target-repo", type=Path)
        if name in {"prepare", "apply"}:
            command.add_argument("--plan", type=Path, help="independent proposal JSON; prepare without it returns the evaluation template")
    compatibility = sub.add_parser("compatibility-report", help="read complete frozen/target differences and optional Supervisor review; does not change a run")
    compatibility.add_argument("run_id")
    compatibility.add_argument("--checkpoint", required=True)
    compatibility.add_argument("--repo", type=Path, default=Path.cwd())
    compatibility.add_argument("--state-root", type=Path)
    compatibility.add_argument("--target-repo", type=Path, help="target rule worktree (default: this executor's repository)")
    compatibility.add_argument("--review", type=Path, help="independent Supervisor review JSON bound to this exact report")
    upgrade = sub.add_parser("upgrade-runtime", help="register a reviewed compatible frozen executor on the same run; does not start it")
    upgrade.add_argument("run_id")
    upgrade.add_argument("--report", type=Path, required=True, help="exact compatibility-report JSON (raw report or its report wrapper)")
    upgrade.add_argument("--review", type=Path, required=True, help="original independent Supervisor compatibility review JSON")
    upgrade.add_argument("--target-repo", type=Path)
    upgrade.add_argument("--repo", type=Path, default=Path.cwd())
    upgrade.add_argument("--state-root", type=Path)
    standalone = sub.add_parser("standalone", help="materialize one node with effective Design and its shared Contract as an independent reading view")
    standalone.add_argument("source", type=Path)
    standalone.add_argument("--output", type=Path, required=True)
    standalone.add_argument("--repo", type=Path, default=Path.cwd())
    standalone.add_argument("--state-root", type=Path)
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
    scan = watch.add_parser("scan", help="one lightweight observation; no task is resumed by default")
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
            command.add_argument("--auto-restart-limit", type=int, help="explicit persistent per-lineage authorization; never refunds use")
            command.add_argument("--codex-bin", help="Codex queue executable, or executable test substitute")
            command.add_argument("--probe-delay", type=int, metavar="SECONDS", help="one same-thread probe in 1..300 seconds; leaves scan cadence unchanged")
        elif name == "ack":
            command.add_argument("wake_id", metavar="ID")
    background = sub.add_parser("_background")
    background.add_argument("run_id")
    background.add_argument("--launch-id", required=True)
    background.add_argument("--repo", type=Path, default=Path.cwd())
    background.add_argument("--state-root", type=Path)
    clock = sub.add_parser("_clock", help=argparse.SUPPRESS)
    clock.add_argument("--launch-id", required=True)
    clock.add_argument("--lock-fd", type=int)
    clock.add_argument("--repo", type=Path, default=Path.cwd())
    clock.add_argument("--state-root", type=Path)
    return result


def _budget_options(command):
    command.add_argument("--add-seconds", type=int, default=0,
                         help="explicitly extend operator and started scope deadlines from max(now, deadline); keep used time")
    command.add_argument("--add-calls", type=int, default=0,
                         help="explicitly add this many calls to each stage/node limit; keep used calls and decisions")


def _frozen_entry(lock: dict, command: list[str], *, store=None, state=None) -> None:
    arguments, directory, environment = (frozen_run_invocation(store, store.load(state["run_id"]), command)
                                         if store is not None and state is not None else frozen_invocation(lock, command))
    if rules_root() == Path(environment["PYTHONPATH"]).resolve():
        if store is not None:
            os.chdir(directory)
        return
    # The absolute entry and import path select frozen code; diagnostics belong
    # in the run directory. -B keeps bytecode out of the immutable snapshot.
    os.chdir(directory)
    os.execve(sys.executable, arguments, environment)


def _print_observations(rows, *, as_json=False):
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    for row in rows:
        if row.get("error"):
            print(f"{row['run_id']} unknown: {row['error']}")
            continue
        executor = row.get("executor", {})
        print(f"{row['run_id']} {row.get('operator_id')} {row.get('disposition', row.get('status'))} {row.get('stage')} "
              f"nodes={row.get('completed_node_count')}/{row.get('planned_node_count')} "
              f"active={len(row.get('active_attempts', {}))} "
              f"observed={row['observation']['status']} at={row['observation']['observed_at']:.0f} "
              f"executor={executor.get('version')}/{str(executor.get('digest') or 'unknown')[:12]} "
              f"gate={'v1' if row.get('gate_capable') else 'unsupported'} "
              f"last={row.get('updated_at')} deadline={row.get('budget', {}).get('deadline')} "
              f"published={row.get('published')} checkpoint={row.get('checkpoint')}")
        if row.get("successor"):
            print(f"  successor={row['successor']['run_id']} status={row['successor']['status']} saved_status={row['status']}")
        for item in row.get("anomalies", []):
            print(f"  {item.get('classification')} {item.get('scope')}: {item.get('summary')} [{item.get('evidence')}]")
            if item.get("recovery_condition"):
                print("  next: " + item["recovery_condition"])


def _launch_context(store, state: dict, *, mode="resume", budget_addition=None) -> dict:
    token = os.environ.get("SCHEMA_EXTRACT_ENTRY_ID")
    entry = store.read_supervisor()["starts"].get(state["run_id"], {}).get("launch", {})
    if token and entry.get("id") == token and entry.get("status") == "claimed" and entry.get("identity", {}).get("pid") == os.getpid():
        return entry
    entry = monitor.reserve_launch(store, state, mode=mode, budget_addition=budget_addition)
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
        result["input_policy"] = "沿用已登记的固定源码、规则和参数，本次未重新读取工作树。源码或参数变更请换输出目录 extract；兼容工具修复经审阅后 upgrade-runtime，语义变化使用 continue。"
    successor = store.read_supervisor()["continuations"].get(run_id)
    launch = registration.get("background") or {}
    if successor:
        outcome, next_step = "superseded", "查看接续任务 " + successor["run_id"] + "；旧任务已停止派发。"
        result["successor"] = successor
    elif result["run_status"] == "ready" and result["published"]:
        outcome, next_step = "published", "交付回执：" + str(Path(registration["output"]) / ".extraction/receipt.json")
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
        next_step = "status " + run_id + " 查看原因；resume 使用已登记执行器，兼容升级用 upgrade-runtime，语义变化用 continue。"
    result.update(result=outcome, next_step=next_step)
    success = (outcome == "published" and state is not None or queued and outcome == "queued" or
               background and outcome in {"start_requested", "already_running", "held"})
    return result, 0 if success else 1


def _busy_result(state: dict, entry: dict) -> int:
    print(json.dumps({"run_id": state["run_id"], "result": "already_running" if entry.get("reused") else "preserved",
                      "status": state["status"], "published": state["published"], "launch": entry,
                      "next_step": "status " + state["run_id"] + " 查看原任务；本命令未执行到发布，也未追加预算。"}, ensure_ascii=False, indent=2))
    return 1


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        args.repo = args.repo.resolve()
        state_root = (args.state_root or args.repo / ".schema_extract/v2").resolve()
        if args.command == "compatibility-report":
            from .compatibility import build_report, verify_review
            report = build_report(Store(state_root), args.run_id, args.checkpoint, args.target_repo)
            result = {"report": report, "review": verify_review(report, read_json(args.review)) if args.review else None}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "upgrade-runtime":
            from .runtime import upgrade_runtime
            report = read_json(args.report)
            report = report.get("report", report)
            result = upgrade_runtime(Store(state_root), args.run_id, report, read_json(args.review),
                                     target_repo=args.target_repo)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "standalone":
            from .standalone import create_view
            print(json.dumps(create_view(args.source, args.output, args.repo), ensure_ascii=False, indent=2))
            return 0
        if args.command == "audit":
            from .audit import _parent_fd, audit_run, format_audit
            with _parent_fd((args.state_root or args.repo / ".schema_extract/v2").absolute() / "state-root-check"):
                pass
            result = audit_run(Store(state_root), args.run_id, attempt_id=args.attempt, session_id=args.session,
                               verify=args.verify, archive=args.archive)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else format_audit(result))
            return 1 if result["verification"]["status"] == "incomplete" or (result["archive"] and result["archive"]["status"] != "archived") else 0
        if args.command == "monitor":
            from .wakeup import Clock
            store = Store(state_root)
            clock = Clock(store, args.repo)
            if args.monitor_command == "scan":
                if args.wake or args.thread:
                    if args.run_ids:
                        raise ExtractError("a bound clock scan uses its registered scope; --run belongs to a manual scan")
                    result = clock.scan(args.wake, args.thread)
                else:
                    result = monitor.scan(store, args.run_ids)
            elif args.monitor_command == "start":
                result = clock.start(args.thread, args.run_ids, auto_restart_limit=args.auto_restart_limit,
                                     codex_binary=args.codex_bin, probe_delay=args.probe_delay)
            elif args.monitor_command == "ack":
                result = clock.ack(args.wake_id, args.thread)
            elif args.monitor_command == "stop":
                result = clock.stop()
            else:
                result = clock.status()
            if args.monitor_command != "scan" or args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                _print_observations(result["runs"])
                print(f"changes={len(result['changes'])} anomalies={len(result['anomalies'])}")
                if result.get("clock"):
                    print(json.dumps(result["clock"], ensure_ascii=False))
            return 0
        if args.command == "_clock":
            from .wakeup import Clock
            return Clock(Store(state_root), args.repo).run(args.launch_id, args.lock_fd)
        if args.command in {"revise", "continue"}:
            from .revision_control import (PENDING, apply_prepared_revision, prepare_revision,
                                           request_revision, revision_status)
            store = Store(state_root)
            if args.command == "continue" and args.full_restart:
                from .continuation import continue_run
                if not args.checkpoint or not args.reason or not args.reason.strip():
                    raise ExtractError("continue --full-restart requires --checkpoint and a nonempty --reason")
                if args.plan or args.target_repo or args.request_id:
                    raise ExtractError("a full restart does not accept semantic revision plan or request options")
                state = continue_run(args.repo, store, args.run_id, args.checkpoint, full_restart=True,
                                     reason=args.reason, add_seconds=args.add_seconds, add_calls=args.add_calls)
                print(json.dumps({**summary(store, state), "action": "full_restart_registered", "delivered": False},
                                 ensure_ascii=False, indent=2))
                return 0
            if args.command == "continue":
                if args.checkpoint or args.add_seconds or args.add_calls:
                    raise ExtractError("continuous revision preserves the current frontier and budget; use resume for budget additions, or explicit --full-restart")
                proposal = read_json(args.plan) if args.plan else None
                current = revision_status(store, args.run_id)
                if current["status"] not in PENDING:
                    if current["maintenance"] and current["maintenance"].get("status") not in {"completed", "complete", "cancelled"}:
                        result = current
                    else:
                        reason = args.reason or (proposal or {}).get("reason") or "continue: evaluate current frozen target rules and preserve valid progress"
                        request_revision(store, args.run_id, reason=reason, target_repo=args.target_repo, request_id=args.request_id)
                        result = prepare_revision(store, args.run_id, proposal=proposal, request_id=args.request_id, repo=args.repo)
                else:
                    if args.reason or args.target_repo:
                        request_revision(store, args.run_id, reason=args.reason or current["request"]["reason"],
                                         target_repo=args.target_repo, request_id=args.request_id)
                    result = prepare_revision(store, args.run_id, proposal=proposal, request_id=args.request_id, repo=args.repo)
                if proposal is not None and result["status"] == "prepared":
                    result = apply_prepared_revision(store, args.run_id, request_id=args.request_id, repo=args.repo)
            elif args.revision_command == "status":
                result = revision_status(store, args.run_id)
            elif args.revision_command == "request":
                result = request_revision(store, args.run_id, reason=args.reason, target_repo=args.target_repo,
                                          request_id=args.request_id)
            elif args.revision_command == "prepare":
                result = prepare_revision(store, args.run_id, proposal=read_json(args.plan) if args.plan else None,
                                          request_id=args.request_id, repo=args.repo)
            else:
                result = prepare_revision(store, args.run_id, proposal=read_json(args.plan),
                                          request_id=args.request_id, repo=args.repo) if args.plan else None
                if result is None or result["status"] in {"prepared", "applied"}:
                    result = apply_prepared_revision(store, args.run_id, request_id=args.request_id, repo=args.repo)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            # Registration and preparation can succeed while delivery remains
            # pending. Malformed requests still use the common error exit 2.
            return 0
        if args.command in {"pause", "resume"} and args.all:
            if getattr(args, "scope", None):
                raise ExtractError("--scope requires a single run UUID")
            if args.command == "resume" and (args.add_seconds or args.add_calls):
                raise ExtractError("budget additions require a single run UUID; releasing a hold does not add budget")
            if args.run_id or (args.command == "resume" and not args.hold):
                raise ExtractError("--all takes no run UUID; resume --all requires --hold ID")
            store = Store(state_root)
            result = monitor.hold_all(store) if args.command == "pause" else monitor.release_hold(store, args.hold, args.repo)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"pause", "resume"} and (not args.run_id or getattr(args, "hold", None)):
            raise ExtractError("single-run control requires a UUID; --hold requires --all")
        if args.command == "resume" and args.scope:
            if args.add_seconds or args.add_calls:
                raise ExtractError("scope recovery preserves existing credit; add budget with an explicit whole-run resume first")
            store = Store(state_root)
            state = store.load(args.run_id)
            if "scoped_recovery.v2" not in state.get("executor", {}).get("capabilities", []):
                raise ExtractError("frozen executor lacks scoped_recovery.v2; use its original resume or explicitly upgrade-runtime first")
            lock = read_json(store.run_dir(args.run_id) / "inputs.lock.json")
            command = ["resume", args.run_id, "--scope", args.scope, "--repo", str(args.repo), "--state-root", str(state_root)]
            _frozen_entry(lock, command, store=store, state=state)
            request = monitor.request_restart(store, args.run_id, args.scope, args.repo, automatic=False)
            print(json.dumps(request, ensure_ascii=False, indent=2))
            return 0 if request["status"] in {"authorized", "consumed"} and request.get("launch", {}).get("status") != "launch_failed" else 1
        if args.command == "_background":
            store = Store(state_root)
            entry = monitor.claim_launch(store, args.run_id, args.launch_id)
            os.environ["SCHEMA_EXTRACT_ENTRY_ID"] = entry["id"]
            state = store.load(args.run_id)
            lock = read_json(store.run_dir(args.run_id) / "inputs.lock.json")
            command = (["inspect", "--start", args.run_id] if entry["mode"] == "queued" and
                       "hold.v1" not in state.get("executor", {}).get("capabilities", []) else ["resume", args.run_id])
            _frozen_entry(lock, [*command, "--repo", str(args.repo), "--state-root", str(state_root)], store=store, state=state)
            state = run(store, args.run_id, resume=entry["mode"] != "revision_drain",
                        queued_only=entry["mode"] == "queued", launch_id=entry["id"])
            print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
            return 0 if state["status"] == "ready" and state["published"] else 1
        if args.command in {"issues", "inspect"}:
            if args.command == "inspect" and args.start:
                store = Store(state_root)
                state = store.load(args.start)
                if state["status"] != "queued":
                    raise ExtractError("inspect --start only accepts queued work; use resume explicitly for an interrupted run")
                lock = read_json(store.run_dir(args.start) / "inputs.lock.json")
                if lock["fingerprint"] != state["input_fingerprint"]:
                    raise ExtractError("run input fingerprint differs from its lock")
                entry = _launch_context(store, state, mode="queued")
                if entry.get("reused") or entry.get("status") == "preserved":
                    return _busy_result(state, entry)
                _frozen_entry(lock, ["inspect", "--start", args.start, "--repo", str(args.repo), "--state-root", str(state_root)], store=store, state=state)
                state = run(store, args.start, resume=True, queued_only=True, launch_id=entry.get("id"))
                print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
                return 0 if state["status"] == "ready" and state["published"] else 1
            result = registry(args.repo) if args.command == "issues" else inspect_runs(Store(state_root), args.repo, args.run_ids)
            if args.command == "inspect":
                _print_observations(result, as_json=args.json)
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command not in {"extract", "import-legacy"}:
            store = Store(state_root)
            state = store.load(args.run_id)
            command = [args.command, args.run_id, "--repo", str(args.repo), "--state-root", str(state_root)]
            if getattr(args, "full", False):
                command.append("--full")
            if args.command == "export":
                command.extend(["--output", str(args.output.absolute())])
            lock = read_json(store.run_dir(args.run_id) / "inputs.lock.json")
            if lock["fingerprint"] != state["input_fingerprint"]:
                raise ExtractError("run input fingerprint differs from its lock")
            entry, request = None, None
            if args.command in {"resume", "export"}:
                budget_addition = addition(args.add_seconds, args.add_calls) if args.command == "resume" else addition()
                if any(budget_addition.values()):
                    if BUDGET_CAPABILITY not in state.get("executor", {}).get("capabilities", []):
                        raise ExtractError("this frozen executor cannot add budget; adopt a reviewed compatible "
                                           "runtime with upgrade-runtime, then resume with --add-*; or explicitly use "
                                           "continue --full-restart with --checkpoint, --reason and --add-*")
                    command.extend(["--add-seconds", str(args.add_seconds), "--add-calls", str(args.add_calls)])
                store.assert_startable(args.run_id)
                entry = _launch_context(store, state, mode=args.command, budget_addition=budget_addition)
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
            _frozen_entry(lock, command, store=store, state=state)
            if args.command == "status":
                result = state if args.full else summary(store, state)
                code = 0
            elif args.command in {"pause", "cancel"}:
                try:
                    state = run(store, args.run_id)
                except RunnerBusy:
                    state = store.load(args.run_id)
                result, code = {"request": request, "run": summary(store, state)}, 0
            elif args.command == "export":
                result = export_checkpoint(store, args.run_id, args.output)
                code = 0 if result["status"] == "published" else 1
            elif args.command == "candidates":
                result, code = candidate_report(store, state, args.repo), 0
            else:
                state = run(store, args.run_id, resume=entry.get("mode") != "revision_drain",
                            queued_only=entry.get("mode") == "queued", launch_id=entry.get("id"))
                result = summary(store, state)
                code = 0 if state["status"] == "ready" and state["published"] else 1
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return code
        config = {key: getattr(args, key) for key in ("worker_effort", "reviewer_effort", "model", "revision_limit",
                                                     "operator_seconds", "stage_seconds", "call_seconds", "max_calls")}
        config["codex_binary"] = args.codex_bin
        if args.ids:
            if args.command == "import-legacy" or args.output or not args.output_root or not (args.background or args.queue):
                raise ExtractError("--ids requires --output-root and --background or --queue")
            store, results = Store(state_root), []
            targets = monitor.catalog_targets(args.repo, args.ids, args.output_root)
            for target in targets:
                if target.get("error"):
                    results.append(target)
                    continue
                try:
                    registered, state = monitor.registered_extract(args.repo, store, target["output"], operator_id=target["operator_id"], config=config, queued=args.queue)
                    if state is not None and args.background and registered["authorized"]:
                        registered["background"] = monitor.start_background(store, state["run_id"], args.repo)
                    result, code = _registration_result(store, registered, state, queued=args.queue, background=args.background)
                    results.append({"operator_id": target["operator_id"], **result, "exit_code": code})
                except (ExtractError, OSError, KeyError, TypeError, AttributeError) as exc:
                    results.append({"operator_id": target["operator_id"], "error": str(exc)})
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
            return 1 if any(value.get("error") or value.get("exit_code") for value in results) else 0
        if not args.output or args.output_root:
            raise ExtractError("single-source extraction requires --output and does not take --output-root")
        if args.command == "import-legacy":
            from .legacy import import_legacy
            if args.background:
                raise ExtractError("import-legacy supports --queue followed by explicit resume; --background belongs to extract")
            if args.operator_id:
                raise ExtractError("import-legacy requires --source with explicit legacy frozen-source bindings")
            store, state, _ = import_legacy(args.repo, state_root, args.output, source_task=args.source_task,
                                             checkpoint_id=args.checkpoint_id, source=args.source,
                                             bindings=args.bindings, config=config, queued=args.queue)
        else:
            store = Store(state_root)
            registered, state = monitor.registered_extract(args.repo, store, args.output, operator_id=args.operator_id,
                                                          source=args.source, bindings=args.bindings, config=config, queued=args.queue)
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
        if entry.get("reused") or entry.get("status") == "preserved":
            return _busy_result(state, entry)
        _frozen_entry(read_json(store.run_dir(state["run_id"]) / "inputs.lock.json"),
                      ["resume", state["run_id"], "--repo", str(args.repo), "--state-root", str(state_root)], store=store, state=state)
        state = run(store, state["run_id"], resume=True, queued_only=entry.get("mode") == "queued", launch_id=entry.get("id"))
        print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
        return 0 if state["status"] == "ready" and state["published"] else 1
    except (ExtractError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
