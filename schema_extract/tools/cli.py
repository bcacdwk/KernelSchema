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
from .store import Store
from . import monitor


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
                            ("resume", "reconcile old processes and resume with the original budgets"),
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
    for name in ("issues", "inspect"):
        command = sub.add_parser(name, help="read-only framework registry" if name == "issues" else "read-only run summaries; does not install a scheduler")
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-root", type=Path)
        if name == "inspect":
            command.add_argument("--start", metavar="RUN_UUID", help="request execution of one explicitly selected queued run")
            command.add_argument("--run", nargs="+", dest="run_ids")
            command.add_argument("--json", action="store_true")
    continued = sub.add_parser("continue", help="queue a same-source successor using the current frozen executor")
    continued.add_argument("run_id")
    continued.add_argument("--checkpoint", required=True)
    continued.add_argument("--repo", type=Path, default=Path.cwd())
    continued.add_argument("--state-root", type=Path)
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


def _frozen_entry(lock: dict, command: list[str]) -> None:
    arguments, rules, environment = frozen_invocation(lock, command)
    if rules_root() == rules.resolve():
        return
    # cwd and the first import path both point at the locked package. -B keeps
    # Python from creating __pycache__ inside the immutable snapshot.
    os.chdir(rules)
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


def _launch_context(store, state: dict, *, mode="resume") -> dict:
    token = os.environ.get("SCHEMA_EXTRACT_ENTRY_ID")
    entry = store.read_supervisor()["starts"].get(state["run_id"], {}).get("launch", {})
    if token and entry.get("id") == token and entry.get("status") == "claimed" and entry.get("identity", {}).get("pid") == os.getpid():
        return entry
    entry = monitor.reserve_launch(store, state, mode=mode)
    if not entry.get("reused") and entry.get("id"):
        os.environ["SCHEMA_EXTRACT_ENTRY_ID"] = entry["id"]
    return entry


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        args.repo = args.repo.resolve()
        state_root = (args.state_root or args.repo / ".schema_extract/v2").resolve()
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
        if args.command == "continue":
            from .continuation import continue_run
            state = continue_run(args.repo, Store(state_root), args.run_id, args.checkpoint)
            print(json.dumps(summary(Store(state_root), state), ensure_ascii=False, indent=2))
            return 0 if state["status"] == "queued" else 1
        if args.command in {"pause", "resume"} and args.all:
            if args.run_id or (args.command == "resume" and not args.hold):
                raise ExtractError("--all takes no run UUID; resume --all requires --hold ID")
            store = Store(state_root)
            result = monitor.hold_all(store) if args.command == "pause" else monitor.release_hold(store, args.hold, args.repo)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"pause", "resume"} and (not args.run_id or getattr(args, "hold", None)):
            raise ExtractError("single-run control requires a UUID; --hold requires --all")
        if args.command == "_background":
            store = Store(state_root)
            entry = monitor.claim_launch(store, args.run_id, args.launch_id)
            os.environ["SCHEMA_EXTRACT_ENTRY_ID"] = entry["id"]
            state = store.load(args.run_id)
            lock = read_json(store.run_dir(args.run_id) / "inputs.lock.json")
            command = (["inspect", "--start", args.run_id] if entry["mode"] == "queued" and
                       "hold.v1" not in state.get("executor", {}).get("capabilities", []) else ["resume", args.run_id])
            _frozen_entry(lock, [*command, "--repo", str(args.repo), "--state-root", str(state_root)])
            state = run(store, args.run_id, resume=True, queued_only=entry["mode"] == "queued", launch_id=entry["id"])
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
                if entry.get("reused"):
                    print(json.dumps(entry, ensure_ascii=False, indent=2))
                    return 0
                _frozen_entry(lock, ["inspect", "--start", args.start, "--repo", str(args.repo), "--state-root", str(state_root)])
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
                store.assert_startable(args.run_id)
                entry = _launch_context(store, state, mode=args.command)
                if entry.get("reused"):
                    print(json.dumps(entry, ensure_ascii=False, indent=2))
                    return 0
            elif args.command in {"pause", "cancel"}:
                request_id = os.environ.get("SCHEMA_EXTRACT_CONTROL_REQUEST")
                if request_id:
                    request = next((value for value in store.control_requests(args.run_id)
                                    if value["id"] == request_id and value["action"] == args.command), None)
                if request is None:
                    request = store.request_control(args.run_id, args.command)
                    os.environ["SCHEMA_EXTRACT_CONTROL_REQUEST"] = request["id"]
            _frozen_entry(lock, command)
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
                state = run(store, args.run_id, resume=True, queued_only=entry.get("mode") == "queued", launch_id=entry.get("id"))
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
                    results.append({"operator_id": target["operator_id"], **registered})
                except (ExtractError, OSError, KeyError, TypeError, AttributeError) as exc:
                    results.append({"operator_id": target["operator_id"], "error": str(exc)})
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
            return 1 if any(value.get("error") or (value.get("background") or {}).get("status") == "launch_failed" for value in results) else 0
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
                print(json.dumps(registered, ensure_ascii=False, indent=2))
                return 1 if (registered.get("background") or {}).get("status") == "launch_failed" else 0
        print(json.dumps({"run_id": state["run_id"], "state_directory": str(store.run_dir(state["run_id"])),
                          "status": state["status"]}, ensure_ascii=False), file=sys.stderr, flush=True)
        if args.queue:
            print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
            return 0
        entry = _launch_context(store, state, mode="queued" if state["status"] == "queued" else "resume")
        if entry.get("reused") or entry.get("status") == "preserved":
            print(json.dumps(entry, ensure_ascii=False, indent=2))
            return 0
        _frozen_entry(read_json(store.run_dir(state["run_id"]) / "inputs.lock.json"),
                      ["resume", state["run_id"], "--repo", str(args.repo), "--state-root", str(state_root)])
        state = run(store, state["run_id"], resume=True, queued_only=entry.get("mode") == "queued", launch_id=entry.get("id"))
        print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
        return 0 if state["status"] == "ready" and state["published"] else 1
    except (ExtractError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
