#!/usr/bin/env python3
"""Extract a schema from a selected kernel with the local Codex CLI."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema_extract.tools.codex_cli import CodexCLI, write_json
from schema_extract.tools.execution import Runner, publish, task_status
from schema_extract.tools.framework import candidate_report, registry
from schema_extract.tools.planning import DEFAULTS, make_plan
from schema_extract.tools.records import StateError, Store, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--state-root", type=Path, help="default: <repo>/.schema_extract")
    parser.add_argument("--codex", default="codex", help="local Codex executable")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("discover")
    sub.add_parser("issues", help="list and count pending maintenance proposals and archived decisions")
    plan_cmd = sub.add_parser("plan")
    plan_cmd.add_argument("--id", action="append", required=True)
    plan_cmd.add_argument("--run-id")
    plan_cmd.add_argument("--seed-run", help="reuse verified accepted drafts from this stopped run; preserve usage and re-review Contract")
    plan_cmd.add_argument("--work-root", required=True, type=Path)
    plan_cmd.add_argument("--output-root", required=True, type=Path)
    plan_cmd.add_argument("--source", action="append", default=[], help="extra fixed repository-relative material")
    plan_cmd.add_argument("--model", default=DEFAULTS["model"])
    plan_cmd.add_argument("--jobs", type=int, choices=(1, 2), default=DEFAULTS["jobs"])
    for name in ("run", "status", "candidates", "pause", "cancel", "resume", "retry", "export"):
        command = sub.add_parser(name)
        command.add_argument("run_id")
        if name == "status":
            command.add_argument("--full", action="store_true", help="show complete authoritative task records instead of compact progress")
        if name in {"resume", "retry", "export"}:
            command.add_argument("--id", action="append", required=True)
        if name in {"resume", "retry"}:
            command.add_argument("--reason")
        if name == "export":
            command.add_argument("--to", type=Path, help="only with one selected task; must be a new directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "issues":
            records = registry(args.repo)
            pending = {key: value for key, value in records.items() if value["status"] == "pending"}
            print(json.dumps({"pending": len(pending), "records": pending, "archived": len(records) - len(pending)},
                             ensure_ascii=False, indent=2))
            return 0
        store = Store(args.repo, args.state_root or args.repo / ".schema_extract")
        if args.command == "discover":
            entries = read_json(store.repo / "kernel_wiki/catalog.json")["operators"]
            print(json.dumps([{k: entry.get(k) for k in ("id", "title", "category")} for entry in entries], ensure_ascii=False, indent=2))
            return 0
        if args.command == "plan":
            config = {**DEFAULTS, "model": args.model, "jobs": args.jobs}
            with store.lock():
                plan = make_plan(store, args.id, args.work_root, args.output_root, args.source, config, args.run_id,
                                 seed_run=args.seed_run)
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        directory, plan = store.load_plan(args.run_id)
        if args.command == "candidates":
            print(json.dumps(candidate_report(store, plan), ensure_ascii=False, indent=2))
            return 0
        if args.command == "status":
            states = [store.load(key) for key in plan["tasks"]]
            print(json.dumps(states if args.full else [task_status(store, state) for state in states], ensure_ascii=False, indent=2))
            return 0
        if args.command in {"pause", "cancel"}:
            write_json(directory / "control.json", {"mode": args.command, "requested_at": time.time()})
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
                print(json.dumps({"review_framework_candidates": ["candidates", plan["run_id"]]}), flush=True)
            finally:
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
        return 0
    except (StateError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"schema_extract: {error}", file=sys.stderr)
        return 2



if __name__ == "__main__":
    raise SystemExit(main())
