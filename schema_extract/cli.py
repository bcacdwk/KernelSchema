"""Supervisor controls route to each run's own immutable executor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from . import __version__
from .common import ExtractError, RunnerBusy, read_json
from .runner import DEFAULT_CONFIG, create_run, export_checkpoint, run, summary
from .snapshots import verify_snapshot
from .store import Store


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Schema Extract v2: P1-P6 extraction with concurrent ready nodes and an independently approved checkpoint.")
    result.add_argument("--version", action="version", version=f"Schema Extract {__version__}")
    sub = result.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="freeze inputs and independently approve all six stages")
    select = extract.add_mutually_exclusive_group(required=True)
    select.add_argument("--id", dest="operator_id")
    select.add_argument("--source", help="catalog directory or schema_extract.source.v2 JSON descriptor")
    extract.add_argument("--bindings", type=Path, help="descriptor for git:<ref>:<directory> shorthand")
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--queue", action="store_true", help="freeze and queue only; resume starts the fixed-budget run")
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
        command.add_argument("run_id")
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-root", type=Path)
        if name == "status":
            command.add_argument("--full", action="store_true", help="print the entire authoritative state")
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
    for name in ("issues", "inspect"):
        command = sub.add_parser(name, help="read-only framework registry" if name == "issues" else "read-only run summaries; does not install a scheduler")
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-root", type=Path)
        if name == "inspect":
            command.add_argument("--start", metavar="RUN_UUID", help="request execution of one explicitly selected queued run")
    return result


def _frozen_entry(lock: dict, command: list[str]) -> None:
    verify_snapshot(lock)
    rules = Path(lock["snapshot"]) / "rules"
    if Path(__file__).resolve().parent.parent == rules.resolve():
        return
    # cwd and the first import path both point at the locked package. -B keeps
    # Python from creating __pycache__ inside the immutable snapshot.
    os.chdir(rules)
    environment = {**os.environ, "PYTHONPATH": str(rules), "PYTHONDONTWRITEBYTECODE": "1"}
    os.execve(sys.executable, [sys.executable, "-B", "-m", "schema_extract", *command], environment)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        args.repo = args.repo.resolve()
        state_root = (args.state_root or args.repo / ".schema_extract/v2").resolve()
        if args.command in {"issues", "inspect"}:
            from .maintenance import inspect_runs, registry
            if args.command == "inspect" and args.start:
                store = Store(state_root)
                state = store.load(args.start)
                if state["status"] != "queued":
                    raise ExtractError("inspect --start only accepts queued work; use resume explicitly for an interrupted run")
                lock = read_json(store.run_dir(args.start) / "inputs.lock.json")
                if lock["fingerprint"] != state["input_fingerprint"]:
                    raise ExtractError("run input fingerprint differs from its lock")
                _frozen_entry(lock, ["inspect", "--start", args.start, "--repo", str(args.repo), "--state-root", str(state_root)])
                state = run(store, args.start, resume=True, queued_only=True)
                print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
                return 0 if state["status"] == "ready" and state["published"] else 1
            result = registry(args.repo) if args.command == "issues" else inspect_runs(Store(state_root), args.repo)
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
            _frozen_entry(lock, command)
            if args.command == "status":
                result = state if args.full else summary(store, state)
                code = 0
            elif args.command in {"pause", "cancel"}:
                request = store.request_control(args.run_id, args.command)
                try:
                    state = run(store, args.run_id)
                except RunnerBusy:
                    state = store.load(args.run_id)
                result, code = {"request": request, "run": summary(store, state)}, 0
            elif args.command == "export":
                result = export_checkpoint(store, args.run_id, args.output)
                code = 0 if result["status"] == "published" else 1
            elif args.command == "candidates":
                from .maintenance import candidate_report
                result, code = candidate_report(store, state, args.repo), 0
            else:
                state = run(store, args.run_id, resume=True)
                result = summary(store, state)
                code = 0 if state["status"] == "ready" and state["published"] else 1
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return code
        config = {key: getattr(args, key) for key in ("worker_effort", "reviewer_effort", "model", "revision_limit",
                                                     "operator_seconds", "stage_seconds", "call_seconds", "max_calls")}
        config["codex_binary"] = args.codex_bin
        if args.command == "import-legacy":
            from .legacy import import_legacy
            if args.operator_id:
                raise ExtractError("import-legacy requires --source with explicit legacy frozen-source bindings")
            store, state, _ = import_legacy(args.repo, state_root, args.output, source_task=args.source_task,
                                             checkpoint_id=args.checkpoint_id, source=args.source,
                                             bindings=args.bindings, config=config, queued=args.queue)
        else:
            store, state, _ = create_run(args.repo, state_root, args.output, operator_id=args.operator_id,
                                         source=args.source, bindings=args.bindings, config=config, queued=args.queue)
        print(json.dumps({"run_id": state["run_id"], "state_directory": str(store.run_dir(state["run_id"])),
                          "status": state["status"]}, ensure_ascii=False), file=sys.stderr, flush=True)
        if args.queue:
            print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
            return 0
        _frozen_entry(read_json(store.run_dir(state["run_id"]) / "inputs.lock.json"),
                      ["resume", state["run_id"], "--repo", str(args.repo), "--state-root", str(state_root)])
        state = run(store, state["run_id"])
        print(json.dumps(summary(store, state), ensure_ascii=False, indent=2))
        return 0 if state["status"] == "ready" and state["published"] else 1
    except (ExtractError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
