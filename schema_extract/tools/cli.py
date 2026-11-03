"""Supervisor controls route to each run's own immutable executor."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
import sys

from . import __version__
from .common import ExtractError, RunnerBusy, digest, read_json, rules_root
from .runner import DEFAULT_CONFIG, create_run, export_checkpoint, run, summary
from .snapshots import frozen_invocation
from .store import Store


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


def inspect_runs(store, repo: Path) -> list[dict]:
    """Ask each frozen executor for its summary; inspection never starts a call."""
    summaries = []
    for path in sorted((store.root / "runs").glob("*/state.json")):
        run_id = path.parent.name
        try:
            lock = read_json(path.parent / "inputs.lock.json")
            if lock["fingerprint"] != store.load(run_id)["input_fingerprint"]:
                raise ExtractError("run input fingerprint differs from its lock")
            command, rules, environment = frozen_invocation(lock, ["status", run_id,
                "--repo", str(repo.resolve()), "--state-root", str(store.root)])
            process = subprocess.run(command, cwd=rules, env=environment,
                                     capture_output=True, text=True, timeout=30)
            if process.returncode:
                raise ExtractError(process.stderr.strip())
            summaries.append({"run": json.loads(process.stdout), "executor": str(rules)})
        except (ExtractError, ValueError, OSError, subprocess.SubprocessError) as exc:
            summaries.append({"run_id": run_id, "error": str(exc)})
    return summaries


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
    arguments, rules, environment = frozen_invocation(lock, command)
    if rules_root() == rules.resolve():
        return
    # cwd and the first import path both point at the locked package. -B keeps
    # Python from creating __pycache__ inside the immutable snapshot.
    os.chdir(rules)
    os.execve(sys.executable, arguments, environment)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        args.repo = args.repo.resolve()
        state_root = (args.state_root or args.repo / ".schema_extract/v2").resolve()
        if args.command in {"issues", "inspect"}:
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
