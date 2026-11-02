"""Local CLI transport. No API credentials, shared daemon or implicit last session."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Callable


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_table() -> dict[int, dict]:
    """Record only PID identities and relationships, never command/environment text."""
    out = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,pgid=,lstart=,stat="], text=True
    )
    result = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 9:
            result[int(parts[0])] = {
                "pid": int(parts[0]), "ppid": int(parts[1]), "pgid": int(parts[2]),
                "started": " ".join(parts[3:8]), "stat": parts[8],
            }
    return result


def owned_processes(identity: dict, known: list[dict] = ()) -> list[dict]:
    table = process_table()
    root = table.get(identity["pid"])
    same_root = root and root["started"] == identity["started"]
    selected = {
        p["pid"] for p in known
        if p["pid"] in table and table[p["pid"]]["started"] == p["started"]
    }
    if same_root:
        selected.add(identity["pid"])
    # The group was created solely for this invocation. The root's saved identity
    # prevents targeting an unrelated process after a crash and PID reuse.
    if selected or same_root:
        selected.update(p for p, item in table.items() if item["pgid"] == identity["pid"])
    while True:
        children = {p for p, item in table.items() if item["ppid"] in selected}
        if children <= selected:
            break
        selected.update(children)
    return [table[p] for p in sorted(selected) if not table[p]["stat"].startswith("Z")]


def terminate_owned(identity: dict, known: list[dict] = (), grace: float = 3) -> dict:
    """Stop the isolated process group and observed descendants, then confirm."""
    seen = list(known)
    actions = []
    table = process_table()
    root = table.get(identity["pid"])
    matched = (root and root["started"] == identity["started"]) or any(
        p["pid"] in table and table[p["pid"]]["started"] == p["started"] for p in known)
    unknown_group = [p for p in table.values() if p["pgid"] == identity["pid"] and not p["stat"].startswith("Z")]
    if unknown_group and not matched:
        return {"confirmed": False, "actions": [], "survivors": unknown_group,
                "reason": "process group exists but ownership cannot be safely identified"}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = owned_processes(identity, seen)
        seen = list({(p["pid"], p["started"]): p for p in seen + alive}.values())
        for p in reversed(alive):
            try:
                os.kill(p["pid"], sig)
                actions.append({"pid": p["pid"], "signal": sig.name})
            except ProcessLookupError:
                pass
        until = time.monotonic() + grace
        while time.monotonic() < until and owned_processes(identity, seen):
            time.sleep(0.1)
        if not owned_processes(identity, seen):
            return {"confirmed": True, "actions": actions, "survivors": []}
    return {"confirmed": False, "actions": actions, "survivors": owned_processes(identity, seen)}


class CodexCLI:
    def __init__(self, executable: str = "codex", model: str = "gpt-6-astra"):
        self.executable = executable
        self.model = model
        self.effort = "xhigh"

    def version(self) -> str:
        return subprocess.check_output([self.executable, "--version"], text=True, timeout=15).strip()

    def command(self, cwd: Path, schema: Path, output: Path, role: str, thread_id: str | None) -> list[str]:
        if role not in {"worker", "reviewer", "supervisor"}:
            raise ValueError("unknown CLI role")
        if thread_id:
            from uuid import UUID
            UUID(thread_id)  # Never accept --last or a loose thread-name lookup.
        command = [
            self.executable, "--no-daemon", "-a", "never", "-s",
            "workspace-write" if role == "worker" else "read-only",
            "-C", str(cwd), "-c", 'model_reasoning_effort="xhigh"',
            "-c", "sandbox_workspace_write.exclude_tmpdir_env_var=true",
            "-c", "sandbox_workspace_write.exclude_slash_tmp=true",
            "-c", "sandbox_workspace_write.network_access=false", "exec",
        ]
        if thread_id:
            command += ["resume", thread_id]
        command += [
            "--ignore-user-config", "--skip-git-repo-check",
            "--model", self.model, "--json", "--output-schema", str(schema),
            "--output-last-message", str(output), "-",
        ]
        return command

    def run(
        self, *, cwd: Path, records: Path, prompt: str, schema: dict,
        role: str = "worker", thread_id: str | None = None,
        cancel: Callable[[], bool] = lambda: False,
        on_process: Callable[[dict], None] = lambda _: None,
    ) -> dict:
        cwd = cwd.resolve()
        records = records.resolve()
        records.mkdir(parents=True, exist_ok=False)
        schema_path, output_path = records / "response.schema.json", records / "response.json"
        write_json(schema_path, schema)
        (records / "prompt.txt").write_text(prompt)
        command = self.command(cwd, schema_path, output_path, role, thread_id)
        invocation = {"cli_version": None, "argv": command, "cwd": str(cwd),
                      "model": self.model, "effort": self.effort, "role": role,
                      "resume_thread_id": thread_id, "started_at": time.time()}
        started = time.monotonic()

        def not_started(outcome, error):
            result = {**invocation, "thread_id": None, "exit_code": None,
                      "elapsed_seconds": round(time.monotonic() - started, 3),
                      "event_types": [], "jsonl_error_lines": [], "usage": {}, "usage_scope": "thread_cumulative",
                      "termination": {"confirmed": True, "actions": [], "survivors": [], "reason": "process_not_started"},
                      "outcome": outcome, "error": str(error), "response": None}
            write_json(records / "result.json", result)
            return result

        try:
            invocation["cli_version"] = self.version()
        except (OSError, subprocess.SubprocessError) as error:
            write_json(records / "invocation.json", invocation)
            return not_started("preflight_error", error)
        write_json(records / "invocation.json", invocation)
        reason, termination, known = None, None, []
        last_scan, previous_processes = 0, None
        with (records / "events.jsonl").open("w") as out, (records / "stderr.txt").open("w") as err:
            try:
                process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=out,
                                           stderr=err, text=True, start_new_session=True)
            except OSError as error:
                return not_started("spawn_error", error)
            identity = process_table().get(process.pid)
            if not identity:
                process.wait(timeout=10)
                identity = {"pid": process.pid, "started": "already-exited"}
            try:
                write_json(records / "process.json", {"identity": identity, "descendants": []})
                on_process({"identity": identity, "descendants": [], "records": str(records)})
                process.stdin.write(prompt)
                process.stdin.close()
                while process.poll() is None:
                    if time.monotonic() - last_scan >= 1:
                        alive = owned_processes(identity, known)
                        known = list({(p["pid"], p["started"]): p for p in known + alive}.values())
                        process_record = {"identity": identity, "descendants": known}
                        if process_record != previous_processes:
                            write_json(records / "process.json", process_record)
                            previous_processes = process_record
                        last_scan = time.monotonic()
                    if cancel():
                        reason = "cancelled"
                        termination = terminate_owned(identity, known)
                        break
                    time.sleep(0.15)
            except BaseException:
                terminate_owned(identity, known)
                process.wait(timeout=10)
                raise
            if termination is None:
                # A successful parent exit is not proof its tool subprocesses stopped.
                termination = terminate_owned(identity, known)
            if termination and not termination["confirmed"]:
                reason = "termination_unconfirmed"
            try:
                exit_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                exit_code = None
                reason = "termination_unconfirmed"
        events, parse_errors, usage, found_thread, completed = [], [], {}, None, False
        for number, line in enumerate((records / "events.jsonl").read_text().splitlines(), 1):
            try:
                event = json.loads(line)
                events.append(event.get("type"))
                if event.get("type") == "thread.started":
                    found_thread = event.get("thread_id")
                if event.get("type") == "turn.completed":
                    completed = True
                    usage = event.get("usage", {})
            except (ValueError, AttributeError):
                parse_errors.append(number)
        response = None
        try:
            response = json.loads(output_path.read_text())
        except (OSError, ValueError):
            pass
        if not reason and (exit_code != 0 or not completed or not found_thread or response is None or parse_errors):
            reason = "cli_error"
        if thread_id and found_thread != thread_id and reason not in {"cancelled", "termination_unconfirmed"}:
            reason = "session_mismatch"
        result = {**invocation, "thread_id": found_thread, "exit_code": exit_code,
                  "elapsed_seconds": round(time.monotonic() - started, 3),
                  "event_types": events, "jsonl_error_lines": parse_errors, "usage": usage,
                  "usage_scope": "thread_cumulative",
                  "termination": termination or {"confirmed": True, "actions": [], "survivors": []},
                  "outcome": reason or "completed", "response": response}
        write_json(records / "result.json", result)
        return result
