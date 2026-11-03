"""Direct independent Codex CLI calls and exact UUID resumes, preserving provider config."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time
from uuid import UUID

from .audit import parse_events, result_usage
from .common import ExtractError, digest, read_json, sha256, write_json
from .processes import owned_processes, process_table, terminate_owned


class CodexCLI:
    def __init__(self, executable: str = "codex", model: str | None = None):
        self.executable, self.model = executable, model

    def command(self, *, cwd: Path, schema: Path, output: Path, role: str,
                effort: str, session_id: str | None) -> list[str]:
        if role not in {"worker", "reviewer"}:
            raise ExtractError("unknown agent role")
        if session_id:
            try:
                if str(UUID(session_id)) != session_id:
                    raise ValueError()
            except (ValueError, TypeError, AttributeError) as exc:
                raise ExtractError("resume requires the exact role session UUID") from exc
        # Global options precede exec/resume; resume itself has no -s or -C.
        command = [self.executable, "--no-daemon", "-a", "never", "-s",
                   "workspace-write" if role == "worker" else "read-only", "-C", str(cwd),
                   "-c", f'model_reasoning_effort="{effort}"',
                   "-c", "sandbox_workspace_write.exclude_tmpdir_env_var=true",
                   "-c", "sandbox_workspace_write.exclude_slash_tmp=true",
                   "-c", "sandbox_workspace_write.network_access=false", "exec"]
        if session_id:
            command += ["resume", session_id]
        command += ["--skip-git-repo-check", "--json", "--output-schema", str(schema),
                    "--output-last-message", str(output)]
        if self.model:
            command += ["--model", self.model]
        return command + ["-"]

    def run(self, *, attempt: dict, prompt: str, schema: dict, effort: str,
            timeout: float, on_process=lambda _: None, cancel_event=None) -> dict:
        directory, cwd = Path(attempt["directory"]), Path(attempt["workspace"])
        schema_path, response_path = directory / "response.schema.json", directory / "response.json"
        write_json(schema_path, schema)
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        command = self.command(cwd=cwd, schema=schema_path, output=response_path,
                               role=attempt["role"], effort=effort, session_id=attempt["session_id"])
        invocation = {"argv": command, "role": attempt["role"], "effort": effort, "model": self.model,
                      "resume_session_id": attempt["session_id"], "started_at": time.time(), "cli_version": None}
        invocation.update({key: attempt.get(key) for key in ("run_id", "attempt_id", "stage", "node", "session_group")})
        invocation.update(prompt_sha256=sha256(prompt.encode("utf-8")), response_schema_digest=digest(schema))
        started = time.monotonic()
        termination = {"confirmed": True, "actions": [], "survivors": [], "reason": "not_started"}
        exit_code, outcome, process = None, "completed", None
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("cancel requested before invocation launch")
            invocation["cli_version"] = subprocess.check_output([self.executable, "--version"], text=True, timeout=min(15, timeout)).strip()
            write_json(directory / "invocation.json", invocation)
            # A file supplies stdin so a stalled child cannot block a pipe write.
            with (directory / "prompt.txt").open("r", encoding="utf-8") as stdin, \
                    (directory / "events.jsonl").open("w", encoding="utf-8") as stdout, \
                    (directory / "stderr.txt").open("w", encoding="utf-8") as stderr:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("cancel requested before invocation launch")
                # A crash in the launch/identity window must fence, never assume
                # that a missing process.json proves that nothing was started.
                write_json(directory / "launch.json", {"status": "launching", "at": time.time()})
                process = subprocess.Popen(command, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                termination = {"confirmed": False, "actions": [], "survivors": [], "reason": "process_started"}
                identity = {"pid": process.pid, "started": "unobserved"}
                known = []
                def record():
                    nonlocal known
                    known = list({(item["pid"], item["started"]): item for item in
                                  [*known, *owned_processes(identity, known)]}.values())
                    value = {"identity": identity, "descendants": known}
                    write_json(directory / "process.json", value)
                    on_process(value)
                try:
                    identity = process_table().get(process.pid) or {"pid": process.pid, "started": "already_exited"}
                    record()
                    next_scan = time.monotonic() + 1
                    while process.poll() is None:
                        if cancel_event is not None and cancel_event.is_set():
                            outcome = "cancelled"
                            break
                        if time.monotonic() - started >= timeout:
                            outcome = "timeout"
                            break
                        if time.monotonic() >= next_scan:
                            record()
                            next_scan = time.monotonic() + 1
                        time.sleep(0.05)
                finally:
                    try:
                        termination = terminate_owned(identity, known)
                    except (OSError, subprocess.SubprocessError) as exc:
                        termination = {"confirmed": False, "actions": [], "survivors": [],
                                       "reason": f"cannot enumerate invocation processes: {exc}"}
                    if process.poll() is None and not termination["confirmed"]:
                        # Popen still owns this live child and its start_new_session group.
                        # Kill it even if process-table collection itself failed.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                            termination["actions"].append({"pid": process.pid, "signal": "SIGKILL"})
                        except ProcessLookupError:
                            pass
                    try:
                        exit_code = process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        termination["confirmed"] = False
                    if not termination["confirmed"]:
                        outcome = "termination_unconfirmed"
        except InterruptedError as exc:
            outcome = "cancelled"
            invocation["error"] = str(exc)
        except (OSError, subprocess.SubprocessError) as exc:
            outcome = "cli_error" if termination["confirmed"] else "termination_unconfirmed"
            invocation["error"] = str(exc)
        parsed = parse_events(directory / "events.jsonl")
        trace = parsed["trace"]
        events, session_ids = list(trace["event_counts"]), trace["session_ids"]
        parse_errors = trace["parse_error_lines"]
        if trace["truncated_line"]:
            parse_errors = [*parse_errors, trace["truncated_line"]]
        completed = bool(trace["event_counts"].get("turn.completed"))
        session_id = session_ids[0] if len(set(session_ids)) == 1 else None
        try:
            response = read_json(response_path)
        except ExtractError:
            response = None
        if outcome == "completed":
            if exit_code != 0 or not completed or not session_id or parse_errors or trace["error"] or not trace["stable"]:
                outcome = "cli_error"
            elif attempt["session_id"] and session_id != attempt["session_id"]:
                outcome = "session_mismatch"
            elif response is None:
                outcome = "receipt_error"
        result = {**invocation, "outcome": outcome, "exit_code": exit_code, "session_id": session_id,
                  "event_types": events, "parse_error_lines": parse_errors, "termination": termination,
                  "elapsed_seconds": time.monotonic() - started, "completed_at": time.time(), "response": response,
                  "trace": trace, "usage": result_usage(parsed, outcome=outcome, termination=termination),
                  "timing_source": "elapsed_seconds: monotonic CLI transport including setup, termination and event parsing; processing.json: Runner checks/processing"}
        write_json(directory / "result.json", result)
        return result
