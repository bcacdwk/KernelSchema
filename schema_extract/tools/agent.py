"""Shared gated lifecycle, exact scoped sessions, and verified native CLI results."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from uuid import uuid4

from jsonschema import validators

from . import processes
from .audit import _open_file, parse_events, result_usage
from .backends import backend_for, exact_uuid, strict_json
from .common import ExtractError, digest, read_json, real_directory, regular_file, sha256, write_json
from .artifacts import artifact_inventory
from .observation import begin_usage_observation, finish_usage_observation
from .processes import (GATE_PROTOCOL, child_exited, identity_key, mark_not_started,
                        owned_processes, process_table, terminate_owned)


def _binding(attempt):
    for key in ("backend", "model", "effort", "run_id", "attempt_id", "stage", "node", "role",
                "directory", "workspace", "session_directory", "session_scope", "session_id", "writable_files"):
        if key not in attempt:
            raise ExtractError("attempt is missing its registered " + key)
    backend_for(attempt["backend"])
    if attempt["role"] not in {"worker", "reviewer"}:
        raise ExtractError("unknown agent role")
    if any(not isinstance(attempt[key], str) or not attempt[key] for key in ("model", "effort", "stage", "node")):
        raise ExtractError("agent identity fields must be nonempty strings")
    exact_uuid(attempt["run_id"])
    exact_uuid(attempt["attempt_id"])
    if attempt["session_id"] is not None:
        exact_uuid(attempt["session_id"])
    scope = attempt["session_scope"]
    if (not isinstance(scope, dict) or set(scope) != {"stage", "node", "role", "operation"} or
            any(scope[key] != attempt[key] for key in ("stage", "node", "role")) or
            not isinstance(scope["operation"], str) or not scope["operation"]):
        raise ExtractError("session_scope must bind this stage, node, role and operation")
    return {key: attempt[key] for key in ("run_id", "backend", "model", "effort", "session_scope")}


@contextmanager
def _session(attempt):
    binding = _binding(attempt)
    directory = Path(attempt["session_directory"])
    if not directory.is_absolute() or directory.is_relative_to(Path(attempt["directory"])):
        raise ExtractError("session_directory must be stable and outside the attempt")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    real_directory(directory)
    descriptor = os.open(directory / "session.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExtractError("this role session is already owned by another invocation") from exc
        path = directory / "session.json"
        record = read_json(path) if path.exists() else {"format": "schema_extract.session.v1", "binding": binding,
                 "status": "empty", "session_id": None, "retired_sessions": []}
        if (record.get("format") != "schema_extract.session.v1" or
                record.get("status") not in {"empty", "ready", "running", "retired", "fenced"} or
                "session_id" not in record or not isinstance(record.get("retired_sessions"), list)):
            raise ExtractError("registered session record is incomplete")
        if record.get("session_id") is not None:
            exact_uuid(record["session_id"])
        if record.get("binding") != binding:
            raise ExtractError("registered backend/model/effort/session scope differs")
        yield path, record
    finally:
        os.close(descriptor)


def retire_session(attempt: dict, *, termination: dict, reason: str):
    """Recovery may retire a session only after proving the invocation has ended."""
    if termination.get("confirmed") is not True:
        raise ExtractError("cannot retire a role session while termination is unconfirmed")
    with _session(attempt) as (path, record):
        if record.get("attempt_id") not in {None, attempt["attempt_id"]}:
            raise ExtractError("cannot retire a session owned by another attempt")
        _retire(record, reason)
        write_json(path, record)


def _retire(record, reason):
    if record.get("session_id") and record["session_id"] not in record["retired_sessions"]:
        record["retired_sessions"].append(record["session_id"])
    record.update(status="retired", session_id=None, retired_reason=reason)


def _settle_session(record, result, *, not_started):
    if result["termination"].get("confirmed") is not True:
        record["status"] = "fenced"
    elif result["outcome"] in {"completed", "receipt_error"}:
        record.update(status="ready", session_id=result["session_id"])
    elif not_started:
        record["status"] = "ready" if record.get("session_id") else "empty"
    else:
        for identity in result["trace"]["session_ids"]:
            if identity not in record["retired_sessions"]:
                record["retired_sessions"].append(identity)
        _retire(record, result["outcome"])


def settle_session(attempt: dict, *, result: dict):
    """Finish a recovered, already verified result without guessing a CLI session."""
    if result.get("attempt_sha256") != digest(attempt):
        raise ExtractError("recovered result does not bind this attempt")
    with _session(attempt) as (path, record):
        if record.get("attempt_id") != attempt["attempt_id"]:
            raise ExtractError("another invocation owns this session settlement")
        _settle_session(record, result, not_started=processes.launch_not_started(attempt))
        write_json(path, record)
        return {"status": record["status"], "session_id": record.get("session_id")}


def restore_unstarted_session(attempt: dict):
    """Release an interrupted gate's session only with positive no-exec proof."""
    if not processes.launch_not_started(attempt):
        raise ExtractError("cannot preserve a role session without confirmed no-exec evidence")
    with _session(attempt) as (path, record):
        if record.get("session_id") != attempt["session_id"]:
            raise ExtractError("unstarted invocation differs from its registered role session")
        if record["status"] in {"empty", "ready"}:
            return
        if record.get("attempt_id") != attempt["attempt_id"]:
            raise ExtractError("another invocation owns this role session")
        record["status"] = "ready" if record.get("session_id") else "empty"
        write_json(path, record)


def _schema_check(schema, response, attempt_id):
    def local_refs(value):
        if isinstance(value, dict):
            if "$ref" in value and not str(value["$ref"]).startswith("#"):
                raise ExtractError("response schemas cannot resolve external references")
            for child in value.values():
                local_refs(child)
        elif isinstance(value, list):
            for child in value:
                local_refs(child)
    local_refs(schema)
    validator = validators.validator_for(schema)
    validator.check_schema(schema)
    errors = list(validator(schema).iter_errors(response))
    return {"schema_valid": not errors, "nonce_valid": isinstance(response, dict) and response.get("attempt_id") == attempt_id,
            "schema_errors": [{"path": list(error.absolute_path), "validator": error.validator} for error in errors[:12]]}


def read_result(attempt: dict, *, exit_code: int | None, termination: dict, outcome="completed") -> dict:
    """Normal execution and recovery use this same parser; no exit status is guessed."""
    _binding(attempt)
    directory = Path(attempt["directory"])
    invocation = read_json(directory / "invocation.json")
    if (invocation.get("attempt_sha256") != digest(attempt) or
            invocation.get("cwd") != str(Path(attempt["session_directory"]) / "cwd") or
            invocation.get("workspace") != attempt["workspace"] or
            any(invocation.get(key) != attempt[key] for key in ("backend", "model", "effort", "run_id", "attempt_id", "session_scope"))):
        raise ExtractError("recorded invocation differs from its registered attempt")
    schema = read_json(directory / "response.schema.json")
    if digest(schema) != invocation["response_schema_digest"]:
        raise ExtractError("response schema changed after invocation registration")
    parsed = parse_events(directory / "events.jsonl", backend=attempt["backend"])
    response = parsed["response"]
    checked = _schema_check(schema, response, attempt["attempt_id"])
    output_match = True
    if attempt["backend"] == "codex":
        try:
            path = directory / "cli-tmp/response.json"
            with _open_file(path) as stream:
                output_match = strict_json(stream.read()) == response
        except (OSError, ValueError):
            output_match = False
    if (termination.get("confirmed") is True and
            any(os.path.lexists(directory / name) for name in
                ("launch.json", "process.json", "exit.json", "cli-exit.json", "events.jsonl")) and
            not processes.gate_reaped(attempt) and not processes.launch_not_started(attempt)):
        termination = {**termination, "confirmed": False,
                       "reason": "invocation anchor exited without confirmed descendant closure"}
    if termination.get("confirmed") is not True:
        outcome = "termination_unconfirmed"
    elif outcome in {"completed", "receipt_error"}:
        # A proposal can be repaired in this session only while the native
        # execution itself still satisfies every success gate on reread.
        if (exit_code != 0 or not parsed["native_status"]["success"] or
                invocation.get("cwd_before") != [] or invocation.get("cwd_after") != []):
            outcome = "cli_error"
        elif attempt["session_id"] is not None and parsed["session_id"] != attempt["session_id"]:
            outcome = "session_mismatch"
        elif not output_match or not checked["schema_valid"] or not checked["nonce_valid"]:
            outcome = "receipt_error"
    return {**invocation, "outcome": outcome, "exit_code": exit_code, "termination": termination,
            "session_id": parsed["session_id"], "response": response, "native_status": parsed["native_status"],
            "requested_model": attempt["model"], "requested_effort": attempt["effort"],
            **{key: parsed[key] for key in ("reported_model", "reported_models", "reported_effort", "cli_reported_model")},
            "response_validation": {**checked, "native_output_matches": output_match},
            "trace": parsed["trace"], "event_types": list(parsed["trace"]["event_counts"]),
            "parse_error_lines": parsed["trace"]["parse_error_lines"],
            "usage": result_usage(parsed, outcome=outcome, termination=termination),
            "raw_logs": {"events": str(directory / "events.jsonl"), "stderr": str(directory / "stderr.txt")}}


class AgentCLI:
    def __init__(self, *, backend: str, executable: str, model: str):
        self.backend = backend_for(backend)
        self.executable = str(Path(shutil.which(executable) or executable).resolve(strict=True))
        if not isinstance(model, str) or not model:
            raise ExtractError("model must be explicitly registered")
        self.model = model

    def run(self, *, attempt: dict, prompt: str, schema: dict, timeout: float,
            on_process=lambda _: None, cancel_event=None) -> dict:
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
            raise ExtractError("invocation timeout must be finite and positive")
        if attempt.get("backend") != self.backend.name or attempt.get("model") != self.model:
            raise ExtractError("CLI backend/model differs from the registered attempt")
        if attempt.get("executable") is not None and str(Path(attempt["executable"]).resolve()) != self.executable:
            raise ExtractError("CLI executable differs from the registered attempt")
        with _session(attempt) as (session_path, session):
            if session["status"] in {"running", "fenced"}:
                raise ExtractError("the prior role invocation requires termination recovery")
            if attempt["session_id"] != session.get("session_id"):
                raise ExtractError("resume requires this exact registered role session UUID")
            return self._run(attempt, prompt, schema, timeout, on_process, cancel_event, session_path, session)

    def _run(self, attempt, prompt, schema, timeout, on_process, cancel_event, session_path, session):
        directory = Path(attempt["directory"])
        home = session_path.parent / "home"
        home.mkdir(mode=0o700, exist_ok=True)
        real_directory(home)
        schema_path, prompt_path = directory / "response.schema.json", directory / "prompt.txt"
        _schema_check(schema, {}, attempt["attempt_id"])
        write_json(schema_path, schema)
        prompt_path.write_text(prompt, encoding="utf-8")
        invocation = {key: attempt.get(key) for key in ("run_id", "attempt_id", "stage", "node", "role", "session_group",
                      "session_scope", "session_directory", "backend", "model", "effort")}
        invocation.update(resume_session_id=attempt["session_id"], executable=self.executable,
                          requested_model=self.model, requested_effort=attempt["effort"], started_at=time.time(), cli_version=None,
                          delegation_policy="forbidden", prompt_sha256=sha256(prompt.encode()), response_schema_digest=digest(schema),
                          launch_id=str(uuid4()), attempt_sha256=digest(attempt), cwd=str(session_path.parent / "cwd"), workspace=attempt["workspace"])
        started = time.monotonic()
        termination = {"confirmed": True, "actions": [], "survivors": [], "reason": "not_started"}
        outcome, exit_code, process, usage_start, observed_seconds = "completed", None, None, None, 0.0
        release_attempted, known, identity, threads, pipe_errors = False, [], None, [], []
        session_started = False
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("cancel requested before invocation launch")
            workspace = real_directory(Path(attempt["workspace"]))
            real_directory(directory)
            artifact_inventory(workspace)
            writable = attempt["writable_files"]
            if (not isinstance(writable, list) or any(not isinstance(name, str) for name in writable) or
                    len(set(writable)) != len(writable) or (attempt["role"] == "reviewer" and writable)):
                raise ExtractError("role writable_files must be a distinct list, empty for reviewer")
            for name in writable:
                if Path(name).name != name or name.startswith("."):
                    raise ExtractError("writable artifact names must be directly in the workspace")
                regular_file(workspace, name)
            cwd = session_path.parent / "cwd"
            cwd.mkdir(mode=0o700, exist_ok=True)
            real_directory(cwd)
            invocation["cwd_before"] = sorted(path.name for path in cwd.iterdir())
            if invocation["cwd_before"]:
                raise ExtractError("the stable session cwd must remain empty; artifacts belong to the attempt workspace")
            temporary = directory / "cli-tmp"
            temporary.mkdir(mode=0o700)
            environment, fingerprint = self.backend.environment(home, attempt["effort"])
            environment["TMPDIR"] = str(temporary)
            if session.get("configuration_digest") not in {None, fingerprint}:
                raise ExtractError("role provider connection configuration changed")
            invocation["configuration_digest"] = fingerprint
            invocation["cli_version"] = subprocess.check_output([self.executable, "--version"], text=True,
                                    cwd=directory, env=environment, stderr=subprocess.PIPE,
                                    timeout=min(15, timeout)).strip()
            if attempt.get("cli_version") is not None and attempt["cli_version"] != invocation["cli_version"]:
                raise ExtractError("CLI version differs from the registered attempt")
            if (session.get("executable") not in {None, self.executable} or
                    session.get("cli_version") not in {None, invocation["cli_version"]}):
                raise ExtractError("role CLI executable or version changed")
            command = self.backend.command(self.executable, model=self.model, effort=attempt["effort"], role=attempt["role"],
                          session_id=attempt["session_id"], schema=schema, cwd=cwd, workspace=workspace,
                          schema_path=schema_path, output_path=temporary / "response.json", writable_files=writable,
                          read_directories=attempt["read_directories"],
                          check=Path(attempt["check_command"]) if attempt.get("check_command") else None)
            invocation["argv"] = command
            write_json(directory / "invocation.json", invocation)
            if self.backend.name == "codex":
                observed = time.monotonic()
                try:
                    usage_start = begin_usage_observation(attempt, invocation, home=home / ".codex")
                except Exception as exc:
                    invocation["usage_observation_error"] = "baseline_" + type(exc).__name__
                observed_seconds += time.monotonic() - observed
            creator = process_table(cwd=directory).get(os.getpid())
            if creator is None:
                raise OSError("cannot establish launch creator identity")
            launch = {"protocol": GATE_PROTOCOL, "status": "gated", "at": time.time(), "creator": creator,
                      **{key: invocation[key] for key in ("run_id", "attempt_id", "launch_id", "attempt_sha256")}}
            write_json(directory / "launch.json", launch)
            session.update(status="running", attempt_id=attempt["attempt_id"], configuration_digest=fingerprint,
                           executable=self.executable, cli_version=invocation["cli_version"])
            write_json(session_path, session)
            session_started = True
            start_read, start_write = os.pipe()

            def record():
                nonlocal known
                known = list({identity_key(item): item for item in [*known, *owned_processes(identity, known, cwd=directory)]}.values())
                value = {"identity": identity, "descendants": known, "launch_id": launch["launch_id"]}
                write_json(directory / "process.json", value)
                on_process(value)

            def drain(pipe, name):
                try:
                    with (directory / name).open("wb") as output, pipe:
                        shutil.copyfileobj(pipe, output)
                        output.flush()
                        os.fsync(output.fileno())
                except Exception as exc:
                    pipe_errors.append(type(exc).__name__)

            try:
                wrapper = [sys.executable, "-B", str(Path(processes.__file__).resolve()), "--launch-gate",
                           str(start_read), str(directory / "launch.json"), launch["launch_id"], launch["attempt_id"],
                           launch["attempt_sha256"], str(cwd), *command]
                with prompt_path.open("rb") as stdin:
                    process = subprocess.Popen(wrapper, cwd=directory, env=environment, stdin=stdin,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                              pass_fds=(start_read,), bufsize=0)
                termination = {"confirmed": False, "actions": [], "survivors": [], "reason": "process_started"}
                for pipe, name in ((process.stdout, "events.jsonl"), (process.stderr, "stderr.txt")):
                    thread = threading.Thread(target=drain, args=(pipe, name), daemon=True)
                    thread.start()
                    threads.append(thread)
                identity = process_table(cwd=directory).get(process.pid)
                if identity is None:
                    raise OSError("cannot establish gated invocation identity")
                record()
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("cancel requested before invocation release")
                if time.monotonic() - started >= timeout:
                    outcome = "timeout"
                else:
                    write_json(directory / "launch.json", {**launch, "status": "released", "identity": identity,
                                                           "released_at": time.time()})
                    release_attempted = True
                    os.write(start_write, b"1")
                next_scan = 0
                while outcome == "completed" and not child_exited(process):
                    if (directory / "cli-exit.json").exists():
                        cli_exit = read_json(directory / "cli-exit.json")
                        if any(cli_exit.get(key) != launch[key] for key in ("protocol", "launch_id", "attempt_id", "attempt_sha256")):
                            raise ExtractError("CLI exit differs from the registered launch gate")
                        break  # Main CLI exited; reap/terminate any remaining descendants below.
                    if cancel_event is not None and cancel_event.is_set():
                        outcome = "cancelled"
                        break
                    if time.monotonic() - started >= timeout:
                        outcome = "timeout"
                        break
                    if time.monotonic() >= next_scan:
                        record()
                        next_scan = time.monotonic() + 0.5
                    time.sleep(0.05)
            finally:
                os.close(start_read)
                os.close(start_write)
                if not release_attempted:
                    mark_not_started(directory / "launch.json", launch, "parent_aborted" if process else "spawn_failed")
                if process is not None:
                    try:
                        termination = terminate_owned(identity or {"pid": process.pid, "started": "unobserved"}, known, cwd=directory)
                    except (OSError, subprocess.SubprocessError) as exc:
                        termination = {"confirmed": False, "actions": [], "survivors": [], "reason": type(exc).__name__}
                    try:
                        exit_code = process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        termination["confirmed"] = False
                    if (termination["confirmed"] and release_attempted and not processes.gate_reaped(attempt) and
                            not processes.launch_not_started(attempt)):
                        termination.update(confirmed=False, reason="invocation anchor exited without confirmed descendant closure")
                    if not release_attempted and exit_code is not None:
                        termination.update(confirmed=True, survivors=[], reason="gate never released; wrapper reaped")
                for thread in threads:
                    thread.join(timeout=3)
                if any(thread.is_alive() for thread in threads):
                    termination["confirmed"] = False
                if process is not None:
                    write_json(directory / "exit.json", {"exit_code": exit_code, "termination": termination,
                               "launch_id": launch["launch_id"], "attempt_sha256": digest(attempt)})
        except InterruptedError as exc:
            outcome = "cancelled"
            invocation["error"] = str(exc)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            outcome = "cli_error"
            invocation["error"] = str(exc)
        if "cwd_before" in invocation:
            try:
                invocation["cwd_after"] = sorted(path.name for path in real_directory(Path(invocation["cwd"])).iterdir())
                if invocation["cwd_after"]:
                    outcome = "cli_error"
                    invocation["error"] = "stable session cwd contains unexpected files"
            except (OSError, ValueError) as exc:
                outcome = "cli_error"
                invocation["error"] = str(exc)
        if pipe_errors:
            outcome = "cli_error"
            invocation["log_error"] = pipe_errors
        write_json(directory / "invocation.json", invocation)
        result = read_result(attempt, exit_code=exit_code, termination=termination, outcome=outcome)
        result.update(elapsed_seconds=time.monotonic() - started, completed_at=time.time(),
                      timing_source="monotonic CLI setup/transport/termination/parsing; Runner processing is separate")
        if isinstance(result["response"], dict):
            write_json(directory / "response.json", result["response"])
        observed = time.monotonic()
        if usage_start is not None:
            try:
                parsed = {"trace": result["trace"], "usage": result["usage"]}
                observation = finish_usage_observation(usage_start, parsed, result)
                write_json(directory / "usage-observation.json", observation)
                result["usage_observation_sha256"] = digest(observation)
            except Exception as exc:
                result["usage_observation_error"] = type(exc).__name__
        result["usage_observation_seconds"] = observed_seconds + time.monotonic() - observed
        result.update(elapsed_seconds=time.monotonic() - started, completed_at=time.time())
        if session_started:
            _settle_session(session, result, not_started=processes.launch_not_started(attempt))
        result["session_status"] = session["status"]
        write_json(directory / "result.json", result)
        if session_started:
            write_json(session_path, session)
        return result
