"""Recorded usage and explicit, read-only evidence inspection and packaging."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import tempfile
import time
from uuid import UUID, uuid4

from .common import ExtractError, digest, json_bytes, manifest, relative_name, source_root
from .observation import FORMAT as OBSERVATION_FORMAT, interval_observation, observation_summary, observation_totals


TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")
COUNTER_SCOPE = "provider_counter_semantics_unverified"


def _uuid(value, label):
    try:
        if str(UUID(value)) == value:
            return value
    except (ValueError, TypeError, AttributeError):
        pass
    raise ExtractError(label + " must be an exact UUID")


@contextmanager
def _parent_fd(path: Path):
    path = path.absolute()
    parent = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in path.parts[1:-1]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        yield parent
    finally:
        os.close(parent)


@contextmanager
def _open_file(path: Path):
    """Open through directory descriptors; no parent or final symlink is followed."""
    with _parent_fd(path) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ExtractError("audit evidence is not a regular file: " + str(path))
            yield stream


def _stamp(stream):
    info = os.fstat(stream.fileno())
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def parse_events(path: Path, *, sink=None) -> dict:
    """One streaming pass; an unterminated final line is evidence, not a consumed event.

    No live-log consumer needs a cursor. Byte/line anchors are retained for an
    explicit later audit, not for repeated monitoring or session-history scans.
    """
    trace = {"format": "schema_extract.trace.v1", "path": "events.jsonl", "sha256": None,
             "bytes": 0, "complete_bytes": 0, "line_count": 0, "event_counts": {},
             "parse_error_lines": [], "parse_error_count": 0, "truncated_line": None,
             "session_ids": [], "stable": False, "error": None}
    reports, counts, checksum = [], Counter(), hashlib.sha256()
    try:
        with _open_file(path) as stream:
            before = _stamp(stream)
            for number, line in enumerate(stream, 1):
                offset = trace["bytes"]
                checksum.update(line)
                if sink:
                    sink(line)
                trace["bytes"] += len(line)
                if not line.endswith(b"\n"):
                    trace["truncated_line"] = number
                    break
                trace["complete_bytes"] = trace["bytes"]
                trace["line_count"] = number
                try:
                    event = json.loads(line)
                    kind = event["type"]
                    if not isinstance(kind, str):
                        raise ValueError("event type is not a string")
                    counts[kind] += 1
                    if kind == "thread.started":
                        session_id = _uuid(event["thread_id"], "event thread")
                        if session_id not in trace["session_ids"]:
                            trace["session_ids"].append(session_id)
                    if kind == "turn.completed":
                        raw = event.get("usage")
                        tokens = {field: raw.get(field) if isinstance(raw, dict) and
                                  type(raw.get(field)) is int and raw[field] >= 0 else None
                                  for field in TOKEN_FIELDS}
                        reports.append({"line": number, "offset": offset, "end_offset": trace["bytes"],
                                        "event_sha256": hashlib.sha256(line).hexdigest(),
                                        "raw": raw, "tokens": tokens})
                except (ValueError, KeyError, TypeError, AttributeError):
                    trace["parse_error_count"] += 1
                    if len(trace["parse_error_lines"]) < 32:
                        trace["parse_error_lines"].append(number)
            trace["stable"] = before == _stamp(stream)
            trace["sha256"] = checksum.hexdigest()
    except (ExtractError, OSError) as exc:
        trace["error"] = str(exc)
    trace["event_counts"] = dict(counts)
    issues = []
    if trace["error"] or not trace["stable"]:
        issues.append("missing_or_changing_trace")
    if trace["truncated_line"]:
        issues.append("unterminated_final_line")
    if trace["parse_error_count"]:
        issues.append("malformed_events")
    if not reports or any(item["raw"] is None for item in reports):
        issues.append("missing_usage")
    if any(any(item["tokens"][field] is None for field in TOKEN_FIELDS) for item in reports):
        issues.append("missing_or_invalid_token_fields")
    if any(item["tokens"]["input_tokens"] is not None and item["tokens"]["cached_input_tokens"] is not None and
           item["tokens"]["cached_input_tokens"] > item["tokens"]["input_tokens"] for item in reports):
        issues.append("cached_input_exceeds_input")
    reported = any(item["raw"] is not None for item in reports)
    usage = {"format": "schema_extract.usage.v1", "status": "partial" if reported and issues else
             "reported" if reported else "unknown", "source": "events.jsonl:turn.completed.usage",
             "counter_scope": COUNTER_SCOPE, "cached_input_is_input_subset": True,
             "additive_tokens": {field: None for field in TOKEN_FIELDS},
             "reports": reports, "issues": issues}
    return {"trace": trace, "usage": usage}


def result_usage(parsed: dict, *, outcome: str, termination: dict) -> dict:
    value = deepcopy(parsed["usage"])
    if outcome != "completed" or not termination.get("confirmed"):
        value["issues"].append("invocation_not_successfully_closed")
        if value["status"] == "reported":
            value["status"] = "partial"
    return value


def usage_summary(usage: dict | None) -> dict:
    """A small ledger entry; raw provider objects remain in result/events only."""
    usage = usage or {}
    reports = usage.get("reports", [])
    return {"status": usage.get("status", "unknown"),
            "counter_scope": usage.get("counter_scope", COUNTER_SCOPE),
            "source": usage.get("source", "usage_not_recorded"),
            "report_count": len(reports),
            "last_reported_tokens": deepcopy(reports[-1]["tokens"]) if reports else {field: None for field in TOKEN_FIELDS},
            "additive_tokens": {field: None for field in TOKEN_FIELDS},
            "issues": list(usage.get("issues", ["usage_not_recorded"]))}


def _json(path):
    try:
        with _open_file(path) as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise ExtractError(f"cannot read audit record {path}: {exc}") from exc


def _state(store, run_id, files=None):
    from .runtime import STATE_FORMAT as RUNTIME_STATE_FORMAT
    from .revisions import STATE_FORMAT as SEMANTIC_STATE_FORMAT
    path = store.run_dir(_uuid(run_id, "run")) / "state.json"
    value = files.file(path, as_json=True) if files else _json(path)
    if (not isinstance(value, dict) or value.get("format") not in {"schema_extract.run.v2", RUNTIME_STATE_FORMAT, SEMANTIC_STATE_FORMAT} or
            value.get("run_id") != run_id or "active_attempts" not in value or "reopen_requests" not in value):
        raise ExtractError("audit requires a matching v2 run record; older fields remain unavailable")
    return value


def _sum_time(rows, field):
    values = [row.get("elapsed_seconds") if field == "elapsed_seconds" else row.get("timing", {}).get(field)
              for row in rows]
    known = [value for value in values if type(value) in {int, float} and value >= 0]
    return {"seconds": sum(known) if known else None, "known_attempts": len(known),
            "unknown_attempts": len(values) - len(known)}


def _totals(rows):
    return {"attempt_count": len(rows), "usage_status": dict(Counter(row["usage"]["status"] for row in rows)),
            "additive_tokens": {field: None for field in TOKEN_FIELDS},
            "root_usage": observation_totals(rows),
            "accounted_seconds": _sum_time(rows, "elapsed_seconds"),
            "cli_seconds": _sum_time(rows, "cli_seconds"),
            "processing_seconds": _sum_time(rows, "processing_seconds")}


def _checks(ledger):
    clock = ledger.get("clock", {})
    # Shared checks have no per-operator time allocation. Keep the bounded
    # original anchors, including message IDs that can arrive after the scan.
    records = {}
    for item in [*clock.get("history", []), clock.get("last_reminder", {}), clock.get("reminder", {})]:
        if isinstance(item, dict) and item.get("scan"):
            check = deepcopy(item["scan"])
            check["ack"] = item.get("ack")
            records[check["id"]] = check
    if clock.get("last_scan"):
        check = deepcopy(clock["last_scan"])
        records[check["id"]] = {**records.get(check["id"], {}), **check}
    reminders = {}
    for item in [*clock.get("history", []), clock.get("last_reminder", {}), clock.get("reminder", {}), clock.get("probe", {})]:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        reminders[item["id"]] = {key: deepcopy(item.get(key)) for key in ("id", "due_at", "ack")}
        reminders[item["id"]]["attempts"] = [{key: attempt.get(key) for key in ("started_at", "finished_at", "status")}
                                              for attempt in item.get("attempts", [])]
    return {"allocation": "shared_state_root_checks_not_charged_to_this_run",
            "retention": "bounded_history_not_all_checks", "records": list(records.values()),
            "reminder_timing": list(reminders.values()),
            "latest_observation": {key: clock.get("last_observation", {}).get(key) for key in
                                   ("id", "due_at", "started_at", "completed_at", "recovery_completed_at")},
            "latency_scope": "recorded due/queue/ack boundaries; client-side cause remains unknown",
            "known_seconds": sum(item.get("elapsed_seconds", 0) for item in records.values()) if records else None}


def _waits(state, ledger, now):
    """Read S3 lifecycle anchors; scopes may wait while sibling calls work."""
    run_id, control = state["run_id"], state.get("control", {})
    groups = {key: [] for key in ("pause", "hold", "start_queue", "retry_backoff", "recovery_queue")}
    sources = []
    for request_id, request in control.get("requests", {}).items():
        if request.get("action") != "pause":
            continue
        start, end = request.get("stopped_at"), request.get("resumed_at")
        current = request_id == control.get("last_request_id")
        if start is None and current:
            start = control.get("stopped_at")
        active = current and state.get("status") == "paused" and control.get("action") == "pause"
        kind = "hold" if request.get("hold_id") else "pause"
        if kind == "hold" and end is None:
            end = ledger.get("holds", {}).get(request["hold_id"], {}).get("released_at")
        groups[kind].append((start, end, active))
        sources.append({"source": "control.requests", "id": request_id, "kind": kind,
                        "requested_at": request.get("created_at"), "started_at": start, "ended_at": end})
    launch = ledger.get("starts", {}).get(run_id, {}).get("launch", {})
    if state.get("status") == "queued" or launch.get("mode") == "queued":
        groups["start_queue"].append((state.get("created_at"), launch.get("created_at"), state.get("status") == "queued"))
        sources.append({"source": "supervisor.starts.launch", "id": launch.get("id"), "kind": "start_queue",
                        "started_at": state.get("created_at"), "ended_at": launch.get("created_at"),
                        "scope": "initial queue to launch reservation; launch/initialization time excluded"})
    for identity, request in ledger.get("retries", {}).items():
        if request.get("run_id") != run_id:
            continue
        consumed, active = request.get("consumed_at"), request.get("status") == "authorized"
        due = request.get("next_due_at")
        stop = consumed or request.get("completed_at") or (now if active else None)
        end = min(due, stop) if type(due) in {int, float} and type(stop) in {int, float} else None
        if type(due) in {int, float} and due > 0:
            groups["retry_backoff"].append((request.get("failure_observed_at"), end, active))
        groups["recovery_queue"].append((request.get("created_at"), consumed, active))
        sources.append({"source": "supervisor.retries", "id": identity,
                        **{key: request.get(key) for key in ("scope", "failure_observed_at", "created_at", "next_due_at", "consumed_at")}})
    return {"by_kind": {key: interval_observation(value, now=now) for key, value in groups.items()},
            "known_union": interval_observation([interval for values in groups.values() for interval in values], now=now),
            "sources": sources, "retention": "partial lifecycle history; missing boundaries remain unknown",
            "scope": "union of observed scope waits, possibly concurrent with sibling CLI work; do not subtract from wall time"}


def _rows(store, state):
    origins = state.get("continuation", {}).get("attempt_origins", {})
    usage = state.get("attempt_usage", {})
    active = {*state.get("active_attempts", {}).values(), *state.get("unsettled_attempts", {}),
              *state.get("unconfirmed_attempts", {})}
    rows = {}
    for attempt_id in dict.fromkeys([*usage, *state.get("attempts", [])]):
        _uuid(attempt_id, "attempt")
        entry = deepcopy(usage.get(attempt_id, {}))
        entry.pop("handoffs", None)  # Session progress belongs to its origin state.
        origin = entry.get("origin_run_id") or origins.get(attempt_id) or state["run_id"]
        _uuid(origin, "origin run")
        entry.update(attempt_id=attempt_id, origin_run_id=origin, inherited=origin != state["run_id"],
                     active=origin == state["run_id"] and attempt_id in active,
                     directory=str(store.run_dir(origin) / "attempts" / attempt_id))
        entry.setdefault("usage", usage_summary(None))
        entry.setdefault("timing", {})
        rows[origin, attempt_id] = entry
    return sorted(rows.values(), key=lambda row: (row.get("created_at", float("inf")), row["origin_run_id"], row["attempt_id"]))


class _Evidence:
    """Read selected source files once; spool only when a package is requested."""
    def __init__(self, root, portable):
        self.root, self.portable = root, portable
        self.entries, self.cache, self.objects = {}, {}, {}

    def close(self):
        for stream in self.objects.values():
            stream.close()

    def projection(self, name, value):
        data = json_bytes(value)
        checksum = hashlib.sha256(data).hexdigest()
        self.entries[name] = {"name": name, "sha256": checksum, "bytes": len(data), "source": "recorded observation projection"}
        if self.portable and checksum not in self.objects:
            stream = tempfile.TemporaryFile()
            stream.write(data)
            self.objects[checksum] = stream

    def file(self, path, *, as_json=False, events=False, sha256=None, json_digest=None):
        path = path.absolute()
        if not path.is_relative_to(self.root):
            raise ExtractError("evidence leaves this runtime store: " + str(path))
        name = path.relative_to(self.root).as_posix()
        relative_name(name)
        # Runtime/revision evidence may first register a JSON file by its raw
        # hash. A later checkpoint/receipt read still needs the parsed value.
        if name not in self.cache or ((as_json or events) and self.cache[name] is None):
            spool = tempfile.TemporaryFile() if self.portable else None
            try:
                if events:
                    value = parse_events(path, sink=spool.write if spool else None)
                    trace = value["trace"]
                    if trace["error"] or not trace["stable"]:
                        raise ExtractError("missing or changing events: " + str(path))
                    checksum, size = trace["sha256"], trace["bytes"]
                else:
                    checksum, size, chunks = hashlib.sha256(), 0, []
                    with _open_file(path) as stream:
                        before = _stamp(stream)
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            checksum.update(block)
                            size += len(block)
                            if spool:
                                spool.write(block)
                            if as_json:
                                chunks.append(block)
                        if before != _stamp(stream):
                            raise ExtractError("evidence changed while reading: " + str(path))
                    value = json.loads(b"".join(chunks)) if as_json else None
                    checksum = checksum.hexdigest()
                if name in self.entries and self.entries[name]["sha256"] != checksum:
                    raise ExtractError("evidence changed between selected reads: " + str(path))
                self.entries[name] = {"name": name, "sha256": checksum, "bytes": size, "source": str(path)}
                self.cache[name] = value
                if spool:
                    if checksum in self.objects:
                        spool.close()
                    else:
                        self.objects[checksum] = spool
                    spool = None
            finally:
                if spool:
                    spool.close()
        entry, value = self.entries[name], self.cache[name]
        if sha256 is not None and entry["sha256"] != sha256:
            raise ExtractError("evidence SHA-256 mismatch: " + str(path))
        if json_digest is not None and digest(value) != json_digest:
            raise ExtractError("recorded JSON digest mismatch: " + str(path))
        return value


def _receipt_response(result, receipt, attempt, *, lock=None, artifacts=None):
    """Prove the receipt came from this raw response and the bytes it concerned."""
    from .receipts import call_identity, normalize_response
    response = result.get("response")
    response_format = attempt.get("response_format", "schema_extract.receipt.v2")
    if response_format == "schema_extract.proposal.v1":
        if lock is None or artifacts is None:
            raise ExtractError("proposal audit requires its fixed inputs and original artifact checkpoint")
        response = normalize_response(response, lock, attempt, artifacts)
    elif response_format != "schema_extract.receipt.v2":
        raise ExtractError("unknown recorded response format")
    if response != receipt:
        raise ExtractError("accepted receipt differs from the recorded CLI response")
    if receipt.get("identity") != call_identity(attempt):
        raise ExtractError("receipt invocation identity mismatch")


def _digest_id(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ExtractError("invalid " + label + " digest")
    return value


def _inputs(store, state, files):
    from .snapshots import verify_run_inputs
    lock = files.file(store.run_dir(state["run_id"]) / "inputs.lock.json", as_json=True)
    snapshot = store.root / "snapshots" / _digest_id(lock["fingerprint"], "input")
    if Path(lock["snapshot"]).absolute() != snapshot or Path(state["snapshot"]).absolute() != snapshot:
        raise ExtractError("fixed snapshot path leaves the registered runtime snapshot")
    files.file(snapshot / "inputs.lock.json", as_json=True, json_digest=digest(lock))
    private_git = manifest(snapshot / "source.git") if lock["source"].get("ref_kind") == "private_snapshot_commit" else {}
    # Explicit deep audit uses the real gate, including aliases/private Git.
    # Packaging then streams selected bytes; normal observation does neither.
    verify_run_inputs(state, lock)
    for kind in ("source", "rules"):
        for name, checksum in lock[kind]["files"].items():
            relative_name(name)
            files.file(snapshot / kind / name, sha256=_digest_id(checksum, kind))
    for name, checksum in private_git.items():
        files.file(snapshot / "source.git" / name, sha256=checksum)
    return lock


def _checkpoint(store, state, checkpoint_id, files):
    _digest_id(checkpoint_id, "checkpoint")
    directory = store.run_dir(state["run_id"]) / "checkpoints" / checkpoint_id
    recorded = files.file(directory / "checkpoint.json", as_json=True)
    with _parent_fd(directory / "artifacts" / "_selected_tree"):
        artifacts, value = store.verify_checkpoint(state, checkpoint_id)
    if recorded != value:
        raise ExtractError("checkpoint changed during selected verification")
    for name, checksum in value["files"].items():
        relative_name(name)
        files.file(directory / "artifacts" / name, sha256=_digest_id(checksum, "artifact"))
    return value


def _verify_attempt(store, state, row, files, notes):
    directory = Path(row["directory"])
    hashes = row.get("evidence", {})
    attempt = files.file(directory / "attempt.json", as_json=True, json_digest=hashes.get("attempt.json"))
    row["execution_binding"] = attempt.get("execution_binding")
    for key in ("semantic_revision", "maintenance_task_id", "purpose", "maintenance_step"):
        if key in row and row[key] != attempt.get(key):
            raise ExtractError("attempt ledger context differs: " + key)
        row[key] = attempt.get(key)
    for key, expected in (("run_id", row["origin_run_id"]), ("attempt_id", row["attempt_id"]),
                           ("input_fingerprint", state["input_fingerprint"]), ("directory", str(directory)),
                           ("workspace", str(directory / "workspace"))):
        if attempt.get(key) != expected:
            raise ExtractError("recorded attempt identity differs: " + key)
    for key in ("stage", "node", "role", "session_group", "base_checkpoint"):
        if key in row and row[key] != attempt.get(key):
            raise ExtractError("attempt ledger scope differs: " + key)
    if "requested_session_id" in row and row["requested_session_id"] != attempt.get("session_id"):
        raise ExtractError("attempt ledger requested session differs")
    records = {"attempt.json": attempt}
    for name in ("invocation.json", "result.json", "response.schema.json", "receipt.json", "validated.json",
                 "check.json", "merge_check.json", "event.json", "processing.json", "finished.json",
                 "repair.json", "recovery.json", "rejected.json", "stale.json", "handoff.json",
                 "termination.json", "launch.json", "process.json", "usage-observation.json"):
        path = directory / name
        if name in hashes or os.path.lexists(path):
            records[name] = files.file(path, as_json=True, json_digest=hashes.get(name))
    result = records.get("result.json")
    if not result:
        if records.get("finished.json", {}).get("status") == "not_dispatched":
            notes.append({"attempt_id": row["attempt_id"], "reason": "dispatch_denied; no provider usage exists"})
            return attempt, None, None
        raise ExtractError("selected attempt has no result.json")
    for key in ("run_id", "attempt_id", "stage", "node", "role", "session_group"):
        if result.get(key) is not None and result[key] != attempt.get(key):
            raise ExtractError("result invocation identity differs: " + key)
    parsed = files.file(directory / "events.jsonl", events=True)
    if result.get("trace"):
        if parsed["trace"] != result["trace"]:
            raise ExtractError("events differ from the closed trace summary")
    else:
        notes.append({"attempt_id": row["attempt_id"], "reason": "closed_trace_digest_not_recorded"})
    if result.get("usage"):
        expected = result_usage(parsed, outcome=result["outcome"], termination=result.get("termination", {}))
        if result["usage"] != expected or row["usage"] != usage_summary(expected):
            raise ExtractError("usage differs from selected events or attempt ledger")
    else:
        notes.append({"attempt_id": row["attempt_id"], "reason": "usage_not_recorded; inspected events do not rewrite history"})
    if result.get("usage_observation_sha256"):
        observed = records.get("usage-observation.json", {})
        if (observed.get("format") != OBSERVATION_FORMAT or digest(observed) != result["usage_observation_sha256"] or
                observed.get("identity") != {key: result.get(key) for key in ("run_id", "attempt_id", "launch_id", "attempt_sha256")} or
                observed.get("identity", {}).get("attempt_sha256") != digest(attempt) or
                observed.get("session_id") != result.get("session_id") or
                ("delegation" in observed and (observed["delegation"].get("policy") != result.get("delegation_policy", "unknown") or
                 result.get("delegation_policy") != records.get("invocation.json", {}).get("delegation_policy"))) or
                observed.get("stream_reports") != parsed["usage"]["reports"] or
                observed.get("stream_trace_sha256") != parsed["trace"]["sha256"] or
                row.get("usage_observation") != observation_summary(observed)):
            raise ExtractError("usage observation differs from its bound invocation, stream or ledger")
        notes.append({"attempt_id": row["attempt_id"],
                      "delegation_status": observed.get("delegation", {}).get("status", "unavailable"),
                      "reason": "bound native root counters and optional delegation identities; child-session tokens and whole-invocation coverage remain unknown"})
    elif row.get("usage_observation"):
        raise ExtractError("attempt ledger has an unbound usage observation")
    actual, requested = result.get("session_id"), attempt.get("session_id")
    if requested and actual and requested != actual:
        raise ExtractError("actual session differs from exact requested UUID")
    found = parsed["trace"]["session_ids"]
    if actual and (len(found) != 1 or found[0] != _uuid(actual, "actual session")):
        raise ExtractError("result session differs from selected events")
    if row.get("session_id") and row["session_id"] != actual:
        raise ExtractError("attempt ledger session differs from its result")
    files.file(directory / "prompt.txt", sha256=result.get("prompt_sha256"))
    if result.get("response_schema_digest") is not None and result["response_schema_digest"] != (digest(records["response.schema.json"]) if "response.schema.json" in records else None):
        raise ExtractError("response schema differs from the recorded invocation")
    if os.path.lexists(directory / "stderr.txt"):
        files.file(directory / "stderr.txt")
    if result.get("response") is not None:
        response = files.file(directory / "response.json", as_json=True, json_digest=hashes.get("response.json"))
        if response != result["response"]:
            raise ExtractError("result response differs from response.json")
    elif os.path.lexists(directory / "response.json"):
        if "response.json" in hashes:
            response = files.file(directory / "response.json", as_json=True, json_digest=hashes["response.json"])
            if response is not None:
                raise ExtractError("result response differs from response.json")
        else:
            files.file(directory / "response.json")
            notes.append({"attempt_id": row["attempt_id"], "reason": "unbound invalid/null response preserved as bytes"})
    receipt = records.get("receipt.json")
    accepted = state.get("accepted_receipts", {}).get(row["attempt_id"])
    if accepted is not None:
        if receipt is None or digest(receipt) != accepted:
            raise ExtractError("accepted receipt differs from the recorded digest")
        if records.get("validated.json", {}).get("receipt_sha256") != accepted:
            raise ExtractError("validation is not bound to the accepted receipt")
        if attempt.get("semantic_revision"):
            for key in ("semantic_revision", "maintenance_task_id", "purpose", "maintenance_step"):
                if records.get("validated.json", {}).get(key) != attempt.get(key):
                    raise ExtractError("validation context differs from its original attempt: " + key)
    if row.get("receipt_sha256") is not None and row["receipt_sha256"] != accepted:
        raise ExtractError("attempt accounting refers to a different accepted receipt")
    if row.get("elapsed_seconds") is not None and records.get("finished.json", {}).get("elapsed_seconds", 0) > row["elapsed_seconds"]:
        raise ExtractError("attempt time ledger is below its durable finished high-water mark")
    if result.get("elapsed_seconds") is not None and row.get("timing", {}).get("cli_seconds") is not None and row["timing"]["cli_seconds"] != result["elapsed_seconds"]:
        raise ExtractError("CLI timing differs from the recorded result")
    processing = records.get("processing.json", {})
    for disk_key, ledger_key in (("elapsed_seconds", "processing_seconds"), ("estimated_seconds", "processing_estimated_seconds")):
        known = row.get("timing", {}).get(ledger_key)
        if known is not None and processing.get(disk_key, 0) != known:
            raise ExtractError("processing timing differs from its durable high-water mark")
    return attempt, result, receipt


def _verify(store, state, rows, files, *, whole_run):
    from .receipts import _anchor
    from .revisions import resolve_context, verify_revisions
    errors, notes, selected, origins, checkpoints = [], [], [], {state["run_id"]: state}, {}
    def failure(scope, exc):
        errors.append({"scope": scope, "reason": str(exc)})
    for row in rows:
        try:
            origin = row["origin_run_id"]
            if origin not in origins:
                origins[origin] = _state(store, origin, files)
            selected.append((row, *_verify_attempt(store, origins[origin], row, files, notes)))
            for checkpoint_id in {row.get("base_checkpoint"), row.get("result_checkpoint"),
                                  *row.get("checkpoints", []), selected[-1][1]["base_checkpoint"]} - {None}:
                checkpoints[origin, checkpoint_id] = None
        except (ExtractError, OSError, ValueError, KeyError, TypeError) as exc:
            failure(row["attempt_id"], exc)
    if whole_run:
        for checkpoint_id in {state.get("checkpoint"), state.get("merged_checkpoint")} - {None}:
            checkpoints[state["run_id"], checkpoint_id] = None
    locks = {}
    for origin, original in origins.items():
        try:
            locks[origin] = _inputs(store, original, files)
            revision_records = verify_revisions(store, original, evidence=files.file)
            for entry in revision_records:
                snapshot = Path(entry["target_lock"]["snapshot"])
                if entry["target_lock"]["source"].get("ref_kind") == "private_snapshot_commit":
                    for name, checksum in manifest(snapshot / "source.git").items():
                        files.file(snapshot / "source.git" / name, sha256=checksum)
            if original.get("execution_binding") or original.get("execution_bindings") or original.get("format") == "schema_extract.run.v2.runtime.v1":
                from .runtime import verify_execution
                verify_execution(store, original, attempts=[attempt for row, attempt, _, _ in selected
                                                            if row["origin_run_id"] == origin], evidence=files.file)
        except (ExtractError, OSError, ValueError, KeyError, TypeError) as exc:
            failure(origin, exc)
    for (origin, checkpoint_id) in checkpoints:
        try:
            checkpoints[origin, checkpoint_id] = _checkpoint(store, origins[origin], checkpoint_id, files)
        except (ExtractError, OSError, ValueError, KeyError, TypeError) as exc:
            failure(checkpoint_id, exc)
    sessions = {}
    for row, attempt, result, receipt in selected:
        try:
            actual = (result or {}).get("session_id")
            if actual:
                owner = row["origin_run_id"], attempt["session_group"], attempt["role"]
                if actual in sessions and sessions[actual] != owner:
                    raise ExtractError("one session is used by different roles/stages/nodes or continuation runs")
                sessions[actual] = owner
                for handoff in origins[row["origin_run_id"]].get("session_handoffs", []):
                    if (actual == handoff.get("previous_session_id") and attempt.get("created_at", 0) > handoff.get("at", float("inf"))):
                        raise ExtractError("selected attempt reused a retired session")
            context = resolve_context(store, origins[row["origin_run_id"]], attempt)
            row["context"] = context["identity"]
            # This checks raw response provenance against its original source
            # and checkpoint. It does not run current semantic acceptance rules
            # or turn a historical receipt into an effective current approval.
            lock = context["source_lock"]
            if receipt and lock:
                if attempt["attempt_id"] in origins[row["origin_run_id"]].get("accepted_receipts", {}):
                    if attempt.get("response_format") == "schema_extract.proposal.v1":
                        baseline = checkpoints.get((row["origin_run_id"], attempt["base_checkpoint"]))
                        if not baseline or baseline["files"] != attempt["dependency_digests"]:
                            raise ExtractError("proposal baseline differs from the registered dependency bytes")
                        checkpoint_id = (attempt["base_checkpoint"] if attempt["role"] == "reviewer"
                                         else row.get("result_checkpoint"))
                        if not checkpoints.get((row["origin_run_id"], checkpoint_id)):
                            raise ExtractError("proposal has no verified original artifact checkpoint")
                        artifacts = store.run_dir(row["origin_run_id"]) / "checkpoints" / checkpoint_id / "artifacts"
                        _receipt_response(result, receipt, attempt, lock=lock, artifacts=artifacts)
                    else:
                        _receipt_response(result, receipt, attempt)
                for anchor in receipt.get("source_evidence", []):
                    if anchor.get("source_id") != lock["source"]["id"] or lock["source"]["files"].get(anchor["path"]) != anchor["sha256"]:
                        raise ExtractError("source evidence differs from the selected fixed source")
                    _anchor(anchor, source_root(lock), lock["source"]["files"])
                candidates = [value for (origin, _), value in checkpoints.items() if origin == row["origin_run_id"] and value]
                for anchor in receipt.get("artifact_evidence", []):
                    match = next((value for value in candidates if value["files"].get(anchor["path"]) == anchor["sha256"]), None)
                    if match is None:
                        raise ExtractError("artifact evidence has no matching selected immutable checkpoint")
                    _anchor(anchor, store.run_dir(row["origin_run_id"]) / "checkpoints" / match["id"] / "artifacts", match["files"])
        except (ExtractError, KeyError, TypeError) as exc:
            failure(row["attempt_id"], exc)
    # Atomic state replacement during collection makes this an incomplete
    # observation, even if each individually selected payload was readable.
    for origin, original in origins.items():
        try:
            if _state(store, origin) != original:
                failure(origin, "state changed during the selected audit")
        except (ExtractError, OSError) as exc:
            failure(origin, exc)
    active = any(row["active"] for row in rows) or (whole_run and (state["status"] == "running" or store.writer_status(state["run_id"]) == "held"))
    if active:
        failure(state["run_id"], "selected scope is active or unsettled; this is not a complete evidence snapshot")
    chosen = {(row["origin_run_id"], row["attempt_id"]) for row in rows}
    handoffs = [{"origin_run_id": origin, **item} for origin, original in origins.items()
                for item in original.get("session_handoffs", []) if (origin, item["attempt_id"]) in chosen or
                (origin, item.get("next_attempt_id")) in chosen]
    return {"status": "incomplete" if errors else "partial" if notes else "verified", "errors": errors,
            "unavailable": notes, "authority": "evidence_only; no approval, dispatch or publication",
            "session_handoffs": handoffs,
            "checkpoint_ids": [{"origin_run_id": origin, "checkpoint": identity} for origin, identity in sorted(checkpoints)]}


def _archive_manifest(path):
    """Verify a package without extracting paths or following archive links."""
    recorded, objects, seen = None, {}, set()
    with _open_file(path) as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.name in seen:
                raise ExtractError("archive contains a link, special file or duplicate member")
            seen.add(member.name)
            payload = archive.extractfile(member)
            if member.name == "manifest.json":
                if member.size > 16 * 1024 * 1024:
                    raise ExtractError("archive manifest is too large")
                recorded = json.load(payload)
            elif member.name.startswith("objects/"):
                expected = _digest_id(member.name.removeprefix("objects/"), "archive object")
                checksum, size = hashlib.sha256(), 0
                for block in iter(lambda: payload.read(1024 * 1024), b""):
                    checksum.update(block)
                    size += len(block)
                if checksum.hexdigest() != expected:
                    raise ExtractError("archive object checksum mismatch")
                objects[expected] = size
            else:
                raise ExtractError("archive has an unexpected member")
    if (not isinstance(recorded, dict) or recorded.get("format") != "schema_extract.archive.v1" or
            digest({key: value for key, value in recorded.items() if key != "id"}) != recorded.get("id")):
        raise ExtractError("archive manifest identity mismatch")
    required = {}
    names = set()
    for entry in recorded["entries"]:
        name = relative_name(entry["name"])
        if name in names:
            raise ExtractError("archive manifest has duplicate evidence names")
        names.add(name)
        required[entry["sha256"]] = entry["bytes"]
    if required != objects:
        raise ExtractError("archive payloads differ from the manifest")
    return recorded


def _archive(path, files, scope, verification):
    path = path.absolute()
    if ".." in path.parts:
        raise ExtractError("archive destination must not contain '..' path components")
    if path.is_relative_to(files.root):
        raise ExtractError("archive target must be outside the runtime store")
    body = {"format": "schema_extract.archive.v1", "scope": scope,
            "verification_status": verification["status"], "unavailable": verification["unavailable"],
            "checkpoint_ids": verification["checkpoint_ids"],
            "entries": sorted(files.entries.values(), key=lambda item: item["name"]),
            "objects": "objects/<sha256>; identical selected bytes are stored once",
            "authority": "selected recorded evidence; does not confer extraction approval",
            "token_scope": COUNTER_SCOPE}
    manifest = {**body, "id": digest(body)}
    reused = False
    # The destination's parent must exist. Descriptor-relative creation and
    # link publication keep both symlinks and overwrite races out of this path.
    with _parent_fd(path) as parent:
        try:
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            info = None
        if info is not None:
            if not stat.S_ISREG(info.st_mode) or _archive_manifest(path)["id"] != manifest["id"]:
                raise ExtractError("archive target already contains different evidence; preserved: " + str(path))
            reused = True
        else:
            temporary = ".audit-" + uuid4().hex + ".tar.gz"
            try:
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
                with os.fdopen(fd, "wb") as stream:
                    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                        entries = [("manifest.json", io.BytesIO(json_bytes(manifest)))]
                        entries.extend(("objects/" + checksum, source) for checksum, source in sorted(files.objects.items()))
                        for name, source in entries:
                            source.seek(0, os.SEEK_END)
                            info = tarfile.TarInfo(name)
                            info.size, info.mode = source.tell(), 0o444
                            source.seek(0)
                            archive.addfile(info, source)
                    stream.flush()
                    os.fsync(stream.fileno())
                if _archive_manifest(path.parent / temporary)["id"] != manifest["id"]:
                    raise ExtractError("new archive failed manifest verification")
                try:
                    os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                except FileExistsError:
                    if _archive_manifest(path)["id"] != manifest["id"]:
                        raise ExtractError("archive target changed before publication; preserved")
                    reused = True
                os.fsync(parent)
            finally:
                os.unlink(temporary, dir_fd=parent)
    return {"status": "archived", "path": str(path), "manifest_sha256": manifest["id"],
            "reused": reused, "object_count": len(files.objects), "entry_count": len(files.entries),
            "verification_status": verification["status"]}


def audit_run(store, run_id, *, attempt_id=None, session_id=None, verify=False, archive=None):
    if attempt_id and session_id:
        raise ExtractError("choose either --attempt or --session")
    if attempt_id:
        _uuid(attempt_id, "attempt")
    if session_id:
        _uuid(session_id, "session")
    files = _Evidence(store.root, archive is not None) if verify or archive is not None else None
    try:
        state = _state(store, run_id, files)
        all_rows = _rows(store, state)
        rows = [row for row in all_rows if (not attempt_id or row["attempt_id"] == attempt_id) and
                (not session_id or session_id in {row.get("session_id"), row.get("requested_session_id")} )]
        unindexed = sum("requested_session_id" not in row and "session_id" not in row for row in all_rows) if session_id else 0
        if not rows and (attempt_id or session_id) and not unindexed:
            raise ExtractError("selected attempt/session is not in this run's recorded ledger")
        supervisor = store.root / "supervisor.json"
        ledger = _json(supervisor) if os.path.lexists(supervisor) else {}
        checks = _checks(ledger)
        scope = {"run_id": run_id, "attempt_id": attempt_id, "session_id": session_id,
                 "usage_scope": "this run's registered CLI attempts, including deduplicated inherited origins; excludes Supervisor and external billing",
                 "attempts": [{"origin_run_id": row["origin_run_id"], "attempt_id": row["attempt_id"]} for row in rows],
                 "unindexed_session_attempts": unindexed,
                 "checkpoint_policy": "selected call input/result checkpoints" if attempt_id or session_id else "call input/result and current/final checkpoints"}
        groups = {}
        for row in rows:
            key = tuple(row.get(field) or "unknown" for field in ("stage", "role", "node"))
            groups.setdefault(key, []).append(row)
        selected_ids = {(row["origin_run_id"], row["attempt_id"]) for row in rows}
        handoffs = [{"origin_run_id": run_id, **item} for item in state.get("session_handoffs", []) if
                    (run_id, item["attempt_id"]) in selected_ids or (run_id, item.get("next_attempt_id")) in selected_ids]
        inherited_handoffs = [{"origin_run_id": origin, "path": str(store.run_dir(origin) / "state.json") + "#session_handoffs",
                               "availability": "not_checked"} for origin in sorted({row["origin_run_id"] for row in rows if row["inherited"]})]
        now = time.time()
        created = state.get("continuation", {}).get("original_created_at", state.get("created_at"))
        commit = ledger.get("commits", {}).get(run_id, {})
        completed = commit.get("at") if state.get("published") and commit.get("checkpoint") == state.get("checkpoint") and commit.get("target") == state.get("output") else None
        waits = _waits(state, ledger, now)
        report = {"format": "schema_extract.audit.v1", "scope": scope, "observed_at": now,
                  "operator_id": state.get("operator_id"), "status": state["status"], "executor": state.get("executor"),
                  "execution_binding": state.get("execution_binding"),
                  "execution_bindings": list(state.get("execution_bindings", [])),
                  "semantic_revision": state.get("semantic_revision"),
                  "semantic_revisions": list(state.get("semantic_revisions", [])),
                  "semantic_context_sources": [{"revision": identity,
                      "record": str(store.run_dir(run_id) / "revisions" / identity / "revision.json"),
                      "plan": str(store.run_dir(run_id) / "revisions" / identity / "plan.json"),
                      "target_lock": str(store.run_dir(run_id) / "revisions" / identity / "target.lock.json")}
                      for identity in state.get("semantic_revisions", [])],
                  "approval_adoptions": deepcopy(state.get("approval_adoptions", {})),
                  "semantic_request": deepcopy(ledger.get("semantic_requests", {}).get(run_id)),
                  "maintenance": deepcopy(state.get("maintenance")),
                  "state": str(store.run_dir(run_id) / "state.json"), "totals": _totals(rows),
                  "segments": {"current": _totals([row for row in rows if not row["inherited"]]),
                               "inherited": _totals([row for row in rows if row["inherited"]])},
                  "by_scope": [{"stage": key[0], "role": key[1], "node": key[2], **_totals(value)} for key, value in sorted(groups.items())],
                  "attempts": rows, "session_handoffs": handoffs, "inherited_handoff_sources": inherited_handoffs,
                  "supervisor_checks": checks,
                  "time": {"operator_created_at": created, "publication_committed_at": completed,
                           "run_created_at": state.get("created_at"),
                           "run_wall_seconds": (completed or now) - state["created_at"] if state.get("created_at") is not None else None,
                           "run_wall_end": "publication" if completed is not None else "observation",
                           "operator_wall_seconds": completed - created if completed is not None and created is not None else None,
                           "creation_to_observation_seconds": now - created if created is not None else None,
                           "wall_includes": "queue, pause and waiting; never a sum of concurrent calls",
                           "budget_policy": "wall_clock_deadline", "waiting_consumes_budget": True,
                           "pause_wait_seconds": None, "model_execution_seconds": None,
                           "observed_waits": waits,
                           "unavailable": "historical missing wait boundaries, child-session usage, and model-only time",
                           "budget_deadline": state.get("deadline"),
                           "budget_extensions": deepcopy(state.get("budget_extensions", [])),
                           "cli_scope": "cumulative monotonic transport including setup, termination, parsing and usage observation; parallel calls overlap",
                           "processing_scope": "Runner checks/processing; estimated crash intervals are separately marked"},
                  "verification": {"status": "not_requested"}, "archive": None}
        if files:
            checked = _verify(store, state, rows, files, whole_run=not (attempt_id or session_id))
            if unindexed:
                checked["status"] = "incomplete"
                checked["errors"].append({"scope": session_id, "reason": "legacy session fields are unavailable; selection may omit calls"})
            report["verification"] = checked
            report["session_handoffs"] = checked["session_handoffs"]
            if archive is not None:
                if checked["errors"]:
                    report["archive"] = {"status": "not_written", "reason": "selected evidence is incomplete; runtime retained"}
                else:
                    files.projection("observations/supervisor-checks.json", checks)
                    files.projection("observations/wait-intervals.json", waits)
                    try:
                        report["archive"] = _archive(Path(archive), files, scope, checked)
                    except (ExtractError, OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
                        report["archive"] = {"status": "not_written", "reason": str(exc), "path": str(archive)}
        return report
    finally:
        if files:
            files.close()


def format_audit(report):
    total, segments = report["totals"], report["segments"]
    def seconds(value):
        return "unknown" if value["seconds"] is None else f"{value['seconds']:.3f}s ({value['unknown_attempts']} unknown calls)"
    lines = [f"{report['scope']['run_id']} {report['status']} attempts={total['attempt_count']} "
             f"current={segments['current']['attempt_count']} inherited={segments['inherited']['attempt_count']}",
             "whole-invocation tokens=unknown; cached input is part of input",
             "usage scope=this run's registered CLI attempts, including deduplicated inherited origins",
             "budget policy=wall_clock_deadline; queue, pause and waiting consume budget",
             f"CLI={seconds(total['cli_seconds'])} processing={seconds(total['processing_seconds'])} "
             f"budget-accounted={seconds(total['accounted_seconds'])}"]
    root = total["root_usage"]
    tokens = root["additive_tokens"]
    lines.append(f"root {root['root_status']} input/cached/output={tokens['input_tokens']}/{tokens['cached_input_tokens']}/{tokens['output_tokens']} "
                 f"({root['exact_attempts']} exact, {root['partial_attempts']} partial, {root['unknown_attempts']} unknown calls; "
                 f"known={root['known_attempts']})")
    delegation = root["delegation"]["attempt_status"]
    lines.append(f"delegation evidence: {delegation['observed']} observed, {delegation['not_observed_in_range']} not observed in range, "
                 f"{delegation['unavailable']} unavailable calls; child-session tokens and full coverage unknown")
    lines.append("cost=unknown; no reliable pricing or billing source")
    waits = report["time"]["observed_waits"]["known_union"]
    lines.append(f"run wall={report['time']['run_wall_seconds']}s; observed scope-wait union={waits['seconds']}s "
                 f"({waits['unknown_intervals']} unknown boundaries; may overlap CLI work)")
    for item in report["by_scope"][:20]:
        lines.append(f"{item['stage']}/{item['role']}/{item['node']} calls={item['attempt_count']} "
                     f"CLI={seconds(item['cli_seconds'])} usage={item['usage_status']}")
    if report["scope"]["attempt_id"] or report["scope"]["session_id"]:
        for row in report["attempts"][:20]:
            tokens = row["usage"]["last_reported_tokens"]
            lines.append(f"attempt={row['attempt_id']} session={row.get('session_id') or 'unknown'} "
                         f"reported input/cached/output={tokens['input_tokens']}/{tokens['cached_input_tokens']}/{tokens['output_tokens']} "
                         f"evidence={row['directory']}")
    lines.append("state=" + report["state"] + "; --attempt UUID / --session UUID / --json locate recorded evidence")
    if report["scope"]["unindexed_session_attempts"]:
        lines.append("session selection is incomplete: older calls have no recorded session fields")
    if report["verification"]["status"] != "not_requested":
        checked = report["verification"]
        lines.append(f"verification={checked['status']} errors={len(checked['errors'])} unavailable={len(checked['unavailable'])}")
        lines.extend(str(item["scope"]) + ": " + item["reason"] for item in checked["errors"][:5])
    if report["archive"]:
        lines.append("archive=" + json.dumps(report["archive"], ensure_ascii=False))
    return "\n".join(lines)
