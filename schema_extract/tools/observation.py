"""Bounded root-session usage evidence; observation never supplies extra budget.

Only the requested UUID is read. Native conversation text is neither retained nor
passed to the agent. A fresh pre-invocation sample replaces cross-call cursors, so
an unavailable observation cannot cause a later resume to replay its usage.
"""

from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import time
from uuid import UUID

from .common import digest


TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")
FORMAT = "schema_extract.usage_observation.v1"
TAIL_BYTES = 1024 * 1024
INVOCATION_BYTES = 8 * 1024 * 1024
MAX_RECORDS = 4096
DELEGATION_TOOLS = {"spawn_agent"}
COLLAB_TOOLS = {"wait"}


class ObservationError(ValueError):
    pass


def _tokens(raw):
    if not isinstance(raw, dict) or any(type(raw.get(key)) is not int or raw[key] < 0 for key in TOKEN_FIELDS):
        raise ObservationError("missing_or_invalid_token_fields")
    value = {key: raw[key] for key in TOKEN_FIELDS}
    if value["cached_input_tokens"] > value["input_tokens"]:
        raise ObservationError("cached_input_exceeds_input")
    if "total_tokens" in raw and (type(raw["total_tokens"]) is not int or
                                  raw["total_tokens"] != value["input_tokens"] + value["output_tokens"]):
        raise ObservationError("native_total_does_not_reconcile")
    return value


def _stamp(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _session_uuid(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ObservationError("native_delegation_invalid_uuid")
    return value


def _open(path):
    # Native files are private evidence, never archive inputs. Reject symlinks
    # at every component before opening the final regular file without following.
    if path.resolve(strict=True) != path.absolute():
        raise ObservationError("native_path_contains_symlink")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ObservationError("native_source_not_regular")
    return os.fdopen(fd, "rb")


def _locate(home, session_id, started_at):
    identity = UUID(session_id)
    if str(identity) != session_id:
        raise ObservationError("invalid_session_uuid")
    root = home / "sessions"
    database = home / "state_5.sqlite"
    if database.is_file():
        try:
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=0.2)) as connection:
                row = connection.execute("SELECT rollout_path FROM threads WHERE id=?", (session_id,)).fetchone()
            if row:
                path = Path(row[0]).absolute()
                if not path.is_relative_to(root) or not path.name.endswith(session_id + ".jsonl"):
                    raise ObservationError("native_index_identity_conflict")
                return path
        except sqlite3.Error:
            pass  # Older CLI layouts can still use the bounded UUID/date lookup.
    dates = {datetime.fromtimestamp(started_at, timezone.utc).date()}
    if identity.version == 7:
        dates.add(datetime.fromtimestamp((identity.int >> 80) / 1000, timezone.utc).date())
    directories = {root / day.strftime("%Y/%m/%d") for date in dates
                   for day in (date - timedelta(days=1), date, date + timedelta(days=1))}
    matches, inspected = [], 0
    for directory in sorted(directories):
        if not directory.is_dir():
            continue
        with os.scandir(directory) as entries:
            for entry in entries:
                inspected += 1
                if inspected > 4096:
                    raise ObservationError("native_discovery_limit")
                if entry.name.endswith(session_id + ".jsonl") and entry.is_file(follow_symlinks=False):
                    matches.append(Path(entry.path))
    if len(matches) != 1:
        raise ObservationError("native_session_not_uniquely_located")
    return matches[0]


def _metadata(stream, session_id):
    stream.seek(0)
    line = stream.readline(64 * 1024 + 1)
    if not line.endswith(b"\n") or len(line) > 64 * 1024:
        raise ObservationError("native_metadata_missing_or_oversized")
    event = json.loads(line)
    body = event.get("payload", {})
    if event.get("type") != "session_meta" or body.get("id") != session_id:
        raise ObservationError("native_session_identity_conflict")
    # No prompts, transcript content or arbitrary metadata leave this reader.
    metadata = {key: body.get(key) for key in ("id", "cwd", "cli_version")}
    source = body.get("source")
    metadata["source"] = source if source in ("cli", "exec", "vscode", "unknown") else "unknown"
    if body.get("parent_thread_id") is not None:
        try:
            metadata["parent_thread_id"] = _session_uuid(body["parent_thread_id"])
        except (ObservationError, ValueError):
            pass  # Optional ancestry metadata must not invalidate root counters.
    return metadata


def _delegation_record(event):
    """Project recognized markers only; never read arguments or activity text."""
    body = event.get("payload") or {}
    if event.get("type") == "response_item" and body.get("type") == "function_call":
        if body.get("name") in DELEGATION_TOOLS:
            return {"type": "function_call", "tool": body["name"]}
    if event.get("type") != "event_msg" or body.get("type") != "item_completed":
        return None
    item = body["item"]
    kind = item["type"]
    if kind == "SubAgentActivity":
        return {"type": kind, "agent_thread_id": _session_uuid(item["agent_thread_id"])}
    if kind == "CollabAgentToolCall":
        if item.get("tool") not in COLLAB_TOOLS or not isinstance(item.get("receiver_thread_ids"), list):
            raise ObservationError("native_delegation_format_unrecognized")
        return {"type": kind, "tool": item["tool"], "sender_thread_id": _session_uuid(item["sender_thread_id"]),
                "receiver_thread_ids": [_session_uuid(identity) for identity in item["receiver_thread_ids"]]}
    if not isinstance(kind, str) or "SubAgent" in kind or "Collab" in kind:
        raise ObservationError("native_delegation_format_unrecognized")
    return None


def _records(payload, offset, *, delegation=None):
    if payload and not payload.endswith(b"\n"):
        raise ObservationError("native_truncated_line")
    records = []
    for line in payload.splitlines(keepends=True):
        event = json.loads(line)
        body = event.get("payload") or {}
        kind = body.get("type") if isinstance(body, dict) else None
        if delegation is not None:
            try:
                marker = _delegation_record(event)
                if marker:
                    if len(delegation["records"]) >= MAX_RECORDS:
                        raise ObservationError("native_delegation_record_limit")
                    delegation["records"].append({"offset": offset, "end_offset": offset + len(line),
                        "event_sha256": hashlib.sha256(line).hexdigest(), **marker})
            except Exception as exc:
                # Marker diagnostics cannot reject otherwise proved root usage.
                delegation["issues"].append(str(exc) if isinstance(exc, ObservationError) else "native_delegation_format_unrecognized")
        if event.get("type") == "event_msg" and kind in {"token_count", "task_started", "task_complete"}:
            record = {"offset": offset, "end_offset": offset + len(line),
                      "event_sha256": hashlib.sha256(line).hexdigest(), "type": kind}
            if kind == "token_count":
                info = body.get("info")
                if info is None:
                    offset += len(line)
                    continue  # Rate-limit-only updates contain no usage counter.
                record["raw"] = {key: deepcopy(info.get(key)) for key in ("total_token_usage", "last_token_usage")}
                record["tokens"] = _tokens(record["raw"]["total_token_usage"])
                record["last_tokens"] = _tokens(record["raw"]["last_token_usage"])
            else:
                record["turn_id"] = body.get("turn_id")
            records.append(record)
            if len(records) > MAX_RECORDS:
                raise ObservationError("native_usage_record_limit")
        offset += len(line)
    return records


def _baseline(path, session_id):
    with _open(path) as stream:
        before = _stamp(os.fstat(stream.fileno()))
        metadata = _metadata(stream, session_id)
        offset = max(0, before[2] - TAIL_BYTES)
        stream.seek(offset)
        payload = stream.read(TAIL_BYTES)
        if offset:
            boundary = payload.find(b"\n") + 1
            if not boundary:
                raise ObservationError("native_baseline_line_limit")
            payload, offset = payload[boundary:], offset + boundary
        records = _records(payload, offset)
        if before != _stamp(os.fstat(stream.fileno())):
            raise ObservationError("native_baseline_changed_during_read")
    counters = [record for record in records if record["type"] == "token_count"]
    if not counters:
        raise ObservationError("native_resume_baseline_unavailable")
    return {"path": str(path), "file_stamp": before, "offset": before[2], "metadata": metadata,
            "tail_offset": offset, "tail_sha256": hashlib.sha256(payload).hexdigest(),
            "counter": counters[-1]}


def begin_usage_observation(attempt, invocation, *, home=None):
    """Sample resume history before launch; never infer a missing baseline as 0."""
    value = {"format": FORMAT, "identity": {key: invocation.get(key) for key in
             ("run_id", "attempt_id", "launch_id", "attempt_sha256")},
             "requested_session_id": attempt.get("session_id"), "workspace": attempt["workspace"],
             "started_at": invocation["started_at"], "cli_version": invocation.get("cli_version"),
             "baseline": None, "issues": []}
    value["delegation"] = {"policy": "forbidden" if invocation.get("delegation_policy") == "forbidden" else "unknown",
                           "status": "unavailable", "records": [], "issues": [],
                           "child_usage_reason": "child_session_tokens_and_complete_source_set_not_observed"}
    try:
        value["home"] = str((home or Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))).expanduser().absolute())
        if attempt.get("session_id"):
            path = _locate(Path(value["home"]), attempt["session_id"], invocation["started_at"])
            value["baseline"] = _baseline(path, attempt["session_id"])
    except Exception as exc:
        value["issues"].append(str(exc) if isinstance(exc, ObservationError) else "native_baseline_" + type(exc).__name__)
    value["sampled_at"] = time.time()
    return value


def _invocation_records(start, session_id):
    baseline = start["baseline"]
    path = Path(baseline["path"]) if baseline else _locate(Path(start["home"]), session_id, start["started_at"])
    with _open(path) as stream:
        before = _stamp(os.fstat(stream.fileno()))
        metadata = _metadata(stream, session_id)
        offset = baseline["offset"] if baseline else 0
        if baseline:
            if before[:2] != baseline["file_stamp"][:2] or before[2] < offset:
                raise ObservationError("native_resume_cursor_invalidated")
            stream.seek(baseline["tail_offset"])
            if hashlib.sha256(stream.read(offset - baseline["tail_offset"])).hexdigest() != baseline["tail_sha256"]:
                raise ObservationError("native_resume_prefix_changed")
        elif Path(metadata.get("cwd") or "").absolute() != Path(start["workspace"]).absolute():
            raise ObservationError("native_new_session_workspace_conflict")
        if before[2] - offset > INVOCATION_BYTES:
            raise ObservationError("native_invocation_byte_limit")
        stream.seek(offset)
        payload = stream.read(INVOCATION_BYTES + 1)
        if len(payload) != before[2] - offset or before != _stamp(os.fstat(stream.fileno())):
            raise ObservationError("native_changed_during_read")
    delegation = {"records": [], "issues": []}
    records = _records(payload, offset, delegation=delegation)
    return {"path": str(path), "file_stamp": before, "metadata": metadata, "offset": offset,
            "end_offset": before[2], "range_sha256": hashlib.sha256(payload).hexdigest()}, records, delegation


def _finish_delegation(value, records, result):
    starts = [record for record in records if record["type"] == "task_started"]
    ends = [record for record in records if record["type"] == "task_complete"]
    complete = (len(starts) == len(ends) == 1 and starts[0].get("turn_id") and
                starts[0]["turn_id"] == ends[0]["turn_id"] and starts[0]["end_offset"] <= ends[0]["offset"] and
                result.get("outcome") == "completed" and result.get("termination", {}).get("confirmed"))
    if not complete or any(record["offset"] < starts[0]["end_offset"] or record["end_offset"] > ends[0]["offset"]
                           for record in value["records"]):
        value["issues"].append("native_delegation_range_incomplete")
    if not value["issues"]:
        value["status"] = "observed" if value["records"] else "not_observed_in_range"
    value["issues"] = sorted(set(value["issues"]))


def finish_usage_observation(start, parsed, result):
    """Return evidence independently of CLI success and without raising into it."""
    value = {key: deepcopy(item) for key, item in start.items() if key not in {"home", "workspace"}}
    value.update(status="unavailable", root_status="unavailable", cursor_valid=False,
                 scope="current_root_session_invocation_delta; child_session_coverage_unobserved",
                 cached_input_is_input_subset=True, additive_tokens={key: None for key in TOKEN_FIELDS},
                 session_id=result.get("session_id"), completed_at=result.get("completed_at"),
                 cli_version=result.get("cli_version"), source="native token_count reconciled with events.jsonl",
                 stream_reports=deepcopy(parsed["usage"]["reports"]), stream_trace_sha256=parsed["trace"]["sha256"])
    try:
        if value["issues"]:
            raise ObservationError("resume_baseline_unavailable")
        session_id = result.get("session_id")
        if not session_id or parsed["trace"]["session_ids"] != [session_id]:
            raise ObservationError("stream_session_identity_unavailable")
        if start["requested_session_id"] and (start["requested_session_id"] != session_id or not start["baseline"]):
            raise ObservationError("resume_baseline_identity_conflict")
        native, records, delegation = _invocation_records(start, session_id)
        value["native"] = native
        value["records"] = records
        value["delegation"].update(delegation)
        try:
            _finish_delegation(value["delegation"], records, result)
        except Exception:
            value["delegation"].update(status="unavailable", issues=["native_delegation_observation_failed"])
        counters = [record for record in records if record["type"] == "token_count"]
        if not counters:
            raise ObservationError("native_invocation_usage_unavailable")
        tasks = [record["turn_id"] for record in records if record["type"] == "task_started"]
        if len(tasks) != 1 or not tasks[0]:
            raise ObservationError("native_invocation_turn_boundary_unavailable")
        value["turn_id"] = tasks[0]
        task_start = next(record["end_offset"] for record in records if record["type"] == "task_started")
        endings = [record for record in records if record["type"] == "task_complete"]
        if (any(record["offset"] < task_start for record in counters) or
                any(record["turn_id"] != tasks[0] for record in endings) or len(endings) > 1 or
                (endings and any(record["offset"] > endings[0]["offset"] for record in counters))):
            raise ObservationError("native_usage_outside_invocation_turn")
        previous = start["baseline"]["counter"]["tokens"] if start["baseline"] else None
        if previous is None:
            # New-session zero has evidence: the first cumulative counter equals
            # the first response, under this fresh UUID and workspace identity.
            if counters[0]["tokens"] != counters[0]["last_tokens"]:
                raise ObservationError("native_new_session_zero_baseline_unproven")
            previous = {key: 0 for key in TOKEN_FIELDS}
        value["baseline_tokens"] = deepcopy(previous)
        total, deltas = {key: 0 for key in TOKEN_FIELDS}, 0
        for record in counters:
            current = record["tokens"]
            delta = {key: current[key] - previous[key] for key in TOKEN_FIELDS}
            if any(number < 0 for number in delta.values()):
                raise ObservationError("native_counter_regression")
            if any(delta.values()):
                if delta != record["last_tokens"]:
                    raise ObservationError("native_counter_delta_conflict")
                total = {key: total[key] + delta[key] for key in TOKEN_FIELDS}
                deltas += 1
            previous = current
        value["final_tokens"] = previous
        if not deltas:
            raise ObservationError("native_no_new_usage_delta")
        reports = parsed["usage"]["reports"]
        trace = parsed["trace"]
        if reports:
            if len(reports) != 1 or _tokens(reports[0]["raw"]) != previous:
                raise ObservationError("native_stream_cumulative_conflict")
        else:
            value["issues"].append("stream_terminal_usage_unavailable")
        if trace["error"] or not trace["stable"] or trace["truncated_line"] or trace["parse_error_count"]:
            raise ObservationError("stream_trace_incomplete")
        closed = (result.get("outcome") == "completed" and result.get("termination", {}).get("confirmed") and
                  [record["turn_id"] for record in records if record["type"] == "task_complete"] == tasks)
        if not closed:
            value["issues"].append("invocation_not_successfully_closed")
        value.update(additive_tokens=total, root_status="exact" if closed and reports else "partial",
                     status="partial", cursor_valid=True, delta_count=deltas)
        value["issues"].append("child_session_coverage_unobserved")
    except Exception as exc:
        value["issues"].append(str(exc) if isinstance(exc, ObservationError) else "native_observation_" + type(exc).__name__)
    if value["delegation"]["status"] == "unavailable" and not value["delegation"]["issues"]:
        value["delegation"]["issues"].append("native_delegation_range_unavailable")
    value["issues"] = sorted(set(value["issues"]))
    value["observed_at"] = time.time()
    return value


def observation_summary(value):
    """The existing attempt ledger retains a small immutable observation binding."""
    keys = ("format", "identity", "session_id", "turn_id", "status", "root_status", "scope",
            "source", "cursor_valid", "additive_tokens", "baseline_tokens", "final_tokens", "issues")
    summary = {key: deepcopy(value[key]) for key in keys if key in value}
    if "delegation" in value:
        delegation = value["delegation"]
        records = delegation.get("records", [])
        related = {identity for record in records for identity in
                   [record.get("agent_thread_id"), record.get("sender_thread_id"), *record.get("receiver_thread_ids", [])]
                   if identity and identity != value.get("session_id")}
        summary["delegation"] = {key: deepcopy(delegation[key]) for key in
                                  ("policy", "status", "issues", "child_usage_reason") if key in delegation}
        summary["delegation"].update(record_count=len(records), related_thread_count=len(related))
    return summary | {"sha256": digest(value)}


def observation_totals(rows):
    """Sum only proved, disjoint root deltas; unknown children stay unknown."""
    selected, duplicates, conflicts = {}, 0, set()
    for row in rows:
        identity = row.get("origin_run_id"), row.get("attempt_id")
        if identity in selected:
            duplicates += 1
            if selected[identity].get("usage_observation") != row.get("usage_observation"):
                conflicts.add(identity)
        else:
            selected[identity] = row
    usable, spans = {}, {}
    for identity, row in selected.items():
        observed = row.get("usage_observation", {})
        if not observed.get("cursor_valid") or observed.get("format") != FORMAT:
            continue
        try:
            tokens = _tokens(observed.get("additive_tokens"))
            before, after = _tokens(observed.get("baseline_tokens")), _tokens(observed.get("final_tokens"))
            if tokens != {key: after[key] - before[key] for key in TOKEN_FIELDS}:
                continue
            lower, upper = before["input_tokens"] + before["output_tokens"], after["input_tokens"] + after["output_tokens"]
            if not observed.get("session_id") or upper <= lower:
                continue
            span = observed["session_id"], lower, upper
            for other, previous in spans.items():
                if span[0] == previous[0] and max(lower, previous[1]) < min(upper, previous[2]):
                    conflicts.update((identity, other))
            spans[identity], usable[identity] = span, tokens
        except (ObservationError, KeyError, TypeError):
            continue
    known = {identity: tokens for identity, tokens in usable.items() if identity not in conflicts}
    exact = sum(selected[identity]["usage_observation"].get("root_status") == "exact" for identity in known)
    delegation_counts = {key: 0 for key in ("observed", "not_observed_in_range", "unavailable")}
    policy_counts = {key: 0 for key in ("forbidden", "unknown")}
    for identity, row in selected.items():
        observed = row.get("usage_observation") or {}
        delegation = observed.get("delegation", {}) if observed.get("format") == FORMAT and identity not in conflicts else {}
        state = delegation.get("status", "unavailable")
        delegation_counts[state if state in delegation_counts else "unavailable"] += 1
        policy_counts["forbidden" if delegation.get("policy") == "forbidden" else "unknown"] += 1
    return {"status": "partial" if known else "unavailable", "scope": "root_session_known_subtotal",
            "attempt_scope": "registered_cli_attempts_in_this_run_including_deduplicated_origins",
            "root_status": "exact" if selected and exact == len(selected) else "partial" if known else "unavailable",
            "additive_tokens": {key: sum(tokens[key] for tokens in known.values()) if known else None for key in TOKEN_FIELDS},
            "exact_attempts": exact, "partial_attempts": len(known) - exact,
            "known_attempts": len(known), "unknown_attempts": len(selected) - len(known),
            "duplicate_attempts": duplicates, "conflicting_attempts": len(conflicts),
            "delegation": {"attempt_status": delegation_counts, "policy_counts": policy_counts},
            "child_session_coverage": "unavailable", "whole_invocation_tokens": None,
            "whole_invocation_reason": "root_deltas_and_delegation_markers_do_not_prove_complete_child_session_usage",
            "cached_input_is_input_subset": True, "cost": None,
            "cost_reason": "no_reliable_pricing_or_billing_source"}


def interval_observation(intervals, *, now):
    """Union known intervals only; an unknown boundary is never replaced by 0."""
    known, unknown, opened = [], 0, 0
    for start, end, active in intervals:
        if active and end is None:
            end, opened = now, opened + 1
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in (start, end)) or end < start:
            unknown += 1
        else:
            known.append((start, end))
    merged = []
    for start, end in sorted(known):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return {"seconds": sum(end - start for start, end in merged) if known else None,
            "status": "partial" if known else "unavailable", "known_intervals": len(known),
            "unknown_intervals": unknown, "open_intervals": opened}
