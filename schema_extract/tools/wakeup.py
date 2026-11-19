"""A persistent clock for one existing Supervisor thread, not a run scheduler."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from uuid import UUID, uuid4

from .common import ExtractError, SupervisorBusy, digest, rules_root
from .processes import child_exited, process_identity, process_alive, process_table, observe_owned, terminate_owned
from .prompts import render_operation
from . import monitor


CLOCK_FORMAT = "schema_extract.clock.v1"
EARLY_SECONDS = 300
REGULAR_SECONDS = 1800
RETRY_SECONDS = 120
MAX_QUEUE_ATTEMPTS = 3
HEARTBEAT_SECONDS = 15
OBSERVATION_SECONDS = 30
MAX_OBSERVED_RUNS = 16
ATTENTION_SECONDS = 1800
HEARTBEAT_STALE_SECONDS = 75


def _uuid(value: str, label="thread") -> str:
    try:
        if str(UUID(value)) == value:
            return value
    except (ValueError, TypeError, AttributeError):
        pass
    raise ExtractError(label + " must be a canonical UUID")


def _alive(identity, table=None) -> bool | None:
    try:
        if table is None:
            if not isinstance(identity, dict) or type(identity.get("pid")) is not int or identity["pid"] <= 1:
                return None
            current = process_identity(identity["pid"])
            table = {identity["pid"]: current} if current else {}
        return process_alive(identity, table)
    except OSError:
        return None


def format_beijing_time(timestamp) -> str:
    if timestamp is None:
        return "未记录"
    return datetime.fromtimestamp(timestamp, timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S 北京时间（UTC+8）")


def clock_display(result: dict) -> dict:
    """Add human-readable times to an output copy, never to scheduler evidence."""
    result = deepcopy(result)

    def times(record, keys):
        if isinstance(record, dict):
            for key in keys:
                if key in record:
                    record[key + "_display"] = format_beijing_time(record[key])

    for record in (result, result.get("clock")):
        if not isinstance(record, dict):
            continue
        times(record, ("observed_at", "next_due_at"))
        if record.get("enabled") is False and record.get("next_due_at") is not None:
            record["next_due_at_display"] += "（已停用，保留记录）"
        for key in ("check", "last_scan"):
            times(record.get(key), ("started_at", "due_at", "ack_at", "observed_at", "completed_at"))
        reminder = record.get("reminder")
        times(reminder, ("due_at", "next_attempt_at", "first_queued_at", "last_queued_at"))
        if isinstance(reminder, dict):
            times(reminder.get("ack"), ("at",))
        times(record.get("ack"), ("at",))
        times(record.get("daemon"), ("heartbeat_at",))
    return result


def _clock(ledger: dict) -> dict:
    clock = ledger.get("clock", {"format": CLOCK_FORMAT, "enabled": False, "generation": 0,
                                 "thread_id": None, "last_check_number": -1, "next_due_at": None})
    if not isinstance(clock, dict) or clock.get("format") != CLOCK_FORMAT:
        raise ExtractError("unsupported Supervisor clock record")
    return clock


def _binding(clock: dict) -> dict:
    return {key: clock[key] for key in ("generation", "thread_id")}


def _matches(clock: dict, binding: dict) -> bool:
    return bool(clock.get("enabled") and all(clock.get(key) == value for key, value in binding.items()))


def _item(clock: dict, number: int, due: float) -> dict:
    identity = {**_binding(clock), "number": number, "due_at": due}
    return {**identity, "id": f"check-{clock['generation']}-{number}-{digest(identity)[:24]}",
            "status": "scheduled", "attempts": [], "ack": None}


def _archive(clock: dict, item: dict | None) -> None:
    if item:
        clock["history"] = (clock.get("history", []) + [deepcopy(item)])[-64:]


def _arm(clock: dict) -> None:
    if clock.get("enabled") and clock["last_check_number"] >= 0 and not clock.get("reminder"):
        clock["reminder"] = _item(clock, clock["last_check_number"] + 1, clock["next_due_at"])


def _current(clock: dict, item: dict | None, now: float) -> bool:
    if not item or not _matches(clock, _binding(item)) or now < item["due_at"]:
        return False
    return (item["number"] == clock["last_check_number"] + 1 and item["due_at"] == clock["next_due_at"] and
            item["id"] == _item(clock, item["number"], item["due_at"])["id"])


def _next_send_at(item: dict | None) -> float | None:
    """Only a completed attempt can make another delivery eligible."""
    if (not item or item.get("ack") or len(item["attempts"]) >= MAX_QUEUE_ATTEMPTS or
            any(attempt.get("status") == "sending" for attempt in item["attempts"])):
        return None
    if not item["attempts"]:
        return item["due_at"]
    attempt = item["attempts"][-1]
    delay = ATTENTION_SECONDS if attempt.get("status") == "queued" else RETRY_SECONDS
    return max(attempt["started_at"], attempt.get("finished_at", 0)) + delay


class QueueError(ExtractError):
    pass


class Clock:
    def __init__(self, store, repo: Path, *, now=None, sender=None, on_wait=None):
        self.store, self.repo = store, Path(repo).resolve()
        self.now = now or time.time
        self.sender = sender
        self.on_wait = on_wait
        self.stopping = threading.Event()
        self.launch_id = None
        self.inflight = None
        self.observation_claim = None

    def _transaction(self, phase):
        callback = (lambda elapsed: self.on_wait(phase, elapsed)) if self.on_wait else None
        return self.store.control_transaction(on_wait=callback)

    @property
    def lock_path(self) -> Path:
        return self.store.root / "monitor-clock.lock"

    @property
    def runtime_directory(self) -> Path:
        directory = self.store.root / "runtime" / "clock"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _lock_status(self) -> str:
        try:
            with self.lock_path.open("r") as stream:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return "free"
                except BlockingIOError:
                    return "held"
        except FileNotFoundError:
            return "absent"
        except OSError:
            return "unknown"

    def start(self, thread: str, run_ids=None, *, codex_binary=None, launch=True) -> dict:
        """The foreground start performs scan #0; the detached child only ticks.

        launch=False is an in-process test seam, never a CLI timer option.
        """
        thread = _uuid(thread)
        if run_ids is not None:
            run_ids = sorted(set(run_ids))
            for run_id in run_ids:
                self.store.run_dir(run_id)
        self.store.initialize()
        identity = process_identity(os.getpid())
        table = process_table()
        if identity is None:
            raise ExtractError("cannot establish clock launcher identity")
        # The same open-file description is inherited by the daemon. Closing
        # the parent's descriptor is safe; LOCK_UN would also unlock the child.
        with self.lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                current = _clock(self.store.read_supervisor())
                self._check_start(current, thread, run_ids, codex_binary)
                if not current.get("enabled"):
                    raise ExtractError("previous clock is still stopping; inspect monitor status before start")
                binding = _binding(current)
                daemon = current.get("daemon", {})
                if (_alive(daemon.get("identity"), table) is not True and
                        not (daemon.get("status") == "starting" and _alive(daemon.get("launcher"), table) is True)):
                    raise ExtractError("clock lock is held but its live owner is unconfirmed; inspect monitor status")
                result = self.status()
                if not _matches(result, binding):
                    return {**result, "reused": False, "started": False, "reason": "start_invalidated"}
                return {**result, "reused": True}
            with self.store.supervisor() as ledger:
                clock = _clock(ledger)
                self._check_start(clock, thread, run_ids, codex_binary)
                daemon = clock.get("daemon", {})
                if (daemon and daemon.get("status") not in {"stopped", "launch_failed"} and
                        _alive(daemon.get("identity"), table) is not False):
                    raise ExtractError("recorded clock owner is not confirmed exited; inspect monitor status")
                for _, pending in self._sending(clock):
                    if (not pending.get("identity") or
                            not observe_owned(pending["identity"], pending.get("descendants", []), table=table)["confirmed"]):
                        raise ExtractError("previous queue delivery is not confirmed exited; inspect monitor status")
                if not clock.get("enabled"):
                    _archive(clock, clock.pop("reminder", None))
                    clock.update(enabled=True, generation=clock["generation"] + 1, thread_id=thread,
                                 run_ids=run_ids, started_at=self.now())
                clock.setdefault("run_ids", run_ids)
                clock.update(repo=str(self.repo), entrypoint=str(rules_root() / "schema_extract/extract_yaml.py"),
                             python=sys.executable, queue_backend="codex", codex_binary=codex_binary or clock.get("codex_binary", "codex"))
                launch_id = str(uuid4())
                clock["daemon"] = {"id": launch_id, "status": "starting", "launcher": identity,
                                   "created_at": self.now(), "identity": None, "heartbeat_at": None}
                _arm(clock)
                ledger["clock"] = clock
                binding, selected = _binding(clock), clock["run_ids"]
                initial = clock["last_check_number"] < 0
            if initial:
                self._initial_scan(binding)
            clock = _clock(self.store.read_supervisor())
            if not _matches(clock, binding) or clock.get("daemon", {}).get("id") != launch_id:
                return {**self.status(), "started": False, "reason": "start_invalidated"}
            if not launch:
                with self.store.supervisor() as ledger:
                    current = _clock(ledger)
                    if current.get("daemon", {}).get("id") == launch_id:
                        current["daemon"]["status"] = "stopped"
                return {**self.status(), "started": False}
            command = [sys.executable, "-B", clock["entrypoint"], "_clock", "--launch-id", launch_id,
                       "--lock-fd", str(lock.fileno()), "--repo", str(self.repo), "--state-root", str(self.store.root)]
            try:
                log_path = self.store.root / "monitor-clock.log"
                with log_path.open("ab") as log:
                    process = subprocess.Popen(command, cwd=self.runtime_directory, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                               close_fds=True, pass_fds=(lock.fileno(),), start_new_session=True)
                try:
                    child_identity, identity_error = process_identity(process.pid), None
                except OSError as exc:
                    child_identity, identity_error = None, str(exc)
                with self.store.supervisor() as ledger:
                    current = _clock(ledger)
                    daemon = current.get("daemon", {})
                    if daemon.get("id") == launch_id and daemon["status"] == "starting":
                        daemon.update(pid=process.pid, identity=child_identity, log=str(log_path))
                        if identity_error:
                            daemon["observation_error"] = identity_error
                return {**self.status(), "started": True, "pid": process.pid}
            except OSError as exc:
                with self.store.supervisor() as ledger:
                    daemon = _clock(ledger).get("daemon", {})
                    if daemon.get("id") == launch_id:
                        daemon.update(status="launch_failed", error=str(exc))
                raise ExtractError("clock launch failed; explicit start can resume: " + str(exc)) from exc

    @staticmethod
    def _check_start(clock, thread, run_ids, binary):
        if clock.get("enabled"):
            if thread != clock["thread_id"]:
                raise ExtractError("clock is bound to another thread; use monitor stop before start")
            if run_ids is not None and run_ids != clock.get("run_ids"):
                raise ExtractError("changing the monitored run scope requires monitor stop/start")
            if binary is not None and binary != clock.get("codex_binary", "codex"):
                raise ExtractError("changing the queue executable requires monitor stop/start")

    def _initial_scan(self, binding) -> None:
        clock = _clock(self.store.read_supervisor())
        started = self.now()
        rows = monitor.inspect_runs(self.store, clock.get("run_ids"))
        observed_at = self.now()
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            if not _matches(clock, binding) or clock["last_check_number"] >= 0:
                return
            now = self.now()
            result = monitor.record_scan(ledger, rows, observed_at)
            check = self._complete(clock, result, 0, f"initial-{clock['generation']}", started, now)
            clock["next_observation_at"] = now + OBSERVATION_SECONDS

    @staticmethod
    def _complete(clock, result, number, wake_id, started, now) -> dict:
        item = clock.pop("reminder", None)
        check = {"id": wake_id, "number": number, **_binding(clock), "started_at": started,
                 "due_at": (item or {}).get("due_at"), "ack_at": (item or {}).get("ack", {}).get("at") if (item or {}).get("ack") else None,
                 "observed_at": result["observed_at"], "completed_at": now, "elapsed_seconds": max(0, now - started),
                 "run_count": len(result["runs"]), "change_count": len(result["changes"]),
                 "anomaly_count": len(result["anomalies"]),
                 "message_ids": [attempt["message_id"] for attempt in (item or {}).get("attempts", []) if attempt.get("message_id")]}
        interval = EARLY_SECONDS if number < 3 else REGULAR_SECONDS
        clock.update(last_check_number=number, last_scan=check, next_due_at=now + interval)
        if item:
            item.update(status="completed", completed_at=now, scan=check)
            _archive(clock, item)
        _arm(clock)
        return check

    def ack(self, wake_id: str, thread: str) -> dict:
        thread = _uuid(thread)
        with self._transaction("ack") as ledger:
            now = self.now()
            clock = _clock(ledger)
            if not clock.get("enabled"):
                return {"accepted": False, "reason": "disabled"}
            if thread != clock["thread_id"]:
                return {"accepted": False, "reason": "wrong_thread"}
            item = clock.get("reminder")
            if not _current(clock, item, now) or item["id"] != wake_id:
                return {"accepted": False, "reason": "stale_or_not_due"}
            if item.get("ack"):
                return {"accepted": False, "reason": "already_acknowledged"}
            if not item["attempts"]:
                return {"accepted": False, "reason": "not_sent"}
            item.update(status="acknowledged", ack={"thread_id": thread, "at": now})
            return {"accepted": True, "id": wake_id, "number": item["number"], "ack": deepcopy(item["ack"])}

    def scan(self, wake_id=None, thread=None) -> dict:
        """Only an explicit thread can finish that thread's accepted reminder."""
        thread = _uuid(thread)
        identity = process_identity(os.getpid())
        if identity is None:
            raise ExtractError("cannot establish scan process identity")
        with self._transaction("scan_claim") as ledger:
            clock = _clock(ledger)
            if not clock.get("enabled") or thread != clock["thread_id"]:
                raise ExtractError("clock disabled or scan thread does not match")
            last = clock.get("last_scan", {})
            if wake_id and last.get("id") == wake_id and last.get("generation") == clock["generation"]:
                return {"runs": [], "changes": [], "anomalies": [], "clock": {"committed": False, "reused": True,
                        "check": deepcopy(last), "next_due_at": clock["next_due_at"]}}
            item = clock.get("reminder")
            if not _current(clock, item, self.now()) or (wake_id and item["id"] != wake_id) or not item.get("ack"):
                raise ExtractError("scan requires the current accepted reminder; acknowledge it first")
            previous = item.get("scan")
            if previous and _alive(previous.get("owner")) is not False:
                return {"runs": [], "changes": [], "anomalies": [], "clock": {"committed": False,
                        "status": "scan_busy", "id": item["id"], "owner": previous.get("owner")}}
            claim = {"id": str(uuid4()), "owner": identity, "started_at": self.now()}
            item["scan"] = claim
            wake_id, binding, selected = item["id"], _binding(clock), clock.get("run_ids")
        try:
            rows = monitor.inspect_runs(self.store, selected)
            observed_at = self.now()
            with self._transaction("scan_commit") as ledger:
                clock = _clock(ledger)
                item = clock.get("reminder")
                if (not _matches(clock, binding) or not item or item["id"] != wake_id or
                        item.get("scan", {}).get("id") != claim["id"]):
                    return {"runs": rows, "changes": [], "anomalies": [],
                            "clock": {"committed": False, "status": "scan_invalidated", "id": wake_id}}
                now = self.now()
                result = monitor.record_scan(ledger, rows, observed_at)
                check = self._complete(clock, result, item["number"], wake_id, claim["started_at"], now)
                result["clock"] = {"committed": True, "check": deepcopy(check), "next_due_at": clock["next_due_at"]}
        except Exception as exc:
            with self._transaction("scan_cleanup") as ledger:
                clock = _clock(ledger)
                item = clock.get("reminder")
                if item and item["id"] == wake_id and item.get("scan", {}).get("id") == claim["id"]:
                    item.pop("scan", None)
                    item["scan_error"] = str(exc)
            raise
        return result

    def observe(self) -> dict:
        """Use this daemon's bounded observation pass independently of delivery."""
        now = self.now()
        current = _clock(self.store.read_supervisor())
        if not current.get("enabled") or now < current.get("next_observation_at", 0):
            return {"status": "waiting"}
        run_ids = current.get("run_ids")
        run_ids = sorted(run_ids if run_ids is not None else self.store.run_ids())
        identity = process_identity(os.getpid())
        table = process_table()
        if identity is None:
            raise ExtractError("cannot establish observation process identity")
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            if (not _matches(clock, _binding(current)) or now < clock.get("next_observation_at", 0) or
                    (self.launch_id and clock.get("daemon", {}).get("id") != self.launch_id)):
                return {"status": "invalidated"}
            previous = clock.get("observation_claim")
            if previous and previous.get("id") != self.observation_claim and _alive(previous.get("owner"), table) is not False:
                return {"status": "busy"}
            offset = clock.get("observation_cursor", 0) % max(1, len(run_ids))
            selected = (run_ids[offset:] + run_ids[:offset])[:MAX_OBSERVED_RUNS]
            claim = {"id": str(uuid4()), "owner": identity, "started_at": now,
                     "due_at": clock.get("next_observation_at", now), **_binding(clock)}
            clock["observation_claim"] = claim
        self.observation_claim = claim["id"]
        try:
            rows = monitor.inspect_runs(self.store, selected)
            observed_at = self.now()
            with self.store.supervisor() as ledger:
                clock = _clock(ledger)
                if (not _matches(clock, _binding(claim)) or clock.get("observation_claim", {}).get("id") != claim["id"]):
                    return {"status": "invalidated"}
                now = self.now()
                result = monitor.record_scan(ledger, rows, observed_at)
                clock["last_observation"] = {**claim, "observed_at": observed_at, "completed_at": now, "run_count": len(rows),
                                             "anomaly_count": len(result["anomalies"])}
                clock.update(next_observation_at=now + OBSERVATION_SECONDS,
                             observation_cursor=(offset + len(selected)) % max(1, len(run_ids)))
                clock.pop("observation_claim", None)
                clock.pop("observation_error", None)
            self.observation_claim = None
            return {"status": "observed", "id": claim["id"]}
        except SupervisorBusy:
            raise
        except Exception as exc:
            with self.store.supervisor() as ledger:
                clock = _clock(ledger)
                if clock.get("observation_claim", {}).get("id") == claim["id"]:
                    clock.pop("observation_claim", None)
                    clock.update(observation_error=str(exc), next_observation_at=self.now() + OBSERVATION_SECONDS)
            raise

    def message(self, clock: dict, item: dict) -> str:
        entry = [clock["python"], "-B", clock["entrypoint"], "monitor"]
        where = ["--repo", clock["repo"], "--state-root", str(self.store.root)]
        ack = shlex.join([*entry, "ack", item["id"], "--thread", clock["thread_id"], *where])
        status = shlex.join([*entry, "status", *where])
        due = format_beijing_time(item["due_at"])
        context = {"reminder_id": item["id"], "thread_id": clock["thread_id"], "due_at": due,
                   "ack_command": ack, "status_command": status}
        scan = shlex.join([*entry, "scan", "--wake", item["id"], "--thread", clock["thread_id"], *where])
        manual = Path(clock["entrypoint"]).with_name("README.md")
        return render_operation("wakeup", {**context, "scan_command": scan, "manual": str(manual)})

    @staticmethod
    def _sending(clock):
        return [(record, attempt) for record in [clock.get("reminder"), *clock.get("history", [])]
                if record for attempt in record.get("attempts", []) if attempt.get("status") == "sending"]

    def _record_send(self, item, number, outcome):
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            current = clock.get("reminder")
            for record in [current, *clock.get("history", [])]:
                if not record or record["id"] != item["id"]:
                    continue
                attempt = next((value for value in record["attempts"] if value["number"] == number), None)
                if attempt is not None:
                    attempt.update(outcome)
                if (record is current and _matches(clock, _binding(item)) and not record.get("ack") and
                        record["status"] == "sending" and record["attempts"][-1]["number"] == number):
                    record["status"] = ("sending" if outcome["status"] == "sending" else
                                        "queued" if outcome["status"] == "queued" else "retry_pending")
                if outcome.get("message_id") and record.get("scan", {}).get("completed_at") is not None:
                    ids = record["scan"].setdefault("message_ids", [])
                    if outcome["message_id"] not in ids:
                        ids.append(outcome["message_id"])
            last = clock.get("last_scan", {})
            if last.get("id") == item["id"] and outcome.get("message_id"):
                if outcome["message_id"] not in last.get("message_ids", []):
                    last.setdefault("message_ids", []).append(outcome["message_id"])

    def _send(self, clock: dict, item: dict) -> None:
        number = len(item["attempts"])
        attempt = item["attempts"][-1]
        message = self.message(clock, item)
        if self.sender:
            # A synchronous fake sender is only an offline seam. Real delivery
            # always uses the nonblocking child below.
            self.inflight = {"item": item, "number": number, "facts": {"status": "sending"}}
            try:
                message_id = _uuid(self.sender(clock["thread_id"], message), "queue message")
                outcome = {"status": "queued", "message_id": message_id}
            except Exception as exc:
                outcome = {"status": "uncertain", "error": str(exc)}
            self.inflight["outcome"] = {**outcome, "finished_at": self.now()}
            return
        command = [clock["codex_binary"], "queue", "--thread", clock["thread_id"], "--message", message, "--resume"]
        stdout_path, stderr_path = Path(attempt["stdout"]), Path(attempt["stderr"])
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       start_new_session=True, cwd=self.runtime_directory)
        self.inflight = {"item": item, "number": number, "process": process,
                         "facts": {"status": "sending", "pid": process.pid}, "thread": clock["thread_id"]}
        try:
            self.inflight["facts"]["identity"] = process_identity(process.pid)
        except OSError as exc:
            self.inflight["facts"]["observation_error"] = str(exc)

    def _poll_send(self):
        pending = self.inflight
        if pending is None:
            # A restarted clock cannot recover a Popen wait status. Preserve an
            # unfinished delivery until its recorded processes are truly gone.
            for item, attempt in self._sending(_clock(self.store.read_supervisor())):
                identity = attempt.get("identity")
                if not identity:
                    return {"status": "sending", "id": item["id"], "reason": "queue identity unconfirmed"}
                observed = observe_owned(identity, attempt.get("descendants", []))
                if not observed["confirmed"]:
                    self._record_send(item, attempt["number"], {"status": "sending", "termination": observed,
                                      "descendants": observed.get("descendants", [])})
                    return {"status": "sending", "id": item["id"]}
                outcome = {"status": "uncertain", "finished_at": self.now(), "termination": observed,
                           "error": "queue owner exited before recording its result"}
                self._record_send(item, attempt["number"], outcome)
                return {"id": item["id"], **outcome}
            return None
        item, facts = pending["item"], pending["facts"]
        process = pending.get("process")
        if process is not None and "outcome" not in pending:
            exited = child_exited(process)
            if not facts.get("identity"):
                try:
                    facts["identity"] = process_identity(process.pid)
                except OSError as exc:
                    facts["observation_error"] = str(exc)
            identity = facts.get("identity")
            observed = (observe_owned(identity, facts.get("descendants", [])) if identity else
                        {"confirmed": False, "actions": [], "survivors": [], "reason": "queue identity unconfirmed"})
            facts.update(termination=observed, descendants=observed.get("descendants", []))
            if exited and identity and not observed["confirmed"]:
                observed = terminate_owned(identity, facts.get("descendants", []), include_anchor=True)
                facts.update(termination=observed, descendants=observed.get("descendants", []))
            if not exited or not observed["confirmed"]:
                self._record_send(item, pending["number"], facts)
                return {"status": "sending", "id": item["id"]}
            code = process.poll()
            attempt = item["attempts"][-1]
            stdout = Path(attempt["stdout"]).read_text(errors="replace")
            stderr = Path(attempt["stderr"]).read_text(errors="replace")
            match = re.search(r"Queued message ([0-9a-f-]{36}) for thread ([0-9a-f-]{36})", stdout)
            try:
                if code or not match or match[2] != pending["thread"]:
                    raise QueueError(f"queue did not confirm the bound thread and message ID (exit={code}); "
                                     + (stderr or stdout)[-500:])
                outcome = {"status": "queued", "message_id": _uuid(match[1], "queue message")}
            except ExtractError as exc:
                outcome = {"status": "uncertain", "error": str(exc)}
            pending["outcome"] = {**outcome, "exit_code": code, "finished_at": self.now()}
        outcome = {**facts, **pending["outcome"]}
        self._record_send(item, pending["number"], outcome)
        self.inflight = None
        return {"id": item["id"], **outcome}

    def tick(self) -> dict:
        try:
            return self._tick()
        except SupervisorBusy:
            return {"status": "busy", "committed": False}
        except OSError as exc:
            return {"status": "observation_unknown", "error": str(exc)}

    def _tick(self) -> dict:
        owner = process_identity(os.getpid())
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            if self.launch_id and clock.get("enabled") and clock.get("daemon", {}).get("id") == self.launch_id:
                clock["daemon"].update(heartbeat_at=self.now(), status="running")
        observation = self.observe()
        delivery = self._poll_send()
        now = self.now()
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            if not clock.get("enabled") or (self.launch_id and clock.get("daemon", {}).get("id") != self.launch_id):
                return {"status": "disabled", "delivery": delivery}
            if self.launch_id:
                clock["daemon"].update(heartbeat_at=now, status="running")
                clock["daemon"].pop("error", None)
            if delivery:
                return {**delivery, "observation": observation}
            _arm(clock)
            item = clock.get("reminder")
            next_send_at = _next_send_at(item)
            if (self.inflight or self._sending(clock) or not _current(clock, item, now) or
                    next_send_at is None or now < next_send_at):
                return {"status": "waiting", "observation": observation}
            attempts = item["attempts"]
            prefix = self.runtime_directory / (item["id"] + "-" + str(len(attempts) + 1))
            attempts.append({"number": len(attempts) + 1, "started_at": now, "status": "sending",
                             "owner": owner, "launch_id": self.launch_id,
                             "stdout": str(prefix) + ".stdout", "stderr": str(prefix) + ".stderr"})
            item["status"] = "sending"
            clock["queue_attempts_total"] = clock.get("queue_attempts_total", 0) + 1
            item, sent_clock = deepcopy(item), deepcopy(clock)
        current = _clock(self.store.read_supervisor())
        current_item = current.get("reminder")
        if (not _matches(current, _binding(sent_clock)) or not _current(current, current_item, self.now()) or
                current_item["id"] != item["id"] or current_item.get("ack")):
            self._record_send(item, len(item["attempts"]), {"status": "not_started", "finished_at": self.now()})
            return {"status": "invalidated", "id": item["id"]}
        try:
            self._send(sent_clock, item)
            self._record_send(item, len(item["attempts"]), self.inflight["facts"])
        except SupervisorBusy:
            raise
        except Exception as exc:
            if self.inflight is None:
                self.inflight = {"item": item, "number": len(item["attempts"]), "facts": {},
                                 "outcome": {"status": "uncertain", "error": str(exc), "finished_at": self.now()}}
            else:
                raise
        return {**self._poll_send(), "observation": observation}

    def run(self, launch_id: str, lock_fd=None) -> int:
        self.store.initialize()
        lock = os.fdopen(lock_fd, "a+") if lock_fd is not None else self.lock_path.open("a+")
        with lock:
            actual, expected = os.fstat(lock.fileno()), self.lock_path.stat()
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise ExtractError("inherited clock lock does not match this state root")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 1
            identity = process_identity(os.getpid())
            if identity is None:
                raise ExtractError("cannot establish clock daemon identity")
            with self.store.supervisor() as ledger:
                clock = _clock(ledger)
                if not clock.get("enabled") or clock.get("daemon", {}).get("id") != launch_id or clock["last_check_number"] < 0:
                    return 0
                self.launch_id = launch_id
                clock["daemon"].update(identity=identity, pid=os.getpid(), status="running",
                                       started_at=self.now(), heartbeat_at=self.now())
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda *_: self.stopping.set())
            try:
                while not self.stopping.is_set():
                    try:
                        if self.tick()["status"] == "disabled":
                            break
                    except Exception as exc:
                        try:
                            with self.store.supervisor() as ledger:
                                daemon = _clock(ledger).get("daemon", {})
                                if daemon.get("id") == launch_id:
                                    daemon.update(status="clock_error", error=str(exc))
                        except SupervisorBusy:
                            pass
                    clock = _clock(self.store.read_supervisor())
                    due_times = [clock.get("next_observation_at", self.now() + OBSERVATION_SECONDS)]
                    next_send_at = _next_send_at(clock.get("reminder"))
                    if next_send_at is not None:
                        due_times.append(next_send_at)
                    delay = min([HEARTBEAT_SECONDS, *(max(0.05, due - self.now()) for due in due_times)])
                    self.stopping.wait(delay)
            finally:
                cleanup = None
                if self.inflight and self.inflight["facts"].get("identity"):
                    cleanup = terminate_owned(self.inflight["facts"]["identity"],
                                              self.inflight["facts"].get("descendants", []), include_anchor=True)
                    try:
                        self._poll_send()
                    except (SupervisorBusy, OSError):
                        pass
                try:
                    with self.store.supervisor() as ledger:
                        daemon = _clock(ledger).get("daemon", {})
                        if daemon.get("id") == launch_id:
                            daemon.update(status="stopped", stopped_at=self.now())
                            if cleanup is not None:
                                daemon["queue_cleanup"] = cleanup
                except SupervisorBusy:
                    pass
        return 0

    def stop(self) -> dict:
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            if clock.get("enabled"):
                clock.update(enabled=False, generation=clock["generation"] + 1, stopped_at=self.now())
                clock.pop("observation_claim", None)
                _archive(clock, clock.pop("reminder", None))
                ledger["clock"] = clock
            daemon = deepcopy(clock.get("daemon", {}))
            sends = deepcopy(self._sending(clock))
        # Invalidation commits before any signal, and never controls Extract.
        outcomes = []
        for item, attempt in sends:
            pending = self.inflight
            identity = attempt.get("identity")
            known = attempt.get("descendants", [])
            if pending and pending["item"]["id"] == item["id"] and pending["number"] == attempt["number"]:
                identity = pending["facts"].get("identity", identity)
                known = pending["facts"].get("descendants", known)
            try:
                outcome = (terminate_owned(identity, known, include_anchor=True) if identity else
                           {"confirmed": False, "actions": [], "survivors": [], "reason": "queue identity unconfirmed"})
            except OSError as exc:
                outcome = {"confirmed": False, "actions": [], "survivors": [], "reason": str(exc)}
            try:
                self._record_send(item, attempt["number"], {"status": "sending", "stop_requested": True, "termination": outcome,
                                                          "descendants": outcome.get("descendants", [])})
            except SupervisorBusy:
                outcome.update(confirmed=False, evidence_committed=False)
            outcomes.append(outcome)
        identity = daemon.get("identity")
        if identity:
            try:
                outcomes.append(terminate_owned(identity, include_anchor=True))
            except OSError as exc:
                outcomes.append({"confirmed": False, "reason": str(exc)})
        elif daemon.get("status") not in {None, "stopped", "launch_failed"}:
            outcomes.append({"confirmed": False, "reason": "clock identity unconfirmed"})
        self.stopping.set()
        result = self.status()
        confirmed = all(item["confirmed"] for item in outcomes) and result["process_lock"] in {"free", "absent"}
        result.update(committed=True, termination={"confirmed": confirmed, "targets": outcomes},
                      stop_status="confirmed" if confirmed else "pending")
        return result

    def status(self) -> dict:
        clock = deepcopy(_clock(self.store.read_supervisor()))
        now = self.now()
        daemon = clock.get("daemon", {})
        alive, lock = _alive(daemon.get("identity")), self._lock_status()
        heartbeat = daemon.get("heartbeat_at")
        heartbeat_age = max(0, now - heartbeat) if heartbeat is not None else None
        if alive is True and lock == "held":
            process_status = "starting" if heartbeat is None else "running"
            if heartbeat_age is not None and heartbeat_age > HEARTBEAT_STALE_SECONDS:
                process_status = "heartbeat_stale"
            if daemon.get("status") == "clock_error":
                process_status = "clock_error"
        elif alive is True:
            process_status = "identity_lock_mismatch"
        elif lock in {"free", "absent"} and (alive is False or daemon.get("status") in {None, "stopped", "launch_failed"}):
            process_status = "stopped_requires_start" if clock.get("enabled") else "stopped"
        else:
            process_status = "unknown"
        if not clock.get("enabled") and lock == "held":
            process_status = "stopping"
        reasons = []
        item = clock.get("reminder")
        if item:
            attempts = item["attempts"]
            queued_at = [attempt["finished_at"] for attempt in attempts if attempt.get("status") == "queued"]
            item["attempt_count"] = len(attempts)
            item["attempts_remaining"] = max(0, MAX_QUEUE_ATTEMPTS - len(attempts))
            item["next_attempt_at"] = _next_send_at(item)
            item["first_queued_at"] = min(queued_at) if queued_at else None
            item["last_queued_at"] = max(queued_at) if queued_at else None
            item["ack_age_seconds"] = max(0, now - item["ack"]["at"]) if item.get("ack") else None
            item["overdue_seconds"] = max(0, now - item["due_at"])
            if item.get("ack"):
                item["delivery"] = "acknowledged"
                item["status"] = "acknowledged_unfinished"
                if now - item["ack"]["at"] >= ATTENTION_SECONDS:
                    reasons.append("acknowledged_unfinished")
            elif queued_at:
                item["delivery"] = "queued_unacknowledged"
                if now - item["first_queued_at"] >= ATTENTION_SECONDS:
                    reasons.append("queued_without_ack")
            else:
                item["delivery"] = ("sending" if any(attempt["status"] == "sending" for attempt in attempts) else
                                    "uncertain" if attempts else "not_sent")
            if (not item.get("ack") and len(attempts) >= MAX_QUEUE_ATTEMPTS and
                    not any(attempt["status"] == "sending" for attempt in attempts)):
                reasons.append("queue_attempts_exhausted")
        if clock.get("enabled") and process_status not in {"running", "starting"}:
            reasons.append(process_status)
        clock["history_count"] = len(clock.pop("history", []))
        clock["evidence"] = str(self.store.root / "supervisor.json")
        clock.update(process_alive=alive, process_lock=lock, process_status=process_status,
                     heartbeat_age_seconds=heartbeat_age, needs_attention=bool(reasons), attention_reasons=reasons,
                     observed_at=now, overdue_seconds=max(0, now - clock["next_due_at"]) if clock.get("next_due_at") is not None else None,
                     restart_instruction="主机重启或时钟退出后显式 monitor start 恢复原进度；入队回执不证明 app-server 仍存活或该线程已确认。")
        if process_status in {"identity_lock_mismatch", "unknown"}:
            clock["restart_instruction"] = "进程身份与锁证据不足或不一致；先核对原进程和锁，不能据此另起实例。"
        return clock
