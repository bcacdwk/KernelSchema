"""A persistent clock for one existing Supervisor thread, not a run scheduler."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
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

from .common import ExtractError, digest, rules_root
from .processes import process_table
from . import monitor


CLOCK_FORMAT = "schema_extract.clock.v1"
EARLY_SECONDS = 300
REGULAR_SECONDS = 1800
RETRY_SECONDS = 120
MAX_QUEUE_ATTEMPTS = 3
QUEUE_TIMEOUT_SECONDS = 30
HEARTBEAT_SECONDS = 15
ATTENTION_SECONDS = 1800
PROC_IDENTITIES = Path("/proc/sys/kernel/random/boot_id").is_file()


def _uuid(value: str, label="thread") -> str:
    try:
        if str(UUID(value)) == value:
            return value
    except (ValueError, TypeError, AttributeError):
        pass
    raise ExtractError(label + " must be a canonical UUID")


def process_identity(pid=None, *, table=None) -> dict | None:
    """Prefer Linux boot/start ticks; reuse the existing ps identity on Unix."""
    pid = os.getpid() if pid is None else pid
    if not PROC_IDENTITIES:
        if table is False:
            return None
        table = process_table() if table is None else table
        current = table.get(pid)
        if not current or current["stat"].startswith("Z"):
            return None
        return {"pid": pid, "started": current["started"]}
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except FileNotFoundError:
        return None


def process_alive(identity, *, table=None) -> bool | None:
    if (not isinstance(identity, dict) or type(identity.get("pid")) is not int or
            not ((identity.get("start_ticks") and identity.get("boot_id")) or identity.get("started")) or table is False):
        return None
    try:
        return process_identity(identity["pid"], table=table) == identity
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _control_processes():
    """Collect a Unix fallback before entering any control transaction."""
    if PROC_IDENTITIES:
        return None
    try:
        return process_table()
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


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


def _item(clock: dict, kind: str, number: int, due: float) -> dict:
    identity = {**_binding(clock), "kind": kind, "number": number, "due_at": due}
    return {**identity, "id": f"{kind}-{clock['generation']}-{number}-{digest(identity)[:24]}",
            "status": "scheduled", "attempts": [], "ack": None}


def _archive(clock: dict, item: dict | None) -> None:
    if item:
        clock["last_" + item["kind"]] = deepcopy(item)
        clock["history"] = (clock.get("history", []) + [deepcopy(item)])[-64:]


def _arm(clock: dict) -> None:
    if clock.get("enabled") and clock["last_check_number"] >= 0 and not clock.get("reminder"):
        clock["reminder"] = _item(clock, "check", clock["last_check_number"] + 1, clock["next_due_at"])


def _current(clock: dict, item: dict | None, now: float) -> bool:
    if not item or not _matches(clock, _binding(item)) or now < item["due_at"]:
        return False
    if item["kind"] == "check":
        return (item["number"] == clock["last_check_number"] + 1 and item["due_at"] == clock["next_due_at"] and
                item["id"] == _item(clock, "check", item["number"], item["due_at"])["id"])
    return item["kind"] == "probe" and item["id"] == _item(clock, "probe", 0, item["due_at"])["id"]


class QueueError(ExtractError):
    pass


class Clock:
    def __init__(self, store, repo: Path, *, now=None, sender=None):
        self.store, self.repo = store, Path(repo).resolve()
        self.now = now or time.time
        self.sender = sender
        self.stopping = threading.Event()
        self.launch_id = None

    @property
    def lock_path(self) -> Path:
        return self.store.root / "monitor-clock.lock"

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

    def start(self, thread: str, run_ids=None, *, auto_restart_limit=None, codex_binary=None,
              probe_delay=None, launch=True) -> dict:
        """The foreground start performs scan #0; the detached child only ticks.

        launch=False is an in-process test seam, never a CLI timer option.
        """
        thread = _uuid(thread)
        if probe_delay is not None and (type(probe_delay) is not int or not 1 <= probe_delay <= 300):
            raise ExtractError("probe delay must be an integer from 1 to 300 seconds")
        if auto_restart_limit is not None and (type(auto_restart_limit) is not int or auto_restart_limit < 0):
            raise ExtractError("auto-restart limit must be a nonnegative integer")
        if run_ids is not None:
            run_ids = sorted(set(run_ids))
            for run_id in run_ids:
                self.store.run_dir(run_id)
        self.store.initialize()
        table = _control_processes()
        identity = process_identity(table=table)
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
                if (process_alive(daemon.get("identity"), table=table) is not True and
                        not (daemon.get("status") == "starting" and process_alive(daemon.get("launcher"), table=table) is True)):
                    raise ExtractError("clock lock is held but its live owner is unconfirmed; inspect monitor status")
                if probe_delay is not None:
                    with self.store.supervisor() as ledger:
                        current = _clock(ledger)
                        self._check_start(current, thread, run_ids, codex_binary)
                        if not _matches(current, binding):
                            raise ExtractError("clock binding changed while start was being checked")
                        if not current.get("probe"):
                            current["probe"] = _item(current, "probe", 0, self.now() + probe_delay)
                if auto_restart_limit is not None:
                    selected = current.get("run_ids")
                    monitor.configure_auto_restart(self.store, selected if selected is not None else self.store.run_ids(),
                                                   auto_restart_limit, clock_binding=binding)
                result = self.status()
                if not _matches(result, binding):
                    return {**result, "reused": False, "started": False, "reason": "start_invalidated"}
                return {**result, "reused": True}
            with self.store.supervisor() as ledger:
                clock = _clock(ledger)
                self._check_start(clock, thread, run_ids, codex_binary)
                daemon = clock.get("daemon", {})
                if daemon.get("status") not in {"stopped", "launch_failed"} and process_alive(daemon.get("identity"), table=table) is True:
                    raise ExtractError("recorded clock process is live without its lock; inspect monitor status")
                if not clock.get("enabled"):
                    _archive(clock, clock.pop("reminder", None))
                    _archive(clock, clock.pop("probe", None))
                    clock.update(enabled=True, generation=clock["generation"] + 1, thread_id=thread,
                                 run_ids=run_ids, verified=False, verified_at=None, started_at=self.now())
                clock.setdefault("run_ids", run_ids)
                clock.update(repo=str(self.repo), entrypoint=str(rules_root() / "schema_extract/extract.py"),
                             python=sys.executable, codex_binary=codex_binary or clock.get("codex_binary", "codex"))
                if probe_delay is not None and not clock.get("probe"):
                    clock["probe"] = _item(clock, "probe", 0, self.now() + probe_delay)
                launch_id = str(uuid4())
                clock["daemon"] = {"id": launch_id, "status": "starting", "launcher": identity,
                                   "created_at": self.now(), "identity": None, "heartbeat_at": None}
                _arm(clock)
                ledger["clock"] = clock
                binding, selected = _binding(clock), clock["run_ids"]
                initial = clock["last_check_number"] < 0
            if auto_restart_limit is not None:
                monitor.configure_auto_restart(self.store, selected if selected is not None else self.store.run_ids(),
                                               auto_restart_limit, clock_binding=binding)
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
                    process = subprocess.Popen(command, cwd=self.repo, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                               close_fds=True, pass_fds=(lock.fileno(),), start_new_session=True)
                child_identity = process_identity(process.pid)
                with self.store.supervisor() as ledger:
                    current = _clock(ledger)
                    daemon = current.get("daemon", {})
                    if daemon.get("id") == launch_id and daemon["status"] == "starting":
                        daemon.update(pid=process.pid, identity=child_identity, log=str(log_path))
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
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            if not _matches(clock, binding) or clock["last_check_number"] >= 0:
                return
            now = self.now()
            result = monitor.record_scan(ledger, rows, now)
            check = self._complete(clock, result, 0, f"initial-{clock['generation']}", started, now)
        monitor.authorized_restarts(self.store, rows, self.repo, clock_binding={**binding, "check_id": check["id"]})

    @staticmethod
    def _complete(clock, result, number, wake_id, started, now) -> dict:
        item = clock.pop("reminder", None)
        check = {"id": wake_id, "number": number, **_binding(clock), "started_at": started,
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
        now = self.now()
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            if not clock.get("enabled"):
                return {"accepted": False, "reason": "disabled"}
            if thread != clock["thread_id"]:
                return {"accepted": False, "reason": "wrong_thread"}
            item = next((item for item in (clock.get("reminder"), clock.get("probe")) if item and item["id"] == wake_id), None)
            if not _current(clock, item, now):
                return {"accepted": False, "reason": "stale_or_not_due"}
            if item.get("ack"):
                return {"accepted": False, "reason": "already_acknowledged", "unfinished": item["kind"] == "check"}
            if not item["attempts"]:
                return {"accepted": False, "reason": "not_sent"}
            item.update(status="acknowledged", ack={"thread_id": thread, "at": now})
            clock.update(verified=True, verified_at=now)
            return {"accepted": True, "id": wake_id, "kind": item["kind"], "number": item["number"],
                    "ack": deepcopy(item["ack"]), "scan_required": item["kind"] == "check"}

    def scan(self, wake_id=None, thread=None) -> dict:
        """Only an explicit thread can finish that thread's accepted reminder."""
        thread = _uuid(thread)
        table = _control_processes()
        identity = process_identity(table=table)
        if identity is None:
            raise ExtractError("cannot establish scan process identity")
        with self.store.supervisor() as ledger:
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
            if previous and process_alive(previous.get("owner"), table=table) is not False:
                return {"runs": [], "changes": [], "anomalies": [], "clock": {"committed": False,
                        "status": "scan_busy", "id": item["id"], "owner": previous.get("owner")}}
            claim = {"id": str(uuid4()), "owner": identity, "started_at": self.now()}
            item["scan"] = claim
            wake_id, binding, selected = item["id"], _binding(clock), clock.get("run_ids")
        try:
            rows = monitor.inspect_runs(self.store, selected)
            with self.store.supervisor() as ledger:
                clock = _clock(ledger)
                item = clock.get("reminder")
                if (not _matches(clock, binding) or not item or item["id"] != wake_id or
                        item.get("scan", {}).get("id") != claim["id"]):
                    return {"runs": rows, "changes": [], "anomalies": [],
                            "clock": {"committed": False, "status": "scan_invalidated", "id": wake_id}}
                now = self.now()
                result = monitor.record_scan(ledger, rows, now)
                check = self._complete(clock, result, item["number"], wake_id, claim["started_at"], now)
                result["clock"] = {"committed": True, "check": deepcopy(check), "next_due_at": clock["next_due_at"]}
        except Exception as exc:
            with self.store.supervisor() as ledger:
                clock = _clock(ledger)
                item = clock.get("reminder")
                if item and item["id"] == wake_id and item.get("scan", {}).get("id") == claim["id"]:
                    item.pop("scan", None)
                    item["scan_error"] = str(exc)
            raise
        result["automatic_restarts"] = monitor.authorized_restarts(
            self.store, result["runs"], self.repo, clock_binding={**binding, "check_id": check["id"]})
        return result

    def message(self, clock: dict, item: dict) -> str:
        entry = [clock["python"], "-B", clock["entrypoint"], "monitor"]
        where = ["--repo", clock["repo"], "--state-root", str(self.store.root)]
        ack = shlex.join([*entry, "ack", item["id"], "--thread", clock["thread_id"], *where])
        due = datetime.fromtimestamp(item["due_at"], timezone.utc).isoformat()
        prefix = f"Schema Extract 同线程提醒 {item['id']}，线程 {clock['thread_id']}，到期 {due}。先执行：{ack}。"
        if item["kind"] == "probe":
            return (prefix + "accepted=false 时结束此提醒；accepted=true 时用 monitor status 核对该线程确认，简短报告后结束本轮。"
                    "这是一次唤醒验证，不执行巡检或控制提取，不改变正式 300/1800 秒周期。不要另建线程、闹钟或在对话中 sleep。")
        scan = shlex.join([*entry, "scan", "--wake", item["id"], "--thread", clock["thread_id"], *where])
        manual = Path(clock["entrypoint"]).with_name("README.md")
        return (prefix + f"accepted=false 时结束此提醒；accepted=true 时执行一次：{scan}。"
                f"按[巡检手册]({manual})查看各算子的阶段、在途调用和异常；正常简报即可。"
                "需框架修复时按手册先 hold 并核清范围，再按已有授权交工程处理。"
                "巡检完成并登记 next_due_at 后结束本次提醒；尊重用户最新暂停或讨论，继续处理其尚未结束的已授权事项。"
                "ack 后若中断，同线程显式执行这条 scan 续完，不重复 ack。"
                "后台只负责下一次提醒；不要另建线程、闹钟或在对话中 sleep，已有提取授权和预算保持原边界。")

    def _send(self, thread: str, message: str, binary: str) -> str:
        command = [binary, "queue", "--thread", thread, "--message", message, "--resume"]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            deadline = time.monotonic() + QUEUE_TIMEOUT_SECONDS
            while True:
                if self.stopping.is_set() or time.monotonic() >= deadline:
                    raise QueueError("queue stopped or timed out; delivery is uncertain")
                try:
                    stdout, stderr = process.communicate(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # A detached descendant can retain stdout after the queue parent
            # exits. Reap only our parent with a bound; never wait for pipe EOF
            # in cleanup or Popen.__exit__, and do not chase other groups.
            try:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    if process.poll() is not None:
                        break
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        match = re.search(r"Queued message ([0-9a-f-]{36}) for thread ([0-9a-f-]{36})", stdout)
        if process.returncode or not match or match[2] != thread:
            raise QueueError(f"queue did not confirm the bound thread and message ID (exit={process.returncode}); "
                             + (stderr or stdout)[-500:])
        return _uuid(match[1], "queue message")

    def tick(self) -> dict:
        """Reserve one send, release the control lock, then call codex queue."""
        now = self.now()
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            if not clock.get("enabled") or (self.launch_id and clock.get("daemon", {}).get("id") != self.launch_id):
                return {"status": "disabled"}
            if self.launch_id:
                clock["daemon"]["heartbeat_at"] = now
            _arm(clock)
            selected = None
            for key in ("probe", "reminder"):
                item = clock.get(key)
                if not _current(clock, item, now) or item.get("ack"):
                    continue
                attempts = item["attempts"]
                if any(attempt.get("status") == "queued" for attempt in attempts) or len(attempts) >= MAX_QUEUE_ATTEMPTS:
                    continue
                if attempts and now < max(attempts[-1]["started_at"], attempts[-1].get("finished_at", 0)) + RETRY_SECONDS:
                    continue
                attempt = {"number": len(attempts) + 1, "started_at": now, "status": "sending"}
                attempts.append(attempt)
                item["status"] = "sending"
                clock["queue_attempts_total"] = clock.get("queue_attempts_total", 0) + 1
                selected = key, deepcopy(item), deepcopy(clock)
                break
            if selected is None:
                if self.launch_id:
                    clock["daemon"]["status"] = "running"
                    clock["daemon"].pop("error", None)
                return {"status": "waiting"}
        key, item, sent_clock = selected
        try:
            # Stop may win after reservation; never manufacture another ID.
            current = _clock(self.store.read_supervisor())
            if not _matches(current, _binding(sent_clock)):
                return {"status": "invalidated", "id": item["id"]}
            message = self.message(sent_clock, item)
            message_id = _uuid(self.sender(sent_clock["thread_id"], message) if self.sender else
                               self._send(sent_clock["thread_id"], message, sent_clock["codex_binary"]), "queue message")
            outcome = {"status": "queued", "message_id": message_id, "finished_at": self.now()}
        except Exception as exc:
            outcome = {"status": "uncertain", "error": str(exc), "finished_at": self.now()}
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            current = clock.get(key)
            # A fast receiver may already have acked, completed, or stopped.
            # Supplement only this attempt's facts, including its archived
            # copies; never replace a newer reminder or undo an acceptance.
            records = [current, clock.get("last_" + item["kind"]), *clock.get("history", [])]
            for record in records:
                if not record or record["id"] != item["id"]:
                    continue
                attempt = next((attempt for attempt in record["attempts"]
                                if attempt["number"] == len(item["attempts"])), None)
                if attempt is not None:
                    attempt.update(outcome)
                if not record.get("ack") and record["status"] == "sending":
                    record["status"] = "queued" if outcome["status"] == "queued" else "retry_pending"
                if outcome.get("message_id") and record.get("scan", {}).get("completed_at") is not None:
                    ids = record["scan"].setdefault("message_ids", [])
                    if outcome["message_id"] not in ids:
                        ids.append(outcome["message_id"])
            last = clock.get("last_scan", {})
            if last.get("id") == item["id"] and outcome.get("message_id") not in last.get("message_ids", []):
                if outcome.get("message_id"):
                    last.setdefault("message_ids", []).append(outcome["message_id"])
            if self.launch_id and clock.get("daemon", {}).get("id") == self.launch_id:
                clock["daemon"].update(heartbeat_at=self.now(), status="running")
                clock["daemon"].pop("error", None)
        return {"id": item["id"], **outcome}

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
            identity = process_identity()
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
                        with self.store.supervisor() as ledger:
                            daemon = _clock(ledger).get("daemon", {})
                            if daemon.get("id") == launch_id:
                                daemon.update(status="clock_error", error=str(exc))
                    clock = _clock(self.store.read_supervisor())
                    deadlines = []
                    for item in (clock.get("probe"), clock.get("reminder")):
                        if not item or item.get("ack") or len(item["attempts"]) >= MAX_QUEUE_ATTEMPTS:
                            continue
                        attempts = item["attempts"]
                        if any(attempt.get("status") == "queued" for attempt in attempts):
                            continue
                        deadlines.append(max(attempts[-1]["started_at"], attempts[-1].get("finished_at", 0)) + RETRY_SECONDS
                                         if attempts else item["due_at"])
                    delay = min([HEARTBEAT_SECONDS, *(max(0.05, due - self.now()) for due in deadlines)])
                    self.stopping.wait(delay)
            finally:
                with self.store.supervisor() as ledger:
                    daemon = _clock(ledger).get("daemon", {})
                    if daemon.get("id") == launch_id:
                        daemon.update(status="stopped", stopped_at=self.now())
        return 0

    def stop(self) -> dict:
        with self.store.supervisor() as ledger:
            clock = _clock(ledger)
            if clock.get("enabled"):
                clock.update(enabled=False, generation=clock["generation"] + 1, verified=False, stopped_at=self.now())
                _archive(clock, clock.pop("reminder", None))
                _archive(clock, clock.pop("probe", None))
                ledger["clock"] = clock
            identity = deepcopy(clock.get("daemon", {}).get("identity"))
        # No control lock is held while signalling or waiting for the clock.
        if process_alive(identity) is True:
            for sig, grace in ((signal.SIGTERM, 2), (signal.SIGKILL, 1)):
                if process_alive(identity) is not True:
                    break
                try:
                    os.kill(identity["pid"], sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + grace
                while time.monotonic() < deadline and process_alive(identity) is True:
                    time.sleep(0.05)
        return self.status()

    def status(self) -> dict:
        clock = deepcopy(_clock(self.store.read_supervisor()))
        now = self.now()
        daemon = clock.get("daemon", {})
        alive, lock = process_alive(daemon.get("identity")), self._lock_status()
        heartbeat = daemon.get("heartbeat_at")
        heartbeat_age = max(0, now - heartbeat) if heartbeat is not None else None
        if alive is True and lock == "held":
            process_status = "starting" if heartbeat is None else "running"
            if heartbeat_age is not None and heartbeat_age > HEARTBEAT_SECONDS * 3 + QUEUE_TIMEOUT_SECONDS:
                process_status = "heartbeat_stale"
            if daemon.get("status") == "clock_error":
                process_status = "clock_error"
        elif alive is True:
            process_status = "identity_lock_mismatch"
        elif lock in {"free", "absent"} and (alive is False or not daemon.get("identity")):
            process_status = "stopped_requires_start" if clock.get("enabled") else "stopped"
        else:
            process_status = "unknown"
        if not clock.get("enabled") and lock == "held":
            process_status = "stopping"
        reasons = []
        for key in ("reminder", "probe"):
            item = clock.get(key)
            if not item:
                continue
            attempts = item["attempts"]
            queued = next((attempt for attempt in attempts if attempt.get("status") == "queued"), None)
            item["attempt_count"] = len(attempts)
            item["overdue_seconds"] = max(0, now - item["due_at"])
            if item.get("ack"):
                item["delivery"] = "acknowledged"
                item["status"] = "acknowledged_unfinished" if key == "reminder" else "verified"
                if key == "reminder" and now - item["ack"]["at"] >= ATTENTION_SECONDS:
                    reasons.append("acknowledged_unfinished")
            elif queued:
                item["delivery"] = "queued_unacknowledged"
                if now - queued["finished_at"] >= ATTENTION_SECONDS:
                    reasons.append("queued_without_ack")
            else:
                item["delivery"] = "uncertain" if attempts else "not_sent"
                if len(attempts) >= MAX_QUEUE_ATTEMPTS:
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
