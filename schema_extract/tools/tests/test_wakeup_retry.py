"""Reminder delivery uses isolated state, a virtual clock and fake processes."""

import fcntl
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from schema_extract.tools import audit, monitor, wakeup
from schema_extract.tools.store import Store
from schema_extract.tools.common import SupervisorBusy


class ReminderFixture:
    def __init__(self, root, outcomes):
        self.store = Store(root)
        self.now = 1000.0
        self.thread = str(uuid4())
        self.outcomes = iter(outcomes)
        self.sent = []
        self.on_send = None
        self.restart()

    def restart(self):
        self.clock = wakeup.Clock(self.store, self.store.root, now=lambda: self.now, sender=self.send)
        self.clock.start(self.thread, [], launch=False)

    def send(self, thread, message):
        self.sent.append((thread, message))
        self.now += 2  # Queue completion, rather than its start, sets the next delay.
        if self.on_send:
            action, self.on_send = self.on_send, None
            action()
        if next(self.outcomes) == "uncertain":
            raise wakeup.QueueError("fixture: delivery uncertain")
        return str(uuid4())

    def reminder(self):
        return self.clock.status()["reminder"]

    def first_send(self):
        self.now = self.reminder()["due_at"]
        self.wake_id = self.reminder()["id"]
        self.clock.tick()


class WakeupRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target, value in (("process_identity", {"pid": 123456789, "start_token": "linux:fixture:1", "stat": "S", "ppid": 1, "pgid": 123456789}),
                              ("_alive", False)):
            patcher = patch.object(wakeup, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target in ("inspect_runs",):
            patcher = patch.object(monitor, target, return_value=[])
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(wakeup.Clock, "observe", return_value={"status": "observed"})
        self.observation = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(wakeup.subprocess, "Popen", side_effect=AssertionError("real queue forbidden"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_latest_attempt_sets_delay_and_all_outcomes_share_persistent_limit(self):
        for second in ("queued", "uncertain"):
            with self.subTest(second=second):
                fixture = ReminderFixture(self.root / second, ["queued", second, "queued"])
                fixture.first_send()
                first = fixture.now
                item = fixture.reminder()
                self.assertEqual((item["first_queued_at"], item["last_queued_at"]), (first, first))
                self.assertEqual((item["next_attempt_at"], item["attempts_remaining"]), (first + 1800, 2))
                fixture.now = first + 1799
                self.assertEqual(fixture.clock.tick()["status"], "waiting")

                # Run one in-process iteration with fake waiting; no daemon or real sleep.
                class OneWait:
                    stopped = False
                    delays = []

                    def is_set(self):
                        return self.stopped

                    def wait(self, delay):
                        self.delays.append(delay)
                        self.stopped = True

                fixture.clock.stopping = OneWait()
                with fixture.store.supervisor() as ledger:
                    ledger["clock"]["next_observation_at"] = fixture.now + 30
                    launch_id = ledger["clock"]["daemon"]["id"]
                with patch.object(wakeup.signal, "signal"):
                    self.assertEqual(fixture.clock.run(launch_id), 0)
                self.assertEqual(fixture.clock.stopping.delays, [1])
                fixture.restart()
                self.assertEqual(fixture.reminder()["attempt_count"], 1)
                fixture.now = first + 1800
                self.assertEqual(fixture.clock.tick()["status"], second)
                next_send = fixture.now + (1800 if second == "queued" else 120)
                self.assertEqual(fixture.reminder()["next_attempt_at"], next_send)
                fixture.now = next_send - 1
                self.assertEqual(fixture.clock.tick()["status"], "waiting")
                fixture.now = next_send
                self.assertEqual(fixture.clock.tick()["status"], "queued")
                item = fixture.reminder()
                self.assertEqual((item["first_queued_at"], item["last_queued_at"]), (first, fixture.now))
                self.assertEqual((item["attempt_count"], item["attempts_remaining"], item["next_attempt_at"]),
                                 (3, 0, None))
                self.assertEqual(item["delivery"], "queued_unacknowledged")
                self.assertIn("queue_attempts_exhausted", fixture.clock.status()["attention_reasons"])
                fixture.restart()
                fixture.now += 3600
                observations = self.observation.call_count
                self.assertEqual(fixture.clock.tick()["status"], "waiting")
                self.assertEqual(self.observation.call_count, observations + 1)
                self.assertEqual(len(fixture.sent), 3)
                self.assertEqual(fixture.reminder()["id"], fixture.wake_id)
                self.assertEqual(fixture.clock.status()["last_check_number"], 0)
                self.assertTrue(all(thread == fixture.thread and fixture.wake_id in message
                                    for thread, message in fixture.sent))

    def test_ack_scan_and_stop_win_before_send_and_before_late_receipts(self):
        fixture = ReminderFixture(self.root / "preflight", ["queued"])
        fixture.first_send()
        fixture.now = fixture.reminder()["next_attempt_at"]
        read = fixture.store.read_supervisor
        reads = 0

        def ack_before_send():
            nonlocal reads
            reads += 1
            if reads == 4:  # Heartbeat, pending lookup and reservation precede the unlocked preflight.
                self.assertTrue(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
            return read()

        with patch.object(fixture.store, "read_supervisor", side_effect=ack_before_send):
            self.assertEqual(fixture.clock.tick()["status"], "invalidated")
        self.assertEqual(len(fixture.sent), 1)
        fixture.now += 1800
        self.assertEqual(fixture.clock.tick()["status"], "waiting")
        item = fixture.reminder()
        self.assertEqual((item["status"], item["ack_age_seconds"], item["next_attempt_at"]),
                         ("acknowledged_unfinished", 1800, None))
        self.assertIn("acknowledged_unfinished", fixture.clock.status()["attention_reasons"])
        self.assertTrue(fixture.clock.scan(fixture.wake_id, fixture.thread)["clock"]["committed"])
        due = fixture.reminder()["due_at"]
        self.assertTrue(fixture.clock.scan(fixture.wake_id, fixture.thread)["clock"]["reused"])
        self.assertEqual((fixture.clock.status()["last_check_number"], fixture.reminder()["due_at"]), (1, due))

        for action in ("scan", "stop"):
            with self.subTest(late_receipt=action):
                fixture = ReminderFixture(self.root / action, ["queued", "queued", "queued"])
                fixture.first_send()
                generation = fixture.clock.status()["generation"]
                fixture.now = fixture.reminder()["next_attempt_at"]
                if action == "scan":
                    def finish():
                        self.assertTrue(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
                        self.assertTrue(fixture.clock.scan(fixture.wake_id, fixture.thread)["clock"]["committed"])
                    fixture.on_send = finish
                else:
                    fixture.clock.launch_id = fixture.clock.status()["daemon"]["id"]
                    fixture.on_send = fixture.clock.stop
                result = fixture.clock.tick()
                self.assertEqual(result["status"], "queued")
                clock = fixture.store.read_supervisor()["clock"]
                record = clock["history"][-1]
                self.assertEqual(record["id"], fixture.wake_id)
                self.assertEqual(record["attempts"][-1]["message_id"], result["message_id"])
                observed = next(item for item in audit._checks({"clock": clock})["reminder_timing"]
                                if item["id"] == fixture.wake_id)
                self.assertEqual(observed["thread_id"], fixture.thread)
                self.assertEqual(observed["attempts"][-1]["message_id"], result["message_id"])
                if action == "scan":
                    self.assertEqual(clock["last_check_number"], 1)
                    self.assertIn(result["message_id"], clock["last_scan"]["message_ids"])
                    self.assertTrue(fixture.clock.scan(fixture.wake_id, fixture.thread)["clock"]["reused"])
                else:
                    self.assertFalse(clock["enabled"])
                    self.assertEqual(clock["generation"], generation + 1)
                    self.assertEqual(fixture.clock.tick()["status"], "disabled")
                    self.assertFalse(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
                    fixture.restart()
                    self.assertFalse(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
                    self.assertNotEqual(fixture.reminder()["id"], fixture.wake_id)
                    self.assertEqual(fixture.clock.tick()["status"], "queued")
                    self.assertNotIn(fixture.wake_id, fixture.sent[-1][1])

    def fake_queue(self, fixture):
        fixture.clock.sender = None
        owner = {"pid": 87654321, "start_token": "linux:fixture:2", "ppid": 1, "pgid": 87654321, "stat": "S"}

        class Process:
            pid = owner["pid"]
            returncode = None
            options = None

            def poll(self):
                return self.returncode

        process = Process()

        def start(command, **options):
            self.assertNotIn("timeout", options)
            self.assertNotEqual(options["stdout"], wakeup.subprocess.PIPE)
            self.assertNotEqual(options["stderr"], wakeup.subprocess.PIPE)
            process.options = options
            return process

        def observed(*_args, **_kwargs):
            return {"confirmed": process.returncode is not None, "actions": [],
                    "survivors": [owner] if process.returncode is None else [], "descendants": [owner]}

        for mocker in (patch.object(wakeup.subprocess, "Popen", side_effect=start),
                       patch.object(wakeup, "child_exited", side_effect=lambda value: value.returncode is not None),
                       patch.object(wakeup, "process_identity", return_value=owner),
                       patch.object(wakeup, "observe_owned", side_effect=observed)):
            mocker.start()
            self.addCleanup(mocker.stop)
        return process, owner

    def test_live_queue_never_retries_and_heartbeat_observation_continue(self):
        fixture = ReminderFixture(self.root / "live", [])
        process, owner = self.fake_queue(fixture)
        fixture.clock.launch_id = fixture.clock.status()["daemon"]["id"]
        fixture.first_send()
        attempt = fixture.reminder()["attempts"][0]
        self.assertEqual(attempt["identity"], owner)
        self.assertEqual(fixture.reminder()["delivery"], "sending")
        observations = self.observation.call_count
        with patch.object(wakeup, "terminate_owned", side_effect=AssertionError("age cannot kill queue")):
            for elapsed in (31, 121, 1801, 100000):
                fixture.now += elapsed
                result = fixture.clock.tick()
                self.assertEqual(result["status"], "sending")
                self.assertEqual(fixture.reminder()["attempt_count"], 1)
                self.assertIsNone(fixture.reminder()["next_attempt_at"])
                self.assertEqual(fixture.clock.status()["daemon"]["heartbeat_at"], fixture.now)
        self.assertEqual(self.observation.call_count, observations + 4)
        Path(attempt["stdout"]).write_text("Queued message " + str(uuid4()) + " for thread " + fixture.thread)
        process.returncode = 0
        self.assertEqual(fixture.clock.tick()["status"], "queued")
        self.assertEqual(fixture.reminder()["next_attempt_at"], fixture.now + wakeup.ATTENTION_SECONDS)

    def test_inflight_archived_by_scan_blocks_a_second_queue_and_records_late_result(self):
        fixture = ReminderFixture(self.root / "archived", [])
        process, _ = self.fake_queue(fixture)
        fixture.first_send()
        old = fixture.reminder()
        fixture.clock.ack(old["id"], fixture.thread)
        fixture.clock.scan(old["id"], fixture.thread)
        fixture.now = fixture.reminder()["due_at"] + 100000
        self.assertEqual(fixture.clock.tick()["status"], "sending")
        self.assertEqual(fixture.reminder()["attempt_count"], 0)
        message_id = str(uuid4())
        Path(old["attempts"][0]["stdout"]).write_text("Queued message " + message_id + " for thread " + fixture.thread)
        process.returncode = 0
        self.assertEqual(fixture.clock.tick()["message_id"], message_id)
        self.assertIn(message_id, fixture.clock.status()["last_scan"]["message_ids"])

    def test_stop_invalidates_before_signal_and_never_waits_for_queue(self):
        fixture = ReminderFixture(self.root / "stop-pending", [])
        _, owner = self.fake_queue(fixture)
        fixture.first_send()
        generation = fixture.clock.status()["generation"]
        def stop(identity, known=(), **options):
            clock = fixture.store.read_supervisor()["clock"]
            self.assertFalse(clock["enabled"])
            self.assertEqual(clock["generation"], generation + 1)
            self.assertTrue(options["include_anchor"])
            self.assertEqual(identity, owner)
            return {"confirmed": False, "survivors": [owner], "descendants": [owner], "actions": []}
        with patch.object(wakeup, "terminate_owned", side_effect=stop):
            result = fixture.clock.stop()
        self.assertEqual(result["stop_status"], "pending")
        self.assertTrue(result["committed"])
        with self.assertRaisesRegex(wakeup.ExtractError, "not confirmed exited"):
            fixture.restart()

    def test_busy_control_lock_does_not_ack_or_block_the_tick(self):
        fixture = ReminderFixture(self.root / "busy", ["queued"])
        fixture.first_send()
        with (fixture.store.root / "supervisor.lock").open("a") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(fixture.clock.tick()["status"], "busy")
            with self.assertRaises(SupervisorBusy):
                fixture.clock.ack(fixture.wake_id, fixture.thread)
        self.assertIsNone(fixture.reminder()["ack"])
        self.assertTrue(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])

    def test_initialization_lock_is_immediate_and_preserves_uninitialized_store(self):
        store = Store(self.root / "initializing")
        store.root.mkdir()
        with (store.root / "supervisor.lock").open("a") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(SupervisorBusy):
                store.initialize()
            self.assertFalse((store.root / "format.json").exists())
        store.initialize()



class QueueOwnershipTests(unittest.TestCase):
    def test_exited_queue_keeps_its_anchor_until_unrecorded_child_closes(self):
        from schema_extract.tools.processes import process_identity
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child_path = root / "child.pid"
            binary = root / "queue-fixture"
            message_id, thread_id = str(uuid4()), str(uuid4())
            binary.write_text("#!" + sys.executable + "\nimport os, pathlib, time\n"
                "pid = os.fork()\nif pid == 0:\n while True: time.sleep(1)\n"
                "pathlib.Path(" + repr(str(child_path)) + ").write_text(str(pid))\n"
                "print(" + repr("Queued message " + message_id + " for thread " + thread_id) + ", flush=True)\n")
            binary.chmod(0o700)
            now = time.time()
            clock = wakeup.Clock(Store(root / "state"), root, now=lambda: now)
            clock.start(thread_id, [], codex_binary=str(binary), launch=False)
            now = clock.status()["next_due_at"]
            send = clock._send
            child = None
            def exited_before_observation(*args):
                nonlocal child
                send(*args)
                until = time.monotonic() + 8
                while not wakeup.child_exited(clock.inflight["process"]) and time.monotonic() < until:
                    time.sleep(0.01)
                self.assertTrue(wakeup.child_exited(clock.inflight["process"]))
                child = process_identity(int(child_path.read_text()))
            process = None
            try:
                with patch.object(clock, "_send", side_effect=exited_before_observation):
                    result = clock.tick()
                pending = clock.inflight
                process = pending["process"] if pending else None
                until = time.monotonic() + 8
                while result["status"] == "sending" and time.monotonic() < until:
                    time.sleep(0.01)
                    result = clock.tick()
                self.assertEqual(result["status"], "queued", result)
                self.assertEqual(result["message_id"], message_id)
                self.assertEqual(clock.status()["reminder"]["attempt_count"], 1)
                self.assertIsNotNone(child)
                self.assertFalse(wakeup.process_alive(child, wakeup.process_table()))
            finally:
                if child:
                    wakeup.terminate_owned(child, include_anchor=True)
                clock.stop()
                if process is not None:
                    process.wait(timeout=8)


if __name__ == "__main__":
    unittest.main()
