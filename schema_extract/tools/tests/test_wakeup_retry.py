"""Reminder delivery, sampling and interruption use isolated state."""

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from schema_extract.tools import audit, monitor, wakeup
from schema_extract.tools.store import Store, control_error
from schema_extract.tools.common import SupervisorBusy
from schema_extract.tools.wakeup import _alive


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

    def test_start_commit_facts_at_lock_spawn_and_writeback_boundaries(self):
        for moment in ("before_registration", "before_spawn", "after_spawn"):
            with self.subTest(moment=moment):
                store = Store(self.root / moment)
                store.initialize()
                clock = wakeup.Clock(store, self.root, now=lambda: 1000)
                thread = str(uuid4())
                held = None
                class Child:
                    pid = 98765432
                def spawn(*args, **kwargs):
                    nonlocal held
                    if moment == "before_spawn":
                        raise SupervisorBusy("injected before Popen")
                    held = (store.root / "supervisor.lock").open("a")
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return Child()
                if moment == "before_registration":
                    held = (store.root / "supervisor.lock").open("a")
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    with patch.object(wakeup.subprocess, "Popen", side_effect=spawn) as popen:
                        with self.assertRaises(SupervisorBusy) as caught:
                            clock.start(thread, [])
                    result = control_error(caught.exception)
                    self.assertEqual(result["committed"], moment != "before_registration")
                    self.assertEqual(popen.call_count, 0 if moment == "before_registration" else 1)
                    if moment != "before_registration":
                        saved = store.read_supervisor()["clock"]
                        self.assertEqual((result["launch_id"], result["generation"]), (saved["daemon"]["id"], saved["generation"]))
                    if moment == "after_spawn":
                        self.assertEqual(result["delivery"]["pid"], Child.pid)
                        self.assertFalse(result["delivery"]["evidence_committed"])
                finally:
                    if held:
                        held.close()
                if moment != "before_registration":
                    # Without evidence that the old child exited, another start
                    # must retain its reservation, even if the returned PID was lost.
                    with patch.object(wakeup, "_alive", return_value=None), \
                            patch.object(wakeup.subprocess, "Popen") as repeated:
                        with self.assertRaises(wakeup.ExtractError):
                            clock.start(thread, [])
                    repeated.assert_not_called()

    def test_tick_commit_survives_delivery_record_busy_without_sending_twice(self):
        fixture = ReminderFixture(self.root, ["queued"])
        fixture.now = fixture.reminder()["due_at"]
        wake_id = fixture.reminder()["id"]
        record = fixture.clock._record_send
        def busy_record(*args, **kwargs):
            with (fixture.store.root / "supervisor.lock").open("a") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return record(*args, **kwargs)
        with patch.object(fixture.clock, "_record_send", side_effect=busy_record):
            result = fixture.clock.tick()
        self.assertTrue(result["committed"])
        self.assertEqual(result["id"], wake_id)
        self.assertEqual(len(fixture.sent), 1)
        self.assertEqual(fixture.store.read_supervisor()["clock"]["queue_attempts_total"], 1)
        fixture.clock.tick()
        self.assertEqual(len(fixture.sent), 1)
        self.assertEqual(fixture.reminder()["id"], wake_id)
        self.assertEqual(fixture.reminder()["attempt_count"], 1)

    def test_disabled_start_runs_initial_scan_and_schedules_from_new_completion(self):
        fixture = ReminderFixture(self.root, ["queued"])
        fixture.first_send()
        fixture.clock.ack(fixture.wake_id, fixture.thread)
        fixture.clock.scan(fixture.wake_id, fixture.thread)
        old = fixture.clock.status()
        fixture.clock.stop()
        fixture.now += 10000
        started = fixture.now
        def initial(store, selected):
            clock = store.read_supervisor()["clock"]
            self.assertEqual(clock["last_check_number"], -1)
            self.assertIsNone(clock["next_due_at"])
            self.assertNotIn("reminder", clock)
            fixture.now += 7
            return []
        with patch.object(monitor, "inspect_runs", side_effect=initial) as inspection:
            fixture.restart()
        inspection.assert_called_once()
        current = fixture.clock.status()
        self.assertGreater(current["generation"], old["generation"])
        self.assertEqual(current["last_check_number"], 0)
        self.assertEqual(current["last_scan"]["started_at"], started)
        self.assertEqual(current["next_due_at"], started + 7 + wakeup.EARLY_SECONDS)
        self.assertNotEqual(current["reminder"]["id"], old["reminder"]["id"])
        self.assertFalse(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
        self.assertEqual(len(fixture.sent), 1)

    def test_enabled_daemon_recovery_keeps_original_ack_claim_and_due(self):
        fixture = ReminderFixture(self.root, ["queued"])
        fixture.first_send()
        fixture.clock.ack(fixture.wake_id, fixture.thread)
        with fixture.store.supervisor() as ledger:
            clock = ledger["clock"]
            clock["reminder"]["scan"] = {"id": str(uuid4()), "owner": {"pid": 123456789}, "started_at": fixture.now}
            before = json.loads(json.dumps(clock))
        fixture.now += 10000
        with patch.object(monitor, "inspect_runs", side_effect=AssertionError("same generation does not restart initial scan")):
            fixture.restart()
        current = fixture.store.read_supervisor()["clock"]
        for key in ("generation", "reminder", "last_scan", "last_check_number", "next_due_at"):
            self.assertEqual(current[key], before[key])
        fixture.clock.tick()
        self.assertEqual(len(fixture.sent), 1)
        self.assertEqual(fixture.reminder()["id"], fixture.wake_id)

    def test_new_generation_remains_fenced_by_unconfirmed_old_send(self):
        fixture = ReminderFixture(self.root, ["queued"])
        fixture.first_send()
        with fixture.store.supervisor() as ledger:
            ledger["clock"]["reminder"]["attempts"][-1].update(status="sending", identity=None)
        fixture.clock.stop()
        before = fixture.store.read_supervisor()
        with self.assertRaisesRegex(wakeup.ExtractError, "previous queue delivery"):
            fixture.restart()
        self.assertEqual(fixture.store.read_supervisor(), before)
        self.assertEqual(len(fixture.sent), 1)

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
                    self.assertEqual(fixture.clock.tick()["status"], "waiting")
                    fixture.now = fixture.reminder()["due_at"]
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

    def test_ack_retries_lock_acquisition_while_tick_and_stop_remain_immediate(self):
        fixture = ReminderFixture(self.root / "busy", ["queued"])
        fixture.first_send()
        before = fixture.reminder()
        callbacks = []
        fixture.clock.on_wait = lambda phase, elapsed: callbacks.append((phase, elapsed))
        with (fixture.store.root / "supervisor.lock").open("a") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(fixture.clock.tick()["status"], "busy")
            with self.assertRaises(SupervisorBusy):
                fixture.clock.stop()
            self.assertIsNone(fixture.reminder()["ack"])

            def unlock(delay):
                fixture.now += delay
                fcntl.flock(held, fcntl.LOCK_UN)

            with patch("schema_extract.tools.store.time.sleep",
                       side_effect=unlock) as wait:
                self.assertTrue(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
        wait.assert_called_once_with(0.05)
        self.assertEqual([phase for phase, _ in callbacks], ["ack"])
        self.assertGreaterEqual(callbacks[0][1], 0)
        self.assertEqual(fixture.reminder()["ack"]["at"], fixture.now)
        self.assertEqual(fixture.reminder()["attempts"], before["attempts"])
        self.assertEqual(fixture.clock.status()["next_due_at"], before["due_at"])
        with patch("schema_extract.tools.store.time.sleep", side_effect=AssertionError("rejection must not retry")):
            self.assertEqual(fixture.clock.ack(fixture.wake_id, fixture.thread)["reason"], "already_acknowledged")
            self.assertEqual(fixture.clock.ack(fixture.wake_id, str(uuid4()))["reason"], "wrong_thread")

    def test_scan_keeps_a_live_claim_created_while_waiting_for_the_lock(self):
        fixture = ReminderFixture(self.root / "new-owner", ["queued"])
        fixture.first_send()
        fixture.clock.ack(fixture.wake_id, fixture.thread)
        owner = wakeup.process_identity(123456789)
        processes, after_claim = {}, []
        with (fixture.store.root / "supervisor.lock").open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def claim_during_wait(delay):
                fcntl.flock(held, fcntl.LOCK_UN)
                processes[owner["pid"]] = owner
                with fixture.store.supervisor() as ledger:
                    ledger["clock"]["reminder"]["scan"] = {
                        "id": str(uuid4()), "owner": owner, "started_at": fixture.now}
                after_claim.append(fixture.store.read_supervisor())

            with (patch.object(wakeup, "_alive", _alive),
                  patch.object(wakeup, "process_table", side_effect=AssertionError("claim checks only its owner PID")),
                  patch.object(wakeup, "process_identity", side_effect=lambda pid: owner if pid == os.getpid() else processes.get(pid)),
                  patch.object(monitor, "inspect_runs", side_effect=AssertionError("occupied scan must not run")),
                  patch("schema_extract.tools.store.time.sleep", side_effect=claim_during_wait) as wait):
                result = fixture.clock.scan(fixture.wake_id, fixture.thread)
        wait.assert_called_once_with(0.05)
        self.assertEqual(result["clock"]["status"], "scan_busy")
        self.assertEqual(fixture.store.read_supervisor(), after_claim[0])

    def test_ack_rereads_stop_and_generation_changes_after_waiting(self):
        for restart in (False, True):
            with self.subTest(restart=restart):
                fixture = ReminderFixture(self.root / str(restart), ["queued"])
                fixture.first_send()
                after_stop = []
                with (fixture.store.root / "supervisor.lock").open("a") as held:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

                    def invalidate(delay):
                        fcntl.flock(held, fcntl.LOCK_UN)
                        fixture.clock.stop()
                        if restart:
                            fixture.restart()
                        after_stop.append(fixture.store.read_supervisor())

                    with patch("schema_extract.tools.store.time.sleep", side_effect=invalidate) as wait:
                        result = fixture.clock.ack(fixture.wake_id, fixture.thread)
                wait.assert_called_once_with(0.05)
                self.assertEqual(result, {"accepted": False, "reason": "stale_or_not_due" if restart else "disabled"})
                self.assertEqual(fixture.store.read_supervisor(), after_stop[0])

    def test_scan_retries_claim_commit_and_error_cleanup_without_repeating_inspection(self):
        for phase in ("claim", "commit", "cleanup"):
            with self.subTest(phase=phase):
                fixture = ReminderFixture(self.root / phase, ["queued"])
                fixture.first_send()
                self.assertTrue(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
                callbacks = []
                fixture.clock.on_wait = lambda phase, elapsed: callbacks.append((phase, elapsed))
                with (fixture.store.root / "supervisor.lock").open("a") as held:
                    if phase == "claim":
                        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

                    def inspect(*args):
                        if phase != "claim":
                            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if phase == "cleanup":
                            raise wakeup.ExtractError("fixture scan failed")
                        return []

                    with (patch.object(monitor, "inspect_runs", side_effect=inspect) as inspection,
                          patch("schema_extract.tools.store.time.sleep",
                                side_effect=lambda delay: fcntl.flock(held, fcntl.LOCK_UN)) as wait):
                        if phase == "cleanup":
                            with self.assertRaisesRegex(wakeup.ExtractError, "fixture scan failed"):
                                fixture.clock.scan(fixture.wake_id, fixture.thread)
                        else:
                            self.assertTrue(fixture.clock.scan(fixture.wake_id, fixture.thread)["clock"]["committed"])
                wait.assert_called_once_with(0.05)
                self.assertEqual([name for name, _ in callbacks], ["scan_" + phase])
                inspection.assert_called_once()
                if phase == "cleanup":
                    item = fixture.reminder()
                    self.assertIsNotNone(item["ack"])
                    self.assertNotIn("scan", item)
                    self.assertEqual(item["scan_error"], "fixture scan failed")
                    self.assertEqual(fixture.clock.status()["last_check_number"], 0)
                else:
                    self.assertEqual(fixture.clock.status()["last_check_number"], 1)
                    self.assertTrue(fixture.clock.scan(fixture.wake_id, fixture.thread)["clock"]["reused"])

    def test_delayed_commit_preserves_sampling_time_and_newer_observation(self):
        fixture = ReminderFixture(self.root / "old-observation", ["queued"])
        fixture.first_send()
        fixture.clock.ack(fixture.wake_id, fixture.thread)
        sampled_at = fixture.now
        old = {"run_id": "operator", "state_revision": 1, "status": "running", "anomalies": [],
               "observation": {"observed_at": sampled_at}}
        with (fixture.store.root / "supervisor.lock").open("a") as held:
            def inspect(*args):
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return [old]

            def newer_observation(delay):
                fcntl.flock(held, fcntl.LOCK_UN)
                fixture.now += 60
                with fixture.store.supervisor() as ledger:
                    monitor.record_scan(ledger, [{**old, "state_revision": 2,
                        "observation": {"observed_at": fixture.now}}], fixture.now)
                fixture.now += 60

            with (patch.object(monitor, "inspect_runs", side_effect=inspect) as inspection,
                  patch("schema_extract.tools.store.time.sleep", side_effect=newer_observation)):
                result = fixture.clock.scan(fixture.wake_id, fixture.thread)
        inspection.assert_called_once()
        self.assertEqual(result["observed_at"], sampled_at)
        self.assertEqual(result["clock"]["check"]["completed_at"], sampled_at + 120)
        self.assertEqual(result["clock"]["next_due_at"], sampled_at + 420)
        self.assertEqual(result["superseded_run_ids"], ["operator"])
        self.assertEqual(result["changes"], [])
        latest = fixture.store.read_supervisor()["observations"]["latest"]["operator"]
        self.assertEqual((latest["state_revision"], latest["observed_at"]), (2, sampled_at + 60))

    def test_scan_rereads_stop_and_generation_changes_after_each_lock_wait(self):
        for phase in ("claim", "commit"):
            for restart in (False, True):
                with self.subTest(phase=phase, restart=restart):
                    fixture = ReminderFixture(self.root / (phase + str(restart)), ["queued"])
                    fixture.first_send()
                    self.assertTrue(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
                    generation = fixture.clock.status()["generation"]
                    after_stop = []
                    with (fixture.store.root / "supervisor.lock").open("a") as held:
                        if phase == "claim":
                            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

                        def inspect(*args):
                            # Inject contention in the old scan only. An explicit
                            # restart now performs its own fresh initial scan.
                            if fixture.store.read_supervisor()["clock"]["generation"] == generation:
                                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            return []

                        def invalidate(delay):
                            fcntl.flock(held, fcntl.LOCK_UN)
                            fixture.clock.stop()
                            if restart:
                                fixture.restart()
                            after_stop.append(fixture.store.read_supervisor())

                        with (patch.object(monitor, "inspect_runs", side_effect=inspect) as inspection,
                              patch("schema_extract.tools.store.time.sleep", side_effect=invalidate) as wait):
                            if phase == "claim":
                                with self.assertRaisesRegex(wakeup.ExtractError, "clock disabled|current accepted reminder"):
                                    fixture.clock.scan(fixture.wake_id, fixture.thread)
                            else:
                                result = fixture.clock.scan(fixture.wake_id, fixture.thread)
                                self.assertEqual(result["clock"], {"committed": False, "status": "scan_invalidated", "id": fixture.wake_id})
                    wait.assert_called_once_with(0.05)
                    self.assertEqual(inspection.call_count, (0 if phase == "claim" else 1) + int(restart))
                    self.assertEqual(fixture.store.read_supervisor(), after_stop[0])

    def test_initialization_lock_is_immediate_and_preserves_uninitialized_store(self):
        store = Store(self.root / "initializing")
        store.root.mkdir()
        with (store.root / "supervisor.lock").open("a") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(SupervisorBusy):
                store.initialize()
            self.assertFalse((store.root / "format.json").exists())
        store.initialize()



class ObservationTimingTests(unittest.TestCase):
    def test_per_run_sampling_order_and_liveness_do_not_depend_on_revision(self):
        ledger = {"observations": {}}

        def row(run_id, observed_at, alive=True):
            return {"run_id": run_id, "state_revision": 1, "status": "running", "stage": "P1",
                    "state_directory": "/fixture", "observation": {"observed_at": observed_at}, "anomalies": [],
                    "runner_observation": {"lock": "held", "alive": alive}}

        monitor.record_scan(ledger, [row("a", 100), row("b", 100)], 100)
        result = monitor.record_scan(ledger, [row("a", 200, False)], 200)
        self.assertEqual([item["run_id"] for item in result["changes"]], ["a"])
        result = monitor.record_scan(ledger, [row("a", 150), row("b", 150)], 150)
        self.assertEqual(result["superseded_run_ids"], ["a"])
        self.assertEqual(result["changes"], [])
        latest = ledger["observations"]["latest"]
        self.assertEqual((latest["a"]["observed_at"], latest["b"]["observed_at"]), (200, 150))
        self.assertFalse(latest["a"]["runner_observation"]["alive"])
        self.assertEqual(ledger["observations"]["observed_at"], 200)
        result = monitor.record_scan(ledger, [row("b", 399)], 999)
        self.assertEqual(result["changes"], [])
        self.assertEqual(result["anomalies"], [])
        result = monitor.record_scan(ledger, [row("b", 400)], 1000)
        self.assertEqual(result["anomalies"][0]["classification"], "possible_stall")

    def test_time_display_preserves_machine_values_and_limits_its_scope(self):
        stamp = 1798732800  # 2026-12-31 16:00 UTC.
        self.assertEqual(wakeup.format_beijing_time(stamp), "2027-01-01 00:00:00 北京时间（UTC+8）")
        self.assertEqual(wakeup.format_beijing_time(0), "1970-01-01 08:00:00 北京时间（UTC+8）")
        self.assertEqual(wakeup.format_beijing_time(None), "未记录")
        clock = {"generation": 1, "thread_id": str(uuid4()), "enabled": False, "next_due_at": stamp,
                 "observed_at": stamp, "last_scan": {"completed_at": stamp, "observed_at": stamp - 90},
                 "daemon": {"heartbeat_at": None}, "history": [{"due_at": stamp}],
                 "runs": [{"observed_at": stamp}], "elapsed_seconds": 120}
        clock["reminder"] = wakeup._item(clock, 1, stamp)
        before = json.dumps(clock, sort_keys=True)
        displayed = wakeup.clock_display(clock)
        self.assertEqual(json.dumps(clock, sort_keys=True), before)
        self.assertEqual(displayed["reminder"]["id"], wakeup._item(clock, 1, stamp)["id"])
        self.assertEqual(displayed["next_due_at"], stamp)
        self.assertIn("已停用", displayed["next_due_at_display"])
        self.assertEqual(displayed["daemon"]["heartbeat_at_display"], "未记录")
        self.assertEqual(displayed["history"], clock["history"])
        self.assertEqual(displayed["runs"], clock["runs"])
        self.assertNotIn("elapsed_seconds_display", displayed)
        bound = wakeup.clock_display({"clock": {"check": clock["last_scan"], "next_due_at": stamp}})
        self.assertEqual(bound["clock"]["check"]["observed_at"], stamp - 90)
        self.assertIn("北京时间", bound["clock"]["check"]["completed_at_display"])

    def test_single_pid_liveness_keeps_unknown_identity_conservative(self):
        identity = {"pid": 123, "start_token": "linux:fixture:1", "stat": "S"}
        with patch.object(wakeup, "process_table", side_effect=AssertionError("no full process scan")):
            with patch.object(wakeup, "process_identity", return_value={**identity, "start_token": "linux:fixture:2"}):
                self.assertIs(_alive(identity), False)
            with patch.object(wakeup, "process_identity", side_effect=PermissionError("unreadable")):
                self.assertIsNone(_alive(identity))
            self.assertIsNone(_alive(None))


class RealWakeupInterruptionTests(unittest.TestCase):
    def wait_for_file(self, path, child):
        until = time.monotonic() + 8
        while not path.exists() and child.poll() is None and time.monotonic() < until:
            time.sleep(0.01)
        self.assertTrue(path.exists(), "child did not reach " + path.name)

    def test_real_lock_wait_interrupt_and_same_wake_reconcile(self):
        with tempfile.TemporaryDirectory() as temporary:
            for phase in ("ack", "scan_commit", "after_commit"):
                with self.subTest(phase=phase):
                    fixture = ReminderFixture(Path(temporary) / phase, ["queued"])
                    fixture.first_send()
                    if phase != "ack":
                        fixture.clock.ack(fixture.wake_id, fixture.thread)
                    root = fixture.store.root
                    script = """
import sys
from pathlib import Path
from schema_extract.tools import monitor, wakeup
from schema_extract.tools.store import Store
root, phase, wake, thread, now = sys.argv[1:]
root = Path(root)
def waiting(phase, elapsed):
    (root / 'waiting').write_text(phase)
def inspect(*args):
    (root / 'inspected').write_text('once')
    if phase == 'scan_commit':
        sys.stdin.readline()
    return []
monitor.inspect_runs = inspect
clock = wakeup.Clock(Store(root), root, now=lambda: float(now), on_wait=waiting)
if phase == 'ack':
    clock.ack(wake, thread)
else:
    clock.scan(wake, thread)
    (root / 'committed').write_text('yes')
    sys.stdin.readline()
"""
                    child = None
                    with (root / "supervisor.lock").open("a") as held:
                        if phase == "ack":
                            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        before = (root / "supervisor.json").read_bytes()
                        try:
                            child = subprocess.Popen([sys.executable, "-B", "-c", script, str(root), phase,
                                                      fixture.wake_id, fixture.thread, str(fixture.now)],
                                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                     text=True)
                            if phase == "scan_commit":
                                self.wait_for_file(root / "inspected", child)
                                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                child.stdin.write("continue\n")
                                child.stdin.flush()
                            self.wait_for_file(root / ("committed" if phase == "after_commit" else "waiting"), child)
                            if phase != "after_commit":
                                self.assertEqual((root / "waiting").read_text(), phase)
                            child.send_signal(signal.SIGINT)
                            stdout, stderr = child.communicate(timeout=8)
                            self.assertNotEqual(child.returncode, 0)
                            self.assertEqual(stdout, "")
                            self.assertIn("KeyboardInterrupt", stderr)
                            if phase == "ack":
                                self.assertEqual((root / "supervisor.json").read_bytes(), before)
                            elif phase == "scan_commit":
                                state = fixture.store.read_supervisor()["clock"]
                                self.assertEqual(state["last_check_number"], 0)
                                self.assertIsNotNone(state["reminder"]["ack"])
                                self.assertEqual(state["reminder"]["scan"]["owner"]["pid"], child.pid)
                        finally:
                            fcntl.flock(held, fcntl.LOCK_UN)
                            if child is not None and child.poll() is None:
                                child.kill()
                                child.communicate(timeout=8)
                    if phase == "ack":
                        self.assertTrue(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
                    with patch.object(monitor, "inspect_runs", return_value=[]) as inspection:
                        result = fixture.clock.scan(fixture.wake_id, fixture.thread)
                        self.assertTrue(result["clock"].get("reused" if phase == "after_commit" else "committed"))
                        self.assertTrue(fixture.clock.scan(fixture.wake_id, fixture.thread)["clock"]["reused"])
                        self.assertEqual(inspection.call_count, 0 if phase == "after_commit" else 1)
                    self.assertEqual(fixture.clock.status()["last_check_number"], 1)


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
