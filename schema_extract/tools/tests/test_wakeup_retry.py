"""Bounded reminder retries use isolated state, a fake clock and a fake queue."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from schema_extract.tools import monitor, wakeup
from schema_extract.tools.store import Store


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
        for target, value in (("process_identity", {"pid": 123456789, "started": "fixture"}),
                              ("process_alive", False)):
            patcher = patch.object(wakeup, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target in ("inspect_runs", "authorized_restarts"):
            patcher = patch.object(monitor, target, return_value=[])
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(wakeup.Clock, "observe", return_value={"status": "observed"})
        self.observation = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(wakeup.Clock, "_send", side_effect=AssertionError("real queue forbidden"))
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
            if reads == 2:  # The reservation transaction has released its lock.
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
                for record in (clock["last_check"], clock["history"][-1]):
                    self.assertEqual(record["id"], fixture.wake_id)
                    self.assertEqual(record["attempts"][-1]["message_id"], result["message_id"])
                    self.assertEqual(record["status"], "completed" if action == "scan" else "sending")
                if action == "scan":
                    self.assertEqual(clock["last_check_number"], 1)
                    self.assertIn(result["message_id"], clock["last_scan"]["message_ids"])
                    self.assertTrue(fixture.clock.scan(fixture.wake_id, fixture.thread)["clock"]["reused"])
                else:
                    self.assertFalse(clock["enabled"])
                    self.assertEqual((clock["generation"], clock["daemon"]["status"]), (generation + 1, "stopped"))
                    self.assertEqual(fixture.clock.tick()["status"], "disabled")
                    self.assertFalse(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
                    fixture.restart()
                    self.assertFalse(fixture.clock.ack(fixture.wake_id, fixture.thread)["accepted"])
                    self.assertNotEqual(fixture.reminder()["id"], fixture.wake_id)
                    self.assertEqual(fixture.clock.tick()["status"], "queued")
                    self.assertNotIn(fixture.wake_id, fixture.sent[-1][1])


if __name__ == "__main__":
    unittest.main()
