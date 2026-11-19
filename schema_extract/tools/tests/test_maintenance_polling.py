"""Idle Runner polls leave the Supervisor lock available for foreground control."""

from contextlib import contextmanager
from copy import deepcopy
import fcntl
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

from schema_extract.tools import maintenance
from schema_extract.tools.common import ExtractError, digest, write_json
from schema_extract.tools.store import Store


class MaintenancePollingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name))
        self.store.initialize()
        self.state = {"run_id": str(uuid4()), "input_fingerprint": "fixed-input"}

    def request(self):
        body = {**self.state, "issue_id": "fixture", "tasks": []}
        request = {**body, "id": str(uuid4()), "request_sha256": digest(body), "status": "requested", "created_at": 1.0}
        with self.store.supervisor() as ledger:
            ledger["maintenance_requests"][self.state["run_id"]] = request
        return request

    def test_idle_poll_finishes_while_another_control_transaction_holds_the_lock(self):
        for status in (None, "applied"):
            with self.subTest(status=status):
                with self.store.supervisor() as ledger:
                    ledger["maintenance_requests"] = ({self.state["run_id"]: {"status": status}} if status else {})
                before, returned, errors = self.store.read_supervisor(), [], []
                finished = threading.Event()

                def poll():
                    try:
                        returned.append(maintenance.consume_requests(self.store, self.state))
                    except Exception as exc:
                        errors.append(exc)
                    finally:
                        finished.set()

                with (self.store.root / "supervisor.lock").open("r") as held:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    worker = threading.Thread(target=poll)
                    worker.start()
                    completed_while_held = finished.wait(0.5)
                worker.join(timeout=2)
                self.assertTrue(completed_while_held, "idle maintenance poll waited for the control lock")
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertIs(returned[0], self.state)
                self.assertEqual(self.store.read_supervisor(), before)

    def test_pending_request_is_rechecked_after_the_preliminary_read(self):
        self.request()
        transaction = self.store.control_transaction

        @contextmanager
        def completed_before_acquisition():
            with self.store.supervisor() as ledger:
                ledger["maintenance_requests"][self.state["run_id"]]["status"] = "applied"
            with transaction() as ledger:
                yield ledger

        with patch.object(self.store, "control_transaction", completed_before_acquisition):
            self.assertIs(maintenance.consume_requests(self.store, self.state), self.state)
        self.assertNotIn("maintenance_request", self.state)

    def test_pending_request_still_requires_its_sealed_identity(self):
        request = self.request()
        write_json(self.store.root / "maintenance_requests" / self.state["run_id"] / (request["id"] + ".json"), {})
        before = deepcopy(self.store.read_supervisor())
        with self.assertRaisesRegex(ExtractError, "sealed current-run identity"):
            maintenance.consume_requests(self.store, self.state)
        self.assertEqual(self.store.read_supervisor(), before)
        self.assertNotIn("maintenance_request", self.state)
