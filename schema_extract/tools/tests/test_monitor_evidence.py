"""A concurrent branch's diagnostic must not point at its sibling's attempt."""

from copy import deepcopy
from contextlib import redirect_stdout
import fcntl
import io
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
from unittest.mock import patch

from schema_extract.tools.monitor import classify


class MonitorEvidenceTests(unittest.TestCase):
    def run_view(self):
        return {"run_id": "operator", "state_directory": "/runtime/runs/operator", "stage": "P5",
                "status": "running", "runner_observation": {"lock": "held", "alive": True},
                "last_attempt_id": "healthy-sibling", "node_details": {
                    "first": {"reason": {"code": "proposal_rejected", "summary": "First branch failed.",
                                          "attempt_id": "failed-first"}},
                    "second": {"reason": {"code": "proposal_rejected", "summary": "Second branch failed.",
                                           "attempt_id": "failed-second"}}}}

    def test_parallel_failures_retain_their_own_attempts(self):
        observations = {item["scope"]: item for item in classify(self.run_view())}
        self.assertEqual(set(observations), {"P5:first", "P5:second"})
        for name in ("first", "second"):
            self.assertEqual(observations["P5:" + name]["engineering_handoff"]["read_only_evidence"],
                             ["/runtime/runs/operator/state.json", "/runtime/runs/operator/attempts/failed-" + name])

    def test_branch_without_identity_does_not_guess_the_latest_attempt(self):
        view = self.run_view()
        for detail in view["node_details"].values():
            detail["reason"].pop("attempt_id")
        for item in classify(view):
            self.assertEqual(item["engineering_handoff"]["read_only_evidence"], ["/runtime/runs/operator/state.json"])

    def test_sequential_failure_uses_an_available_last_attempt(self):
        view = self.run_view()
        view.update(stage="P4", status="invalid_receipt", node_details={},
                    reason={"code": "proposal_rejected", "summary": "Root failed."})
        for latest in ("failed-root", None):
            view["last_attempt_id"] = latest
            expected = ["/runtime/runs/operator/state.json"]
            if latest:
                expected.append("/runtime/runs/operator/attempts/" + latest)
            self.assertEqual(classify(view)[0]["engineering_handoff"]["read_only_evidence"], expected)


class MonitorLaunchTests(unittest.TestCase):
    def test_exited_background_before_claim_does_not_reuse_living_launcher(self):
        from schema_extract.tools import monitor
        from schema_extract.tools.store import Store

        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root))
            store.initialize()
            run_id = str(uuid4())
            store.run_dir(run_id).mkdir(parents=True)
            state = {"run_id": run_id, "status": "queued", "input_fingerprint": "frozen-input", "executor": {"digest": "frozen-code"},
                     "output": str(Path(root) / "output"), "created_at": 1, "control": {"action": None, "requests": {}}}
            identity = {"pid": os.getpid(), "start_token": "linux:fixture:1", "stat": "S", "ppid": 1, "pgid": os.getpid()}
            class Exited:
                pid = 87654321
                def poll(self):
                    return 1
            with (patch.object(store, "load", return_value=state),
                  patch("schema_extract.tools.runtime.frozen_run_invocation"),
                  patch("schema_extract.tools.runtime.verify_execution"),
                  patch.object(monitor, "process_table", return_value={identity["pid"]: identity}),
                  patch.object(monitor.subprocess, "Popen", return_value=Exited())):
                launch = monitor.start_background(store, run_id, Path(root))
                self.assertEqual((launch["status"], launch["exit_code"]), ("launch_failed", 1))
                retry = monitor.reserve_launch(store, state, mode="queued", background=True)
            self.assertFalse(retry.get("reused", False))
            self.assertNotEqual(retry["id"], launch["id"])
            self.assertEqual(state["status"], "queued")


class CurrentObservationTests(unittest.TestCase):
    def test_late_failure_is_history_and_clock_counts_the_current_ready_sample(self):
        from schema_extract.tools import cli, monitor, wakeup
        from schema_extract.tools.store import Store
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root))
            run_id = str(uuid4())
            failure = {"run_id": run_id, "status": "invalid_receipt", "stage": "P1", "state_revision": 1,
                       "observation": {"observed_at": 100, "status": "current"}, "active_attempts": {},
                       "anomalies": [{"run_id": run_id, "scope": "P1", "classification": "framework_or_protocol",
                                      "summary": "old failure", "evidence": "fixture", "recovery_condition": "old recovery"}]}
            ready = {**failure, "status": "ready", "state_revision": 2,
                     "observation": {"observed_at": 200, "status": "current"}, "anomalies": []}
            with store.supervisor() as ledger:
                monitor.record_scan(ledger, [ready], 200)
            current = deepcopy(store.read_supervisor()["observations"]["latest"])
            with store.supervisor() as ledger:
                result = monitor.record_scan(ledger, [failure], 300)
            self.assertEqual(store.read_supervisor()["observations"]["latest"], current)
            self.assertEqual(result["anomalies"], [])
            self.assertEqual(result["changes"], [])
            self.assertEqual(result["superseded_run_ids"], [run_id])
            historical = result["runs"][0]
            self.assertEqual(historical["observation_role"], "historical")
            self.assertEqual(historical["anomalies"], failure["anomalies"])
            self.assertEqual(historical["superseded_by_observed_at"], 200)
            self.assertNotIn("observation_role", failure)
            shown = io.StringIO()
            with redirect_stdout(shown):
                cli._print_observations(result["runs"])
            self.assertTrue(shown.getvalue().startswith("历史采样"))
            self.assertIn("old failure", shown.getvalue())
            self.assertNotIn("next:", shown.getvalue())
            self.assertNotIn("old recovery", shown.getvalue())
            clock = {"enabled": True, "generation": 1, "thread_id": str(uuid4()), "last_check_number": 0, "next_due_at": 300}
            check = wakeup.Clock._complete(clock, result, 1, "check", 300, 300)
            self.assertEqual(check["anomaly_count"], 0)

    def test_old_ready_cannot_hide_a_new_failure_from_current_anomalies(self):
        from schema_extract.tools import cli, monitor, wakeup
        ledger = {"observations": {}}
        anomaly = {"run_id": "fixture", "classification": "execution_failure", "summary": "new failure",
                   "recovery_condition": "current recovery"}
        current = {"run_id": "fixture", "state_revision": 2, "status": "cli_error",
                   "observation": {"observed_at": 200}, "anomalies": [anomaly]}
        monitor.record_scan(ledger, [current], 200)
        result = monitor.record_scan(ledger, [{"run_id": "fixture", "state_revision": 1, "status": "ready",
            "observation": {"observed_at": 100, "status": "current"}, "anomalies": []}], 300)
        self.assertEqual(result["anomalies"], [anomaly])
        self.assertEqual(result["runs"][0]["observation_role"], "historical")
        shown = io.StringIO()
        with redirect_stdout(shown):
            cli._print_observations(result["runs"], current_anomalies=result["anomalies"])
        self.assertIn("current recovery", shown.getvalue())
        self.assertIn("current fixture execution_failure", shown.getvalue())
        clock = {"enabled": True, "generation": 1, "thread_id": str(uuid4()), "last_check_number": 0, "next_due_at": 300}
        self.assertEqual(wakeup.Clock._complete(clock, result, 1, "check", 300, 300)["anomaly_count"], 1)


class CancelDeliveryTests(unittest.TestCase):
    def setUp(self):
        from schema_extract.tools import monitor
        from schema_extract.tools.store import Store
        self.monitor = monitor
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name))
        self.store.initialize()
        self.run_id = str(uuid4())
        self.identity = {"pid": 87654321, "start_token": "linux:fixture:runner", "stat": "S", "ppid": 1, "pgid": 87654321}
        self.state = {"run_id": self.run_id, "input_fingerprint": "fixed-input", "executor": {"digest": "fixed-code"},
                      "runner": {"identity": self.identity}, "attempts": [], "active_attempts": {},
                      "unconfirmed_attempts": {}, "unsettled_attempts": {}}
        self.store.run_dir(self.run_id).mkdir(parents=True)
        mocker = patch.object(self.store, "load", side_effect=lambda _: deepcopy(self.state))
        mocker.start()
        self.addCleanup(mocker.stop)

    def test_request_is_durable_before_kill_without_waiting_for_runner_lock(self):
        request = self.store.request_control(self.run_id, "cancel")
        self.assertEqual(self.store.request_control(self.run_id, "cancel")["id"], request["id"])
        signals = []
        alive = {self.identity["pid"]: self.identity}
        def signal(identity):
            ledger = self.store.read_supervisor()
            self.assertEqual(ledger["controls"][self.run_id]["action"], "cancel")
            self.assertIn(request["id"], ledger["controls"][self.run_id]["deliveries"])
            signals.append(identity)
            return True
        with (self.store.writer(self.run_id),
              patch.object(self.monitor, "process_table", return_value=alive),
              patch("schema_extract.tools.processes.process_table", return_value=alive),
              patch.object(self.monitor, "signal_process", side_effect=signal)):
            delivery = self.monitor.deliver_cancel(self.store, request)
        self.assertEqual(signals, [self.identity])
        self.assertEqual((delivery["status"], delivery["confirmed"]), ("pending", False))
        with patch.object(self.monitor, "process_table", return_value={}):
            delivery = self.monitor.deliver_cancel(self.store, request)
        self.assertEqual((delivery["status"], delivery["confirmed"]), ("confirmed", True))
        self.assertEqual(self.state["runner"]["identity"], self.identity)

    def test_later_pause_does_not_revoke_cancel_delivery(self):
        request = self.store.request_control(self.run_id, "cancel")
        pause = self.store.request_control(self.run_id, "pause")
        alive = {self.identity["pid"]: self.identity}
        with (patch.object(self.monitor, "process_table", return_value=alive),
              patch("schema_extract.tools.processes.process_table", return_value=alive),
              patch.object(self.monitor, "signal_process", return_value=True) as signal):
            self.assertFalse(self.monitor.deliver_cancel(self.store, request)["confirmed"])
            self.assertEqual(signal.call_count, 1)
            self.assertFalse(self.monitor.deliver_cancel(self.store, request)["confirmed"])
            self.assertEqual(signal.call_count, 2)
        with patch.object(self.monitor, "process_table", return_value={}):
            self.assertTrue(self.monitor.deliver_cancel(self.store, request)["confirmed"])
        current = self.store.read_supervisor()["controls"][self.run_id]
        self.assertEqual((current["id"], current["action"]), (pause["id"], "pause"))

    def test_confirmed_cancel_stays_closed_and_new_cancel_targets_resumed_runner(self):
        request = self.store.request_control(self.run_id, "cancel")
        with patch.object(self.monitor, "process_table", return_value={}):
            closed = self.monitor.deliver_cancel(self.store, request)
        self.assertTrue(closed["confirmed"])
        self.state.update(status="cancelled", output=str(self.store.root / "output"), created_at=1,
                          control={"action": "cancel", "requests": {request["id"]: request}})
        launcher = {**self.identity, "pid": os.getpid(), "pgid": os.getpid(), "start_token": "linux:fixture:launcher"}
        with (patch.object(self.monitor, "process_table", return_value={launcher["pid"]: launcher}),
              patch("schema_extract.tools.runtime.verify_execution")):
            launch = self.monitor.reserve_launch(self.store, self.state, mode="resume")
        self.state["runner"] = {"identity": launcher, "launch_id": launch["id"]}
        with patch.object(self.monitor, "signal_process", side_effect=AssertionError("closed delivery cannot signal")):
            self.assertEqual(self.monitor.deliver_cancel(self.store, request), closed)
        fresh = self.store.request_control(self.run_id, "cancel")
        self.assertNotEqual(fresh["id"], request["id"])
        runner = {**launcher, "pid": self.identity["pid"], "pgid": self.identity["pid"]}
        self.state["runner"]["identity"] = runner
        with self.store.supervisor() as ledger:
            ledger["starts"][self.run_id]["launch"]["identity"] = runner
        with (patch.object(self.monitor, "process_table", return_value={runner["pid"]: runner}),
              patch("schema_extract.tools.processes.process_table", return_value={runner["pid"]: runner}),
              patch.object(self.monitor, "signal_process", return_value=True) as signal):
            self.assertFalse(self.monitor.deliver_cancel(self.store, fresh)["confirmed"])
            signal.assert_called_once_with(runner)

    def test_gate_anchor_survives_cancel_until_real_descendant_closure(self):
        from schema_extract.tools.common import digest, write_json
        from schema_extract.tools.processes import GATE_PROTOCOL
        attempt_id, launch_id = str(uuid4()), str(uuid4())
        directory = self.store.run_dir(self.run_id) / "attempts" / attempt_id
        directory.mkdir(parents=True)
        gate = {**self.identity, "pid": 87654322, "pgid": 87654322, "ppid": self.identity["pid"], "start_token": "linux:fixture:gate"}
        attempt = {"run_id": self.run_id, "attempt_id": attempt_id, "input_fingerprint": "fixed-input", "directory": str(directory)}
        launch = {"protocol": GATE_PROTOCOL, "status": "setup", "launch_id": launch_id,
                  "run_id": self.run_id, "attempt_id": attempt_id, "attempt_sha256": digest(attempt), "identity": gate}
        write_json(directory / "attempt.json", attempt)
        write_json(directory / "launch.json", launch)
        write_json(directory / "invocation.json", launch)
        write_json(directory / "process.json", {"identity": gate, "descendants": [], "launch_id": launch_id})
        self.state["attempts"] = [attempt_id]
        self.state["active_attempts"] = {"P1": attempt_id}
        request = self.store.request_control(self.run_id, "cancel")
        signals = []
        table = {self.identity["pid"]: self.identity, gate["pid"]: gate}
        def signal(identity):
            signals.append(identity["pid"])
            table.pop(identity["pid"], None)
            return True
        def terminate(identity, known=(), **options):
            self.assertEqual(identity, gate)
            self.assertFalse(options.get("include_anchor", False))
            self.assertIn(attempt_id, self.store.read_supervisor()["controls"][self.run_id]["deliveries"][request["id"]]["targets"])
            return {"confirmed": True, "actions": [], "survivors": [], "descendants": []}
        with (patch.object(self.monitor, "process_table", side_effect=lambda: dict(table)),
              patch("schema_extract.tools.processes.process_table", side_effect=lambda: dict(table)),
              patch.object(self.monitor, "signal_process", side_effect=signal),
              patch.object(self.monitor, "terminate_owned", side_effect=terminate),
              patch.object(self.monitor, "gate_reaped", return_value=False),
              patch.object(self.monitor, "launch_not_started", return_value=False)):
            delivery = self.monitor.deliver_cancel(self.store, request)
        self.assertEqual(signals, [self.identity["pid"]])
        self.assertFalse(delivery["confirmed"])
        self.assertIn("closure", delivery["attempts"][attempt_id]["reason"])

    def test_busy_does_not_register_or_signal_cancel(self):
        from schema_extract.tools.common import SupervisorBusy
        with (self.store.root / "supervisor.lock").open("a") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(SupervisorBusy):
                self.store.request_control(self.run_id, "cancel")
        self.assertEqual(self.store.control_requests(self.run_id), [])

    def test_observer_failure_preserves_pending_without_signals(self):
        request = self.store.request_control(self.run_id, "cancel")
        with (patch.object(self.monitor, "process_table", side_effect=PermissionError("fixture /proc denial")),
              patch.object(self.monitor, "signal_process", side_effect=AssertionError("unknown identity cannot be signalled"))):
            delivery = self.monitor.deliver_cancel(self.store, request)
        self.assertFalse(delivery["confirmed"])
        self.assertEqual(delivery["status"], "pending")
        self.assertIn("denial", delivery["observer_error"])



if __name__ == "__main__":
    unittest.main()
