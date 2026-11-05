"""Scope recovery preserves concurrent work and does not wait for reminder ack."""

from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from schema_extract.tools import monitor
from schema_extract.tools.common import ExtractError, digest, write_json
from schema_extract.tools.runner import _consume_scope_retries, _reconcile_orphans
from schema_extract.tools.stages import new_node, new_scope, next_calls, node_signature
from schema_extract.tools.store import Store
from schema_extract.tools.wakeup import Clock


class ScopedRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / "runtime")
        self.store.initialize()
        self.run_id, self.attempt_id, self.sibling = (str(uuid4()) for _ in range(3))
        plan = [{"id": name, "file": name + ".yaml", "parent": "" if name == "root" else "root"}
                for name in ("root", "a", "b")]
        nodes = {entry["id"]: new_node(entry) for entry in plan}
        nodes["root"].update(status="approved", merged=True, approval={"saved": True}, approved_generation=0)
        nodes["a"].update(status="cli_error", draft_checkpoint="draft-a")
        nodes["a"]["stage"].update(calls=1, cli_failures=1, elapsed_seconds=7)
        self.state = {"format": "schema_extract.run.v2", "run_id": self.run_id, "status": "running", "stage": "P5",
            "node": "root", "role": None, "mode": "normal", "plan": plan, "plan_revision": 1, "nodes": nodes,
            "stages": {key: new_scope() for key in ("P1", "P2", "P3", "P4", "P5", "P6")},
            "executor": {"capabilities": ["scoped_recovery.v2"]}, "input_fingerprint": "original-inputs",
            "checkpoint": "merged", "merged_checkpoint": "merged", "reopening": None, "reopen_requests": {},
            "config": {"operator_seconds": 3600, "stage_seconds": 3600, "call_seconds": 60,
                       "adjudication_seconds": 30, "max_calls": 12, "retry_limit": 2},
            "deadline": time.time() + 3600, "active_attempts": {"P5:b": self.sibling},
            "unconfirmed_attempts": {}, "unsettled_attempts": {self.sibling: "dispatched"},
            "control": {"action": None, "requests": {}, "resume_status": None}, "sessions": {},
            "attempt_usage": {self.attempt_id: {"stage": "P5", "node": "a", "elapsed_seconds": 7, "settled": True}},
            "reason": None, "feedback": [], "recovered_attempts": {}}
        for stage in ("P1", "P2", "P3", "P4"):
            self.state["stages"][stage]["status"] = "approved"
        directory = self.store.run_dir(self.run_id) / "attempts" / self.attempt_id
        directory.mkdir(parents=True)
        self.attempt = {"run_id": self.run_id, "attempt_id": self.attempt_id, "stage": "P5", "node": "a",
            "role": "worker", "mode": "normal", "generation": 0, "plan_revision": 1,
            "input_fingerprint": "original-inputs", "base_checkpoint": "before-failure",
            "directory": str(directory), "workspace": str(directory / "workspace"),
            "node_signature": node_signature(plan[1]), "dependency_digests": {"root.yaml": "approved-root"}}
        result = {"outcome": "cli_error", "exit_code": 75, "termination": {"confirmed": True}}
        write_json(directory / "attempt.json", self.attempt)
        write_json(directory / "result.json", result)
        nodes["a"]["reason"] = {"code": "cli_retry", "classification": "transient_execution", "scope": "P5:a",
                                "attempt_id": self.attempt_id, "result_sha256": digest(result), "next_due_at": 0}
        self.store.save(self.state)
        self.files = {"root.yaml": "approved-root"}
        for mocker in (patch.object(self.store, "verify_checkpoint", side_effect=lambda *_: (directory, {"files": self.files})),
                       patch.object(self.store, "verify_node"), patch.object(self.store, "writer_status", return_value="held"),
                       patch("schema_extract.tools.recovery.process_table", return_value={})):
            mocker.start()
            self.addCleanup(mocker.stop)

    def test_request_survives_sibling_revision_and_preserves_its_fence_and_budget(self):
        request = monitor.request_restart(self.store, self.run_id, "P5:a", self.store.root, automatic=False)
        self.assertEqual(request["delivery"], "active_runner")
        self.state["nodes"]["b"]["stage"]["elapsed_seconds"] += 9
        self.store.save(self.state)
        sibling, usage = deepcopy(self.state["nodes"]["b"]), deepcopy(self.state["attempt_usage"])
        state, due = _consume_scope_retries(self.store, self.state, {}, {self.sibling: object()})
        self.assertIsNone(due)
        self.assertEqual(state["nodes"]["b"], sibling)
        self.assertEqual(state["attempt_usage"], usage)
        self.assertEqual(state["active_attempts"], {"P5:b": self.sibling})
        self.assertEqual(state["nodes"]["a"]["stage"]["calls"], 1)
        self.assertEqual([call["node"] for call in next_calls(state)], ["a"])
        self.assertEqual(self.store.read_supervisor()["retries"][request["id"]]["status"], "consumed")
        self.assertEqual(_consume_scope_retries(self.store, state, {}, {})[0], state)

    def test_parent_change_and_later_pause_prevent_dispatch(self):
        self.files["root.yaml"] = "changed-root"
        with self.assertRaisesRegex(ExtractError, "dependency changed"):
            monitor.request_restart(self.store, self.run_id, "P5:a", self.store.root, automatic=False)
        self.files["root.yaml"] = "approved-root"
        request = monitor.request_restart(self.store, self.run_id, "P5:a", self.store.root, automatic=False)
        self.store.request_control(self.run_id, "pause")
        state, _ = _consume_scope_retries(self.store, self.state, {}, {self.sibling: object()})
        self.assertEqual(state["nodes"]["a"]["status"], "cli_error")
        self.assertEqual(state["control"]["action"], "pause")
        self.assertEqual(self.store.read_supervisor()["retries"][request["id"]]["status"], "superseded")

    def _clock_request_without_ack(self):
        now = [time.time()]
        clock = Clock(self.store, self.store.root, now=lambda: now[0], sender=lambda *_: str(uuid4()))
        with patch.object(monitor, "inspect_runs", return_value=[]):
            clock.start(str(uuid4()), [self.run_id], auto_restart_limit=2, launch=False)
        row = {"run_id": self.run_id, "status": "running", "state_revision": 1,
               "anomalies": [{"classification": "transient_execution", "scope": "P5:a"}]}
        now[0] += 31
        with patch.object(monitor, "inspect_runs", return_value=[row]):
            result = clock.tick()
        self.assertEqual(result["observation"]["status"], "observed")
        ledger = self.store.read_supervisor()
        self.assertIsNone(ledger["clock"]["reminder"]["ack"])
        request = next(iter(ledger["retries"].values()))
        self.assertEqual(request["status"], "authorized")
        return clock, request

    def test_clock_observation_recovers_without_ack(self):
        _, request = self._clock_request_without_ack()
        state, _ = _consume_scope_retries(self.store, self.state, {}, {self.sibling: object()})
        self.assertEqual(state["nodes"]["a"]["status"], "draft")
        self.assertEqual(self.store.read_supervisor()["retries"][request["id"]]["status"], "consumed")

    def test_clock_stop_revokes_pending_request(self):
        clock, request = self._clock_request_without_ack()
        clock.stop()
        state, _ = _consume_scope_retries(self.store, self.state, {}, {self.sibling: object()})
        self.assertEqual(state["nodes"]["a"]["status"], "cli_error")
        self.assertEqual(self.store.read_supervisor()["retries"][request["id"]]["status"], "superseded")

    def test_orphan_timeout_keeps_healthy_sibling_and_original_budget(self):
        self.state["active_attempts"]["P5:a"] = self.attempt_id
        self.state["deadline"] = 105
        directory = self.store.run_dir(self.run_id) / "attempts" / self.sibling
        directory.mkdir()
        sibling = {**self.attempt, "attempt_id": self.sibling, "node": "b", "created_at": 90,
                   "timeout_seconds": 20, "directory": str(directory), "workspace": str(directory / "workspace")}
        write_json(directory / "attempt.json", sibling)
        write_json(Path(self.attempt["directory"]) / "attempt.json",
                   {**self.attempt, "created_at": 50, "timeout_seconds": 20})
        before = deepcopy(self.state)
        observed = []
        def reconcile(store, state, lock, *, scopes):
            observed.append(scopes)
            for scope in scopes:
                state["active_attempts"].pop(scope)
            return state, []
        with patch("schema_extract.tools.runner.termination_observed", return_value=False), \
                patch("schema_extract.tools.runner._reconcile", side_effect=reconcile), \
                patch("schema_extract.tools.runner.time.time", return_value=100):
            state, _ = _reconcile_orphans(self.store, self.state, {}, {}, set())
        self.assertEqual(observed, [{"P5:a"}])
        self.assertEqual(state["active_attempts"], {"P5:b": self.sibling})
        self.assertEqual(state["nodes"], before["nodes"])
        self.assertEqual(state["attempt_usage"], before["attempt_usage"])
        self.assertEqual((state["deadline"], state["config"]), (before["deadline"], before["config"]))
        # The run's original deadline also bounds a sibling whose own saved
        # invocation timeout has not yet elapsed.
        with patch("schema_extract.tools.runner.termination_observed", return_value=False), \
                patch("schema_extract.tools.runner._reconcile", side_effect=reconcile), \
                patch("schema_extract.tools.runner.time.time", return_value=106):
            _reconcile_orphans(self.store, state, {}, {}, set())
        self.assertEqual(observed, [{"P5:a"}, {"P5:b"}])


if __name__ == "__main__":
    unittest.main()
