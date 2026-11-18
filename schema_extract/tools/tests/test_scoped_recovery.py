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
from schema_extract.tools.stages import (new_node, new_scope, next_calls, node_signature,
                                        selection_signature, writing_signature)
from schema_extract.tools.store import Store


class ScopedRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / "runtime")
        self.store.initialize()
        self.run_id, self.attempt_id, self.sibling = (str(uuid4()) for _ in range(3))
        selected = [{"id": name, "instance": "kernel.py::copy(" + name + ")"}
                for name in ("root", "a", "b")]
        writing = [{"id": name, "wait_for": []} for name in ("a", "b")]
        nodes = {entry["id"]: new_node(entry) for entry in selected}
        nodes["root"].update(status="approved", merged=True, approved_generation=0,
                            approval={"stage": "P3", "node": "root", "input_fingerprint": "original-inputs",
                                      "generation": 0, "node_signature": node_signature(selected[0])})
        nodes["a"].update(status="cli_error", draft_checkpoint="draft-a")
        nodes["a"]["stage"].update(calls=1, cli_failures=1, elapsed_seconds=7)
        self.state = {"format": "schema_extract.run.v2", "run_id": self.run_id, "status": "running", "stage": "P5",
            "node": "root", "role": None, "mode": "normal", "selected_designs": selected,
            "writing_plan": writing, "selection_signature": selection_signature(selected),
            "writing_signature": writing_signature(writing), "plan_revision": 1, "nodes": nodes,
            "stages": {key: new_scope() for key in ("P1", "P2", "P3", "P4", "P5", "P6")},
            "executor": {"capabilities": ["scoped_recovery.v2"], "digest": "frozen-executor"}, "input_fingerprint": "original-inputs",
            "checkpoint": "merged", "merged_checkpoint": "merged", "reopening": None, "reopen_requests": {},
            "config": {"backend": "claude", "agent_binary": "/usr/bin/false", "model": "offline-fixture",
                       "worker_effort": "max", "reviewer_effort": "max",
                       "revision_limit": 5, "max_calls": 12, "retry_limit": 2},
            "active_attempts": {"P5:b": self.sibling},
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
            "node_signature": node_signature(selected[1]), "dependency_digests": {"root.yaml": "approved-root"}}
        result = {"outcome": "cli_error", "exit_code": 75, "termination": {"confirmed": True}}
        write_json(directory / "attempt.json", self.attempt)
        write_json(directory / "result.json", result)
        nodes["a"]["reason"] = {"code": "cli_retry", "classification": "transient_execution", "scope": "P5:a",
                                "attempt_id": self.attempt_id, "result_sha256": digest(result), "next_due_at": 0}
        self.store.save(self.state)
        self.files = {"root.yaml": "approved-root"}
        # Native exit/frozen-execution and actual source parsing are exercised
        # by test_current_recovery and test_current_returns. These unit cases
        # isolate the authorization ledger, controls and concurrent-scope fences.
        def source_check(state, attempt, workspace):
            if self.files != attempt["dependency_digests"]:
                raise ExtractError("actual dependency changed")
            return {"dependency_digests": deepcopy(self.files), "adopted_source_identities": {"root": "root-identity"}}
        for mocker in (patch.object(self.store, "verify_checkpoint", side_effect=lambda *_: (directory, {"files": self.files})),
                       patch.object(self.store, "validate_sources", side_effect=source_check),
                       patch.object(self.store, "writer_status", return_value="held"),
                       patch("schema_extract.tools.runtime.verify_execution", return_value={}),
                       patch("schema_extract.tools.recovery.termination_observed", return_value=True),
                       patch("schema_extract.tools.recovery.verified_result", side_effect=lambda attempt, result, termination: result)):
            mocker.start()
            self.addCleanup(mocker.stop)

    def test_request_survives_sibling_revision_and_preserves_its_fence_and_counts(self):
        request = monitor.request_restart(self.store, self.run_id, "P5:a", self.store.root)
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

    def test_actual_source_change_and_later_pause_prevent_dispatch(self):
        self.files["root.yaml"] = "changed-root"
        with self.assertRaisesRegex(ExtractError, "dependency changed"):
            monitor.request_restart(self.store, self.run_id, "P5:a", self.store.root)
        self.files["root.yaml"] = "approved-root"
        request = monitor.request_restart(self.store, self.run_id, "P5:a", self.store.root)
        self.store.request_control(self.run_id, "pause")
        state, _ = _consume_scope_retries(self.store, self.state, {}, {self.sibling: object()})
        self.assertEqual(state["nodes"]["a"]["status"], "cli_error")
        self.assertEqual(state["control"]["action"], "pause")
        self.assertEqual(self.store.read_supervisor()["retries"][request["id"]]["status"], "superseded")

    def test_old_healthy_orphans_keep_sibling_fences_and_original_counts(self):
        self.state["active_attempts"]["P5:a"] = self.attempt_id
        directory = self.store.run_dir(self.run_id) / "attempts" / self.sibling
        directory.mkdir()
        sibling = {**self.attempt, "attempt_id": self.sibling, "node": "b", "created_at": 90,
                   "directory": str(directory), "workspace": str(directory / "workspace")}
        write_json(directory / "attempt.json", sibling)
        write_json(Path(self.attempt["directory"]) / "attempt.json", {**self.attempt, "created_at": 50})
        before = deepcopy(self.state)
        observed = []
        def reconcile(store, state, lock, *, scopes):
            observed.append(scopes)
            for scope in scopes:
                state["active_attempts"].pop(scope)
            return state, []
        with patch("schema_extract.tools.runner.termination_observed", return_value=False), \
                patch("schema_extract.tools.runner._reconcile", side_effect=reconcile), \
                patch("schema_extract.tools.runner.time.time", return_value=10**12):
            state, _ = _reconcile_orphans(self.store, self.state, {}, {}, set())
        self.assertEqual(observed, [])
        self.assertEqual(state, before)
        with patch("schema_extract.tools.runner.termination_observed", side_effect=lambda a, *_, **kw: a["node"] == "a"), \
                patch("schema_extract.tools.runner._reconcile", side_effect=reconcile):
            state, _ = _reconcile_orphans(self.store, state, {}, {}, set())
        self.assertEqual(observed, [{"P5:a"}])
        self.assertEqual(state["active_attempts"], {"P5:b": self.sibling})
        self.assertEqual(state["nodes"], before["nodes"])
        self.assertEqual(state["attempt_usage"], before["attempt_usage"])
        self.assertEqual(state["config"], before["config"])


if __name__ == "__main__":
    unittest.main()
