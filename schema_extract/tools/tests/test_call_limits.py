"""Call additions preserve review reservations, accounting and single decisions."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from schema_extract.tools import stages
from schema_extract.tools.common import ExtractError
from schema_extract.tools.recovery import account
from schema_extract.tools.runner import _resume_failed_scopes
from schema_extract.tools.stages import (new_node, new_scope, next_calls, node_signature,
                                        selection_signature, settle, writing_signature)


def run_state():
    return {"run_id": "test-run", "status": "running", "stage": "P1", "node": "root", "role": "worker", "mode": "normal",
            "config": {"max_calls": 32},
            "stages": {key: new_scope() for key in ("P1", "P2", "P3", "P4", "P5", "P6")},
            "nodes": {}, "attempt_usage": {}, "active_attempts": {}, "feedback": [], "reason": None,
            "call_additions": [], "sessions": {}}


class CallLimitTests(unittest.TestCase):
    def test_elapsed_does_not_change_call_eligibility(self):
        state = run_state()
        with patch.object(stages.time, "time", return_value=10**12):
            state["stages"]["P1"].update(started_at=0, elapsed_seconds=10**11)
            self.assertIsNone(stages.call_exhaustion(state))

    def test_call_counts_preserve_review_and_p6_signoff(self):
        state = run_state()
        state["stages"]["P1"]["calls"] = 29
        self.assertIsNotNone(stages.call_exhaustion(state))
        state["role"] = "reviewer"
        self.assertIsNone(stages.call_exhaustion(state))
        state.update(stage="P6", mode="adjudicate")
        state["stages"]["P6"]["calls"] = 30
        self.assertIsNone(stages.call_exhaustion(state))
        state["stages"]["P6"].update(calls=31, adjudications=1)
        state["mode"] = "signoff"
        self.assertIsNone(stages.call_exhaustion(state))
        state["stages"]["P6"]["signoffs"] = 1
        self.assertIsNotNone(stages.call_exhaustion(state))

    def test_call_addition_is_explicit_idempotent_and_keeps_usage(self):
        state = run_state()
        state.update(status="call_limit_reached", reason={"code": "call_limit_reached"})
        scope = state["stages"]["P1"]
        scope.update(started_at=0, calls=31, revisions=4)
        attempt = {"attempt_id": "old-call", "stage": "P1", "node": "root"}
        account(state, attempt, 370)
        before = deepcopy(state)
        _resume_failed_scopes(state)
        self.assertEqual(state["status"], "call_limit_reached")
        self.assertTrue(stages.add_calls(state, 3, identity="grant", origin="resume"))
        self.assertFalse(stages.add_calls(state, 3, identity="grant", origin="resume"))
        _resume_failed_scopes(state)
        self.assertEqual(state["status"], "running")
        self.assertIsNone(state["reason"])
        self.assertIsNone(stages.call_exhaustion(state))
        self.assertEqual((scope["calls"], scope["revisions"], scope["elapsed_seconds"]), (31, 4, 370))
        self.assertEqual(state["attempt_usage"], before["attempt_usage"])
        self.assertEqual(state["config"], before["config"])
        self.assertEqual(stages.call_limit(state), 35)
        account(state, attempt, 370)
        self.assertEqual(scope["elapsed_seconds"], 370)
        with self.assertRaises(ExtractError):
            stages.add_calls(state, 4, identity="grant", origin="resume")

    def test_additions_do_not_clear_fences_or_used_decisions(self):
        for mode in ("adjudicate", "signoff"):
            state = run_state()
            state.update(stage="P6", status="blocked", role="reviewer", mode=mode,
                         active_attempts={"P6": "unconfirmed"})
            state["stages"]["P6"].update(adjudications=1, signoffs=1)
            stages.add_calls(state, 20, identity="grant", origin="resume")
            _resume_failed_scopes(state)
            self.assertEqual(state["status"], "blocked")
            self.assertIsNotNone(stages.call_exhaustion(state))
            self.assertEqual(state["active_attempts"], {"P6": "unconfirmed"})
        state = run_state()
        state.update(status="blocked", reason={"code": "call_limit_reached", "scopes": ["P6:single_decision_used"]})
        stages.add_calls(state, 20, identity="grant", origin="resume")
        _resume_failed_scopes(state)
        self.assertEqual(state["status"], "blocked")

    def test_dependency_ready_branches_remain_independent_of_call_stops(self):
        state = run_state()
        selected = [{"id": name, "instance": "kernel.py::copy(" + name + ")"} for name in ("root", "a", "b", "child")]
        writing = [{"id": name, "wait_for": waits} for name, waits in (("a", []), ("b", []), ("child", ["a"]))]
        state.update(stage="P5", role=None, selected_designs=selected, writing_plan=writing,
                     selection_signature=selection_signature(selected), writing_signature=writing_signature(writing),
                     input_fingerprint="fixed", checkpoint="merged", merged_checkpoint="merged", reopening=None)
        state["nodes"] = {entry["id"]: new_node(entry) for entry in selected}
        def approval(stage, node="root"):
            return {"stage": stage, "node": node, "input_fingerprint": "fixed", "generation": 0,
                    "node_signature": node_signature(state["nodes"][node]),
                    "selection_signature": state["selection_signature"], "writing_signature": state["writing_signature"]}
        for key in ("P1", "P2", "P3", "P4"):
            state["stages"][key].update(status="approved", approval=approval(key))
        state["nodes"]["root"].update(status="approved", merged=True, approval=approval("P3"), approved_generation=0)
        self.assertEqual({call["node"] for call in next_calls(state)}, {"a", "b"})
        state["nodes"]["b"].update(status="call_limit_reached", reason={"code": "call_limit_reached"})
        state["nodes"]["a"].update(status="approved", merged=True, approval=approval("P5", "a"), approved_generation=0)
        settle(state)
        self.assertEqual([call["node"] for call in next_calls(state)], ["child"])
        self.assertEqual(state["stage"], "P5")
        for node in ("b", "child"):
            state["nodes"][node].update(status="approved", merged=True, approval=approval("P5", node), approved_generation=0)
        settle(state)
        self.assertEqual((state["stage"], state["role"]), ("P6", "reviewer"))
        self.assertIsNone(next_calls(state)[0]["session_id"])


if __name__ == "__main__":
    unittest.main()
