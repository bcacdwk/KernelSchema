"""Finite review allowances and resumable credit without resetting old decisions."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from schema_extract.tools import budget
from schema_extract.tools.common import ExtractError
from schema_extract.tools.recovery import account
from schema_extract.tools.runner import _resume_failed_scopes
from schema_extract.tools.stages import new_node, new_scope, next_calls, settle


def run_state():
    return {"run_id": "test-run", "status": "running", "stage": "P1", "node": "root", "role": "worker", "mode": "normal",
            "config": {"operator_seconds": 600, "stage_seconds": 600, "call_seconds": 240,
                       "adjudication_seconds": 50, "max_calls": 32}, "deadline": 600,
            "stages": {key: new_scope() for key in ("P1", "P2", "P3", "P4", "P5", "P6")},
            "nodes": {}, "attempt_usage": {}, "active_attempts": {}, "feedback": [], "reason": None,
            "budget_extensions": [], "sessions": {}}


class BudgetTests(unittest.TestCase):
    def test_writing_and_checking_preserve_review_then_one_adjudication(self):
        state = run_state()
        state["stages"]["P1"].update(started_at=0, deadline=600)
        with patch.object(budget.time, "time", return_value=0):
            self.assertEqual(budget.call_budget(state), (240, None))
        # Repeated worker continuations cannot eat the subsequent review's 240s.
        with patch.object(budget.time, "time", return_value=310):
            self.assertIsNotNone(budget.call_budget(state)[1])
            self.assertEqual(budget.check_timeout(state), 0)
            state["role"] = "reviewer"
            self.assertEqual(budget.call_budget(state), (240, None))
        with patch.object(budget.time, "time", return_value=550):
            self.assertIsNotNone(budget.call_budget(state)[1])
            state["mode"] = "adjudicate"
            self.assertEqual(budget.call_budget(state), (50, None))

    def test_call_counts_preserve_review_and_p6_signoff(self):
        state = run_state()
        state["stages"]["P1"]["calls"] = 30
        with patch.object(budget.time, "time", return_value=0):
            self.assertIsNotNone(budget.call_budget(state)[1])
            state["role"] = "reviewer"
            self.assertIsNone(budget.call_budget(state)[1])
            state.update(stage="P6", mode="adjudicate")
            state["stages"]["P6"]["calls"] = 30
            self.assertEqual(budget.call_budget(state), (50, None))
            state["stages"]["P6"].update(calls=31, adjudications=1)
            state["mode"] = "signoff"
            self.assertEqual(budget.call_budget(state), (50, None))
            state["stages"]["P6"]["signoffs"] = 1
            self.assertIsNotNone(budget.call_budget(state)[1])

    def test_expired_credit_is_explicit_idempotent_and_keeps_usage(self):
        state = run_state()
        state.update(status="budget_exhausted", reason={"code": "budget_exhausted"})
        scope = state["stages"]["P1"]
        scope.update(started_at=0, deadline=600, calls=31, revisions=4)
        attempt = {"attempt_id": "old-call", "stage": "P1", "node": "root"}
        account(state, attempt, 370)
        before = deepcopy(state)
        with patch.object(budget.time, "time", return_value=10000):
            _resume_failed_scopes(state)
            self.assertEqual(state["status"], "budget_exhausted")
            self.assertTrue(budget.extend(state, budget.addition(600, 2), identity="grant", origin="resume"))
            self.assertFalse(budget.extend(state, budget.addition(600, 2), identity="grant", origin="resume"))
            _resume_failed_scopes(state)
            self.assertEqual(state["status"], "running")
            self.assertIsNone(state["reason"])
            self.assertIsNone(budget.call_budget(state)[1])
        self.assertEqual(state["deadline"], 10600)
        self.assertEqual(scope["deadline"], 10600)
        self.assertIsNone(state["stages"]["P2"]["deadline"])
        self.assertEqual((scope["calls"], scope["revisions"], scope["elapsed_seconds"]), (31, 4, 370))
        self.assertEqual(state["attempt_usage"], before["attempt_usage"])
        self.assertEqual(state["config"], before["config"])
        self.assertEqual(budget.call_limit(state), 34)
        account(state, attempt, 370)
        self.assertEqual(scope["elapsed_seconds"], 370)
        with self.assertRaises(ExtractError):
            budget.extend(state, budget.addition(601, 2), identity="grant", origin="resume")

    def test_additions_do_not_clear_fences_or_used_decisions(self):
        for mode in ("adjudicate", "signoff"):
            state = run_state()
            state.update(stage="P6", status="blocked", role="reviewer", mode=mode,
                         active_attempts={"P6": "unconfirmed"})
            state["stages"]["P6"].update(adjudications=1, signoffs=1)
            with patch.object(budget.time, "time", return_value=10000):
                budget.extend(state, budget.addition(7200, 20), identity="grant", origin="resume")
                _resume_failed_scopes(state)
                self.assertEqual(state["status"], "blocked")
                self.assertIsNotNone(budget.call_budget(state)[1])
                self.assertEqual(state["active_attempts"], {"P6": "unconfirmed"})
        state = run_state()
        state.update(status="blocked", reason={"code": "continuation_budget_exhausted", "scopes": ["P6:single_decision_used"]})
        with patch.object(budget.time, "time", return_value=0):
            budget.extend(state, budget.addition(7200, 20), identity="grant", origin="continue")
            _resume_failed_scopes(state)
        self.assertEqual(state["status"], "blocked")

    def test_dependency_ready_branches_remain_independent_of_budget_stops(self):
        state = run_state()
        state.update(stage="P5", role=None, plan=[{"id": name, "file": name + ".yaml", "parent": parent}
                     for name, parent in (("root", ""), ("a", "root"), ("b", "root"), ("child", "a"))],
                     checkpoint="merged", merged_checkpoint="merged", reopening=None)
        state["nodes"] = {entry["id"]: new_node(entry) for entry in state["plan"]}
        for key in ("P1", "P2", "P3", "P4"):
            state["stages"][key].update(status="approved", approval={"saved": True})
        state["nodes"]["root"].update(status="approved", merged=True, approval={"saved": True}, approved_generation=0)
        self.assertEqual({call["node"] for call in next_calls(state)}, {"a", "b"})
        state["nodes"]["b"].update(status="budget_exhausted", reason={"code": "budget_exhausted"})
        state["nodes"]["a"].update(status="approved", merged=True, approval={"saved": True}, approved_generation=0)
        settle(state)
        self.assertEqual([call["node"] for call in next_calls(state)], ["child"])
        self.assertEqual(state["stage"], "P5")
        for node in ("b", "child"):
            state["nodes"][node].update(status="approved", merged=True, approval={"saved": True}, approved_generation=0)
        settle(state)
        self.assertEqual((state["stage"], state["role"]), ("P6", "reviewer"))
        self.assertIsNone(next_calls(state)[0]["session_id"])


if __name__ == "__main__":
    unittest.main()
