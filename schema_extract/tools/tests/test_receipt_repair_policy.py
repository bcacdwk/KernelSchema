"""Proposal repairs preserve drafts and budgets without issuing review decisions."""

from copy import deepcopy
from pathlib import Path
import unittest

from schema_extract.tools.common import manifest, write_json
from schema_extract.tools.prompts import build_prompt
from schema_extract.tools.runner import _receive
from schema_extract.tools.stages import DOCUMENT_CONCLUSIONS
from schema_extract.tools.tests import test_stage_flow


class ReceiptRepairPolicyTests(unittest.TestCase):
    def setUp(self):
        self.flow = test_stage_flow.StageFlowTests(methodName="runTest")
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)
        self.flow.create()

    def receive(self, attempt, result):
        f = self.flow
        write_json(Path(attempt["directory"]) / "result.json", result)
        f.state = _receive(f.store, f.state, f.lock, attempt, result, None)
        # The deterministic backend has no native process/usage to settle.
        f.state["unsettled_attempts"].pop(attempt["attempt_id"], None)
        f.store.save(f.state)

    def test_worker_and_reviewer_omissions_share_budget_and_keep_only_new_approvals(self):
        f = self.flow
        failed, passed = [], []
        worker_session = None
        for role in ("worker", "reviewer"):
            attempt = f.prepare(stage="P1", role=role)
            result = f.result(attempt)
            failed.append(attempt["attempt_id"])
            workspace = Path(attempt["workspace"])
            original = manifest(workspace)
            accepted_before = deepcopy(f.state["accepted_receipts"])
            if role == "worker":
                result["response"]["source_evidence"][0]["category"] = "host"
                worker_session = result["session_id"]
            else:
                self.assertNotEqual(result["session_id"], worker_session)
                result["response"]["conclusions"] = [item for item in result["response"]["conclusions"]
                                                       if item["subject"] not in DOCUMENT_CONCLUSIONS]
            self.receive(attempt, result)
            self.assertEqual((f.state["status"], f.state["stage"], f.state["role"]), ("running", "P1", role))
            self.assertEqual(f.state["reason"]["code"], "receipt_repair")
            self.assertEqual(f.state["accepted_receipts"], accepted_before)
            self.assertIsNone(f.state["stages"]["P1"]["approval"])
            self.assertEqual(manifest(workspace), original)
            self.assertFalse((Path(attempt["directory"]) / "validated.json").exists())
            reason = f.state["reason"]["summary"]
            next_attempt = f.prepare(stage="P1", role=role)
            self.assertNotEqual(next_attempt["attempt_id"], attempt["attempt_id"])
            self.assertEqual(next_attempt["session_id"], result["session_id"])
            self.assertIn(reason, build_prompt(f.state, f.lock, next_attempt))
            if role == "worker":
                self.assertNotEqual(next_attempt["base_checkpoint"], attempt["base_checkpoint"])
                self.assertEqual(manifest(Path(next_attempt["workspace"])),
                                 manifest(Path(attempt["directory"]) / "recovered-normalized"))
            else:
                self.assertEqual(next_attempt["base_checkpoint"], attempt["base_checkpoint"])
                self.assertEqual(next_attempt["writable_files"], [])
            self.receive(next_attempt, f.result(next_attempt))
            passed.append(next_attempt["attempt_id"])
        scope = f.state["stages"]["P1"]
        self.assertEqual((scope["calls"], scope["receipt_repairs"], scope["invalid_receipts"]), (4, 2, 2))
        self.assertEqual(set(f.state["accepted_receipts"]), set(passed))
        self.assertTrue(set(failed).isdisjoint(f.state["accepted_receipts"]))
        self.assertEqual(scope["approval"]["attempt_id"], passed[-1])
        self.assertEqual(f.state["stage"], "P2")
        self.assertEqual(f.state["call_additions"], [])

    def test_real_revise_and_blocked_decisions_do_not_enter_receipt_repair(self):
        f = self.flow
        for decision in ("revise", "blocked"):
            f.step(stage="P1", role="worker")
            attempt = f.prepare(stage="P1", role="reviewer")
            result = f.result(attempt)
            result["response"].update(decision=decision, conclusions=[], issues=[{
                "kind": "source_misread", "scope": "root.yaml", "detail": "The task interpretation needs correction",
                "evidence": [0], "recovery_condition": "Resolve the source interpretation"}])
            self.receive(attempt, result)
            self.assertIn(attempt["attempt_id"], f.state["accepted_receipts"])
            self.assertEqual(f.state["stages"]["P1"]["receipt_repairs"], 0)
            self.assertEqual(f.state["stages"]["P1"]["invalid_receipts"], 0)
            self.assertIsNone(f.state["stages"]["P1"]["approval"])
            if decision == "revise":
                self.assertEqual((f.state["role"], f.state["stages"]["P1"]["revisions"]), ("worker", 1))
            else:
                self.assertEqual(f.state["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
