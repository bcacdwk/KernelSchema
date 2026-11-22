"""Proposal repairs preserve drafts and budgets without issuing review decisions."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from schema_extract.tools.common import manifest, write_json
from schema_extract.tools.agent import _schema_check
from schema_extract.tools.prompts import build_prompt
from schema_extract.tools.receipts import proposal_schema
from schema_extract.tools.runner import _receive
from schema_extract.tools.stages import DOCUMENT_CONCLUSIONS, call_exhaustion, call_view
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
            prompt = build_prompt(f.state, f.lock, next_attempt)
            invocation = json.loads(prompt.split('<invocation>\n', 1)[1].split('\n</invocation>', 1)[0])
            self.assertIn(reason, [item.get('summary') for item in invocation['feedback'] if isinstance(item, dict)])
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

    def test_third_ordinary_repair_keeps_draft_nonce_counts_and_review_reservations(self):
        f = self.flow
        nonces, checkpoint = [], None
        for count in (1, 2, 3):
            attempt = f.prepare(stage='P1', role='worker')
            nonces.append(attempt['attempt_id'])
            result = f.result(attempt)
            raw = manifest(Path(attempt['workspace']))
            if count == 1:
                result['response']['decision'] = 'approve'
            elif count == 2:
                result['response']['source_evidence'][0]['path'] = 'outside-manifest.py'
            else:
                result['response']['source_evidence'] = [dict(result['response']['source_evidence'][0], line_end='bad') for _ in range(20)]
                checked = _schema_check(proposal_schema(attempt, source_files=f.lock['source']['files']), result['response'], attempt['attempt_id'])
                result.update(outcome='receipt_error', response_validation={**checked, 'native_output_matches': True})
            self.receive(attempt, result)
            self.assertEqual((f.state['status'], f.state['role']), ('running', 'worker'))
            self.assertEqual(f.state['stages']['P1']['receipt_repairs'], count)
            self.assertEqual(f.state['stages']['P1']['invalid_receipts'], count)
            self.assertEqual(f.state['stages']['P1']['calls'], count)
            self.assertFalse(f.state['accepted_receipts'])
            self.assertIsNone(f.state['stages']['P1']['approval'])
            self.assertEqual(manifest(Path(attempt['workspace'])), raw)
            if count == 1:
                self.assertNotEqual(f.state['checkpoint'], attempt['base_checkpoint'])
            if checkpoint is not None: self.assertEqual(f.state['checkpoint'], checkpoint)
            checkpoint = f.state['checkpoint']
            self.assertIsNone(call_exhaustion(call_view(f.state, attempt)))
        self.assertEqual(len(set(nonces)), 3)
        self.assertEqual(f.state['call_additions'], [])
        self.assertIn('proposal.source_evidence[0].line_end', f.state['reason']['summary'])
        self.assertIn('omitted 8 diagnostics', f.state['reason']['summary'])
        self.assertIn('not checked:', f.state['reason']['summary'])
        view = call_view(deepcopy(f.state), attempt)
        view['config']['max_calls'] = 6
        self.assertIsNotNone(call_exhaustion(view))  # worker still reserves reviewer + adjudication + signoff
        view['role'] = 'reviewer'
        view['config']['max_calls'] = 5
        self.assertIsNotNone(call_exhaustion(view))  # reviewer still reserves adjudication + signoff
        corrected = f.prepare(stage='P1', role='worker')
        self.receive(corrected, f.result(corrected))
        self.assertEqual(f.state['role'], 'reviewer')
        self.assertIsNone(f.state['stages']['P1']['approval'])
        self.assertEqual(f.state['stages']['P1']['receipt_repairs'], 3)
        self.assertEqual(set(f.state['accepted_receipts']), {corrected['attempt_id']})

    def test_native_disagreement_precedes_schema_repair_and_retains_finite_cli_limit(self):
        f = self.flow
        for count in (1, 2, 3):
            attempt = f.prepare(stage='P1', role='worker')
            result = f.result(attempt)
            result['response'].pop('summary')
            result.update(outcome='receipt_error', response_validation={
                'schema_valid': False, 'nonce_valid': True, 'native_output_matches': False,
                'schema_errors': [{'path': [], 'validator': 'required'}]})
            self.receive(attempt, result)
            self.assertEqual(f.state['stages']['P1']['cli_failures'], count)
            self.assertEqual(f.state['stages']['P1']['receipt_repairs'], 0)
            self.assertFalse(f.state['accepted_receipts'])
            self.assertIsNone(f.state['stages']['P1']['approval'])
            self.assertIn('native output channels disagree', f.state['reason']['summary'])
            self.assertEqual(f.state['status'], 'running' if count < 3 else 'cli_error')
        self.assertEqual(f.state['reason']['code'], 'cli_retry_limit_reached')

    def test_nonce_failure_cannot_turn_into_ordinary_repair_or_save_a_new_draft(self):
        f = self.flow
        attempt = f.prepare(stage='P1', role='worker')
        result = f.result(attempt)
        checkpoint = f.state['checkpoint']
        result['response']['attempt_id'] = 'another-invocation'
        result['response'].pop('summary')
        result['outcome'] = 'receipt_error'
        self.receive(attempt, result)
        self.assertEqual(f.state['status'], 'invalid_receipt')
        self.assertEqual(f.state['checkpoint'], checkpoint)
        self.assertEqual(f.state['stages']['P1']['receipt_repairs'], 0)
        self.assertFalse(f.state['accepted_receipts'])

    def test_p5_long_handoff_can_approve_the_real_candidate(self):
        f = self.flow
        f.create(('root', 'branch'))
        f.through_root()
        f.step(stage='P5', node='branch', role='worker')
        attempt = f.prepare(stage='P5', node='branch', role='reviewer')
        result = f.result(attempt)
        result['response']['handoff'] = '界\n' * 700
        self.receive(attempt, result)
        self.assertEqual(f.state['nodes']['branch']['approval']['attempt_id'], attempt['attempt_id'])


if __name__ == "__main__":
    unittest.main()
