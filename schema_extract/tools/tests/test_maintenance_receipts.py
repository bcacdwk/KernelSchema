"""Settled native proposals can be corrected within their original maintenance scope."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from schema_extract.tools import maintenance
from schema_extract.tools.agent import AgentCLI
from schema_extract.tools.common import ExtractError, manifest, read_json, write_json
from schema_extract.tools.runner import _resume_maintenance_receipts, run
from schema_extract.tools.tests import test_stage_flow


class MaintenanceReceiptTests(unittest.TestCase):
    def setUp(self):
        self.flow = test_stage_flow.StageFlowTests(methodName="runTest")
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)
        f = self.flow
        source = f.directory / "source/kernel.py"
        source.write_text(source.read_text() + "# fixed evidence context\n" * 16)
        self.source_lines = len(source.read_text().splitlines())
        binary = f.directory / "receipt-cli"
        binary.write_text("#!" + sys.executable + "\n" + '''
import json, sys
from pathlib import Path
from uuid import uuid4
if '--version' in sys.argv:
    print('maintenance-receipt-fixture')
    raise SystemExit()
response = json.load(sys.stdin)
saved = Path.home() / 'receipt-session.json'
if saved.exists():
    session = json.loads(saved.read_text())
    assert sys.argv[sys.argv.index('--resume') + 1] == session
else:
    assert '--resume' not in sys.argv
    session = str(uuid4())
    saved.write_text(json.dumps(session))
print(json.dumps({'type': 'system', 'subtype': 'init', 'session_id': session}))
print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
                  'session_id': session, 'structured_output': response}))
''')
        binary.chmod(0o755)
        f.create(agent_binary=str(binary))
        f.step(stage="P1", role="worker")
        f.step(stage="P1", role="reviewer")
        f.step(stage="P2", role="worker")
        checkpoint, _ = f.store.verify_checkpoint(f.state)
        self.artifact_lines = len((checkpoint / "root.yaml").read_text().splitlines())
        self.candidate = f.state["checkpoint"]
        self.plan = deepcopy(f.state["plan_draft"])
        self.accepted = deepcopy(f.state["accepted_receipts"])
        self.p1 = deepcopy(f.state["stages"]["P1"])
        self.before = deepcopy(f.state["stages"]["P2"])
        self.cli = AgentCLI(backend="claude", executable=str(binary), model=f.state["config"]["model"])
        self.calls = []

    def install(self, *, patch=False, fields=None):
        f = self.flow
        maintenance.submit_request(f.store, f.state["run_id"], issue_id="correct-current-receipt", tasks=[
            {"stage": "P2", "node": "root", "action": "patch" if patch else "review",
             "fields": fields if fields is not None else [{"path": "dispatch.md", "fields": ["*"]}] if patch else [],
             "reason": "Independently review the completed current candidate"}])
        f.state = maintenance.consume_requests(f.store, f.state)
        f.state = maintenance.mark_drained(f.store, f.state)

    def invoke(self, kind, *, resume=False, pause=False):
        test, f = self, self.flow

        class NativeReview:
            def run(self, *, attempt, prompt, schema, **kwargs):
                test.calls.append(deepcopy(attempt))
                if kind == "worker_shape":
                    draft = Path(attempt["workspace"]) / "dispatch.md"
                    draft.write_text(draft.read_text().replace("临时机器行为验证材料", "临时机器行为验证材料，保留本次合法说明"))
                proposal = f.backend.proposal(attempt)
                proposal.pop("reopen", None)
                if attempt.get("purpose") == "maintenance":
                    proposal["revision_scope"] = attempt["revision_scope"]
                if attempt["mode"] == "adjudicate":
                    proposal["adjudication"] = {"action": "select", "checkpoint": attempt["base_checkpoint"], "patches": []}
                if kind == "artifact_range":
                    proposal["artifact_evidence"].extend([f.backend.anchor("root.yaml") for _ in range(28)])
                    proposal["artifact_evidence"][29]["line_end"] = test.artifact_lines + 1
                elif kind == "source_range":
                    proposal["source_evidence"][0]["line_end"] = test.source_lines + 1
                elif kind in {"shape", "worker_shape"}:
                    proposal.pop("summary")
                else:
                    proposal["source_evidence"][0]["line_end"] = 13
                    next(item for item in proposal["artifact_evidence"] if item["path"] == "root.yaml")["line_end"] = 38
                    if kind == "blocked":
                        proposal.update(decision="blocked", issues=[{
                            "kind": "source_misread", "scope": "root.yaml", "detail": "Semantic issue remains",
                            "evidence": [0], "recovery_condition": "Resolve the source interpretation before proceeding"}])
                if resume:
                    test.assertIn(f"file has {test.artifact_lines} lines" if kind == "shape" else "receipt", prompt)
                result = test.cli.run(attempt=attempt, prompt=json.dumps(proposal), schema=schema)
                test.assertTrue(result["native_status"]["success"])
                test.assertEqual(result["response"], proposal)
                if pause:
                    f.store.request_control(f.state["run_id"], "pause")
                return result

        f.state = run(f.store, f.state["run_id"], resume=resume, transport=NativeReview())
        return f.state

    def task(self, state=None):
        return next(iter((state or self.flow.state)["maintenance"]["tasks"].values()))

    def review_result(self, attempt):
        f = self.flow
        proposal = f.backend.proposal(attempt)
        proposal.pop("reopen", None)
        proposal["revision_scope"] = attempt["revision_scope"]
        with patch.object(f.backend, "proposal", return_value=proposal):
            return f.result(attempt)

    def test_empty_revise_reviews_the_installed_plan_and_preserves_real_feedback(self):
        self.install(fields=[{"path": "dispatch.md", "fields": ["*"]}])
        f = self.flow
        attempt = f.prepare(stage="P2", role="reviewer")
        result = self.review_result(attempt)
        result["response"].pop("reopen", None)
        result["response"].update(decision="revise", plan=[], revision_scope=attempt["revision_scope"], issues=[{
            "kind": "schema_conformance", "scope": "dispatch.md", "detail": "Clarify this candidate rationale",
            "evidence": [0], "recovery_condition": "Preserve the submitted source selection"}])
        f.accept(attempt, result)
        self.assertEqual(f.state["stages"]["P2"]["check_failures"], self.before["check_failures"])
        self.assertEqual(read_json(Path(attempt["directory"]) / "check.json")["exit_code"], 0)
        self.assertEqual(f.state["plan_draft"], self.plan)
        self.assertIsNone(f.state["selected_designs"])
        self.assertIsNone(f.state["stages"]["P2"]["approval"])
        self.assertEqual(self.task()["role"], "worker")
        self.assertIn("Clarify this candidate rationale", json.dumps(self.task()))

    def test_maintenance_review_still_rejects_plan_changes_and_raw_writes(self):
        self.install()
        f = self.flow
        attempt = f.prepare(stage="P2", role="reviewer")
        result = self.review_result(attempt)
        result["response"].pop("reopen", None)
        result["response"]["revision_scope"] = attempt["revision_scope"]
        result["response"]["plan"][0]["instance"] = "kernel.py::copy(other-instance)"
        with self.assertRaisesRegex(ExtractError, "reviewer plan differs"):
            f.accept(attempt, result)
        result["response"]["plan"] = deepcopy(self.plan)
        raw = Path(attempt["workspace"]) / "dispatch.md"
        raw.write_text(raw.read_text() + "\nUnreviewed mutation\n")
        with self.assertRaisesRegex(ExtractError, "read-only|outside stage scope"):
            f.accept(attempt, result)
        self.assertIsNone(f.store.load(f.state["run_id"])["stages"]["P2"]["approval"])

    def assert_unapproved(self, count):
        f = self.flow
        self.assertEqual(f.state["status"], "blocked")
        self.assertEqual(f.state["maintenance"]["status"], "blocked")
        self.assertEqual(self.task()["status"], "invalid_receipt")
        self.assertEqual(f.state["checkpoint"], self.candidate)
        self.assertEqual(f.state["plan_draft"], self.plan)
        self.assertIsNone(f.state["selected_designs"])
        self.assertIsNone(f.state["stages"]["P2"]["approval"])
        self.assertEqual(f.state["accepted_receipts"], self.accepted)
        self.assertEqual(f.state["stages"]["P1"], self.p1)
        self.assertEqual(f.state["stages"]["P2"]["calls"], self.before["calls"] + count)
        self.assertEqual(f.state["stages"]["P2"]["invalid_receipts"], count)
        self.assertFalse(any(f.state[key] for key in ("active_attempts", "unconfirmed_attempts", "unsettled_attempts")))

    def test_resume_corrects_range_and_shape_in_exact_session_without_refunding_or_accepting_old_approve(self):
        self.install()
        f = self.flow
        identity = self.task()["task_id"]
        self.invoke("artifact_range")
        self.assert_unapproved(1)
        self.assertIn(f"artifact_evidence[29] root.yaml:1-{self.artifact_lines + 1}", self.task()["reason"]["summary"])
        self.assertIn(f"file has {self.artifact_lines} lines", self.task()["reason"]["summary"])
        first = self.calls[0]
        native = read_json(Path(first["directory"]) / "result.json")
        self.assertEqual(f.state["sessions"][first["session_group"]]["reviewer"], native["session_id"])
        old_usage = deepcopy(f.state["attempt_usage"])
        self.invoke("shape", resume=True)
        self.assert_unapproved(2)
        self.assertEqual(self.task()["reason"]["code"], "receipt_repair")
        self.assertEqual(f.state["stages"]["P2"]["receipt_repairs"], 2)
        self.invoke("approve", resume=True, pause=True)
        self.assertEqual((f.state["status"], f.state["stage"], f.state["role"]), ("paused", "P3", "worker"))
        self.assertEqual(f.state["maintenance"]["status"], "complete")
        self.assertEqual(f.state["selected_designs"], self.plan)
        self.assertEqual(f.state["stages"]["P2"]["approval"]["attempt_id"], self.calls[-1]["attempt_id"])
        self.assertEqual(set(f.state["accepted_receipts"]) - set(self.accepted), {self.calls[-1]["attempt_id"]})
        self.assertEqual((f.state["stages"]["P2"]["calls"], f.state["stages"]["P2"]["invalid_receipts"],
                          f.state["stages"]["P2"]["receipt_repairs"]), (self.before["calls"] + 3, 2, 2))
        self.assertEqual(f.state["call_additions"], [])
        for attempt_id, usage in old_usage.items():
            self.assertEqual(f.state["attempt_usage"][attempt_id], usage)
        for call in self.calls:
            self.assertEqual((call["purpose"], call["maintenance_task_id"], call["maintenance_step"], call["role"]),
                             ("maintenance", identity, "review", "reviewer"))
            self.assertEqual(call["base_checkpoint"], self.candidate)
            self.assertEqual(call["session_group"], first["session_group"])
            self.assertEqual(call["writable_files"], [])
            self.assertEqual(manifest(Path(call["workspace"])), first["dependency_digests"])
            self.assertTrue(f.state["attempt_usage"][call["attempt_id"]]["settled"])
        self.assertEqual([call["session_id"] for call in self.calls], [None, native["session_id"], native["session_id"]])

    def test_retry_requires_unchanged_candidate_settled_usage_and_empty_fences(self):
        self.install()
        self.invoke("source_range")
        self.assert_unapproved(1)
        f = self.flow
        self.assertIn(f"source_evidence[0] kernel.py:1-{self.source_lines + 1}", self.task()["reason"]["summary"])
        self.assertIn(f"file has {self.source_lines} lines", self.task()["reason"]["summary"])
        attempt_id = self.calls[0]["attempt_id"]
        variants = []
        for name in ("active_attempts", "unconfirmed_attempts", "unsettled_attempts"):
            state = deepcopy(f.state)
            state[name] = {"P2" if name == "active_attempts" else attempt_id: attempt_id}
            variants.append(state)
        state = deepcopy(f.state)
        state["attempt_usage"][attempt_id]["settled"] = False
        variants.append(state)
        state = deepcopy(f.state)
        self.task(state).update(status="blocked", reason={"code": "maintenance_review_blocked"})
        variants.append(state)
        state = deepcopy(f.state)
        state["sessions"][self.calls[0]["session_group"]]["reviewer"] = None
        variants.append(state)
        state = deepcopy(f.state)
        self.task(state)["step"] = "changed"
        variants.append(state)
        state = deepcopy(f.state)
        state["stages"]["P2"]["calls"] = state["config"]["max_calls"]
        variants.append(state)
        for state in variants:
            before = deepcopy(state)
            _resume_maintenance_receipts(f.store, state)
            self.assertEqual(state, before)
        for change in ("draft_checkpoint", "role"):
            state = deepcopy(f.state)
            self.task(state)[change] = "changed"
            with self.assertRaises(ExtractError):
                _resume_maintenance_receipts(f.store, state)
            self.assertEqual(self.task(state)["status"], "invalid_receipt")
        state = deepcopy(f.state)
        state["stages"]["P2"]["generation"] += 1
        with self.assertRaises(ExtractError):
            _resume_maintenance_receipts(f.store, state)
        state = deepcopy(f.state)
        state["input_fingerprint"] = "changed"
        with self.assertRaises(ExtractError):
            _resume_maintenance_receipts(f.store, state)
        # Positive process evidence is required even when the state fences are empty.
        gate = Path(self.calls[0]["directory"]) / "cli-exit.json"
        gate.unlink()
        state = deepcopy(f.state)
        _resume_maintenance_receipts(f.store, state)
        self.assertEqual(state, f.state)

    def test_worker_saved_draft_resumes_in_its_session_then_gets_independent_review(self):
        self.install(patch=True)
        f = self.flow
        self.invoke("worker_shape")
        first = self.calls[0]
        task = self.task()
        saved = task["draft_checkpoint"]
        self.assertEqual(task["status"], "invalid_receipt")
        self.assertNotEqual(saved, first["base_checkpoint"])
        self.assertEqual(saved, f.state["attempt_usage"][first["attempt_id"]]["result_checkpoint"])
        self.assertEqual(f.state["accepted_receipts"], self.accepted)
        native = read_json(Path(first["directory"]) / "result.json")
        old_usage = deepcopy(f.state["attempt_usage"][first["attempt_id"]])
        self.invoke("approve", resume=True, pause=True)
        second = self.calls[1]
        self.assertEqual(second["base_checkpoint"], saved)
        self.assertEqual(second["session_id"], native["session_id"])
        self.assertEqual((second["role"], self.task()["step"]), ("worker", "verify"))
        self.assertNotIn(first["attempt_id"], f.state["accepted_receipts"])
        self.assertIsNone(f.state["stages"]["P2"]["approval"])
        self.assertEqual(f.state["attempt_usage"][first["attempt_id"]], old_usage)
        self.invoke("approve", resume=True, pause=True)
        review = self.calls[2]
        self.assertEqual(review["role"], "reviewer")
        self.assertIsNone(review["session_id"])
        self.assertNotEqual(review["session_scope"], second["session_scope"])
        self.assertEqual(f.state["stages"]["P2"]["approval"]["attempt_id"], review["attempt_id"])
        self.assertEqual(self.task()["status"], "done")
        checkpoint, _ = f.store.verify_checkpoint(f.state)
        self.assertIn("保留本次合法说明", (checkpoint / "dispatch.md").read_text())
        self.assertEqual(f.state["stages"]["P2"]["calls"], self.before["calls"] + 3)

    def test_worker_retry_rejects_another_checkpoint_wrong_parent_bytes_and_repair_drift(self):
        self.install(patch=True)
        self.invoke("worker_shape")
        f, first = self.flow, self.calls[0]
        state = deepcopy(f.state)
        saved = self.task()["draft_checkpoint"]
        artifacts, info = f.store.verify_checkpoint(state, saved)
        for attempt_id, parent in (("another-attempt", first["base_checkpoint"]), (first["attempt_id"], saved)):
            other = f.store.checkpoint(state, artifacts, attempt_id=attempt_id, expected_files=info["files"],
                                       parent=parent, stage=first["stage"], node=first["node"])
            changed = deepcopy(state)
            self.task(changed)["draft_checkpoint"] = other
            changed["attempt_usage"][first["attempt_id"]]["result_checkpoint"] = other
            with self.assertRaises(ExtractError):
                _resume_maintenance_receipts(f.store, changed)
            self.assertEqual(self.task(changed)["status"], "invalid_receipt")
        repair_path = Path(first["directory"]) / "repair.json"
        repair = read_json(repair_path)
        write_json(repair_path, {**repair, "changed": True})
        with self.assertRaises(ExtractError):
            _resume_maintenance_receipts(f.store, deepcopy(state))
        write_json(repair_path, repair)
        draft = artifacts / "dispatch.md"
        draft.chmod(0o644)
        draft.write_text(draft.read_text() + "篡改\n")
        with self.assertRaises(ExtractError):
            _resume_maintenance_receipts(f.store, deepcopy(state))

    def test_semantic_blocked_review_stays_blocked_on_formal_resume(self):
        self.install()
        self.invoke("blocked")
        f = self.flow
        self.assertEqual(self.task()["status"], "blocked")
        previous = deepcopy(f.state)
        self.invoke("approve", resume=True)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(f.state["maintenance"], previous["maintenance"])
        self.assertEqual(f.state["attempt_usage"], previous["attempt_usage"])
        self.assertEqual(f.state["accepted_receipts"], previous["accepted_receipts"])
        self.assertIsNone(f.state["stages"]["P2"]["approval"])

    def test_native_adjudication_session_is_recorded_but_rejected_decision_cannot_restart(self):
        f = self.flow
        f.state["mode"] = "adjudicate"
        f.state["stages"]["P2"]["status"] = "adjudicating"
        f.store.save(f.state)
        self.invoke("source_range")
        first = self.calls[0]
        native = read_json(Path(first["directory"]) / "result.json")
        self.assertEqual(f.state["sessions"]["P2:adjudicate"]["reviewer"], native["session_id"])
        self.assertEqual(f.state["status"], "blocked")
        self.assertEqual(f.state["stages"]["P2"]["adjudications"], 1)
        self.invoke("approve", resume=True)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(f.state["accepted_receipts"], self.accepted)
        self.assertEqual(f.state["stages"]["P2"]["adjudications"], 1)


if __name__ == "__main__":
    unittest.main()
