"""Current full-document inputs, distinct sessions and bounded upstream repair."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from schema_extract.tools.common import manifest
from schema_extract.tools.prompts import build_prompt
from schema_extract.tools.receipts import proposal_schema, receipt_schema
from schema_extract.tools.runner import _begin_reopen
from schema_extract.tools.snapshots import rule_files
from schema_extract.tools.stages import (DOCUMENT_CONCLUSIONS, STAGES, new_node, new_scope,
    read_specs, required_conclusions, session_group, session_scope, stage_files)
from schema_extract.tools.tests import test_artifact_transactions as artifact_fixture


REPO = Path(__file__).resolve().parents[3]


class ReviewProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixture = artifact_fixture.ArtifactTransactions(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.directory
        (self.root / "rules").symlink_to(REPO, target_is_directory=True)
        self.workspace = self.fixture.task("workspace", {"root": self.fixture.base, "branch": self.fixture.branch("branch")})
        selected = artifact_fixture.selection("root", "branch")
        self.state = {"run_id": "run", "status": "running", "stage": "P6", "node": "root", "role": "reviewer", "mode": "normal",
            "stages": {stage: new_scope() for stage in STAGES},
            "nodes": {entry["id"]: new_node(entry) for entry in selected},
            "selected_designs": selected, "writing_plan": [{"id": "branch", "wait_for": ["root"]}], "plan_draft": None,
            "plan_revision": 1, "revision_scope": [], "pending_rechecks": [], "feedback": ["Current scoped source correction"],
            "config": {"max_calls": 32, "revision_limit": 5}, "budget_extensions": [{"calls": 3}],
            "checkpoint": "current-checkpoint", "merged_checkpoint": "current-checkpoint", "reopening": None,
            "reopen_requests": {}, "reason": None}
        for scope in self.state["stages"].values():
            scope.update(revisions=2, continues=3, receipt_repairs=1, cli_failures=1, calls=11, deadline=4321.5)
        self.materials = {"directory": str(self.root / "attempt/materials"), "first_draft": False, "candidates": [], "records": [
            {"kind": "receipt", "stage": "P2", "node": "root", "role": "worker", "attempt_id": "current-record",
             "path": str(self.root / "attempt/materials/records/current-record.json"), "purpose": "knowledge_destination", "read_after": "current_artifacts"}]}
        self.attempt = {"run_id": "run", "attempt_id": "current", "stage": "P6", "node": "root", "role": "reviewer",
            "mode": "normal", "directory": str(self.root / "attempt"), "workspace": str(self.workspace),
            "base_checkpoint": "current-checkpoint", "dependency_digests": manifest(self.workspace), "writable_files": [],
            "backend": "claude", "model": "offline-fixture", "effort": "max", "timeout_seconds": 117,
            "check_command": str(self.root / "attempt/check-helper.py"), "materials": self.materials, "session_scope": {"stage": "P6", "node": "root", "role": "reviewer", "operation": "normal"}}
        one = {"path": "kernel.py", "symbol": "copy"}
        self.lock = {"snapshot": str(self.root), "rules": {"id": "current-rules"},
            "source": {"id": "fixed-source", "ref": "fixed", "repository_id": "fixture",
                       "bindings": {**{key: one for key in ("entrypoint", "host", "input")},
                                    "reference": {"path": "reference.py", "symbol": "reference"},
                                    **{key: [one, {"path": key + ".md", "symbol": ""}] for key in ("io", "acceptance", "scope")}}}}

    def invocation(self, *, stage="P6", role="reviewer", **extra):
        node = extra.pop("node", "branch" if stage == "P5" else "root")
        call = {**self.attempt, "stage": stage, "role": role, "node": node, **extra}
        call["session_scope"] = session_scope(call)
        call["writable_files"] = stage_files(stage, node_entry={"id": node}, revision_scope=self.state["revision_scope"]) if role == "worker" else []
        before = deepcopy((self.state, self.lock, call))
        prompt = build_prompt(self.state, self.lock, call)
        self.assertEqual((self.state, self.lock, call), before)
        context = json.loads(prompt.split("<invocation>\n", 1)[1].split("\n</invocation>", 1)[0])
        return prompt, context, call

    def test_every_role_receives_current_task_decisions_sources_scope_and_budget(self):
        for stage in STAGES:
            for role in ("worker", "reviewer"):
                with self.subTest(stage=stage, role=role):
                    _, value, call = self.invocation(stage=stage, role=role)
                    for key in ("run_id", "attempt_id", "stage", "role", "node", "mode", "base_checkpoint", "dependency_digests", "writable_files"):
                        self.assertEqual(value[key], call[key])
                    self.assertEqual(value["workspace"], str(self.workspace))
                    self.assertEqual(value["source_root"], str(self.root / "source"))
                    self.assertEqual(value["workflow_page"], str(self.root / "rules/EXTRACT_SCHEMA.md"))
                    self.assertEqual(value["selected_designs"], self.state["selected_designs"])
                    self.assertEqual(value["writing_plan"], self.state["writing_plan"])
                    self.assertEqual(value["inputs"], self.lock["source"]["bindings"])
                    self.assertEqual(value["implementation_source"], "fixture@fixed:kernel.py::copy")
                    self.assertEqual(value["reference_source"], "fixture@fixed:reference.py::reference")
                    self.assertEqual(value["required_conclusions"], required_conclusions(call))
                    self.assertEqual(value["required_evidence"], STAGES[stage]["evidence"])
                    self.assertEqual(value["session_scope"], session_scope(call))
                    self.assertEqual(value["budget"]["max_calls"], 35)
                    self.assertEqual(value["budget"]["call_seconds"], 117)
                    self.assertEqual(value["reading_order"], ["current_specs", "current_artifacts", "fixed_source", "process_records"])
                    self.assertEqual(value["process_records"], self.materials["records"])

    def test_selection_and_writing_draft_keep_their_distinct_empty_values(self):
        for selected in (None, [], artifact_fixture.selection("root", "branch")):
            self.state["selected_designs"] = selected
            for draft in (None, [], [{"id": "branch", "wait_for": []}]):
                self.state["plan_draft"] = draft
                for stage in ("P1", "P2", "P3", "P4"):
                    with self.subTest(selected=selected, draft=draft, stage=stage):
                        _, value, _ = self.invocation(stage=stage)
                        self.assertEqual(value["selected_designs"], selected)
                        self.assertEqual(value["plan_draft"], draft if stage in {"P2", "P4"} else None)
                        self.assertEqual(value["writing_plan"], self.state["writing_plan"])

    def test_current_maintenance_and_adjudication_keep_explicit_authority(self):
        fields = [{"path": "root.yaml", "fields": ["DESIGN.Math_reshape"]}]
        task = {"task_id": "repair", "action": "patch", "step": "patch", "reason": "Repair the exact current definition",
                "issues": [], "fields": fields, "feedback": ["Keep all other fields"]}
        self.state["maintenance"] = {"tasks": {"repair": task}}
        for role in ("worker", "reviewer"):
            task["step"] = "patch" if role == "worker" else "verify"
            _, value, call = self.invocation(stage="P3", role=role, purpose="maintenance", maintenance_task_id="repair",
                                           maintenance_step=task["step"], input_fingerprint="fixed", plan_revision=1)
            self.assertEqual(value["maintenance"], {key: task[key] for key in ("task_id", "action", "step", "reason", "issues", "fields")})
            self.assertEqual(value["revision_scope"], fields)
            self.assertEqual(value["feedback"], task["feedback"])
            self.assertNotIn("reopen_targets", value)
            for schema in (proposal_schema(call), receipt_schema(call)):
                self.assertNotIn("reopen", schema["properties"])
                self.assertNotIn("reopen", schema["properties"]["decision"]["enum"])
        candidate = {"checkpoint": "provided", "artifacts": str(self.root / "attempt/materials/candidates/provided"), "files": manifest(self.workspace)}
        self.materials["candidates"] = [candidate]
        for mode in ("adjudicate", "signoff"):
            _, value, _ = self.invocation(stage="P3", mode=mode)
            self.assertEqual(value["writable_files"], [])
            self.assertNotIn("reopen_targets", value)
            if mode == "adjudicate":
                self.assertEqual(value["candidates"], [candidate])
            else:
                self.assertNotIn("candidates", value)

    def test_embedded_documents_are_exact_current_frozen_text_with_no_missing_read_specs(self):
        for stage in STAGES:
            for role in ("worker", "reviewer"):
                with self.subTest(stage=stage, role=role):
                    prompt, value, call = self.invocation(stage=stage, role=role)
                    provided = value["provided_documents"]
                    self.assertIn(str(self.root / "rules/schema_extract/prompt/phases" / STAGES[stage]["page"]), provided)
                    self.assertFalse(set(provided) & set(value["read_specs"]))
                    for path in provided:
                        text = Path(path).read_text().rstrip()
                        self.assertIn('<document path=' + json.dumps(path) + '>\n' + text + '\n</document>', prompt)
                    required = {str(self.root / "rules" / name) for name in read_specs(call)}
                    self.assertTrue(required <= set(provided) | set(value["read_specs"]))
                    if stage == "P5":
                        self.assertIn(str(self.root / "rules/schema_extract/prompt/phases/03_root.md"), value["read_specs"])
                    self.assertEqual(value["reading"]["evidence_reader"], str(self.root / "rules/schema_extract/tools/reading.py"))

    def test_all_stage_role_node_and_operation_sessions_are_independent(self):
        sessions = set()
        for stage in STAGES:
            for node in (("one", "two") if stage == "P5" else ("root",)):
                for role in ("worker", "reviewer"):
                    for mode in ("normal", "adjudicate", "signoff"):
                        call = {"stage": stage, "node": node, "role": role, "mode": mode}
                        scope = tuple(sorted(session_scope(call).items()))
                        self.assertNotIn(scope, sessions)
                        sessions.add(scope)
            reviewer = {"role": "reviewer", "stage": stage}
            self.assertTrue(set(DOCUMENT_CONCLUSIONS) <= set(required_conclusions(reviewer)))
            self.assertIn("schema_docs/1.contract.algorithm_requirement.md", read_specs(reviewer))
            self.assertEqual(required_conclusions({**reviewer, "role": "worker"}), STAGES[stage]["conclusions"])
        self.assertNotEqual(session_group({}, {"stage": "P1", "role": "worker", "node": "root"}),
                            session_group({}, {"stage": "P2", "role": "worker", "node": "root"}))
        self.assertIn("schema_extract/prompt/roles/reviewer.md", rule_files(REPO))

    def test_known_source_misread_can_return_once_without_new_budget(self):
        evidence = {"source_id": "source", "path": "kernel.py", "sha256": "b" * 64, "line_start": 1, "line_end": 1}
        receipt = {"reopen": {"issue_id": "missing-current-definition", "target": "P3", "reason": "Correct the fixed source interpretation", "evidence": [0]},
                   "source_evidence": [evidence], "artifact_evidence": [{"path": "root.yaml", "sha256": self.attempt["dependency_digests"]["root.yaml"]}],
                   "issues": [{"kind": "source_misread", "scope": "root.yaml:DESIGN.Math_reshape", "detail": "The current definition omits a known source fact"}]}
        store = Mock()
        store.verify_checkpoint.return_value = (self.workspace, {"files": manifest(self.workspace)})
        store.checkpoint.return_value = "reopened"
        state = deepcopy(self.state)
        state["stages"]["P3"].update(calls=7, adjudications=1, signoffs=1)
        result = _begin_reopen(store, state, self.attempt, receipt)
        self.assertEqual((result["stage"], result["role"], result["status"]), ("P3", "worker", "running"))
        self.assertEqual((result["stages"]["P3"]["calls"], result["stages"]["P3"]["adjudications"], result["stages"]["P3"]["signoffs"]), (7, 1, 1))
        self.assertEqual(result["nodes"]["root"]["generation"], 1)
        receipt["source_evidence"][0]["line_end"] = 2
        repeated = _begin_reopen(store, result, self.attempt, receipt)
        self.assertEqual(repeated["reason"]["code"], "reopen_already_used")
        self.assertEqual(repeated["nodes"]["root"]["generation"], 1)


if __name__ == "__main__":
    unittest.main()
