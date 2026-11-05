"""Document review coverage, independent inputs and bounded upstream repair."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from schema_extract.tools.common import manifest
from schema_extract.tools.prompts import build_prompt
from schema_extract.tools.runner import _begin_reopen
from schema_extract.tools.snapshots import rule_files
from schema_extract.tools.stages import (
    DOCUMENT_CONCLUSIONS, STAGES, new_node, new_scope,
    read_specs, required_conclusions, session_group,
)


REPO = Path(__file__).resolve().parents[3]


class ReviewProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "rules").symlink_to(REPO, target_is_directory=True)
        self.workspace = self.root / "run/attempts/current/workspace"
        self.workspace.mkdir(parents=True)
        for name in ("root.yaml", "branch.yaml", "dispatch.md"):
            (self.workspace / name).write_text("Current artifact: " + name + "\n")
        approval = {"checkpoint": "current-checkpoint", "receipt_sha256": "a" * 64,
                    "attempt_id": "HISTORY_APPROVAL_CANARY"}
        plan = [{"id": name, "file": name + ".yaml", "parent": parent,
                 "workload": "Current scalar task.", "difference": "Current selected design.",
                 "selection": "kernel", "evidence": [0]}
                for name, parent in (("root", ""), ("branch", "root"))]
        self.state = {
            "run_id": "run", "status": "running", "stage": "P6", "node": "root",
            "role": "reviewer", "mode": "normal",
            "stages": {stage: {**new_scope(), "approval": deepcopy(approval)} for stage in STAGES},
            "nodes": {entry["id"]: {**new_node(entry), "approval": deepcopy(approval)} for entry in plan},
            "plan": plan, "plan_revision": 1, "revision_scope": [], "pending_rechecks": [],
            "feedback": ["HISTORY_FEEDBACK_CANARY"],
            "imported_draft": {"note": "HISTORY_IMPORT_CANARY"},
            "reopening": {"note": "HISTORY_REOPEN_CANARY"},
            "config": {"max_calls": 32, "revision_limit": 5}, "budget_extensions": [],
            "checkpoint": "current-checkpoint", "merged_checkpoint": "current-checkpoint",
            "reopen_requests": {}, "reason": None,
        }
        self.attempt = {
            "attempt_id": "current", "stage": "P6", "node": "root", "role": "reviewer",
            "mode": "normal", "directory": str(self.workspace.parent),
            "workspace": str(self.workspace), "base_checkpoint": "current-checkpoint",
            "dependency_digests": manifest(self.workspace), "writable_files": [],
            "recovery_handoff": {"note": "HISTORY_RECOVERY_CANARY"},
        }
        one = {"path": "kernel.py", "symbol": "kernel"}
        self.lock = {
            "snapshot": str(self.root), "rules": {"id": "rules", "files": {
                name: "c" * 64 for name in ("schema_extract/extract.py", "schema_extract/tools/cli.py",
                                          "schema_extract/tools/reading.py")}},
            "source": {"id": "source", "ref": "fixed", "repository_id": "fixture",
                       "bindings": {**{key: one for key in ("entrypoint", "host", "input", "reference")},
                                    **{key: [one] for key in ("io", "acceptance", "scope")}}},
        }

    def invocation(self, attempt=None):
        prompt = build_prompt(self.state, self.lock, attempt or self.attempt)
        value = json.loads(prompt.split("<invocation>\n", 1)[1].split("\n</invocation>", 1)[0])
        return prompt, value

    def test_normal_reviewer_gets_current_artifacts_without_history(self):
        prompt, value = self.invocation()
        self.assertNotIn("HISTORY_", prompt)
        for key in ("feedback", "handoff", "prior_approvals", "node_approvals",
                    "reviewed_artifacts", "imported_draft", "reopening", "recovery_handoff"):
            self.assertNotIn(key, value)
        self.assertEqual(set(value["review_scope"]["files"]), set(self.attempt["dependency_digests"]))
        self.assertTrue(set(DOCUMENT_CONCLUSIONS) <= set(value["required_conclusions"]))
        self.assertEqual(value["review_scope"]["order"], ["current_specs", "current_artifacts", "fixed_source"])
        self.assertIn("schema_docs/1.contract.algorithm_requirement.md", value["read_specs"])
        self.assertIn("schema_docs/5.design.tile_pipeline.md", value["read_specs"])
        self.assertIn((REPO / "schema_extract/REVIEW.md").read_text(), prompt)

    def test_worker_keeps_authoring_feedback_and_adjudication_keeps_candidates(self):
        worker = {**self.attempt, "role": "worker"}
        prompt, value = self.invocation(worker)
        self.assertIn("HISTORY_FEEDBACK_CANARY", prompt)
        self.assertNotIn("review_scope", value)
        candidate = {"checkpoint": "candidate", "plan": []}
        self.state["stages"]["P6"]["candidates"] = [candidate]
        prompt, value = self.invocation({**self.attempt, "mode": "adjudicate"})
        self.assertNotIn("HISTORY_", prompt)
        self.assertEqual(value["candidates"][0]["checkpoint"], "candidate")
        self.assertTrue(set(DOCUMENT_CONCLUSIONS) <= set(value["required_conclusions"]))

    def test_reopened_plan_can_read_current_nodes_without_their_old_verdicts(self):
        self.state["plan_draft"] = deepcopy(self.state["plan"][:1])
        prompt, value = self.invocation({**self.attempt, "stage": "P4"})
        self.assertNotIn("HISTORY_", prompt)
        self.assertEqual(value["plan"], self.state["plan_draft"])
        self.assertEqual(set(value["current_node_artifacts"]), {"root", "branch"})
        self.assertNotIn("prior_approvals", value)
        self.state["plan_draft"] = []
        _, value = self.invocation({**self.attempt, "stage": "P4"})
        self.assertEqual(value["plan"], [])

    def test_stage_reads_are_cumulative_and_p2_reviewer_is_independent(self):
        for stage in STAGES:
            reviewer = {"role": "reviewer", "stage": stage}
            self.assertTrue(set(DOCUMENT_CONCLUSIONS) <= set(required_conclusions(reviewer)))
            self.assertIn("schema_docs/1.contract.algorithm_requirement.md", read_specs(reviewer))
            self.assertNotIn("schema_docs/2.contract.hardware_specification.md", read_specs(reviewer))
            if stage not in {"P1", "P2"}:
                self.assertIn("schema_docs/5.design.tile_pipeline.md", read_specs(reviewer))
            self.assertEqual(required_conclusions({**reviewer, "role": "worker"}), STAGES[stage]["conclusions"])
        self.assertNotEqual(session_group({}, {"role": "reviewer", "stage": "P1"}),
                            session_group({}, {"role": "reviewer", "stage": "P2"}))
        self.assertEqual(session_group({}, {"role": "worker", "stage": "P1"}),
                         session_group({}, {"role": "worker", "stage": "P2"}))
        self.assertIn("schema_extract/REVIEW.md", rule_files(REPO))

    def test_conformance_can_reopen_readonly_upstream_once_without_new_source(self):
        evidence = {"source_id": "source", "path": "kernel.py", "sha256": "b" * 64,
                    "line_start": 1, "line_end": 1}
        receipt = {
            "reopen": {"issue_id": "missing-current-definition", "target": "P3",
                       "reason": "The current root depends on an earlier explanation.", "evidence": [0]},
            "source_evidence": [evidence],
            "artifact_evidence": [{"path": "root.yaml", "sha256": self.attempt["dependency_digests"]["root.yaml"]}],
            "issues": [{"kind": "schema_conformance", "scope": "root.yaml:design.math_reshape",
                        "detail": "The current definition is missing; schema_docs/3 requires its formula."}],
        }
        store = Mock()
        store.verify_approval.return_value = ({"source_evidence": [evidence]}, None)
        store.verify_checkpoint.return_value = (self.workspace, None)
        store.checkpoint.return_value = "reopened"
        state = deepcopy(self.state)
        state["stages"]["P3"]["calls"] = 7
        result = _begin_reopen(store, state, self.attempt, receipt)
        self.assertEqual((result["stage"], result["role"], result["status"]), ("P3", "worker", "running"))
        self.assertEqual(result["stages"]["P3"]["calls"], 7)
        self.assertEqual(len(result["reopen_requests"]), 1)
        self.assertEqual(result["nodes"]["root"]["generation"], 1)
        repeated = _begin_reopen(store, result, self.attempt, receipt)
        self.assertEqual(repeated["reason"]["code"], "reopen_already_used")
        self.assertEqual(repeated["nodes"]["root"]["generation"], 1)

        for change in ({"artifact_evidence": []}, {"issues": [{"kind": "schema_conformance",
                                                            "scope": "missing.yaml", "detail": "No current anchor."}]}):
            with self.subTest(change=change):
                rejected = _begin_reopen(store, deepcopy(self.state), self.attempt, {**receipt, **change})
                self.assertEqual(rejected["reason"]["code"], "reopen_requires_new_evidence")


if __name__ == "__main__":
    unittest.main()
