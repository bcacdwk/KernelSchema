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
        baseline = self.root / "run/checkpoints/current-checkpoint/artifacts"
        baseline.mkdir(parents=True)
        (baseline / "root.yaml").write_bytes((REPO / "schema_template/starter.yaml").read_bytes())
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
        receipt = self.root / "run/attempts/HISTORY_APPROVAL_CANARY/receipt.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({"source_evidence": [{"path": "kernel.py", "line_start": 1}]}))

    def invocation(self, attempt=None, **bindings):
        attempt = attempt or self.attempt
        before = deepcopy((self.state, self.lock, attempt, bindings))
        prompt = build_prompt(self.state, self.lock, attempt, **bindings)
        self.assertEqual((self.state, self.lock, attempt, bindings), before)
        value = json.loads(prompt.split("<invocation>\n", 1)[1].split("\n</invocation>", 1)[0])
        return prompt, value

    def test_root_stages_project_history_without_narrowing_the_task(self):
        self.state["plan"][1]["difference"] = "UNRELATED_PLAN_CANARY"
        self.state["nodes"]["branch"]["approval"]["attempt_id"] = "UNRELATED_APPROVAL_CANARY"
        self.state["reopening"] = {
            "key": "issue-key", "issue_id": "shared-task-repair", "target": "P1",
            "reason": "Preserve all outputs and task paths.", "from_stage": "P5", "from_node": "branch",
            "source_evidence": [{"path": "all_paths.py", "line_start": 9, "line_end": 11}],
            "artifact_evidence": [{"path": "root.yaml", "line_start": 2, "line_end": 3}],
            "issues": [{"scope": "root.yaml:contract", "detail": "Cover every output.", "evidence": [0]}],
            "previous_plan": deepcopy(self.state["plan"]),
            "previous_merged_checkpoint": "OLD_MERGE_CANARY", "previous_plan_revision": 7,
        }
        for key in ("io", "acceptance", "scope"):
            self.lock["source"]["bindings"][key] = [
                {"path": f"{key}/all_paths.md", "symbol": ""},
                {"path": f"{key}/other_path.md", "symbol": ""},
            ]
        self.state["revision_scope"] = [{"path": "root.yaml", "fields": ["contract", "design"]}]
        self.state["pending_rechecks"] = ["root", "branch"]
        self.state["budget_extensions"] = [{"calls": 3}]
        handoff = self.root / "run/stages/P2/handoff.md"
        handoff.parent.mkdir(parents=True)
        handoff.write_text("HISTORY_P2_HANDOFF_CANARY: the currently selected path.\n")
        old = self.root / "predecessor/artifacts"
        old.mkdir(parents=True)
        for name in ("root.yaml", "dispatch.md"):
            (old / name).write_text("HISTORY_CONTINUED_DRAFT_CANARY\n")
        self.state["continuation"] = {"predecessor": "prior-run", "evidence": str(old.parent)}
        for stage in ("P1", "P3"):
            self.state["stages"][stage].update(revisions=2, continues=3, receipt_repairs=1,
                                              cli_failures=1, calls=11, deadline=4321.5)
            for role in ("worker", "reviewer"):
                with self.subTest(stage=stage, role=role):
                    attempt = {**self.attempt, "stage": stage, "role": role, "timeout_seconds": 117,
                               "writable_files": ["root.yaml"] if role == "worker" else []}
                    prompt, value = self.invocation(attempt)
                    self.assertEqual(value["plan"], self.state["plan"][:1])
                    self.assertEqual(value["node_entry"], self.state["plan"][0])
                    self.assertNotIn("UNRELATED_", prompt)
                    self.assertNotIn("OLD_MERGE_CANARY", prompt)
                    self.assertEqual(value["inputs"], self.lock["source"]["bindings"])
                    self.assertEqual(value["implementation_source"], "fixture@fixed:kernel.py::kernel")
                    self.assertEqual(value["reference_source"], "fixture@fixed:kernel.py::kernel")
                    for key in ("attempt_id", "stage", "role", "node", "mode", "base_checkpoint",
                                "dependency_digests", "workspace", "writable_files"):
                        self.assertEqual(value[key], attempt[key])
                    self.assertEqual(value["source_id"], "source")
                    self.assertEqual(value["source_ref"], "fixed")
                    self.assertEqual(value["source_root"], str(self.root / "source"))
                    self.assertEqual(value["rules_id"], "rules")
                    self.assertEqual(value["rules_root"], str(self.root / "rules"))
                    self.assertEqual(value["executor_rules_id"], "rules")
                    self.assertEqual(value["executor_root"], str(self.root / "rules"))
                    self.assertEqual(value["revision_scope"], self.state["revision_scope"])
                    self.assertEqual(value["pending_rechecks"], self.state["pending_rechecks"])
                    self.assertEqual(value["required_evidence"], STAGES[stage]["evidence"])
                    self.assertEqual(value["required_conclusions"], required_conclusions(attempt))
                    self.assertEqual(value["budget"], {"revisions": 2, "continues": 3, "receipt_repairs": 1,
                        "cli_failures": 1, "calls": 11, "deadline": 4321.5, "call_seconds": 117,
                        "max_calls": 35, "revision_limit": 5})
                    if role == "worker":
                        self.assertEqual(set(value["node_approvals"]), {"root"})
                        self.assertEqual(set(value["reviewed_artifacts"]), {"root"})
                        self.assertEqual(set(value["prior_approvals"]), set(STAGES))
                        self.assertEqual(value["reopening"], {key: self.state["reopening"][key] for key in
                            ("key", "issue_id", "target", "reason", "from_stage", "from_node",
                             "source_evidence", "artifact_evidence", "issues")})
                        self.assertEqual(value["feedback"], self.state["feedback"])
                        self.assertEqual(value["handoff"], handoff.read_text() if stage == "P3" else "")
                        self.assertEqual(value["recovery_handoff"], attempt["recovery_handoff"])
                        self.assertEqual(value["imported_draft"], self.state["imported_draft"])
                        self.assertEqual(value["continued_draft"]["predecessor"], "prior-run")
                        self.assertEqual(set(value["continued_draft"]["artifacts"]),
                                         {"root.yaml"} if stage == "P1" else {"root.yaml", "dispatch.md"})
                        self.assertEqual(value["required_artifact_evidence"], [])
                    else:
                        self.assertNotIn("HISTORY_", prompt)
                        for key in ("feedback", "handoff", "prior_approvals", "node_approvals",
                                    "reviewed_artifacts", "reopening", "recovery_handoff",
                                    "imported_draft", "continued_draft"):
                            self.assertNotIn(key, value)
                        self.assertEqual(set(value["required_artifact_evidence"]),
                                         {"root.yaml"} if stage == "P1" else {"root.yaml", "dispatch.md"})

    def test_root_plan_projection_preserves_empty_values_and_draft_fallback(self):
        full_plan = deepcopy(self.state["plan"])
        cases = [(None, None, None), ([], None, None), (None, [], []), ([], [], []),
                 (None, full_plan, full_plan[:1]), (full_plan, [], full_plan[:1])]
        for stage in ("P1", "P3"):
            for role in ("worker", "reviewer"):
                for plan, draft, expected in cases:
                    with self.subTest(stage=stage, role=role, plan=plan, draft=draft):
                        self.state.update(plan=plan, plan_draft=draft, reopening=None)
                        _, value = self.invocation({**self.attempt, "stage": stage, "role": role})
                        self.assertEqual(value["plan"], expected)
                        self.assertEqual(value["node_entry"], full_plan[0] if plan else None)
                        if role == "worker":
                            self.assertIsNone(value["reopening"])
                self.state["reopening"] = {}
                _, value = self.invocation({**self.attempt, "stage": stage, "role": role})
                if role == "worker":
                    self.assertEqual(value["reopening"], {})

    def test_root_stage_maintenance_and_adjudication_keep_their_authority(self):
        fields = [{"path": "root.yaml", "fields": ["meta_data.version", "design.math_reshape"]}]
        task = {"task_id": "repair", "action": "review", "step": "review", "plan_reason": "Fixed rule issue.",
                "reason": "Review the current complete document.", "issues": ["current-issue"],
                "fields": fields, "feedback": ["HISTORY_MAINTENANCE_FEEDBACK_CANARY"]}
        self.state["maintenance"] = {"tasks": {"repair": task}}
        for stage in ("P1", "P3"):
            for role, step in (("worker", "patch"), ("reviewer", "review")):
                with self.subTest(stage=stage, role=role, purpose="maintenance"):
                    task["step"] = step
                    attempt = {**self.attempt, "stage": stage, "role": role, "purpose": "maintenance",
                               "maintenance_task_id": "repair", "writable_files": ["root.yaml"] if role == "worker" else []}
                    prompt, value = self.invocation(attempt)
                    self.assertEqual(value["maintenance"], {key: item for key, item in task.items() if key != "feedback"})
                    self.assertEqual(value["purpose"], "maintenance")
                    self.assertEqual(value["revision_scope"], fields)
                    self.assertEqual(value["writable_files"], attempt["writable_files"])
                    self.assertNotIn("reopening", value)
                    self.assertNotIn("reopen_targets", value)
                    self.assertNotIn("candidates", value)
                    if role == "worker":
                        self.assertEqual(value["feedback"], task["feedback"])
                    else:
                        self.assertNotIn("HISTORY_", prompt)
            candidate = {"checkpoint": "candidate", "plan": deepcopy(self.state["plan"])}
            self.state["stages"][stage]["candidates"] = [candidate]
            with self.subTest(stage=stage, mode="adjudicate"):
                _, value = self.invocation({**self.attempt, "stage": stage, "mode": "adjudicate"})
                self.assertEqual(value["candidates"], [{**candidate,
                    "artifacts": str(self.root / "run/checkpoints/candidate/artifacts")}])
                self.assertEqual(value["writable_files"], [])
                self.assertEqual(value["revision_scope"], self.state["revision_scope"])
                self.assertNotIn("reopen_targets", value)

    def test_other_whole_tree_stages_keep_their_context(self):
        self.state["plan_draft"] = deepcopy(self.state["plan"][:1])
        self.state["pending_rechecks"] = ["root", "branch"]
        self.state["reopening"]["previous_plan"] = deepcopy(self.state["plan"])
        for stage in ("P2", "P4", "P6"):
            for role in ("worker", "reviewer"):
                with self.subTest(stage=stage, role=role):
                    _, value = self.invocation({**self.attempt, "stage": stage, "role": role})
                    self.assertEqual(value["plan"], self.state["plan_draft"] if
                                     (stage, role) == ("P4", "reviewer") else self.state["plan"])
                    self.assertEqual(value["pending_rechecks"], self.state["pending_rechecks"])
                    if stage == "P4":
                        self.assertEqual(set(value["current_node_artifacts"]), {"root", "branch"})
                    if role == "worker":
                        self.assertEqual(set(value["node_approvals"]), {"root", "branch"})
                        self.assertEqual(set(value["reviewed_artifacts"]), {"root", "branch"})
                        self.assertEqual(value["reopening"], self.state["reopening"])

    def test_provided_documents_are_full_current_text_and_only_deduplicate_prompt_reads(self):
        method = (REPO / "schema_extract/REVIEW.md").read_text()
        for stage, definition in STAGES.items():
            page = "schema_extract/phases/" + definition["page"]
            for role in ("worker", "reviewer"):
                with self.subTest(stage=stage, role=role):
                    attempt = {**self.attempt, "stage": stage, "role": role,
                               "node": "branch" if stage == "P5" else "root"}
                    complete_reads = read_specs(attempt)
                    prompt, value = self.invocation(attempt)
                    provided = [page, "schema_extract/REVIEW.md"] if role == "reviewer" else [page]
                    self.assertEqual(value["provided_documents"], provided)
                    self.assertEqual(value["read_specs"], [name for name in complete_reads if name not in provided])
                    self.assertEqual(read_specs(attempt), complete_reads)
                    self.assertIn(f'<current-phase-page path={json.dumps(str(self.root / "rules" / page))}>\n'
                                  + (REPO / page).read_text() + "\n</current-phase-page>", prompt)
                    embedded_method = prompt.split("<document-review-method>\n", 1)[1].split("\n</document-review-method>", 1)[0]
                    self.assertEqual(embedded_method, method if role == "reviewer" else "")
                    if role == "worker":
                        self.assertNotIn(method, prompt)
                    else:
                        self.assertIn("schema_extract/REVIEW.md", complete_reads)
                    if stage == "P5":
                        self.assertIn("schema_extract/phases/03_root.md", value["read_specs"])
                        self.assertNotIn("schema_extract/phases/03_root.md", provided)

    def test_provided_documents_follow_semantic_rules_with_a_separate_executor(self):
        page = "schema_extract/phases/03_root.md"
        documents = [page, "schema_extract/REVIEW.md"]
        contexts = {}
        for kind in ("semantic", "executor"):
            context = deepcopy(self.lock)
            context["snapshot"] = str(self.root / kind)
            context["rules"]["id"] = kind + "-rules"
            for name in documents:
                path = Path(context["snapshot"]) / "rules" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text((REPO / name).read_text() + f"\n{kind.upper()}_FULL_TEXT: {name}\n")
            contexts[kind] = context
        for role in ("worker", "reviewer"):
            with self.subTest(role=role):
                attempt = {**self.attempt, "stage": "P3", "role": role,
                           "semantic_revision": "revision-1", "execution_binding": "executor-1"}
                prompt, value = self.invocation(attempt, rules_lock=contexts["semantic"], executor_lock=contexts["executor"])
                self.assertEqual(value["rules_id"], "semantic-rules")
                self.assertEqual(value["executor_rules_id"], "executor-rules")
                self.assertEqual(value["rules_root"], str(self.root / "semantic/rules"))
                self.assertEqual(value["executor_root"], str(self.root / "executor/rules"))
                self.assertEqual(value["semantic_revision"], "revision-1")
                self.assertEqual(value["execution_binding"], "executor-1")
                self.assertEqual(value["source_root"], str(self.root / "source"))
                self.assertEqual(value["phase_page"], str(self.root / "semantic/rules" / page))
                self.assertNotIn("EXECUTOR_FULL_TEXT", prompt)
                self.assertIn(str(self.root / "executor/rules/schema_extract/tools/reading.py"), prompt)
                for name in value["provided_documents"]:
                    self.assertIn((self.root / "semantic/rules" / name).read_text(), prompt)
                if role == "reviewer":
                    self.assertEqual(value["review_scope"]["method"], str(self.root / "semantic/rules/schema_extract/REVIEW.md"))
                else:
                    self.assertNotIn("SEMANTIC_FULL_TEXT: schema_extract/REVIEW.md", prompt)

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
