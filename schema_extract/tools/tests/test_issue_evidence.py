"""Document-only rejections use their own current-file anchors, not source filler."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from schema_extract.tools.artifacts import index_prefix
from schema_extract.tools.common import ExtractError, manifest
from schema_extract.tools.receipts import normalize_response, validate_receipt
from schema_extract.tools.stages import DOCUMENT_CONCLUSIONS


class IssueEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.artifacts = self.root / "artifacts"
        self.source.mkdir()
        self.artifacts.mkdir()
        (self.root / "rules").symlink_to(Path(__file__).resolve().parents[3], target_is_directory=True)
        (self.source / "kernel.py").write_text("def route(n): return 'branch' if n > 1 else 'root'\n")
        (self.artifacts / "root.yaml").write_text(
            "meta_data:\n  dispatch:\n    id: root\n    when: N > 0\ncontract: {}\n")
        self.plan = [
            {"id": name, "file": name + ".yaml", "parent": parent, "workload": "Positive N.",
             "difference": "The selected route.", "selection": "kernel.py::route", "evidence": [0]}
            for name, parent in (("root", ""), ("branch", "root"))
        ]
        (self.artifacts / "dispatch.md").write_text(index_prefix(self.plan) + "No known gaps.\n")
        self.lock = {"snapshot": str(self.root), "source": {
            "id": "fixed-source", "files": manifest(self.source),
            "bindings": {"host": {"path": "kernel.py", "symbol": "route"}},
        }}
        self.attempt = {
            "run_id": str(uuid4()), "attempt_id": str(uuid4()), "input_fingerprint": "fixed-input",
            "stage": "P4", "role": "reviewer", "node": "root", "plan_revision": 0,
            "base_checkpoint": "base", "dependency_digests": manifest(self.artifacts),
            "mode": "normal", "generation": 0, "node_signature": "", "writable_files": [],
        }
        self.state = {
            "run_id": self.attempt["run_id"], "input_fingerprint": self.attempt["input_fingerprint"],
            "accepted_receipts": {}, "active_attempts": {"P4": self.attempt["attempt_id"]},
            "plan_draft": [],
        }

        def anchor(path, start=1, end=1, **extra):
            return {"path": path, "line_start": start, "line_end": end, "quote_offset": 0,
                    "note": "Read the current scoped content.", **extra}

        self.proposal = {
            "format": "schema_extract.proposal.v1", "attempt_id": self.attempt["attempt_id"],
            "decision": "revise", "summary": "The structured plan is missing while the index has nodes.",
            "source_evidence": [anchor("kernel.py", category="host")],
            "artifact_evidence": [anchor("root.yaml"), anchor("dispatch.md", 3, 6)],
            "conclusions": [], "checks": [], "checks_note": "P4 uses its stage gate.",
            "issues": [{"kind": "submission_format", "scope": "本次 P4 proposal.plan 与 dispatch.md 节点表",
                        "detail": "The current plan is empty, while dispatch.md lists root and branch.",
                        "recovery_condition": "Supply the full structured plan before P4 approval.",
                        "evidence": []}],
            "framework_candidates": [], "handoff": "", "reopen": None, "plan": [],
        }

    def validate(self):
        receipt = normalize_response(self.proposal, self.lock, self.attempt, self.artifacts)
        return validate_receipt(receipt, self.state, self.lock, self.attempt, self.artifacts, [])

    def test_p4_format_revise_can_use_scoped_artifact_evidence(self):
        receipt = self.validate()
        self.assertEqual(receipt["decision"], "revise")
        self.assertEqual(receipt["issues"][0]["evidence"], [])
        self.assertEqual(receipt["plan"], [])

    def test_schema_conformance_revise_can_use_its_current_file(self):
        self.proposal["issues"][0].update(
            kind="schema_conformance", scope="root.yaml:design",
            detail="The current root has no completed Design; the P3 stage page requires a full root.")
        self.assertEqual(self.validate()["decision"], "revise")

    def test_scoped_filename_supports_field_and_prose_delimiters(self):
        for scope in ("dispatch.md:5-6", "检查dispatch.md的节点表", "`dispatch.md` 的节点表"):
            with self.subTest(scope=scope):
                self.proposal["issues"][0]["scope"] = scope
                self.assertEqual(self.validate()["decision"], "revise")

    def test_unrelated_or_partial_filename_cannot_borrow_an_anchor(self):
        for scope in ("", "unknown.yaml", "other-dispatch.md", "dispatch.md.backup",
                      "archive/dispatch.md", "/old/dispatch.md", r"old\dispatch.md",
                      r".\dispatch.md", r"dispatch.md\backup", "artifact_evidence[0]"):
            with self.subTest(scope=scope):
                self.proposal["issues"][0]["scope"] = scope
                with self.assertRaisesRegex(ExtractError, "revision/blocking"):
                    self.validate()

    def test_worker_blocking_needs_an_anchor_for_its_scoped_file(self):
        self.attempt["role"] = "worker"
        self.proposal["decision"] = "blocked"
        self.proposal["artifact_evidence"] = self.proposal["artifact_evidence"][:1]
        with self.assertRaisesRegex(ExtractError, "revision/blocking"):
            self.validate()
        self.proposal["artifact_evidence"] = []
        with self.assertRaisesRegex(ExtractError, "revision/blocking"):
            self.validate()

    def test_bad_artifact_quote_is_not_validated_by_the_issue_kind(self):
        receipt = normalize_response(self.proposal, self.lock, self.attempt, self.artifacts)
        receipt["artifact_evidence"][1]["quote"] = "This quote was not in the current artifact."
        with self.assertRaisesRegex(ExtractError, "quote text does not match"):
            validate_receipt(receipt, self.state, self.lock, self.attempt, self.artifacts, [])

    def test_source_related_kinds_do_not_receive_the_artifact_exception(self):
        for kind in ("source_gap", "schema_gap", "source_defect", "numerical", "selection", "transcription"):
            with self.subTest(kind=kind):
                self.proposal["issues"][0]["kind"] = kind
                with self.assertRaisesRegex(ExtractError, "revision/blocking"):
                    self.validate()
        self.proposal["issues"][0]["evidence"] = [0]
        self.assertEqual(self.validate()["decision"], "revise")

    def test_mixed_issues_keep_the_existing_one_supported_issue_requirement(self):
        document = self.proposal["issues"][0]
        source_gap = {**deepcopy(document), "kind": "source_gap", "detail": "A source fact is missing."}
        self.proposal["issues"].append(source_gap)
        self.assertEqual(self.validate()["decision"], "revise")
        # A source issue cannot use the document issue's artifact alternative.
        document["scope"] = "uncited.yaml"
        with self.assertRaisesRegex(ExtractError, "revision/blocking"):
            self.validate()
        document["scope"] = "dispatch.md"
        document["detail"] = " "
        with self.assertRaisesRegex(ExtractError, "revision/blocking"):
            self.validate()

    def test_invalid_source_indices_still_fail_even_with_document_support(self):
        for indices in ([-1], [1]):
            with self.subTest(indices=indices):
                self.proposal["issues"][0]["evidence"] = indices
                with self.assertRaisesRegex(ExtractError, "outside source_evidence"):
                    self.validate()

    def test_reviewer_still_needs_source_and_all_stage_artifacts(self):
        original = deepcopy(self.proposal)
        self.proposal["source_evidence"] = []
        with self.assertRaisesRegex(ExtractError, "reviewer needs actual source evidence"):
            self.validate()
        self.proposal = original
        self.proposal["artifact_evidence"] = self.proposal["artifact_evidence"][1:]
        with self.assertRaisesRegex(ExtractError, "reviewer must cite all stage artifacts"):
            self.validate()

    def test_revise_may_omit_recovery_but_blocked_still_requires_it(self):
        self.proposal["issues"][0]["recovery_condition"] = " "
        self.assertEqual(self.validate()["decision"], "revise")
        self.proposal["decision"] = "blocked"
        with self.assertRaisesRegex(ExtractError, "revision/blocking"):
            self.validate()
        self.proposal["issues"][0]["recovery_condition"] = "Supply the missing current plan."
        self.assertEqual(self.validate()["decision"], "blocked")

    def test_reopen_keeps_its_existing_source_requirements(self):
        self.proposal.update(decision="reopen", reopen={
            "issue_id": "incomplete-root", "target": "P3", "reason": "The root needs upstream correction.",
            "evidence": [0],
        })
        self.proposal["issues"][0].update(kind="schema_conformance", scope="root.yaml:design")
        with self.assertRaisesRegex(ExtractError, "revision/blocking"):
            self.validate()
        self.proposal["issues"][0]["evidence"] = [0]
        self.proposal["reopen"]["evidence"] = []
        with self.assertRaisesRegex(ExtractError, "reopen requires valid fixed-source evidence"):
            self.validate()
        self.proposal["reopen"]["evidence"] = [0]
        self.assertEqual(self.validate()["decision"], "reopen")

    def test_p4_approval_still_requires_a_full_source_supported_plan(self):
        self.proposal.update(decision="approve", handoff="Follow the two source-selected routes.")
        self.proposal["conclusions"] = [
            {"subject": subject, "text": "The fixed source supports this property.", "evidence": [0]}
            for subject in ("tree", "paths", "coverage")
        ] + [
            {"subject": subject, "text": "The current document meets this criterion.", "evidence": []}
            for subject in DOCUMENT_CONCLUSIONS
        ]
        with self.assertRaisesRegex(ExtractError, "plan"):
            self.validate()
        self.proposal["plan"] = deepcopy(self.plan)
        self.state["plan_draft"] = deepcopy(self.plan)
        self.assertEqual(self.validate()["decision"], "approve")
        self.proposal["plan"][1]["evidence"] = []
        with self.assertRaisesRegex(ExtractError, "each planned node needs valid source evidence"):
            self.validate()
        self.proposal["plan"] = deepcopy(self.plan)
        self.proposal["conclusions"][2]["evidence"] = []
        with self.assertRaisesRegex(ExtractError, "missing supported stage conclusions: coverage"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
