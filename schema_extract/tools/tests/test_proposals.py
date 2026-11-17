"""Current proposal identity, directory decisions and exact evidence boundaries."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from uuid import uuid4

from schema_extract.tools.artifacts import validate_plan, write_yaml
from schema_extract.tools.audit import _receipt_response
from schema_extract.tools.common import ExtractError, manifest
from schema_extract.tools.reading import read_evidence
from schema_extract.tools.receipts import ReceiptFormatError, normalize_response, validate_receipt
from schema_extract.tools.stages import DOCUMENT_CONCLUSIONS, required_conclusions
from schema_extract.tools.tests import test_artifact_transactions as artifact_fixture


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = artifact_fixture.ArtifactTransactions(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.directory
        self.source = self.root / "source"
        self.source.mkdir()
        (self.root / "rules").symlink_to(artifact_fixture.REPO, target_is_directory=True)
        (self.source / "kernel.py").write_text("def route(x): return x\n")
        self.artifacts = self.fixture.task("artifacts")
        self.selected = artifact_fixture.selection("root")
        self.lock = {"snapshot": str(self.root), "source": {"id": "fixed-source", "files": manifest(self.source),
                     "bindings": {"host": {"path": "kernel.py", "symbol": "route"}}}}
        self.attempt = {"run_id": str(uuid4()), "attempt_id": str(uuid4()), "input_fingerprint": "fixed-input",
                        "stage": "P2", "role": "reviewer", "node": "root", "plan_revision": 0,
                        "base_checkpoint": "base", "dependency_digests": manifest(self.artifacts),
                        "mode": "normal", "generation": 0, "node_signature": "", "writable_files": [],
                        "response_format": "schema_extract.proposal.v1"}
        def anchor(path, **extra):
            return {"path": path, "line_start": 1, "line_end": 1, "quote_offset": 0,
                    "note": "The selected route preserves the task.", **extra}
        self.proposal = {"format": "schema_extract.proposal.v1", "attempt_id": self.attempt["attempt_id"],
                         "decision": "approve", "summary": "The compact routing description is faithful.",
                         "source_evidence": [anchor("kernel.py", category="host")],
                         "artifact_evidence": [anchor(name) for name in ("root.yaml", "dispatch.md")],
                         "conclusions": self.conclusions(), "plan": deepcopy(self.selected),
                         "checks": [], "checks_note": "Runner uses the current stage gate.", "issues": [],
                         "framework_candidates": [], "handoff": "Follow the selected concrete design.", "reopen": None}

    def conclusions(self):
        return [{"subject": subject, "text": "Checked this property of the current submission.",
                 "evidence": [] if subject in DOCUMENT_CONCLUSIONS else [0]}
                for subject in required_conclusions(self.attempt)]

    def state(self):
        return {"run_id": self.attempt["run_id"], "input_fingerprint": self.attempt["input_fingerprint"],
                "accepted_receipts": {}, "active_attempts": {self.attempt["stage"]: self.attempt["attempt_id"]},
                "selected_designs": self.selected, "plan_draft": self.proposal.get("plan"),
                "pending_rechecks": [], "revision_scope": []}

    def normalize(self, proposal=None):
        return normalize_response(self.proposal if proposal is None else proposal,
                                  self.lock, self.attempt, self.artifacts)

    def p4_plan(self):
        self.selected = artifact_fixture.selection("root", "branch")
        plan = [{"id": "branch", "wait_for": ["root"]}]
        self.attempt.update(stage="P4", role="worker", writable_files=["dispatch.md"],
                            selection_signature=validate_plan("P2", self.selected, parser_root=artifact_fixture.REPO),
                            dependency_digests=manifest(self.artifacts))
        self.proposal.update(decision="complete", plan=plan, conclusions=self.conclusions())
        self.fixture.index(self.artifacts, {"root", "branch"}, planned={"branch"})
        return (self.artifacts / "dispatch.md").read_text()

    def validate_p4(self, text):
        (self.artifacts / "dispatch.md").write_text(text)
        receipt = self.normalize()
        return validate_receipt(receipt, self.state(), self.lock, self.attempt, self.artifacts, ["dispatch.md"])

    def test_p4_prose_and_layout_do_not_require_a_generated_parent_table(self):
        index = self.p4_plan()
        for text in (index, index + "\n安排只控制开工顺序。\n",
                     index.replace("## 完整设计", "附加说明。\n\n## 完整设计"),
                     index + "\n~~~markdown\n## 完整设计\n~~~\n"):
            with self.subTest(text=text):
                self.assertEqual(self.validate_p4(text)["decision"], "complete")

    def test_p4_current_waiting_and_planned_links_remain_hard_boundaries(self):
        index = self.p4_plan()
        for text in (index.replace("`branch.yaml`（仅计划）", "[branch](branch.yaml)"),
                     index + "\n[unknown](unknown.yaml)\n", index.replace("root.yaml", "missing.yaml")):
            with self.subTest(text=text), self.assertRaises(ExtractError) as caught:
                self.validate_p4(text)
            self.assertNotIsInstance(caught.exception, ReceiptFormatError)
        self.proposal["plan"][0]["wait_for"] = ["missing"]
        with self.assertRaisesRegex(ExtractError, "wait_for"):
            self.validate_p4(index)

    def test_p4_required_sections_are_real_and_unique(self):
        index = self.p4_plan()
        for text in (index.replace("## 范围", "## 其他"),
                     index.replace("## 完整设计", "## 其他"),
                     index + "\n## 范围\n\n冲突范围。\n",
                     "~~~markdown\n" + index + "~~~\n"):
            with self.subTest(text=text), self.assertRaisesRegex(ExtractError, "dispatch|section"):
                self.validate_p4(text)

    def p6_plan(self):
        self.p4_plan()
        write_yaml(self.artifacts / "branch.yaml", self.fixture.branch("branch"), parser_root=artifact_fixture.REPO)
        self.fixture.index(self.artifacts, {"root", "branch"})
        self.attempt.update(stage="P6", role="reviewer", writable_files=[], dependency_digests=manifest(self.artifacts))
        self.proposal.pop("plan")
        self.proposal.update(decision="approve", affected_nodes=["root", "branch"], revision_scope=[], handoff="",
                             conclusions=self.conclusions())
        self.proposal["conclusions"][0]["evidence"] = []
        self.proposal["artifact_evidence"].append(
            {"path": "branch.yaml", "line_start": 1, "line_end": 1, "quote_offset": 0,
             "note": "Checked this current final document."})
        state = self.state()
        state["pending_rechecks"] = ["root", "branch"]
        return state

    def test_p6_directory_can_use_complete_artifact_evidence(self):
        state = self.p6_plan()
        receipt = self.normalize()
        self.assertEqual(validate_receipt(receipt, state, self.lock, self.attempt, self.artifacts, []), receipt)
        self.assertEqual(receipt["conclusions"][0]["evidence"], [])

    def test_p6_requires_each_document_judgment_and_current_file_coverage(self):
        state = self.p6_plan()
        original = deepcopy(self.proposal)
        for subject in DOCUMENT_CONCLUSIONS:
            with self.subTest(subject=subject):
                self.proposal = deepcopy(original)
                self.proposal["conclusions"] = [item for item in self.proposal["conclusions"] if item["subject"] != subject]
                with self.assertRaisesRegex(ExtractError, "missing supported stage conclusions"):
                    validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])
        self.proposal = deepcopy(original)
        self.proposal["artifact_evidence"] = [item for item in self.proposal["artifact_evidence"] if item["path"] != "branch.yaml"]
        with self.assertRaises(ExtractError):
            validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])

    def test_p2_cumulative_review_is_required_before_design_is_complete(self):
        self.assertEqual(validate_receipt(self.normalize(), self.state(), self.lock, self.attempt,
                                         self.artifacts, [])["decision"], "approve")
        self.proposal["conclusions"] = [item for item in self.proposal["conclusions"] if item["subject"] not in DOCUMENT_CONCLUSIONS]
        with self.assertRaisesRegex(ExtractError, "document_self_contained"):
            validate_receipt(self.normalize(), self.state(), self.lock, self.attempt, self.artifacts, [])

    def test_p6_directory_evidence_does_not_replace_source_judgments(self):
        state = self.p6_plan()
        original = deepcopy(self.proposal)
        self.proposal["artifact_evidence"].pop()
        with self.assertRaisesRegex(ExtractError, "missing supported stage conclusions: .*directory"):
            validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])
        self.proposal = original
        self.proposal["conclusions"][1]["evidence"] = []
        with self.assertRaisesRegex(ExtractError, "missing supported stage conclusions: selection"):
            validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])

    def test_p6_rejects_unselected_files_even_when_cited(self):
        state = self.p6_plan()
        (self.artifacts / "extra.yaml").write_text("extra: file\n")
        self.proposal["artifact_evidence"].append(
            {"path": "extra.yaml", "line_start": 1, "line_end": 1, "quote_offset": 0, "note": "Observed this extra file."})
        self.attempt["dependency_digests"] = manifest(self.artifacts)
        with self.assertRaisesRegex(ExtractError, "missing supported stage conclusions: directory"):
            validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])

    def test_compact_approval_retains_raw_response_and_full_receipt_gate(self):
        raw = deepcopy(self.proposal)
        receipt = self.normalize()
        state = {key: self.attempt[key] for key in ("run_id", "input_fingerprint")}
        state.update(accepted_receipts={}, active_attempts={"P2": self.attempt["attempt_id"]},
                     plan_draft=self.proposal["plan"])
        self.assertEqual(validate_receipt(receipt, state, self.lock, self.attempt, self.artifacts, []), receipt)
        self.assertEqual(self.proposal, raw)
        self.assertNotEqual(receipt, raw)
        _receipt_response({"response": raw}, receipt, self.attempt, lock=self.lock, artifacts=self.artifacts)
        changed = deepcopy(receipt)
        changed["summary"] = "A different conclusion."
        with self.assertRaises(ExtractError):
            _receipt_response({"response": raw}, changed, self.attempt, lock=self.lock, artifacts=self.artifacts)

    def test_nonce_is_required_even_when_other_fields_need_repair(self):
        for nonce in (None, str(uuid4())):
            value = {"format": "schema_extract.proposal.v1"}
            if nonce:
                value["attempt_id"] = nonce
            with self.subTest(nonce=nonce), self.assertRaises(ExtractError) as caught:
                self.normalize(value)
            self.assertNotIsInstance(caught.exception, ReceiptFormatError)

    def test_explicit_machine_claims_are_never_repaired(self):
        for field in ("identity", "changed_files", "artifacts"):
            value = {**self.proposal, field: "wrong", "summary": ""}
            with self.subTest(field=field), self.assertRaises(ExtractError) as caught:
                self.normalize(value)
            self.assertNotIsInstance(caught.exception, ReceiptFormatError)
        value = deepcopy(self.proposal)
        value["source_evidence"][0]["quote"] = "invented text"
        with self.assertRaises(ExtractError):
            self.normalize(value)

    def test_source_and_reviewer_artifact_changes_are_rejected(self):
        for path in (self.source / "kernel.py", self.artifacts / "root.yaml"):
            data = path.read_bytes()
            path.write_bytes(data + b"changed\n")
            with self.subTest(path=path.name), self.assertRaises(ExtractError):
                self.normalize()
            path.write_bytes(data)

    def test_long_line_evidence_can_select_a_late_declaration_losslessly(self):
        line = json.dumps({"padding": "x" * 1800, "relative_to": "candidate"}) + "\n"
        (self.source / "acceptance.json").write_text(line)
        offset = line.index('"relative_to"')
        anchor = read_evidence(self.source, "acceptance.json", 1, 1, quote_offset=offset)
        self.assertIn('"relative_to": "candidate"', anchor["quote"])
        self.assertNotIn("relative_to", read_evidence(self.source, "acceptance.json", 1, 1)["quote"])
        with self.assertRaises(ValueError):
            read_evidence(self.source, "acceptance.json", 1, 1, quote_offset=len(line))

    def test_audit_checkpoint_identity_is_stable_across_processes(self):
        # Isolate storage IO; keep the real audit selection and serialization.
        # Changing a process hash seed must not change an archive's identity.
        script = """
import json
from types import SimpleNamespace
from schema_extract.tools import audit
from schema_extract.tools import runtime
state = {"run_id": "run", "status": "ready", "checkpoint": "a" * 64, "merged_checkpoint": "b" * 64}
audit._inputs = lambda *args: {}
audit._checkpoint = lambda store, state, identity, files: {"id": identity, "files": {}}
audit._state = lambda *args: state
runtime.verify_execution = lambda *args, **kwargs: {}
store = SimpleNamespace(writer_status=lambda _: "available")
files = SimpleNamespace(file=lambda *args, **kwargs: None)
print(json.dumps(audit._verify(store, state, [], files, whole_run=True), sort_keys=True))
"""
        outputs = [subprocess.check_output([sys.executable, "-B", "-c", script],
                    cwd=Path(__file__).resolve().parents[3], env={**os.environ, "PYTHONHASHSEED": seed})
                   for seed in ("1", "2", "3")]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])
        verified = json.loads(outputs[0])
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["checkpoint_ids"], [
            {"origin_run_id": "run", "checkpoint": letter * 64} for letter in ("a", "b")])


if __name__ == "__main__":
    unittest.main()
