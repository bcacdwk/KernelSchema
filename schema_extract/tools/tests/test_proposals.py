"""Durable protocol boundaries: nonce, read-only bytes, raw evidence and long lines."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from uuid import uuid4

from schema_extract.tools.artifacts import index_prefix
from schema_extract.tools.audit import _receipt_response
from schema_extract.tools.common import ExtractError, manifest
from schema_extract.tools.reading import read_evidence, read_range
from schema_extract.tools.receipts import ReceiptFormatError, normalize_response, validate_receipt
from schema_extract.tools.stages import DOCUMENT_CONCLUSIONS


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.artifacts = self.root / "artifacts"
        self.source.mkdir()
        self.artifacts.mkdir()
        (self.source / "kernel.py").write_text("def route(x): return x\n")
        (self.artifacts / "root.yaml").write_text("selected_path: route\n")
        (self.artifacts / "dispatch.md").write_text("[root](root.yaml) uses route.\n")
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
                         "conclusions": [{"subject": subject, "text": "One route covers this task.", "evidence": [0]}
                                         for subject in ("root_choice", "paths", "coverage")],
                         "checks": [], "checks_note": "P2 uses its stage gate.", "issues": [],
                         "framework_candidates": [], "handoff": "Follow route for the root design.", "reopen": None}
        self.proposal["conclusions"].extend(
            {"subject": subject, "text": "The current document meets this criterion.", "evidence": []}
            for subject in DOCUMENT_CONCLUSIONS)

    def normalize(self, proposal=None):
        return normalize_response(self.proposal if proposal is None else proposal,
                                  self.lock, self.attempt, self.artifacts)

    def p4_plan(self):
        (self.root / "rules").symlink_to(Path(__file__).resolve().parents[3], target_is_directory=True)
        (self.artifacts / "root.yaml").write_text("meta_data:\n  dispatch:\n    id: root\n    when: N > 0\ncontract: {}\n")
        plan = [{"id": name, "file": name + ".yaml", "parent": parent, "workload": "Positive scalar inputs.",
                 "difference": "Selected route.", "selection": "kernel.py::route", "evidence": [0]}
                for name, parent in (("root", ""), ("branch", "root"))]
        self.attempt.update(stage="P4", role="worker", writable_files=["dispatch.md"],
                            dependency_digests=manifest(self.artifacts))
        self.proposal.update(decision="complete", plan=plan,
                             conclusions=[{"subject": subject, "text": "The source supports this plan.", "evidence": [0]}
                                          for subject in ("tree", "paths", "coverage")])
        return plan

    def validate_p4(self, text):
        (self.artifacts / "dispatch.md").write_text(text)
        receipt = self.normalize()
        state = {key: self.attempt[key] for key in ("run_id", "input_fingerprint")}
        state.update(accepted_receipts={}, active_attempts={"P4": self.attempt["attempt_id"]})
        return validate_receipt(receipt, state, self.lock, self.attempt, self.artifacts, ["dispatch.md"])

    def test_p4_explanation_before_coverage_passes_without_receipt_repair(self):
        prefix = index_prefix(self.p4_plan())
        for index in (prefix, prefix.replace("## 覆盖缺口", "Parent links describe copying.\n\n## 覆盖缺口"),
                      prefix.replace("## 覆盖缺口", "\n\nParent links describe copying.\n\n\n## 覆盖缺口"),
                      prefix.replace("## 覆盖缺口", "~~~markdown\n## 覆盖缺口\n~~~\n\n## 覆盖缺口")):
            with self.subTest(index=index):
                receipt = self.validate_p4(index + "No known gaps.\n")
                self.assertEqual(receipt["decision"], "complete")

    def test_p4_table_changes_remain_hard_errors_with_a_location(self):
        prefix = index_prefix(self.p4_plan())
        changed = prefix.replace("Positive scalar inputs.", "Negative scalar inputs.", 1)
        with self.assertRaises(ExtractError) as caught:
            self.validate_p4(changed + "No known gaps.\n")
        self.assertNotIsInstance(caught.exception, ReceiptFormatError)
        self.assertIn("first difference at line 5, column", str(caught.exception))

    def test_p4_layout_tolerance_does_not_relax_gaps_links_or_parents(self):
        plan = self.p4_plan()
        prefix = index_prefix(plan)
        for body, message in (("<!-- not a coverage statement -->", "requires actual coverage gaps"),
                              ("No known gaps. [unknown](unknown.yaml)\n", "YAML links differ")):
            with self.subTest(body=body), self.assertRaises(ExtractError) as caught:
                self.validate_p4(prefix + body)
            self.assertNotIsInstance(caught.exception, ReceiptFormatError)
            self.assertIn(message, str(caught.exception))
        plan[1]["parent"] = "missing"
        with self.assertRaises(ExtractError) as caught:
            self.validate_p4(index_prefix(plan) + "No known gaps.\n")
        self.assertNotIsInstance(caught.exception, ReceiptFormatError)
        self.assertIn("parent does not exist", str(caught.exception))

    def test_p4_coverage_needs_its_own_unique_section(self):
        prefix = index_prefix(self.p4_plan())
        table = prefix.removesuffix("## 覆盖缺口\n\n")
        for text, message in (
                (table + "No known gaps.\n", "requires one '## 覆盖缺口' section"),
                (table + "<!--\n## 覆盖缺口\n-->\nNo known gaps.\n", "requires one '## 覆盖缺口' section"),
                (table + "```markdown\n## 覆盖缺口\nNo known gaps.\n```\n", "requires one '## 覆盖缺口' section"),
                (table + "No known gaps.\n\n## 覆盖缺口\n\n<!-- incomplete -->\n", "requires actual coverage gaps"),
                (prefix + "```\nNo known gaps.\n```\n", "requires actual coverage gaps"),
                (prefix + "## Other notes\n\nNo known gaps.\n", "requires actual coverage gaps"),
                (prefix + "No known gaps.\n\n## 覆盖缺口\n\nConflicting coverage.\n", "requires one '## 覆盖缺口' section")):
            with self.subTest(text=text), self.assertRaisesRegex(ExtractError, message):
                self.validate_p4(text)

    def p6_plan(self):
        # This fixture exercises receipt and topology gates; the Runner also
        # applies the full schema checker before accepting a real P6 result.
        plan = self.p4_plan()
        (self.artifacts / "branch.yaml").write_text(
            "meta_data:\n  dispatch:\n    id: branch\n    parent: root\n    when: N > 1\ncontract: {}\n")
        (self.artifacts / "dispatch.md").write_text(index_prefix(plan) + "No known gaps.\n")
        self.attempt.update(stage="P6", role="reviewer", writable_files=[],
                            dependency_digests=manifest(self.artifacts))
        self.proposal.pop("plan")
        self.proposal.update(decision="approve", affected_nodes=["root", "branch"], revision_scope=[], handoff="",
                             conclusions=[{"subject": subject, "text": "Checked this final property.",
                                           "evidence": [] if subject == "tree" else [0]}
                                          for subject in ("tree", "selection", "coverage", "references", "unresolved")])
        self.proposal["artifact_evidence"].append(
            {"path": "branch.yaml", "line_start": 1, "line_end": 1, "quote_offset": 0,
             "note": "Checked this node in the final tree."})
        self.proposal["conclusions"].extend(
            {"subject": subject, "text": "Reviewed this criterion across both current nodes.", "evidence": []}
            for subject in DOCUMENT_CONCLUSIONS)
        state = {key: self.attempt[key] for key in ("run_id", "input_fingerprint")}
        state.update(accepted_receipts={}, active_attempts={"P6": self.attempt["attempt_id"]}, plan=plan,
                     pending_rechecks=["root", "branch"], revision_scope=[])
        return state

    def test_p6_tree_can_use_complete_artifact_evidence(self):
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
                self.proposal["conclusions"] = [
                    item for item in self.proposal["conclusions"] if item["subject"] != subject]
                with self.assertRaisesRegex(ExtractError, "missing supported stage conclusions"):
                    validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])
        self.proposal = deepcopy(original)
        self.proposal["artifact_evidence"] = [
            item for item in self.proposal["artifact_evidence"] if item["path"] != "branch.yaml"]
        with self.assertRaises(ExtractError):
            validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])

    def test_p2_cumulative_review_is_required_before_design_is_complete(self):
        state = {"run_id": self.attempt["run_id"], "input_fingerprint": self.attempt["input_fingerprint"],
                 "accepted_receipts": {}, "active_attempts": {"P2": self.attempt["attempt_id"]}}
        self.assertEqual(validate_receipt(self.normalize(), state, self.lock, self.attempt,
                                         self.artifacts, [])["decision"], "approve")
        self.proposal["conclusions"] = [
            item for item in self.proposal["conclusions"] if item["subject"] not in DOCUMENT_CONCLUSIONS]
        with self.assertRaisesRegex(ExtractError, "document_self_contained"):
            validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])

    def test_p6_artifact_tree_still_requires_every_file_and_source_conclusion(self):
        state = self.p6_plan()
        original = deepcopy(self.proposal)
        self.proposal["artifact_evidence"].pop()
        with self.assertRaisesRegex(ExtractError, "missing supported stage conclusions: .*tree"):
            validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])
        self.proposal = original
        self.proposal["conclusions"][1]["evidence"] = []
        with self.assertRaisesRegex(ExtractError, "missing supported stage conclusions: selection"):
            validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])

    def test_p6_artifact_tree_checks_actual_parent_links(self):
        state = self.p6_plan()
        branch = self.artifacts / "branch.yaml"
        branch.write_text(branch.read_text().replace("parent: root", "parent: missing"))
        self.attempt["dependency_digests"] = manifest(self.artifacts)
        with self.assertRaisesRegex(ExtractError, "YAML identity/parent differs"):
            validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])

    def test_p6_artifact_tree_rejects_unplanned_files_even_when_cited(self):
        state = self.p6_plan()
        (self.artifacts / "extra.yaml").write_text("extra: file\n")
        self.proposal["artifact_evidence"].append(
            {"path": "extra.yaml", "line_start": 1, "line_end": 1, "quote_offset": 0, "note": "Observed this extra file."})
        self.attempt["dependency_digests"] = manifest(self.artifacts)
        with self.assertRaisesRegex(ExtractError, "missing supported stage conclusions: tree"):
            validate_receipt(self.normalize(), state, self.lock, self.attempt, self.artifacts, [])

    def test_compact_approval_retains_raw_response_and_full_receipt_gate(self):
        raw = deepcopy(self.proposal)
        receipt = self.normalize()
        state = {key: self.attempt[key] for key in ("run_id", "input_fingerprint")}
        state.update(accepted_receipts={}, active_attempts={"P2": self.attempt["attempt_id"]})
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
        fragments, cursor = [], None
        while True:
            page = read_range(self.source, "acceptance.json", 1, 1, max_bytes=1024, cursor=cursor)
            fragments.extend(item["text"] for item in page["fragments"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        self.assertEqual("".join(fragments), line)

    def test_audit_checkpoint_identity_is_stable_across_processes(self):
        # Isolate storage IO; keep the real audit selection and serialization.
        # Changing a process hash seed must not change an archive's identity.
        script = """
import json
from types import SimpleNamespace
from schema_extract.tools import audit
state = {"run_id": "run", "status": "ready", "checkpoint": "a" * 64, "merged_checkpoint": "b" * 64}
audit._inputs = lambda *args: {}
audit._checkpoint = lambda store, state, identity, files: {"id": identity, "files": {}}
audit._state = lambda *args: state
store = SimpleNamespace(writer_status=lambda _: "available")
print(json.dumps(audit._verify(store, state, [], None, whole_run=True), sort_keys=True))
"""
        outputs = [subprocess.check_output([sys.executable, "-B", "-c", script],
                    cwd=Path(__file__).resolve().parents[3], env={**os.environ, "PYTHONHASHSEED": seed})
                   for seed in ("1", "2", "3")]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])


if __name__ == "__main__":
    unittest.main()
