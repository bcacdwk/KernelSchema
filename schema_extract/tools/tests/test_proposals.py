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

from schema_extract.tools.audit import _receipt_response
from schema_extract.tools.common import ExtractError, manifest
from schema_extract.tools.reading import read_evidence, read_range
from schema_extract.tools.receipts import ReceiptFormatError, normalize_response, validate_receipt


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

    def normalize(self, proposal=None):
        return normalize_response(self.proposal if proposal is None else proposal,
                                  self.lock, self.attempt, self.artifacts)

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
