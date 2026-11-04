"""Source records retain the existing P1 artifact, provenance and comparison gates."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from schema_extract.tools.artifacts import (load_yaml, validate_artifact_sources,
                                           validate_artifacts, write_yaml)
from schema_extract.tools.common import ExtractError, manifest
from schema_extract.tools.receipts import normalize_response, validate_receipt


REPO = Path(__file__).resolve().parents[3]
SOURCE_RECORD = ("Source record: atol=0.01, rtol=0.05. The comparison formula and relative-error basis "
                 "are not specified; the default atol/rtol criterion is not adopted.")


class SourceGapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # These tests exercise gates with the current checker. Snapshot sealing
        # and frozen entry routing are covered by the CLI recovery tests.
        (self.root / "rules").symlink_to(REPO, target_is_directory=True)
        self.source, self.before, self.after = [self.root / name for name in ("source", "before", "after")]
        for directory in (self.source, self.before, self.after):
            directory.mkdir()
        (self.source / "kernel.py").write_text("def kernel(x): return x\ndef reference(x): return x\ndef inputs(): return [1.0]\n")
        self.task = {"description": "Copy the FP32 scalar X to Y.", "atol": 0.01, "rtol": 0.05}
        (self.source / "task.json").write_text(json.dumps(self.task) + "\n")
        locations = {"entrypoint": {"path": "kernel.py", "symbol": "kernel"},
                     "host": {"path": "kernel.py", "symbol": "kernel"},
                     "input": {"path": "kernel.py", "symbol": "inputs"},
                     "reference": {"path": "kernel.py", "symbol": "reference"},
                     **{key: [{"path": "task.json", "symbol": ""}] for key in ("io", "acceptance", "scope")}}
        self.lock = {"snapshot": str(self.root), "source": {"id": "fixed-source", "repository_id": "fixture",
                     "ref": "fixed", "files": manifest(self.source), "bindings": locations}}
        starter = load_yaml(REPO / "schema_template", "starter.yaml")
        fixture = load_yaml(REPO / "schema_template/tools/tests/fixtures/minimal", "root.yaml")
        self.base = deepcopy(starter)
        self.base["meta_data"] = {**fixture["meta_data"], "source": "fixture@fixed:kernel.py::kernel"}
        self.base["contract"]["algorithm_requirement"] = deepcopy(fixture["contract"]["algorithm_requirement"])
        self.base["contract"]["algorithm_requirement"]["golden_reference"]["source"] = "fixture@fixed:kernel.py::reference"
        write_yaml(self.before / "root.yaml", self.base)

    def artifact(self, criterion=SOURCE_RECORD, change=lambda _: None):
        root = deepcopy(self.base)
        root["meta_data"]["version"] = 2
        root["contract"]["algorithm_requirement"]["golden_reference"]["tolerance"]["Y"] = criterion
        change(root["contract"]["algorithm_requirement"])
        write_yaml(self.after / "root.yaml", root)
        validate_artifacts("P1", "worker", self.before, self.after, REPO / "schema_template/starter.yaml", decision="complete")
        validate_artifact_sources(self.after, self.lock, decision="complete")

    def approve(self, *, declared=False):
        attempt = {"run_id": str(uuid4()), "attempt_id": str(uuid4()), "input_fingerprint": "fixed-input",
                   "stage": "P1", "role": "reviewer", "node": "root", "plan_revision": 1,
                   "base_checkpoint": "reviewed-copy", "dependency_digests": manifest(self.after),
                   "mode": "normal", "generation": 0, "node_signature": "", "writable_files": [],
                   "response_format": "schema_extract.proposal.v1"}
        evidence = [{"category": category, "path": path, "line_start": line, "line_end": line,
                     "quote_offset": 0, "note": "The selected source fixes this role's facts."}
                    for category, path, line in (("implementation", "kernel.py", 1), ("io", "task.json", 1),
                        ("input", "kernel.py", 3), ("reference", "kernel.py", 2),
                        ("acceptance", "task.json", 1), ("scope", "task.json", 1))]
        proposal = {"format": "schema_extract.proposal.v1", "attempt_id": attempt["attempt_id"],
                    "decision": "approve", "summary": "The scalar-copy Contract and its source gap are faithful.",
                    "source_evidence": evidence,
                    "artifact_evidence": [{"path": "root.yaml", "line_start": 1, "line_end": 4,
                                           "quote_offset": 0, "note": "Reviewed the current artifact."}],
                    "conclusions": [{"subject": subject, "text": "Copy semantics are known; no unsupported acceptance guarantee is made.",
                                     "evidence": [0, 1, 3, 4]} for subject in ("contract", "coverage")],
                    "checks": [], "checks_note": "Runner validates the P1 artifact gate.",
                    "issues": [{"kind": "source_gap", "scope": "golden_reference.tolerance.Y",
                                "detail": "An external comparison definition is unavailable; it is not needed to describe scalar copy.",
                                "evidence": [4], "recovery_condition": "Obtain the comparator before using this record for implementation acceptance."}],
                    "comparisons": [{"output": "Y", "source_output": "Y", "evidence": [4]}] if declared else [],
                    "framework_candidates": [], "handoff": ""}
        receipt = normalize_response(proposal, self.lock, attempt, self.after)
        state = {"run_id": attempt["run_id"], "input_fingerprint": attempt["input_fingerprint"],
                 "accepted_receipts": {}, "active_attempts": {"P1": attempt["attempt_id"]}}
        return validate_receipt(receipt, state, self.lock, attempt, self.after, [])

    def test_source_record_with_explicit_issue_can_pass_existing_gates(self):
        self.artifact()
        receipt = self.approve()
        self.assertEqual(receipt["decision"], "approve")
        self.assertEqual(receipt["issues"][0]["kind"], "source_gap")
        self.assertEqual(receipt["comparisons"], [])

    def test_source_record_does_not_hide_missing_core_facts_or_false_reference(self):
        changes = (lambda a: a["input"]["X"].pop("dtype"), lambda a: a["output"].pop("Y"),
                   lambda a: a["golden_reference"].update(source="fixture@fixed:kernel.py::kernel"))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ExtractError):
                self.artifact(change=change)

    def test_source_gap_does_not_relax_explicit_comparison_facts(self):
        self.task["outputs"] = {"Y": {"tolerance": {"relative_to": "candidate", "atol": 0.01, "rtol": 0.05, "max_rel_l2": 0.01}}}
        (self.source / "task.json").write_text(json.dumps(self.task) + "\n")
        self.lock["source"]["files"] = manifest(self.source)
        valid = "relative_to=candidate; atol=0.01; rtol=0.05; max_rel_l2=0.01. External relative-L2 normalization is not specified."
        self.artifact(valid)
        self.assertEqual(self.approve(declared=True)["decision"], "approve")
        for invalid in (valid.replace("relative_to=candidate", "relative_to=reference"),
                        valid.replace("atol=0.01", "atol=1")):
            with self.subTest(criterion=invalid), self.assertRaises(ExtractError):
                self.artifact(invalid)
                self.approve(declared=True)


if __name__ == "__main__":
    unittest.main()
