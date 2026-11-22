"""Exercise existing handoff gates on isolated ProposalTests fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from jsonschema import Draft202012Validator
from schema_extract.tools.common import manifest
from schema_extract.tools.receipts import proposal_schema, validate_receipt, ReceiptFormatError
from schema_extract.tools.stages import node_signature
from schema_extract.tools.tests.test_proposals import ProposalTests

rows = []
scratch = Path(__file__).with_name("scratch")
scratch.mkdir(exist_ok=True)
tempfile.tempdir = str(scratch)
try:
    for stage in ("P2", "P3", "P4", "P6"):
        fixture = ProposalTests(methodName="runTest")
        fixture.setUp()
        try:
            if stage == "P4":
                fixture.p4_plan()
                fixture.attempt.update(role="reviewer", writable_files=[], dependency_digests=manifest(fixture.artifacts))
                fixture.proposal.update(decision="approve", conclusions=fixture.conclusions())
                state = fixture.state()
            elif stage == "P6":
                state = fixture.p6_plan()
            else:
                fixture.attempt["stage"] = stage
                if stage == "P3":
                    fixture.attempt["node_signature"] = node_signature(fixture.selected[0])
                    fixture.lock["source"]["bindings"]["entrypoint"] = deepcopy(fixture.lock["source"]["bindings"]["host"])
                    extra = deepcopy(fixture.proposal["source_evidence"][0])
                    extra["category"] = "implementation"
                    fixture.proposal["source_evidence"].append(extra)
                    fixture.proposal.pop("plan")
                fixture.proposal["conclusions"] = fixture.conclusions()
                state = fixture.state()
            base = deepcopy(fixture.proposal)
            cases = [
                ("approve", "unicode_1200", "界" * 1200),
                ("approve", "unicode_1201", "界" * 1201),
                ("approve", "20_nonempty_lines_with_blanks", "\n\n".join(["x"] * 20)),
                ("approve", "21_nonempty_lines", "\n".join(["x"] * 21)),
                ("approve", "empty", ""),
            ]
            if stage != "P6":
                cases.append(("revise", "long_unlimited", "x" * 3000))
            for decision, case, handoff in cases:
                fixture.proposal = deepcopy(base)
                fixture.proposal.update(decision=decision, handoff=handoff)
                if decision == "revise":
                    fixture.proposal["issues"] = [{"kind": "schema_conformance", "scope": "root.yaml", "detail": "Revise the explanation.", "recovery_condition": "", "evidence": [0]}]
                    if "plan" in fixture.proposal:
                        fixture.proposal["plan"] = []
                errors = list(Draft202012Validator(proposal_schema(fixture.attempt, source_files=fixture.lock["source"]["files"])).iter_errors(fixture.proposal))
                assert not errors, errors
                diagnostic = None
                try:
                    validate_receipt(fixture.normalize(), state, fixture.lock, fixture.attempt, fixture.artifacts, [])
                    passed = True
                except ReceiptFormatError as exc:
                    passed = False
                    diagnostic = str(exc)
                    assert "handoff" in diagnostic
                expected = (stage == "P6" or decision != "approve" or case in {"unicode_1200", "20_nonempty_lines_with_blanks"})
                assert passed == expected, (stage, case, passed)
                rows.append({"stage": stage, "decision": decision, "case": case,
                             "characters": len(handoff), "nonempty_lines": sum(bool(x.strip()) for x in handoff.splitlines()),
                             "json_schema_accepts": True, "receipt_gate_accepts": passed, "diagnostic": diagnostic})
        finally:
            fixture.doCleanups()
finally:
    scratch.rmdir()

result = {"scope": "isolated existing fixtures, no run or backend calls", "rows": rows,
          "p5": "not executed in this matrix; existing gate is stage-restricted and real P5 receipt evidence is cited in the report"}
Path(__file__).with_name("receipt_matrix.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"cases": len(rows), "all_assertions_pass": True, "source_kind": "current unmodified product behavior"}))
