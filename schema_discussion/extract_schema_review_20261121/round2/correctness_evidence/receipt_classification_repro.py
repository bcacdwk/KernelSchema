"""Offline classification at Runner's result boundary with the existing fake backend.

No AgentCLI/run loop, native CLI, external source read, or real run is used.
For schema failures we derive response_validation with the production validator,
matching agent.read_result's receipt_error branch. This is not a native-log replay.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from schema_extract.tools.agent import _schema_check
from schema_extract.tools.receipts import proposal_schema
from schema_extract.tools.runner import _receive
from schema_extract.tools.common import write_json
from schema_extract.tools.tests.test_stage_flow import StageFlowTests


def scenario(kind):
    fixture = StageFlowTests()
    fixture.setUp()
    try:
        fixture.create(backend="codex" if kind == "native_output_mismatch" else "claude")
        attempt = fixture.prepare(stage="P1", role="worker")
        result = fixture.result(attempt)
        before = deepcopy(fixture.state)
        if kind == "missing_summary":
            result["response"].pop("summary")
        elif kind == "outside_source_manifest":
            result["response"]["source_evidence"][0]["path"] = "../outside.py"
        elif kind == "wrong_decision":
            result["response"]["decision"] = "approve"
        elif kind == "wrong_nonce":
            result["response"]["attempt_id"] = "another-attempt"
        schema = proposal_schema(attempt, source_files=fixture.lock["source"]["files"])
        checked = _schema_check(schema, result["response"], attempt["attempt_id"])
        checked["native_output_matches"] = kind != "native_output_mismatch"
        result.update(outcome="receipt_error", response_validation=checked)
        write_json(Path(attempt["directory"]) / "result.json", result)
        state = _receive(fixture.store, fixture.state, fixture.lock, attempt, result, None)
        scope = state["stages"]["P1"]
        return {"case": kind, "schema_valid": checked["schema_valid"], "native_output_matches": checked["native_output_matches"],
                "schema_errors": checked["schema_errors"], "status": state["status"], "reason": state["reason"],
                "receipt_repairs": scope["receipt_repairs"], "calls": scope["calls"],
                "draft_checkpoint_changed": state["checkpoint"] != before["checkpoint"],
                "accepted_receipts_unchanged": state["accepted_receipts"] == before["accepted_receipts"],
                "approval": scope["approval"], "real_model_calls": 0}
    finally:
        fixture.doCleanups()


print(json.dumps([scenario(kind) for kind in ("missing_summary", "outside_source_manifest", "wrong_decision", "wrong_nonce", "native_output_mismatch")], ensure_ascii=False, indent=2))
