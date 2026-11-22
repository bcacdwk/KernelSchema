"""Offline S-06 checks; no run state, model invocation, or published file writes.

Run from repository root with PYTHONDONTWRITEBYTECODE=1 python -B <this file>.
Temporary artifact directories are removed by the existing fixture harness.
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from schema_template.tools import normalization, schema, yaml_source
from schema_extract.tools import artifacts
from schema_extract.tools.tests.test_artifact_transactions import ArtifactTransactions, STARTER, selection
from schema_extract.tools.tests.test_stage_flow import StageFlowTests


def coherent_source(math, *, retain_input=False):
    source = (ROOT / "schema_extract/tools/tests/fixtures/review_eval/base.yaml").read_text()
    formula = "y = 2*v + 1" if retain_input else "y = v + 1"
    action = "Double and increment" if retain_input else "Increment"
    source = source.replace("scalar_copy", "scalar_double_increment" if retain_input else "scalar_increment")
    source = source.replace("Copy one scalar", action + " one scalar")
    source = source.replace("Copied scalar", "Computed scalar")
    source = source.replace("Input: {x:", "Input: {v:")
    source = source.replace("Invariant: {x:", "Invariant: {v:")
    source = source.replace("Name: Copy", "Name: " + action)
    source = source.replace("y = x", formula)
    source = source.replace("Thesis: Retain the scalar copy without intermediate values", "Thesis: " + action + " the input scalar")
    source = source.replace("One serial role copies the scalar", "One serial role computes the scalar output")
    source = source.replace("Copy x to y", "Compute y from v")
    source = source.replace("Read x and write y according to Copy", "Read v and produce y according to the defined Math")
    source = source.replace("dtype: fp32", "dtype: int32")
    source = source.replace("Input: {v: {meaning: Input scalar, shape: [], dtype: int32}}",
                            "Input: {v: {meaning: Bounded integer scalar, shape: [], dtype: int32, range: [0, 16]}}")
    contract, design = source.split("DESIGN:", 1)
    return contract + "DESIGN:" + design.replace(formula, math.replace("\n", "\n          ").rstrip(), 1)


def public_case(math, *, retain_input=False):
    source = coherent_source(math, retain_input=retain_input)
    files = {"root": Path("/virtual/root.yaml")}
    document = yaml_source.loads(source, "root", path=files["root"])
    result = {"before_math": document.value["DESIGN"]["Math_reshape"]["Rewrite"][0]["Math"]}
    schema.validate_documents(files, {"root": document}, skip_hardware=True)
    result["input_static_check"] = "passed"
    try:
        plan = normalization.normalize_documents(files, {"root": document}, skip_hardware=True)
    except schema.SchemaError as exc:
        return {**result, "normalization": "rejected", "diagnostic": str(exc)}
    schema.validate_documents(plan.files, plan.documents, skip_hardware=True)
    return {**result, "normalization": "planned", "after_math": plan.documents["root"].value["DESIGN"]["Math_reshape"]["Rewrite"][0]["Math"],
            "output_static_check": "passed", "idempotent": not normalization.normalize_documents(plan.files, plan.documents, skip_hardware=True).candidates,
            "input_object_unchanged": document.source == source,
            "diagnostics": [{"path": item.path, "message": item.message} for item in plan.corrections]}


def artifact_transaction():
    fixture = ArtifactTransactions()
    fixture.setUp()
    try:
        value = yaml_source.loads(coherent_source("y = v + 1"), "root").value
        value["CONTRACT"]["Hardware_specification"] = deepcopy(fixture.base["CONTRACT"]["Hardware_specification"])
        value["META_DATA"].update(Source=fixture.base["META_DATA"]["Source"], Dispatch={"when": "All scalar inputs"})
        fixture.base = value
        before = fixture.task("before")
        original = fixture.copy(before, "original")
        edited = deepcopy(value)
        edited["META_DATA"]["Version"] = 2
        edited["DESIGN"]["Math_reshape"]["Rewrite"][0]["Math"] = "V = v + 1\ny = V\n"
        artifacts.write_yaml(original / "root.yaml", edited, parser_root=ROOT)
        before_bytes = (before / "root.yaml").read_bytes()
        original_bytes = (original / "root.yaml").read_bytes()
        landing = fixture.directory / "landing"
        result = artifacts.normalize_candidate(before, original, landing, STARTER, stage="P3", role="worker", decision="complete", selected_designs=selection("root"))
        landed = artifacts.load_root(landing, parser_root=ROOT)
        return {"status": result["status"], "strict_stage_check_exit": result["check"]["exit_code"],
                "landed_math": landed["DESIGN"]["Math_reshape"]["Rewrite"][0]["Math"],
                "baseline_unchanged": (before / "root.yaml").read_bytes() == before_bytes,
                "original_unchanged": (original / "root.yaml").read_bytes() == original_bytes,
                "independent_reviewer_executed": False, "real_runner_state_changed": False}
    finally:
        fixture.doCleanups()


def full_worker_acceptance():
    """Use real fixed source and all accept_result gates; reviewer judgments are fake."""
    fixture = StageFlowTests()
    fixture.setUp()
    try:
        (fixture.directory / "source/kernel.py").write_text("def copy(v): return 2*v + 1\n\ndef reference(v): return 2*v + 1\n")
        fixture.create()
        value = yaml_source.loads(coherent_source("V = v + 1\ny = V + v", retain_input=True), "root").value
        value["CONTRACT"]["Hardware_specification"] = deepcopy(fixture.backend.base["CONTRACT"]["Hardware_specification"])
        value["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]["Source"] = fixture.backend.reference
        fixture.backend.base = value
        for stage, role in (("P1", "worker"), ("P1", "reviewer"), ("P2", "worker"), ("P2", "reviewer")):
            fixture.step(stage=stage, role=role)
        attempt = fixture.step(stage="P3", role="worker")
        checkpoint, _ = fixture.store.verify_checkpoint(fixture.state)
        landed = artifacts.load_root(checkpoint, parser_root=ROOT)
        original = artifacts.load_root(Path(attempt["workspace"]), parser_root=ROOT)
        return {"worker_receipt_accepted": attempt["attempt_id"] in fixture.state["accepted_receipts"],
                "stage": fixture.state["stage"], "next_role": fixture.state["role"],
                "original_math": original["DESIGN"]["Math_reshape"]["Rewrite"][0]["Math"],
                "checkpoint_math": landed["DESIGN"]["Math_reshape"]["Rewrite"][0]["Math"],
                "p3_approval": fixture.state["stages"]["P3"]["approval"],
                "fixed_source": "temporary bounded-int32 implementation and reference: 2*v+1",
                "sequential_interpretation": [{"v": v, "contract_and_original_y": 2*v+1, "normalized_y": 2*(v+1)} for v in (0, 1, 16)],
                "real_model_calls": 0, "p3_reviewer_executed": False}
    finally:
        fixture.doCleanups()


cases = {"registered_input_and_plain_local": "V = v + 1\ny = V",
         "open_math_definition": "V := v + 1\ny = V",
         "distinct_lowercase_local": "tmp = v + 1\ny = tmp"}
report = {"current_code": {name: public_case(math) for name, math in cases.items()}}
report["current_code"]["original_input_used_after_local_definition"] = public_case("V = v + 1\ny = V + v", retain_input=True)
original_function = normalization._formula_scopes
namespace = dict(normalization.__dict__)
exec(inspect.getsource(original_function).replace("related=match[1].casefold() in established", "related=False"), namespace)
with patch.object(normalization, "_formula_scopes", namespace["_formula_scopes"]):
    report["narrow_hypothesis_disable_formula_related_only"] = {name: public_case(math) for name, math in cases.items()}
report["real_normalization_transaction_in_temporary_fixture"] = artifact_transaction()
report["real_accept_result_with_fake_backend"] = full_worker_acceptance()
print(json.dumps(report, ensure_ascii=False, indent=2))
