"""Targeted, offline evidence for the second schema/contract review.

Reads the exact twelve delivered runs and current sources. Writes only result.json
beside this script; no model, GPU, Runner, monitor, or production mutation.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from schema_template.tools import text_policy, yaml_source
from schema_template.tools.pipeline.expressions import Domains, tree
from schema_template.tools.pipeline.model import GraphError


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


index_path = REPO / "schema_discussion/extract_twelve_20261119/DELIVERY_INDEX.json"
round1_path = REPO / "schema_discussion/extract_schema_review_20261121/round1/SCHEMA_FINDINGS.json"
index = json.loads(index_path.read_text())
round1 = json.loads(round1_path.read_text())
mapping = {c["id"]: c for c in round1["framework_candidate_mapping"]}
occurrences = defaultdict(list)
receipts_read = 0
formal_before = {}
publication_files = 0
tolerance_records = []

for run in index["runs"]:
    directory = REPO / run["output_relative"]
    delivery = directory / ".extraction/receipt.json"
    formal_before[str(delivery.relative_to(REPO))] = sha(delivery)
    proof = json.loads(delivery.read_text())
    assert proof["run_id"] == run["run_uuid"]
    for name, digest in proof["files"].items():
        path = directory / name
        assert sha(path) == digest
        formal_before[str(path.relative_to(REPO))] = digest
        publication_files += 1

    for receipt in sorted((REPO / ".schema_extract/v2/runs" / run["run_uuid"] / "attempts").glob("*/receipt.json")):
        receipts_read += 1
        data = json.loads(receipt.read_text())
        for candidate in data.get("framework_candidates", []):
            occurrences[candidate["id"]].append({
                "receipt": str(receipt.relative_to(REPO)), "sha256": sha(receipt),
            })

    root = yaml_source.load(directory / "root.yaml", "root").value
    current = root["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]["Tolerance"]
    criteria = {}

    def walk(value, pointer="", file=None):
        if isinstance(value, dict):
            tolerance = value.get("tolerance")
            if isinstance(tolerance, dict) and "mode" in tolerance:
                key = json.dumps(tolerance, sort_keys=True)
                record = criteria.setdefault(key, {
                    "source_file": str(file.relative_to(REPO)),
                    "source_sha256": sha(file),
                    "first_pointer": pointer + "/tolerance", "values": tolerance,
                    "source_outputs": [],
                })
                name = value.get("name")
                if name not in record["source_outputs"]:
                    record["source_outputs"].append(name)
            for key, child in value.items():
                walk(child, pointer + "/" + key, file)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, pointer + "/" + str(i), file)

    for entry in run["source"]["bindings"].get("acceptance", []):
        source = REPO / entry["path"]
        if source.suffix == ".json":
            walk(json.loads(source.read_text()), file=source)
    if criteria:
        assert all("mode=" not in text for text in current.values())
        tolerance_records.append({
            "operator_id": run["canonical_id"],
            "root": str((directory / "root.yaml").relative_to(REPO)),
            "current": current, "criteria": list(criteria.values()),
        })

assert set(occurrences) == set(mapping)
for name, rows in occurrences.items():
    expected = {(x["receipt"], x["receipt_sha256"]) for x in mapping[name]["receipts"]}
    actual = {(x["receipt"], x["sha256"]) for x in rows}
    assert expected == actual, name

masked_cases = masked_tiles = 0
max_extra = 0
for capacity in range(128, 1025, 128):
    for count in range(capacity + 1):
        masked_cases += 1
        last_end = 0
        for mb in range((count + 127) // 128):
            masked_tiles += 1
            eff = min(128, count - 128 * mb)
            eff_a = (eff + 15) // 16 * 16
            start = 128 * mb
            assert 1 <= eff <= eff_a <= 128
            assert start >= last_end and start + eff_a <= capacity
            assert 0 <= eff_a - eff <= 15
            last_end = start + eff_a
            max_extra = max(max_extra, eff_a - eff)

quarter_domains = []
for blocks in (1, 2, 6, 16, 28):
    groups = [16 * t + 4 * q + u for q in range(4) for t in range(blocks) for u in range(4)]
    assert sorted(groups) == list(range(16 * blocks))
    quarter_domains.append({"k": 256 * blocks, "groups": len(groups), "each_group_once": True})

source_text = (
    "Source only; atol=0.01, rtol=0.05, max_mismatch_fraction=0.001, "
    "max_rel_l2=0.01, relative_to=candidate, mode=elementwise_and_l2; "
    "comparator definition unavailable; no defaults"
)
bare_six_fields = (
    "atol=.01,rtol=.05,max_mismatch_fraction=.001,max_rel_l2=.01,"
    "relative_to=candidate,mode=elementwise_and_l2"
)
budget_errors = text_policy.check_values({"Golden_reference": {"Tolerance": {"out": source_text}}}, scope="algorithm")
assert len(bare_six_fields) > 100 and budget_errors
assert text_policy.budget_message(source_text, 300, 2) is None
# No global BUDGETS mutation: the proposed setting is checked only by the existing helper.

domain = Domains({}, {}, 1)
short_circuit = []
for expression, expected in (
    ("false and 1 % 0 == 0", False),
    ("true or 1 % 0 == 0", True),
    ("0 > 0 and 1 % 0 == 0", False),
):
    result = domain.evaluate(tree(expression), {})
    assert result is expected
    short_circuit.append({"expression": expression, "result": result})
try:
    domain.evaluate(tree("true and 1 % 0 == 0"), {})
except GraphError as exc:
    short_circuit.append({"expression": "true and 1 % 0 == 0", "error": str(exc)})
else:
    raise AssertionError("actually evaluated zero divisor must fail")

for path, digest in formal_before.items():
    assert sha(REPO / path) == digest

result = {
    "status": "audit_only_no_product_changes",
    "scope": {"delivery_index": str(index_path.relative_to(REPO)), "sha256": sha(index_path)},
    "coverage": {
        "receipts_read": receipts_read,
        "unique_candidate_ids": len(occurrences),
        "candidate_occurrences": sum(map(len, occurrences.values())),
        "round1_coordinates_and_hashes_match": True,
        "candidate_counts": {key: len(occurrences[key]) for key in sorted(occurrences)},
    },
    "publication_bytes": {"deliveries": 12, "yaml_and_dispatch_files": publication_files,
                          "match_original_proof": True, "preserved_files": formal_before},
    "tolerance_source_records_with_omitted_mode": tolerance_records,
    "tolerance_budget": {
        "six_fields_without_caveats": bare_six_fields,
        "six_fields_characters": len(bare_six_fields),
        "complete_example": source_text,
        "complete_example_size": text_policy.text_size(source_text),
        "current_budget_diagnostics": [{"path": d.path, "message": d.message} for d in budget_errors],
        "proposed_300_char_2_line_budget_accepts_example": True,
    },
    "masked_extent": {
        "capacity_domain": "128..1024, multiples of 128",
        "masked_m_domain": "every integer 0..capacity",
        "cases": masked_cases, "nonempty_tiles": masked_tiles, "max_extra_rows": max_extra,
        "proposed_formula": "eff=min(128,masked_m-128*mb); eff_a=ceil_div(eff,16)*16",
        "index_invariants_pass": True,
        "current_counterexample": {"masked_m": 256, "mb": 0, "current_eff": 256, "source_eff": 128},
        "scope": "index arithmetic only, no GPU execution or floating-point equivalence proof",
    },
    "masked_exactness_counterexamples": {
        "overflow": "2^100 * 2^100 = 2^200 exceeds finite FP32",
        "underflow": "2^-127 * 2^-23 = 2^-150 is half the FP32 minimum subnormal",
        "both_decoded_operands_individually_representable": True,
        "claim_falsified": "all summands are exact in FP32 over the stated Contract domain",
        "actual_gpu_failure_demonstrated": False,
    },
    "weight_band_mma_group_coverage": quarter_domains,
    "fp8_fmha_storage_arithmetic": {
        "root_each_m_l": {"formula": "128 threads * MT=2 * 2 f32 * 4 bytes", "expected_bytes": 2048, "current_bytes": 512},
        "decode_each_m_l": {"formula": "128 threads * MT=1 * 2 f32 * 4 bytes", "expected_bytes": 1024, "current_bytes": 256},
        "root_page_regs": {"formula": "4 warp copies * 32 keys * 4 words * 4 bytes", "expected_bytes": 2048, "current_bytes": 512},
        "decode_page_regs": {"active_payload_bytes": 1024, "full_lane_slot_capacity_if_uniform": 2048,
                             "note": "BN=16 writes only 16 lanes per warp; no automatic factor-of-four correction. Compiler allocation is not established by this payload calculation."},
    },
    "short_circuit_evaluator": short_circuit,
    "not_run": ["extract", "resume", "monitor", "repair-runtime", "real_model", "GPU", "new_P6_review"],
}
output = Path(__file__).with_name("result.json")
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(output.relative_to(REPO)), "coverage": result["coverage"],
                  "masked_cases": masked_cases, "tolerance_operators": len(tolerance_records),
                  "tolerance_outputs": sum(len(x["current"]) for x in tolerance_records),
                  "complete_tolerance_size": text_policy.text_size(source_text)}, ensure_ascii=False))
