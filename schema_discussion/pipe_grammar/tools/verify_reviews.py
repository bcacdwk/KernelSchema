#!/usr/bin/env python3
"""Check that per-operator reviews cover the selected nodes and effective pipes."""

import argparse
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identities(rows):
    return {
        (item["node"], item["phase"], item["role"], item["provider"])
        for item in rows
    }


def review_contexts(review):
    if "reviewed_role_contexts" in review:
        rows = review["reviewed_role_contexts"]
        return list(rows.values()) if isinstance(rows, dict) else rows
    rows = []
    for node in review.get("node_reviews", []):
        for role in node.get("reviewed_roles", []):
            rows.append({"node": node["node"], **role})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = load(BASE / "data/summary.json")
    output = {"format": "kernelschema.pipe_reviews.coverage.v1", "operators": {}}
    errors, pending = [], []
    for operator in summary["operators"]:
        source = load(BASE / "data/operators" / (operator + ".json"))
        expected_nodes = {name for name, node in source["files"].items() if node["included"]}
        expected_groups = {group["group_id"] for group in source["exact_pipe_groups"]}
        expected_roles = identities([
            member
            for group in source["exact_pipe_groups"]
            for member in group["members"]
            if "C_effective" in member["scopes"]
        ])
        counts = source["summary"]["counts"]["C_effective"]
        assert len(expected_roles) == counts["roles"], operator
        record = {"expected_nodes": len(expected_nodes), "expected_groups": len(expected_groups),
                  "expected_role_contexts": len(expected_roles)}
        path = BASE / "reviews" / (operator + ".json")
        if not path.exists():
            record["status"] = "pending"
            pending.append(operator)
            output["operators"][operator] = record
            continue
        review = load(path)
        actual_nodes = {item if isinstance(item, str) else item["node"]
                        for item in review["reviewed_nodes"]}
        actual_groups = set(review["reviewed_group_ids"])
        actual_roles = identities(review_contexts(review))
        differences = {}
        for label, expected, actual in [
            ("nodes", expected_nodes, actual_nodes),
            ("groups", expected_groups, actual_groups),
            ("role_contexts", expected_roles, actual_roles),
        ]:
            if expected != actual:
                differences[label] = {"missing": sorted(expected - actual),
                                      "unexpected": sorted(actual - expected)}
        report_exists = (BASE / "operators" / (operator + ".md")).exists()
        finding_ids = [item["id"] for item in review["findings"]]
        unique_finding_ids = len(finding_ids) == len(set(finding_ids))
        record.update({"reviewed_nodes": len(actual_nodes), "reviewed_groups": len(actual_groups),
                       "reviewed_role_contexts": len(actual_roles), "findings": len(finding_ids),
                       "differences": differences, "report_exists": report_exists,
                       "unique_finding_ids": unique_finding_ids})
        if differences or not report_exists or not unique_finding_ids:
            record["status"] = "mismatch"
            errors.append(operator)
        else:
            record["status"] = "covered"
        output["operators"][operator] = record
    output.update({"covered_operators": sum(row["status"] == "covered" for row in output["operators"].values()),
                   "pending_operators": pending, "mismatched_operators": errors})
    target = BASE / "reviews/coverage.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "operators"}, ensure_ascii=False))
    return 1 if errors or (args.require_complete and pending) else 0


if __name__ == "__main__":
    raise SystemExit(main())
