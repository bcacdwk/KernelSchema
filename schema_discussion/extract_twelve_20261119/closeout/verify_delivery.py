"""Read-only delivery verification; writes evidence only beside this script."""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent.parent
CLOSEOUT = BASE / "closeout"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    launch = read(BASE / "launch.json")
    observations = {row["run_id"]: row for row in read(CLOSEOUT / "inspect.json")}
    summary = read(CLOSEOUT / "status_summary.json")
    status_rows = {row["run_id"]: row for row in summary["runs"]}
    initial_monitor = read(BASE / "monitoring/start.stdout")
    expected_ids = {row["run_uuid"] for row in launch["runs"]}
    assert len(expected_ids) == len(launch["runs"]) == 12
    assert expected_ids == set(observations) == set(status_rows)
    assert expected_ids == set(initial_monitor["run_ids"])
    category_counts = dict(Counter(row["category"] for row in launch["runs"]))
    assert len(category_counts) == 6 and set(category_counts.values()) == {2}

    checked_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    entries = []
    pending_delivery = None
    for run in launch["runs"]:
        run_id = run["run_uuid"]
        status_row = status_rows[run_id]
        evidence = status_row["evidence"]
        state = observations[run_id] if evidence == "inspect.json" else read(CLOSEOUT / evidence)
        assert state["run_id"] == run_id and state["operator_id"] == run["id"]
        assert state["status"] == "ready" and state["stage"] == "P6" and state["published"] is True
        assert all(not state[key] for key in ("active_attempts", "unconfirmed_attempts", "unsettled_attempts"))
        assert not observations[run_id]["anomalies"]
        assert state["output"] == run["output"]
        publication = state["publication"]
        assert publication["status"] == "published" and publication["target"] == run["output"]
        output = Path(run["output"])
        receipt_path = output / ".extraction/receipt.json"
        receipt = read(receipt_path)
        assert receipt == publication["proof"]
        assert receipt["run_id"] == run_id
        assert receipt["checkpoint"] == publication["checkpoint"] == state["checkpoint"]
        assert receipt["input_fingerprint"] == run["input_fingerprint"] == state["input_fingerprint"]
        assert receipt["rules_id"] == run["rules_id"]
        assert receipt["executor"] == run["executor"] == state["executor"]
        assert {key: receipt["source"][key] for key in run["source"]} == run["source"]
        assert state["source_identity"] == run["source"]
        bindings = receipt["source"]["bindings"]
        assert bindings["id"] == bindings["selection"]["id"] == run["id"]
        assert bindings["directory"] == str(output.relative_to(ROOT)).replace("schema_wiki/", "kernel_wiki/", 1)
        files = receipt["files"]
        actual_files = {str(p.relative_to(output)): digest(p) for p in output.rglob("*")
                        if p.is_file() and ".extraction" not in p.relative_to(output).parts}
        assert actual_files == files, (run["id"], "formal bytes differ", actual_files, files)
        yaml_files = sorted(name for name in actual_files if name.endswith(".yaml"))
        assert "root.yaml" in yaml_files and "dispatch.md" in actual_files
        node_ids = {Path(name).stem for name in yaml_files}
        assert node_ids == set(receipt["checked_nodes"]) == {node["id"] for node in receipt["selected_designs"]}
        assert len(yaml_files) == len(receipt["selected_designs"]) == len(receipt["checked_nodes"])
        assert receipt["check_policy"] == "skip-hardware"
        assert receipt["approval"]["session_scope"] == {"node": "root", "operation": "normal", "role": "reviewer", "stage": "P6"}
        delivery_path = BASE / "deliveries" / (run["id"] + ".json")
        if run["id"] == "bf16_nvfp4_gemm":
            assert receipt["checkpoint"] == "413b225c62c101418ec23563655ec52a0fc1bddc9018475df7ddf823871eb877"
            assert all(item["path"].startswith(bindings["directory"] + "/") for item in bindings["scope"])
            assert state["stages"]["P6"]["status"] == "approved"
            assert state["stages"]["P6"]["approval"]["artifact_digests"] == files
            delivery = {
                "operator_id": run["id"], "run_id": run_id, "status": "verified_published",
                "output": run["output"], "checked_at_beijing": checked_at,
                "checkpoint": receipt["checkpoint"], "receipt_sha256": digest(receipt_path),
                "receipt_matches_publication_proof": True, "all_calls_settled": True,
                "files": files, "checked_nodes": receipt["checked_nodes"], "check_policy": receipt["check_policy"],
                "authoritative_status_full_confirmed": True,
                "source_identity_and_scope_match_launch": True,
                "verification_method": "current_state_receipt_manifest_and_existing_p6_evidence",
                "checker_rerun": False, "gpu_run": False,
                "status_evidence": "closeout/" + evidence,
            }
            if delivery_path.exists():
                old_delivery = read(delivery_path)
                assert all(old_delivery[key] == delivery[key] for key in ("run_id", "checkpoint", "receipt_sha256", "files"))
            else:
                pending_delivery = (delivery_path, delivery)
        else:
            delivery = read(delivery_path)
            assert delivery["operator_id"] == run["id"] and delivery["run_id"] == run_id
            assert delivery["output"] == run["output"] and delivery["status"] == "verified_published"
            assert delivery["checkpoint"] == receipt["checkpoint"]
            assert delivery["receipt_sha256"] == digest(receipt_path)
            assert delivery["files"] == files and delivery["receipt_matches_publication_proof"] is True
            assert delivery["all_calls_settled"] is True
        entries.append({
            "canonical_id": run["id"], "category": run["category"], "run_uuid": run_id,
            "output": run["output"], "output_relative": str(output.relative_to(ROOT)),
            "status": "ready", "stage": "P6", "published": True, "all_calls_settled": True,
            "checkpoint": receipt["checkpoint"], "source": receipt["source"],
            "yaml_files": yaml_files, "design_count": len(yaml_files),
            "dispatch": str(output / "dispatch.md"),
            "verification_record": str(delivery_path.relative_to(ROOT)),
            "status_evidence": str((CLOSEOUT / evidence).relative_to(ROOT)),
            "receipt": str(receipt_path), "receipt_sha256": digest(receipt_path),
            "receipt_matches_publication_proof": True, "files_sha256": actual_files,
            "matches_existing_delivery_record": run["id"] != "bf16_nvfp4_gemm",
            "check_policy": receipt["check_policy"],
        })
    counts = {"operators": len(entries), "categories": len(category_counts),
              "yaml_designs": sum(row["design_count"] for row in entries),
              "dispatches": sum("dispatch.md" in row["files_sha256"] for row in entries),
              "receipts": len(entries), "active_attempts": 0, "unconfirmed_attempts": 0, "unsettled_attempts": 0}
    assert counts["yaml_designs"] == 33 and counts["dispatches"] == 12
    if pending_delivery:
        save(*pending_delivery)
    result = {"format": "schema_discussion.delivery_verification.v1", "checked_at_beijing": checked_at,
              "status_checked_at_beijing": summary["checked_at_beijing"], "status": "verified_published",
              "counts": counts, "category_counts": category_counts,
              "scope": "exact twelve launch.json runs only", "formal_files_modified": False,
              "checker_rerun": False, "gpu_run": False, "additional_semantic_audit": False,
              "differences": [], "runs": entries}
    save(CLOSEOUT / "delivery_verification.json", result)
    print(json.dumps({"checked_at_beijing": checked_at, "counts": counts, "category_counts": category_counts,
                      "differences": [], "gemm_verified": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
