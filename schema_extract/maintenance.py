"""Read-only Supervisor summaries and a stable-ID view of maintenance history."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .common import ExtractError, digest, read_json
from .snapshots import verify_snapshot


CATEGORIES = {"schema_docs", "schema_template", "kernel_wiki", "extract_workflow", "extract_tools"}
OUTCOMES = {"declined", "resolved", "partially_adopted"}
STABLE_ID = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def registry(repo: Path) -> dict:
    """Legacy files remain historical data, never an imported runtime package."""
    records = {}
    for root in (repo / "schema_extract_old/issues", repo / "schema_extract/issues"):
        for path in sorted(root.rglob("*.json")):
            value = read_json(path)
            identity, category = value.get("id"), value.get("category")
            if (not isinstance(identity, str) or not STABLE_ID.fullmatch(identity) or
                    identity != path.stem or category not in CATEGORIES or identity in records):
                raise ExtractError("invalid or duplicate stable maintenance ID: " + str(path))
            for name in ("problem", "proposal", "necessity"):
                if not isinstance(value.get(name), str) or not value[name].strip():
                    raise ExtractError("maintenance record is missing " + name + ": " + str(path))
            if not isinstance(value.get("evidence"), list) or not value["evidence"] or not all(
                    isinstance(item, str) and item.strip() for item in value["evidence"]):
                raise ExtractError("maintenance record needs reproducible evidence: " + str(path))
            archived = path.parent.relative_to(root).parts[:1] == ("archive",)
            expected = ("archive", category) if archived else (category,)
            if path.parent.relative_to(root).parts != expected:
                raise ExtractError("maintenance record is in the wrong category directory: " + str(path))
            decision = value.get("decision")
            if archived:
                if (not isinstance(decision, dict) or decision.get("outcome") not in OUTCOMES or
                        not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
                    raise ExtractError("archived decision needs its outcome and reason: " + str(path))
                if decision["outcome"] == "partially_adopted" and not all(
                        isinstance(decision.get(key), str) and decision[key].strip() for key in ("adopted", "remaining")):
                    raise ExtractError("partial decision needs adopted and remaining scope: " + str(path))
            elif decision is not None:
                raise ExtractError("decided issues belong in archive: " + str(path))
            records[identity] = {**value, "path": str(path.relative_to(repo)),
                                 "status": decision["outcome"] if archived else "pending"}
    return {"records": records, "pending": sum(item["status"] == "pending" for item in records.values()),
            "archived": sum(item["status"] != "pending" for item in records.values())}


def candidate_report(store, state: dict, repo: Path) -> list[dict]:
    registered, groups = registry(repo)["records"], {}
    for attempt_id, receipt_hash in state["accepted_receipts"].items():
        path = store.run_dir(state["run_id"]) / "attempts" / attempt_id / "receipt.json"
        receipt = read_json(path)
        if digest(receipt) != receipt_hash:
            raise ExtractError("accepted framework proposal receipt has changed: " + str(path))
        for item in receipt.get("framework_candidates", []):
            valid = bool(STABLE_ID.fullmatch(item["id"]))
            group = groups.setdefault(item["id"], {"id": item["id"],
                "registered": registered.get(item["id"]) if valid else None, "reports": []})
            if not valid:
                group["invalid_reason"] = "候选需要小写字母数字及-或_组成的稳定ID；未登记，原回执仍可查看。"
            source = {"run_id": state["run_id"], "attempt_id": attempt_id,
                      "stage": receipt["identity"]["stage"], "node": receipt["identity"]["node"],
                      "role": receipt["identity"]["role"], "receipt": str(path),
                      "evidence": [receipt["source_evidence"][index] for index in item["evidence"]]}
            group["reports"].append({"candidate": item, "source": source})
    return [groups[key] for key in sorted(groups)]


def inspect_runs(store, repo: Path) -> list[dict]:
    """Ask each frozen executor for its summary; inspection never starts a call."""
    summaries = []
    for path in sorted((store.root / "runs").glob("*/state.json")):
        run_id = path.parent.name
        try:
            lock = read_json(path.parent / "inputs.lock.json")
            if lock["fingerprint"] != store.load(run_id)["input_fingerprint"]:
                raise ExtractError("run input fingerprint differs from its lock")
            verify_snapshot(lock)
            rules = Path(lock["snapshot"]) / "rules"
            process = subprocess.run([sys.executable, "-B", "-m", "schema_extract", "status", run_id,
                                      "--repo", str(repo.resolve()), "--state-root", str(store.root)],
                                     cwd=rules, env={**os.environ, "PYTHONPATH": str(rules), "PYTHONDONTWRITEBYTECODE": "1"},
                                     capture_output=True, text=True, timeout=30)
            if process.returncode:
                raise ExtractError(process.stderr.strip())
            summaries.append({"run": json.loads(process.stdout), "executor": str(rules)})
        except (ExtractError, ValueError, OSError, subprocess.SubprocessError) as exc:
            summaries.append({"run_id": run_id, "error": str(exc)})
    return summaries
