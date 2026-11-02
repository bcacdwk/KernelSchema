"""Structured stage receipts and source-evidence validation."""

from copy import deepcopy
from pathlib import Path
import re

from .framework import CANDIDATES, candidates_from
from .records import StateError, issue_disposition


def obj(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


TEXT = {"type": "string"}
TEXTS = {"type": "array", "items": TEXT}
EVIDENCE = obj({"path": {"type": "string", "description": "Plain relative file path: source_evidence is relative to the fixed repository; artifact_evidence to the workspace. No absolute path, repository@commit prefix or line suffix."},
                "line_start": {"type": "integer"}, "line_end": {"type": "integer"},
                "quote": {"type": "string", "description": "Exact substring within the cited lines. Prefer a short single-line excerpt; preserve whitespace exactly."}, "note": TEXT})
ISSUE = obj({"id": TEXT, "kind": {"type": "string", "enum": ["evidence_gap", "implementation_mismatch",
             "requirement_change", "transcription", "extraction_error", "simplification", "tooling"]},
             "disposition": {"type": "string", "enum": ["fix", "note", "needs_input", "ignored"]},
             "scope": TEXT, "detail": TEXT, "evidence": TEXTS, "restart_condition": TEXT, "blocks": TEXTS})
RECEIPT = obj({
    "task_key": TEXT, "fingerprint": TEXT, "base_checkpoint": TEXT, "stage": TEXT, "node": TEXT,
    "status": {"type": "string", "enum": ["complete", "continue", "revise", "needs_input"]},
    "summary": TEXT, "revision_node": TEXT,
    "continuation": {"type": "string", "enum": ["not_assessed", "proceed", "needs_input"]},
    "continuation_reason": TEXT, "unsupported_claims": TEXTS,
    "dispatches": {"type": "array", "items": obj({"id": TEXT, "parent": TEXT, "reason": TEXT})},
    "issues": {"type": "array", "items": ISSUE}, "resolved_issue_ids": TEXTS,
    "framework_candidates": CANDIDATES,
    "source_evidence": {"type": "array", "items": EVIDENCE},
    "artifact_evidence": {"type": "array", "items": EVIDENCE},
    "impact": {"type": "string", "enum": ["none", "index", "node", "root", "contract", "layout"]},
    "correction": {"type": "string", "enum": ["none", "transcription", "requirements"]},
})


def validate_shape(value, schema, where="receipt"):
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(schema["properties"]):
            raise StateError(f"{where}: wrong receipt fields")
        for key, field in schema["properties"].items():
            validate_shape(value[key], field, f"{where}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise StateError(f"{where}: expected array")
        for item in value:
            validate_shape(item, schema["items"], where)
    elif kind == "integer":
        if type(value) is not int:
            raise StateError(f"{where}: expected integer")
    elif not isinstance(value, str):
        raise StateError(f"{where}: expected string")
    if "enum" in schema and value not in schema["enum"]:
        raise StateError(f"{where}: unsupported value")


def validate_receipt(value, state, baseline, artifacts, role):
    if isinstance(value, dict):
        value = {**value, "framework_candidates": candidates_from(value)}
    validate_shape(value, RECEIPT)
    for key, expected in {"task_key": state["key"], "fingerprint": state["fingerprint"],
                          "base_checkpoint": state.get("checkpoint") or "none", "stage": state["stage"],
                          "node": state["node"]}.items():
        if value[key] != expected:
            raise StateError(f"receipt {key} mismatch")
    value = deepcopy(value)  # The raw model response remains unchanged in result.json.
    statuses = {"complete", "continue", "needs_input"} if role == "worker" else {"complete", "revise", "needs_input"}
    if value["status"] not in statuses:
        raise StateError("receipt status does not match role")
    for issue in value["issues"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", issue["id"]) or not issue["detail"]:
            raise StateError("issues need stable IDs and concrete details")
        if issue_disposition(issue) == "needs_input" and not issue["restart_condition"]:
            raise StateError("needs_input requires the missing fundamental definition/source")
    ids = [item["id"] for item in value["dispatches"]]
    if len(ids) != len(set(ids)) or "root" in ids:
        raise StateError("dispatch list must name unique derived nodes only")
    for item in value["dispatches"]:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", item["id"]) or item["parent"] not in {"root", *ids}:
            raise StateError("invalid dispatch ID or parent")
    evidence_errors = []
    for key, directory in (("source_evidence", Path(baseline["snapshot"])), ("artifact_evidence", artifacts)):
        for index, evidence in enumerate(value[key]):
            anchor = f"{key}[{index}]"
            name = evidence["path"]
            prefix = baseline["repository_id"] + "@" + baseline["snapshot_commit"] + ":"
            if key == "source_evidence" and name.startswith(prefix):
                name = name[len(prefix):]
            elif Path(name).is_absolute():
                try:
                    name = Path(name).relative_to(directory).as_posix()
                except ValueError:
                    pass
            evidence["path"] = name
            path = directory / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file():
                evidence_errors.append(f"{anchor}: invalid evidence path: {name}")
                continue
            if key == "source_evidence" and name not in baseline["files"]:
                evidence_errors.append(f"{anchor}: evidence not in fixed sources: {name}")
                continue
            lines = path.read_text().splitlines()
            first, last = evidence["line_start"], evidence["line_end"]
            if first < 1 or last < first or last > len(lines):
                evidence_errors.append(f"{anchor}: invalid evidence range: {name}:{first}-{last}")
                continue
            if not evidence["quote"].strip() or evidence["quote"] not in "\n".join(lines[first - 1:last]):
                evidence_errors.append(f"{anchor}: evidence quote does not match fixed contents: {name}:{first}-{last}")
    if evidence_errors:
        raise StateError("\n".join(evidence_errors))
    if role == "reviewer":
        if value["impact"] == "index" and value["revision_node"]:
            raise StateError("index-only review fixes require an empty revision_node")
        actions = {issue_disposition(issue) for issue in value["issues"]}
        if (value["status"] == "needs_input" or value["continuation"] == "needs_input") and "needs_input" not in actions:
            raise StateError("review needs_input requires an issue identifying the missing fundamental input")
        if value["status"] == "revise" and "fix" not in actions:
            raise StateError("review revise requires a concrete fix issue")
        if state["review_scope"] == "contract" and not actions.intersection({"fix", "needs_input"}) and value["continuation"] != "proceed":
            raise StateError("completed Contract review must explicitly confirm continuation=proceed")
        if not value["source_evidence"]:
            raise StateError("review needs an actual source anchor")
        required = {"root.yaml"} if state["review_scope"] == "contract" else {state["node"] + ".yaml"}
        if state["review_scope"] == "final":
            required = {path.name for path in artifacts.glob("*.yaml")} | {"dispatch.md"}
        if not required <= {e["path"] for e in value["artifact_evidence"]}:
            raise StateError("review must include the current target artifacts")
    return value
