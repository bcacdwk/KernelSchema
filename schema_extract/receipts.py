"""Model output is a scoped proposal; validate identity, hashes and exact evidence."""

from __future__ import annotations

from pathlib import Path

from .common import ExtractError, manifest, regular_file
from .artifacts import validate_plan, validate_revision_scope
from .stages import STAGES, plan_signature, scope_key


def obj(properties):
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties)}


TEXT = {"type": "string"}
TEXTS = {"type": "array", "items": TEXT}
INDICES = {"type": "array", "items": {"type": "integer"}}
ARTIFACT = obj({"path": TEXT, "sha256": TEXT})
ANCHOR_FIELDS = {"path": TEXT, "sha256": TEXT, "line_start": {"type": "integer"},
                 "line_end": {"type": "integer"}, "quote": TEXT, "note": TEXT}
SOURCE_ANCHOR = obj({"category": {"type": "string", "enum": ["implementation", "host", "io", "input", "reference", "acceptance", "scope"]},
                     "source_id": TEXT, **ANCHOR_FIELDS})
IDENTITY_FIELDS = ("run_id", "attempt_id", "input_fingerprint", "stage", "role", "node",
                   "plan_revision", "base_checkpoint", "dependency_digests")


class ReceiptFormatError(ExtractError):
    """An otherwise scoped proposal may repair its JSON shape in the same session."""


class QuoteTextMismatch(ExtractError):
    """A bounded quote does not match its otherwise verified fixed file/range."""


def call_identity(attempt: dict) -> dict:
    return {key: attempt[key] for key in (*IDENTITY_FIELDS,
            *(key for key in ("mode", "generation", "node_signature") if key in attempt))}


REOPEN_TARGETS = {"P2": {"P1"}, "P3": {"P1", "P2"}, "P4": {"P1", "P2", "P3"},
                  "P5": {"P1", "P2", "P3", "P4"}, "P6": {"P1", "P2", "P3", "P4"}}


def evidence_identity(anchor: dict) -> dict:
    # Different notes, categories or excerpts of the same fixed lines are not
    # new evidence and cannot buy another reopening of the same issue.
    return {key: anchor[key] for key in ("source_id", "path", "sha256", "line_start", "line_end")}


def required_artifact_evidence(state: dict, attempt: dict, files) -> set[str]:
    """One source of truth for the reviewer's required file-level citations."""
    if attempt["role"] != "reviewer":
        return set()
    if attempt["stage"] == "P5":
        entry = state["nodes"][attempt["node"]]
        return {"root.yaml", entry["file"], state["nodes"][entry["parent"]]["file"]}
    if attempt["stage"] == "P6":
        return set(files)
    return {"root.yaml"} if attempt["stage"] == "P1" else {"root.yaml", "dispatch.md"}


def receipt_schema(attempt: dict) -> dict:
    identity = {}
    for key, value in call_identity(attempt).items():
        if isinstance(value, dict):
            identity[key] = obj({name: {"type": "string", "enum": [expected]} for name, expected in value.items()})
        else:
            identity[key] = {"type": "integer" if type(value) is int else "string", "enum": [value]}
    fields = {
        "format": {"type": "string", "enum": ["schema_extract.receipt.v2"]},
        "identity": obj(identity),
        "decision": {"type": "string", "enum": ["complete", "continue", "approve", "revise", "blocked"]},
        "summary": TEXT, "changed_files": TEXTS,
        "artifacts": {"type": "array", "items": ARTIFACT},
        "source_evidence": {"type": "array", "items": SOURCE_ANCHOR},
        "artifact_evidence": {"type": "array", "items": obj(ANCHOR_FIELDS)},
        "conclusions": {"type": "array", "items": obj({"subject": TEXT, "text": TEXT, "evidence": INDICES})},
        "checks": {"type": "array", "items": obj({"command": TEXT, "exit_code": {"type": "integer"}, "note": TEXT})},
        "checks_note": TEXT,
        "issues": {"type": "array", "items": obj({"kind": TEXT, "scope": TEXT, "detail": TEXT,
                                                   "recovery_condition": TEXT, "evidence": INDICES})},
        "framework_candidates": {"type": "array", "items": obj({"id": TEXT, "scope": TEXT, "summary": TEXT, "evidence": INDICES})},
        "handoff": TEXT,
    }
    if attempt["role"] == "reviewer":
        fixed = attempt["dependency_digests"]
        known = {"path": {"type": "string", "enum": sorted(fixed)},
                 "sha256": {"type": "string", "enum": sorted(set(fixed.values()))}}
        # These values are already fixed before a read-only call. Constrain
        # transcription at generation; the validators still require exact
        # path-to-digest pairing and independently verify every quotation.
        fields["artifacts"]["items"] = obj(known)
        fields["artifact_evidence"]["items"] = obj({**ANCHOR_FIELDS, **known})
    if attempt["stage"] == "P4":
        fields["plan"] = {"type": "array", "items": obj({"id": TEXT, "file": TEXT, "parent": TEXT,
            "workload": TEXT, "difference": TEXT, "selection": TEXT, "evidence": INDICES})}
    if attempt["stage"] == "P6":
        fields["revision_scope"] = {"type": "array", "items": obj({"path": TEXT, "fields": TEXTS})}
        fields["affected_nodes"] = TEXTS
    if attempt.get("mode") == "adjudicate":
        fields["adjudication"] = obj({"action": {"type": "string", "enum": ["select", "patch", "blocked"]},
            "checkpoint": TEXT, "patches": {"type": "array", "items": obj({"path": TEXT, "old": TEXT, "new": TEXT})}})
        fields["decision"]["enum"] = ["approve", "blocked"]
    if attempt.get("mode") == "signoff":
        fields["decision"]["enum"] = ["approve", "blocked"]
    reopen = attempt["stage"] in REOPEN_TARGETS and attempt.get("mode", "normal") == "normal"
    if reopen:
        fields["decision"]["enum"].append("reopen")
        fields["reopen"] = obj({"issue_id": TEXT, "target": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "external"]},
                                "reason": TEXT, "evidence": INDICES})
        fields["reopen"]["type"] = ["object", "null"]
    return obj(fields)


def validate_shape(value, schema: dict, location="receipt") -> None:
    kind = schema["type"]
    if isinstance(kind, list):
        if value is None and "null" in kind:
            return
        kind = next(option for option in kind if option != "null")
    if kind == "object":
        if not isinstance(value, dict) or not set(schema["required"]) <= set(value) <= set(schema["properties"]):
            raise ReceiptFormatError(f"{location}: wrong fields")
        for key in value:
            validate_shape(value[key], schema["properties"][key], f"{location}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ReceiptFormatError(f"{location}: expected list")
        for i, item in enumerate(value):
            validate_shape(item, schema["items"], f"{location}[{i}]")
    elif kind == "integer":
        if type(value) is not int:
            raise ReceiptFormatError(f"{location}: expected integer")
    elif kind == "string" and not isinstance(value, str):
        raise ReceiptFormatError(f"{location}: expected string")
    if "enum" in schema and value not in schema["enum"]:
        raise ExtractError(f"{location}: value does not match call identity or allowed values")


def _anchor(evidence: dict, directory: Path, hashes: dict) -> None:
    name = evidence["path"]
    if name not in hashes:
        raise ExtractError(f"evidence is not in the fixed input/artifact manifest: {name}")
    path = regular_file(directory, name)
    if evidence["sha256"] != hashes[name]:
        raise ExtractError(f"evidence file digest mismatch: {name}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError as exc:
        raise ExtractError(f"evidence is not UTF-8: {name}") from exc
    first, last, quote = evidence["line_start"], evidence["line_end"], evidence["quote"]
    if not 1 <= first <= last <= len(lines) or last - first >= 12:
        raise ExtractError(f"invalid evidence range (maximum 12 lines): {name}:{first}-{last}")
    if not quote.strip() or len(quote) > 320:
        raise ExtractError(f"evidence quote does not match cited lines or exceeds 320 characters: {name}:{first}-{last}")
    if quote not in "\n".join(lines[first - 1:last]):
        raise QuoteTextMismatch(f"quote text does not match {name}:{first}-{last}")


def validate_receipt(value: dict, state: dict, lock: dict, attempt: dict,
                     artifacts: Path, changed_files: list[str]) -> dict:
    if attempt["attempt_id"] in state["accepted_receipts"]:
        raise ExtractError("duplicate receipt: this attempt was already accepted")
    if state["active_attempts"].get(scope_key(attempt["stage"], attempt["node"])) != attempt["attempt_id"]:
        raise ExtractError("receipt is not from the active attempt")
    for key in ("run_id", "input_fingerprint"):
        if key not in attempt:
            continue
        if attempt[key] != state[key]:
            raise ExtractError(f"stale receipt: {key} changed")
    # A missing formatting field must not disguise a supplied stale identity.
    claimed = value.get("identity")
    if isinstance(claimed, dict):
        for key, expected in call_identity(attempt).items():
            if key in claimed and claimed[key] != expected:
                raise ExtractError("receipt invocation identity mismatch: " + key)
    validate_shape(value, receipt_schema(attempt))
    if value["identity"] != call_identity(attempt):
        raise ExtractError("receipt invocation identity mismatch")
    allowed = {"complete", "continue", "blocked"} if attempt["role"] == "worker" else {"approve", "revise", "blocked"}
    if attempt["stage"] in REOPEN_TARGETS and attempt.get("mode", "normal") == "normal":
        allowed.add("reopen")
    if value["decision"] not in allowed:
        raise ExtractError("receipt decision does not match role")
    if not value["summary"].strip():
        raise ReceiptFormatError("receipt requires a summary")
    if value["changed_files"] != changed_files:
        raise ExtractError("receipt changed_files differs from actual permitted diff")
    files = manifest(artifacts)
    declared = {item["path"]: item["sha256"] for item in value["artifacts"]}
    submitted_files = ({state["nodes"][attempt["node"]]["file"]: files[state["nodes"][attempt["node"]]["file"]]}
                       if attempt["stage"] == "P5" else files)
    if len(declared) != len(value["artifacts"]) or any(
            name not in submitted_files or expected != submitted_files[name]
            for name, expected in declared.items()):
        raise ExtractError("receipt artifact digests differ from actual contents")
    missing_artifacts = sorted(submitted_files.keys() - declared.keys())
    if missing_artifacts and set(missing_artifacts) & set(attempt["writable_files"]):
        raise ExtractError("receipt omitted writable artifacts: " + ", ".join(missing_artifacts))
    source = lock["source"]
    bindings = source["bindings"]
    quote_errors = []
    for index, evidence in enumerate(value["source_evidence"]):
        if evidence["source_id"] != source["id"]:
            raise ExtractError("evidence source version mismatch")
        key = "entrypoint" if evidence["category"] == "implementation" else evidence["category"]
        locators = bindings[key] if isinstance(bindings[key], list) else [bindings[key]]
        # A role may follow calls into helpers in the frozen tree. Completion also
        # anchors each required role's bound entry below, without relabeling helpers.
        try:
            _anchor(evidence, Path(lock["snapshot"]) / "source", source["files"])
        except QuoteTextMismatch as exc:
            quote_errors.append(f"source_evidence[{index}].quote: {exc}")
    for index, evidence in enumerate(value["artifact_evidence"]):
        try:
            _anchor(evidence, artifacts, files)
        except QuoteTextMismatch as exc:
            quote_errors.append(f"artifact_evidence[{index}].quote: {exc}")
    for item in [*value["conclusions"], *value["issues"], *value["framework_candidates"]]:
        if any(index < 0 or index >= len(value["source_evidence"]) for index in item["evidence"]):
            raise ExtractError("conclusion/issue evidence index is outside source_evidence")
    final = value["decision"] in {"complete", "approve"}
    if final:
        categories = {entry["category"] for entry in value["source_evidence"]}
        required = set(STAGES[attempt["stage"]]["evidence"])
        if not required <= categories:
            raise ExtractError("missing source evidence categories: " + ", ".join(sorted(required - categories)))
        for category in required:
            key = "entrypoint" if category == "implementation" else category
            locators = bindings[key] if isinstance(bindings[key], list) else [bindings[key]]
            if not any(item["category"] == category and item["path"] in {entry["path"] for entry in locators}
                       for item in value["source_evidence"]):
                raise ExtractError(f"missing bound {category} entry anchor")
        subjects = {item["subject"] for item in value["conclusions"] if item["text"].strip() and item["evidence"]}
        required_subjects = set(STAGES[attempt["stage"]]["conclusions"])
        if not required_subjects <= subjects:
            raise ExtractError("missing supported stage conclusions: " + ", ".join(sorted(required_subjects - subjects)))
    if attempt["role"] == "reviewer":
        required_artifacts = required_artifact_evidence(state, attempt, files)
        if not required_artifacts <= {item["path"] for item in value["artifact_evidence"]}:
            raise ExtractError("reviewer must cite all stage artifacts")
        if not value["source_evidence"]:
            raise ExtractError("reviewer needs actual source evidence")
    if value["decision"] in {"revise", "blocked", "reopen"}:
        if not any(item["detail"].strip() and item["scope"].strip() and item["evidence"] and
                   (value["decision"] == "revise" or item["recovery_condition"].strip()) for item in value["issues"]):
            raise ExtractError("revision/blocking requires a concrete issue, evidence and recovery condition")
    if value["decision"] == "reopen":
        request = value.get("reopen")
        if not request or not request["issue_id"].strip() or not request["reason"].strip():
            raise ReceiptFormatError("reopen requires a stable issue_id, target, reason and source evidence")
        if request["target"] not in REOPEN_TARGETS[attempt["stage"]] | {"external"}:
            raise ExtractError("reopen must use the permitted upstream route for this stage")
        if not request["evidence"] or any(index < 0 or index >= len(value["source_evidence"]) for index in request["evidence"]):
            raise ExtractError("reopen requires valid fixed-source evidence indices")
    elif value.get("reopen") is not None:
        raise ExtractError("only a reopen decision may carry an upstream request")
    if not value["checks"] and not value["checks_note"].strip():
        raise ReceiptFormatError("explain why no check command was run")
    if value["decision"] == "approve" and attempt["stage"] in {"P2", "P3", "P4"}:
        handoff = value["handoff"]
        if not handoff.strip() or len(handoff) > 1200 or len([line for line in handoff.splitlines() if line.strip()]) > 20:
            raise ReceiptFormatError("approval needs a short handoff (<=1200 characters, <=20 nonempty lines)")
    if attempt["stage"] == "P4" and final:
        if attempt.get("mode", "normal") == "normal":
            validate_plan(value["plan"], artifacts, Path(lock["snapshot"]) / "rules")
        for entry in value["plan"]:
            if not entry["evidence"] or any(type(i) is not int or i < 0 or i >= len(value["source_evidence"])
                                             for i in entry["evidence"]):
                raise ExtractError("each planned node needs valid source evidence indices")
            if not any(value["source_evidence"][i]["category"] == "host" for i in entry["evidence"]):
                raise ExtractError("each planned node needs host selection evidence")
        if attempt["role"] == "reviewer" and attempt.get("mode", "normal") == "normal" and plan_signature(value["plan"]) != plan_signature(state["plan_draft"]):
            raise ExtractError("P4 reviewer plan differs from the worker's submitted plan")
    if attempt["stage"] == "P6":
        planned = {entry["id"] for entry in state["plan"]}
        affected = set(value["affected_nodes"])
        if len(affected) != len(value["affected_nodes"]) or not affected <= planned:
            raise ExtractError("P6 affected_nodes must name unique planned nodes")
        validate_revision_scope(value["revision_scope"], state["plan"])
        if value["decision"] == "revise" and not value["revision_scope"]:
            raise ExtractError("P6 revision requires explicit files/fields")
        if value["decision"] == "revise" and not {
                Path(item["path"]).stem for item in value["revision_scope"] if item["path"].endswith(".yaml")
                } <= affected:
            raise ExtractError("P6 revision must identify every node within its requested scope")
        if attempt["role"] == "worker" and value["revision_scope"] != state["revision_scope"]:
            raise ExtractError("P6 worker must retain the reviewer's exact revision scope")
        changed_nodes = {Path(name).stem for name in changed_files if name.endswith(".yaml")}
        if not changed_nodes <= affected:
            raise ExtractError("P6 modification must identify every affected node")
        if value["decision"] == "approve" and not set(state["pending_rechecks"]) <= affected:
            raise ExtractError("P6 approval/signoff must recheck all affected nodes")
    if attempt.get("mode") == "adjudicate":
        adjudication = value["adjudication"]
        if (adjudication["action"] == "blocked") != (value["decision"] == "blocked"):
            raise ExtractError("adjudication action and decision disagree")
        if adjudication["action"] != "patch" and adjudication["patches"]:
            raise ExtractError("only patch adjudication may contain patches")
        if adjudication["action"] == "patch" and not adjudication["patches"]:
            raise ExtractError("patch adjudication requires an explicit patch")
    # A rejected proposal may be resubmitted only after all other hard gates
    # passed. Never supply a corrected quote or accept the failed proposal.
    repairs = []
    if missing_artifacts:
        repairs.append("receipt artifacts omitted read-only files: " + ", ".join(missing_artifacts))
    if quote_errors:
        repairs.extend(quote_errors)
        repairs.append("重新读取上述固定文件的行范围，重新核对各引用支持的结论；不要沿用不准确的引文或假定仍可批准。")
    if repairs:
        raise ReceiptFormatError("; ".join(repairs) +
                                 "; keep the saved artifacts unchanged and resubmit the complete receipt for full validation.")
    return value
