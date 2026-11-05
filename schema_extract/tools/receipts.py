"""Model output is a scoped proposal; validate identity, hashes and exact evidence."""

from __future__ import annotations

import re
from pathlib import Path

from .common import ExtractError, manifest, regular_file, rules_root, source_root
from .artifacts import (design_context, region_evidence_lines, resolve_workspace,
                        validate_plan, validate_revision_scope)
from .comparators import validate_comparisons
from .reading import read_evidence
from .stages import DOCUMENT_CONCLUSIONS, STAGES, plan_signature, required_conclusions, scope_key


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
    """A scoped proposal may repair JSON shape or an identified artifact layout error."""


def call_identity(attempt: dict) -> dict:
    return {key: attempt[key] for key in (*IDENTITY_FIELDS,
            *(key for key in ("mode", "generation", "node_signature", "semantic_revision",
                              "maintenance_task_id", "purpose", "maintenance_step") if key in attempt))}


REOPEN_TARGETS = {"P2": {"P1"}, "P3": {"P1", "P2"}, "P4": {"P1", "P2", "P3"},
                  "P5": {"P1", "P2", "P3", "P4"}, "P6": {"P1", "P2", "P3", "P4"}}


def evidence_identity(anchor: dict) -> dict:
    # Different notes, categories or excerpts of the same fixed lines are not
    # new evidence and cannot buy another reopening of the same issue.
    return {key: anchor[key] for key in ("source_id", "path", "sha256", "line_start", "line_end")}


def required_artifact_evidence(state: dict, attempt: dict, files, *, resolution: dict | None = None) -> set[str]:
    """One source of truth for the reviewer's required file-level citations."""
    if attempt["role"] != "reviewer":
        return set()
    if attempt["stage"] == "P5":
        entry = state["nodes"][attempt["node"]]
        result = {"root.yaml", entry["file"], state["nodes"][entry["parent"]]["file"]}
        graph = resolution if resolution is not None else attempt.get("design_resolution", {})
        if graph.get("diagnostic"):
            # A malformed migration/draft must still reach its ordinary author.
            return result | set(files)
        for node in (attempt["node"], entry["parent"]):
            result.update(graph.get("nodes", {}).get(node, {}).get("source_manifest", {}))
        return result
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
        "summary": {**TEXT, "description": "Write a brief non-empty semantic conclusion for this invocation."},
        "changed_files": TEXTS,
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
        # A reviewer receipt can only refer to its fixed read-only input bytes.
        fields["artifacts"]["items"] = obj(known)
        fields["artifact_evidence"]["items"] = obj({**ANCHOR_FIELDS, **known})
    if attempt["stage"] == "P1":
        fields["comparisons"] = {"type": "array", "items": obj({"output": TEXT, "source_output": TEXT, "evidence": INDICES})}
    if attempt["stage"] == "P4":
        fields["plan"] = {"type": "array", "items": obj({"id": TEXT, "file": TEXT, "parent": TEXT,
            "workload": TEXT, "difference": TEXT, "selection": TEXT, "evidence": INDICES})}
    if attempt["stage"] == "P6" or attempt.get("purpose") == "maintenance":
        fields["revision_scope"] = {"type": "array", "items": obj({"path": TEXT, "fields": TEXTS})}
        if attempt["stage"] == "P6":
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
        targets = set(REOPEN_TARGETS[attempt["stage"]]) | {"external"}
        if attempt["stage"] == "P5" and attempt.get("purpose") != "maintenance":
            targets.update(attempt.get("provider_reopen_targets", {}))
        fields["reopen"] = obj({"issue_id": TEXT, "target": {"type": "string", "enum": sorted(targets)},
                                "reason": TEXT, "evidence": INDICES})
        fields["reopen"]["type"] = ["object", "null"]
    return obj(fields)


def proposal_schema(attempt: dict, *, source_files: dict[str, str] | None = None) -> dict:
    """Only semantic choices and this invocation's nonce are model-authored."""
    fields = {key: value for key, value in receipt_schema(attempt)["properties"].items()
              if key not in {"format", "identity", "artifacts", "changed_files"}}
    fields = {"format": {"type": "string", "enum": ["schema_extract.proposal.v1"]},
              "attempt_id": {"type": "string", "enum": [attempt["attempt_id"]]}, **fields}
    anchor = {"path": TEXT, "line_start": {"type": "integer"}, "line_end": {"type": "integer"},
              "quote_offset": {"type": "integer", "minimum": 0}, "note": TEXT}
    source = {"category": SOURCE_ANCHOR["properties"]["category"], **anchor}
    if source_files is not None:
        source["path"] = {"type": "string", "enum": sorted(source_files)}
    fields["source_evidence"] = {"type": "array", "items": obj(source)}
    files = sorted(set(attempt["dependency_digests"]) | set(attempt["writable_files"]))
    fields["artifact_evidence"] = {"type": "array", "items": obj({**anchor, "path": {"type": "string", "enum": files}})}
    if attempt.get("mode", "normal") == "normal":
        fields["decision"] = {"type": "string", "enum":
            (["complete", "continue", "blocked"] if attempt["role"] == "worker" else ["approve", "revise", "blocked"]) +
            (["reopen"] if attempt["stage"] in REOPEN_TARGETS else [])}
    return obj(fields)


def check_proposal_identity(value, attempt: dict) -> None:
    """A malformed or late response must not acquire a new invocation identity."""
    if not isinstance(value, dict) or value.get("attempt_id") != attempt["attempt_id"]:
        raise ExtractError("proposal requires the exact current attempt_id nonce")
    if value.get("format") != "schema_extract.proposal.v1":
        raise ExtractError("proposal format does not match this invocation")
    schema = proposal_schema(attempt)
    extra = set(value) - set(schema["properties"])
    if extra:
        raise ExtractError("proposal supplies unexpected or machine-owned fields: " + ", ".join(sorted(extra)))
    for key in ("source_evidence", "artifact_evidence"):
        entries = value.get(key)
        allowed = set(schema["properties"][key]["items"]["properties"])
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and set(entry) - allowed:
                raise ExtractError(f"{key} supplies unexpected or machine-owned fields")


def normalize_response(value: dict, lock: dict, attempt: dict, artifacts: Path) -> dict:
    """Derive a complete receipt from a nonce-bound proposal and the actual bytes."""
    check_proposal_identity(value, attempt)
    validate_shape(value, proposal_schema(attempt, source_files=lock["source"]["files"]), "proposal")
    files = manifest(artifacts)
    changed = sorted(name for name in files.keys() | attempt["dependency_digests"].keys()
                     if files.get(name) != attempt["dependency_digests"].get(name))
    if attempt["role"] == "reviewer" and changed:
        raise ExtractError("reviewer proposal differs from its read-only checkpoint bytes")
    submitted = {attempt["node"] + ".yaml": files[attempt["node"] + ".yaml"]} if attempt["stage"] == "P5" else files
    def anchors(key, root, hashes):
        result = []
        for selection in value[key]:
            path = selection["path"]
            if path not in hashes:
                raise ExtractError("evidence is outside the fixed manifest: " + path)
            try:
                anchor = read_evidence(root, path, selection["line_start"], selection["line_end"],
                                       quote_offset=selection["quote_offset"], expected_sha256=hashes[path])
            except (OSError, ValueError) as exc:
                raise ExtractError(f"{key}: {exc}") from exc
            if key == "source_evidence":
                anchor.update(source_id=lock["source"]["id"], category=selection["category"])
            result.append({**anchor, "note": selection["note"]})
        return result
    return {**{key: item for key, item in value.items() if key != "attempt_id"},
            "format": "schema_extract.receipt.v2", "identity": call_identity(attempt),
            "artifacts": [{"path": path, "sha256": checksum} for path, checksum in sorted(submitted.items())],
            "changed_files": changed,
            "source_evidence": anchors("source_evidence", source_root(lock), lock["source"]["files"]),
            "artifact_evidence": anchors("artifact_evidence", artifacts, files)}


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
        raise ExtractError(f"quote text does not match {name}:{first}-{last}")


def _artifact_issue_supported(issue: dict, anchors: list[dict]) -> bool:
    """Document-only issues must name a file whose current anchor was validated."""
    if issue["kind"] not in {"submission_format", "schema_conformance"}:
        return False
    return any(re.search(r"(?<![A-Za-z0-9_./\\-])" + re.escape(anchor["path"]) +
                         r"(?![A-Za-z0-9_./\\-])", issue["scope"]) for anchor in anchors)


def _region_anchored(resolved, node: str, region: str, anchors: list[dict]) -> bool:
    filename = resolved.source_files[node].name
    lines = region_evidence_lines(resolved, node, region)
    return any(anchor["path"] == filename and anchor["sha256"] == resolved.source_hashes[node] and
               any(anchor["line_start"] <= line <= anchor["line_end"] for line in lines)
               for anchor in anchors)


def provider_request_details(value: dict, state: dict, attempt: dict, artifacts: Path, *, parser_root: Path) -> dict:
    """Recompute eligibility from submitted raw bytes; invocation enums grant no authority."""
    request = value["reopen"]
    if (attempt["stage"] != "P5" or attempt.get("purpose") == "maintenance" or
            attempt.get("mode", "normal") != "normal"):
        raise ExtractError("provider reopen is only available to ordinary P5 worker/reviewer calls")
    context = design_context(state, attempt, artifacts, parser_root=parser_root)
    target = context["provider_reopen_targets"].get(request["target"])
    if not target:
        raise ExtractError("provider reopen requires a current approved strict ancestor that locally defines an inherited region")
    resolved = resolve_workspace(artifacts, parser_root=parser_root)
    provider = target["provider"]
    filename = resolved.source_files[provider].name
    regions = {}
    for region, chain in target["regions"].items():
        field = "design." + region
        issues = [issue for issue in value["issues"] if filename in issue["scope"] and
                  re.search(r"(?<![A-Za-z0-9_])" + re.escape(field) + r"(?![A-Za-z0-9_])", issue["scope"]) and
                  set(issue["evidence"]) & set(request["evidence"])]
        if issues and _region_anchored(resolved, provider, region, value["artifact_evidence"]):
            regions[region] = chain
    if not regions:
        raise ExtractError("provider reopen needs a current raw definition-region anchor and source-supported issue scoped to that region; child applicability is repaired locally")
    return {**target, "regions": regions,
            "consumer_identity": resolved.effective_identity(attempt["node"])}


def _validate_derivation_anchors(resolved, nodes: list[str], anchors: list[dict]) -> None:
    for consumer in nodes:
        for region, origin in resolved.origins[consumer].items():
            if len(origin.chain) < 2:
                continue
            for node in origin.chain:
                if not _region_anchored(resolved, node, region, anchors):
                    raise ExtractError(f"reviewer must cite actual raw reference/provider lines: {resolved.source_files[node].name}:design.{region}")


def validate_receipt(value: dict, state: dict, lock: dict, attempt: dict,
                     artifacts: Path, changed_files: list[str], *, rules_lock: dict | None = None) -> dict:
    if attempt.get("semantic_revision") and rules_lock is None:
        raise ExtractError("semantic receipt validation requires its explicit rules lock")
    semantic = rules_lock or lock
    if attempt.get("semantic_revision") != state.get("semantic_revision"):
        raise ExtractError("receipt belongs to an obsolete semantic context")
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
    if attempt["stage"] in REOPEN_TARGETS and attempt.get("mode", "normal") == "normal" and attempt.get("purpose") != "maintenance":
        allowed.add("reopen")
    if value["decision"] not in allowed:
        raise ExtractError("receipt decision does not match role")
    if not value["summary"].strip():
        raise ReceiptFormatError("receipt requires a summary")
    if value["changed_files"] != changed_files:
        raise ExtractError("receipt changed_files differs from actual permitted diff")
    files = manifest(artifacts)
    resolution = None
    if attempt["stage"] == "P5":
        resolution = design_context(state, attempt, artifacts, parser_root=rules_root(semantic))["design_resolution"]
    declared = {item["path"]: item["sha256"] for item in value["artifacts"]}
    submitted_files = ({state["nodes"][attempt["node"]]["file"]: files[state["nodes"][attempt["node"]]["file"]]}
                       if attempt["stage"] == "P5" else files)
    if len(declared) != len(value["artifacts"]) or declared != submitted_files:
        raise ExtractError("receipt artifact digests differ from actual contents")
    source = lock["source"]
    bindings = source["bindings"]
    for evidence in value["source_evidence"]:
        if evidence["source_id"] != source["id"]:
            raise ExtractError("evidence source version mismatch")
        # A role may follow calls into helpers in the frozen tree. Completion also
        # anchors each required role's bound entry below, without relabeling helpers.
        _anchor(evidence, source_root(lock), source["files"])
    for evidence in value["artifact_evidence"]:
        _anchor(evidence, artifacts, files)
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
        artifact_subjects = set()
        if attempt["role"] == "reviewer":
            required_files = required_artifact_evidence(state, attempt, files, resolution=resolution)
            if required_files <= {entry["path"] for entry in value["artifact_evidence"]}:
                # These conclusions concern the current documents, not source
                # implementation facts. Their anchors are artifact_evidence.
                artifact_subjects.update(DOCUMENT_CONCLUSIONS)
        if attempt["stage"] == "P6" and any(item["subject"] == "tree" and not item["evidence"]
                                             for item in value["conclusions"]):
            planned_files = {entry["file"] for entry in state["plan"]} | {"dispatch.md"}
            cited_files = {entry["path"] for entry in value["artifact_evidence"]}
            if planned_files == set(files) and planned_files <= cited_files:
                # Topology is a fact about the cited artifacts. Source-dependent
                # selection and coverage conclusions still need source indices.
                validate_plan(state["plan"], artifacts, rules_root(semantic), require_files=True)
                artifact_subjects.add("tree")
        subjects = {item["subject"] for item in value["conclusions"] if item["text"].strip()
                    and (item["evidence"] or item["subject"] in artifact_subjects)}
        required_subjects = set(required_conclusions(attempt))
        if not required_subjects <= subjects:
            raise ExtractError("missing supported stage conclusions: " + ", ".join(sorted(required_subjects - subjects)))
        if attempt["stage"] == "P1" and attempt.get("mode", "normal") != "adjudicate":
            validate_comparisons(value, lock, artifacts, parser_root=rules_root(semantic))
    if attempt["role"] == "reviewer":
        required_artifacts = required_artifact_evidence(state, attempt, files, resolution=resolution)
        if not required_artifacts <= {item["path"] for item in value["artifact_evidence"]}:
            raise ExtractError("reviewer must cite all stage artifacts")
        if not value["source_evidence"]:
            raise ExtractError("reviewer needs actual source evidence")
        if final and attempt["stage"] == "P5" and not resolution.get("diagnostic"):
            resolved = resolve_workspace(artifacts, parser_root=rules_root(semantic))
            _validate_derivation_anchors(resolved, [attempt["node"], state["nodes"][attempt["node"]]["parent"]],
                                         value["artifact_evidence"])
    if value["decision"] in {"revise", "blocked", "reopen"}:
        if not any(item["detail"].strip() and item["scope"].strip() and
                   (item["evidence"] or
                    (value["decision"] in {"revise", "blocked"} and
                     _artifact_issue_supported(item, value["artifact_evidence"]))) and
                   (value["decision"] == "revise" or item["recovery_condition"].strip()) for item in value["issues"]):
            raise ExtractError("revision/blocking requires a concrete issue, source evidence or "
                               "(for revise/blocked) a submission_format/schema_conformance issue scoped to "
                               "cited artifact evidence, and recovery condition")
    if value["decision"] == "reopen":
        request = value.get("reopen")
        if not request or not request["issue_id"].strip() or not request["reason"].strip():
            raise ReceiptFormatError("reopen requires a stable issue_id, target, reason and source evidence")
        provider_target = request["target"].startswith("P5:")
        if not provider_target and request["target"] not in REOPEN_TARGETS[attempt["stage"]] | {"external"}:
            raise ExtractError("reopen must use the permitted upstream route for this stage")
        if not request["evidence"] or any(index < 0 or index >= len(value["source_evidence"]) for index in request["evidence"]):
            raise ExtractError("reopen requires valid fixed-source evidence indices")
        if provider_target:
            provider_request_details(value, state, attempt, artifacts, parser_root=rules_root(semantic))
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
            validate_plan(value["plan"], artifacts, rules_root(semantic))
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
    return value
