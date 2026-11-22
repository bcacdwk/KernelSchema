"""Model output is a scoped proposal; validate identity, hashes and exact evidence."""

from __future__ import annotations

import re
from pathlib import Path

from .common import ExtractError, regular_file, rules_root, source_root
from .artifacts import (artifact_manifest, design_context, region_evidence_lines, resolve_workspace,
                        validate_index, validate_plan, validate_revision_scope)
from .comparators import validate_comparisons
from .reading import EvidenceSelectionError, read_evidence
from .stages import DOCUMENT_CONCLUSIONS, STAGES, node_signature, required_conclusions, scope_key


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
    """Proposal shape, required judgments or evidence selection needs model resubmission."""


def call_identity(attempt: dict) -> dict:
    return {key: attempt[key] for key in (*IDENTITY_FIELDS,
            *(key for key in ("mode", "generation", "node_signature", "selection_signature", "writing_signature",
                              "backend", "model", "effort", "session_scope", "source_identities",
                              "maintenance_task_id", "purpose", "maintenance_step", "execution_binding") if key in attempt))}


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
        result = {"root.yaml", attempt["node"] + ".yaml"}
        graph = resolution if resolution is not None else attempt.get("design_resolution", {})
        if graph.get("diagnostic"):
            return result | set(files)
        result.update(graph.get("nodes", {}).get(attempt["node"], {}).get("source_manifest", {}))
        return result
    if attempt["stage"] == "P6":
        return set(files)
    if attempt["stage"] == "P1":
        return {name for name in files if name.endswith(".yaml")}
    return {"root.yaml", "dispatch.md"}


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
        "summary": TEXT,
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
        manifests = [attempt["dependency_digests"], *(item["files"] for item in attempt.get("adjudication_candidates", []))]
        known = {"path": {"type": "string", "enum": sorted({name for fixed in manifests for name in fixed})},
                 "sha256": {"type": "string", "enum": sorted({checksum for fixed in manifests for checksum in fixed.values()})}}
        # A reviewer receipt can only refer to its fixed read-only input bytes.
        fields["artifacts"]["items"] = obj(known)
        fields["artifact_evidence"]["items"] = obj({**ANCHOR_FIELDS, **known})
    if attempt["stage"] == "P1":
        fields["comparisons"] = {"type": "array", "items": obj({"output": TEXT, "source_output": TEXT, "evidence": INDICES})}
    if attempt["stage"] == "P2":
        fields["plan"] = {"type": "array", "items": obj({"id": TEXT, "instance": TEXT})}
    if attempt["stage"] == "P4":
        fields["plan"] = {"type": "array", "items": obj({"id": TEXT, "wait_for": TEXTS})}
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
    reopen = (attempt["stage"] in REOPEN_TARGETS and attempt.get("mode", "normal") == "normal"
              and attempt.get("purpose") != "maintenance")
    if reopen:
        fields["decision"]["enum"].append("reopen")
        targets = set(REOPEN_TARGETS[attempt["stage"]]) | {"external"}
        if attempt["stage"] in {"P5", "P6"}:
            targets.update(attempt.get("provider_reopen_targets", {}))
            targets.update("P5:" + source for source in attempt.get("source_identities", {})
                           if source not in {"root", attempt["node"]})
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
    files = sorted(set(attempt["dependency_digests"]) | set(attempt["writable_files"]) |
                   {name for item in attempt.get("adjudication_candidates", []) for name in item["files"]})
    fields["artifact_evidence"] = {"type": "array", "items": obj({**anchor, "path": {"type": "string", "enum": files}})}
    if attempt.get("mode", "normal") == "normal":
        fields["decision"] = {"type": "string", "enum":
            (["complete", "continue", "blocked"] if attempt["role"] == "worker" else ["approve", "revise", "blocked"]) +
            (["reopen"] if "reopen" in fields else [])}
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


def _artifact_baseline(value: dict, attempt: dict) -> dict:
    fixed = attempt["dependency_digests"]
    if attempt.get("mode") == "adjudicate" and (value.get("adjudication") or {}).get("action") == "select":
        candidates = {entry["checkpoint"]: entry for entry in attempt["adjudication_candidates"]}
        chosen = candidates.get(value["adjudication"]["checkpoint"])
        if not chosen:
            raise ExtractError("selection must name an actually provided canonical candidate")
        fixed = chosen["files"]
    return fixed


def normalize_response(value: dict, lock: dict, attempt: dict, artifacts: Path) -> dict:
    """Derive a complete receipt from a nonce-bound proposal and the actual bytes."""
    check_proposal_identity(value, attempt)
    validate_shape(value, proposal_schema(attempt, source_files=lock["source"]["files"]), "proposal")
    fixed = _artifact_baseline(value, attempt)
    files = artifact_manifest(artifacts, expected=fixed,
                              additions={"dispatch.md"} if (attempt["stage"], attempt["role"]) == ("P2", "worker") else ())
    changed = sorted(name for name in files.keys() | fixed.keys() if files.get(name) != fixed.get(name))
    if attempt["role"] == "reviewer" and changed:
        raise ExtractError("reviewer proposal differs from its read-only checkpoint bytes")
    submitted = {attempt["node"] + ".yaml": files[attempt["node"] + ".yaml"]} if attempt["stage"] == "P5" else files
    def anchors(key, root, hashes):
        result = []
        for index, selection in enumerate(value[key]):
            path = selection["path"]
            location = f"{key}[{index}] {path}:{selection['line_start']}-{selection['line_end']}"
            if path not in hashes:
                raise ExtractError(f"{location}: evidence is outside the fixed manifest")
            try:
                anchor = read_evidence(root, path, selection["line_start"], selection["line_end"],
                                       quote_offset=selection["quote_offset"], expected_sha256=hashes[path])
            except EvidenceSelectionError as exc:
                raise ReceiptFormatError(f"{location}: {exc}") from exc
            except (OSError, ValueError) as exc:
                raise ExtractError(f"{location}: {exc}") from exc
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
        if "minItems" in schema and len(value) < schema["minItems"] or "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ReceiptFormatError(f"{location}: wrong number of items")
    elif kind == "integer":
        if type(value) is not int:
            raise ReceiptFormatError(f"{location}: expected integer")
    elif kind == "string" and not isinstance(value, str):
        raise ReceiptFormatError(f"{location}: expected string")
    if "enum" in schema and value not in schema["enum"]:
        raise ExtractError(f"{location}: value does not match call identity or allowed values")
    if "minimum" in schema and value < schema["minimum"]:
        raise ReceiptFormatError(f"{location}: below minimum")


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
    if not 1 <= first <= last <= len(lines):
        raise ExtractError(f"invalid evidence range (file has {len(lines)} lines): {name}:{first}-{last}")
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
    if (attempt["stage"] not in {"P5", "P6"} or attempt.get("purpose") == "maintenance" or
            attempt.get("mode", "normal") != "normal"):
        raise ExtractError("provider reopen is only available to ordinary P5/P6 calls")
    context = design_context(state, attempt, artifacts, parser_root=parser_root)
    target = context["provider_reopen_targets"].get(request["target"])
    if not target:
        raise ExtractError("provider reopen requires a current approved actual source that locally defines a used region")
    resolved = resolve_workspace(artifacts, parser_root=parser_root, nodes=None if attempt["stage"] == "P6" else [attempt["node"]])
    provider = target["provider"]
    filename = resolved.source_files[provider].name
    regions = {}
    for region, chain in target["regions"].items():
        field = "DESIGN." + region
        issues = [issue for issue in value["issues"] if filename in issue["scope"] and
                  re.search(r"(?<![A-Za-z0-9_])" + re.escape(field) + r"(?![A-Za-z0-9_])", issue["scope"]) and
                  set(issue["evidence"]) & set(request["evidence"])]
        if issues and _region_anchored(resolved, provider, region, value["artifact_evidence"]):
            regions[region] = chain
    if not regions:
        raise ExtractError("provider reopen needs a current raw definition-region anchor and source-supported issue scoped to that region; child applicability is repaired locally")
    return {**target, "regions": regions,
            "consumer_identity": {node: resolved.effective_identity(node) for node in target["consumers"]},
            "consumer_context": context["design_resolution"]["nodes"].get(attempt["node"]) if attempt["stage"] == "P5" else None}


def _validate_derivation_anchors(resolved, nodes: list[str], anchors: list[dict]) -> None:
    for consumer in nodes:
        for region, origin in resolved.origins[consumer].items():
            if len(origin.chain) < 2:
                continue
            for node in origin.chain:
                if not _region_anchored(resolved, node, region, anchors):
                    raise ExtractError(f"reviewer must cite actual raw reference/provider lines: {resolved.source_files[node].name}:DESIGN.{region}")


def validate_receipt(value: dict, state: dict, lock: dict, attempt: dict,
                     artifacts: Path, changed_files: list[str], *, rules_lock: dict | None = None,
                     semantic_artifacts: Path | None = None) -> dict:
    """Validate original receipt evidence; optional worker semantics use landed raw."""
    semantic = rules_lock or lock
    semantic_artifacts = semantic_artifacts or artifacts
    parser_root = rules_root(semantic)
    fixed = _artifact_baseline(value, attempt)
    files = artifact_manifest(artifacts, expected=fixed,
                              additions={"dispatch.md"} if (attempt["stage"], attempt["role"]) == ("P2", "worker") else ())
    if attempt["role"] == "reviewer" and artifact_manifest(semantic_artifacts, expected=fixed) != files:
        raise ExtractError("reviewer evidence and semantic check must refer to the same final raw bytes")
    if attempt["stage"] in {"P4", "P6"}:
        selection_signature = validate_plan("P2", state["selected_designs"], parser_root=parser_root)
        if attempt.get("selection_signature") != selection_signature:
            raise ExtractError("receipt belongs to a different selected design set or instance")
    if attempt["stage"] in {"P3", "P5"}:
        entry = next((entry for entry in state["selected_designs"] if entry["id"] == attempt["node"]), None)
        if not entry or attempt.get("node_signature") != node_signature(entry):
            raise ExtractError("receipt belongs to another selected concrete instance")
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
    resolution = None
    if attempt["stage"] == "P5":
        resolution = design_context(state, attempt, semantic_artifacts, parser_root=parser_root)["design_resolution"]
    declared = {item["path"]: item["sha256"] for item in value["artifacts"]}
    submitted_files = ({attempt["node"] + ".yaml": files[attempt["node"] + ".yaml"]}
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
            raise ReceiptFormatError("conclusion/issue evidence index is outside source_evidence")
    final = value["decision"] in {"complete", "approve"}
    if final:
        if resolution and resolution.get("diagnostic"):
            raise ExtractError("completion requires valid actual Design sources: " + resolution["diagnostic"])
        categories = {entry["category"] for entry in value["source_evidence"]}
        required = set(STAGES[attempt["stage"]]["evidence"])
        if not required <= categories:
            raise ReceiptFormatError("missing source evidence categories: " + ", ".join(sorted(required - categories)))
        for category in required:
            key = "entrypoint" if category == "implementation" else category
            locators = bindings[key] if isinstance(bindings[key], list) else [bindings[key]]
            if not any(item["category"] == category and item["path"] in {entry["path"] for entry in locators}
                       for item in value["source_evidence"]):
                raise ReceiptFormatError(f"missing bound {category} entry anchor")
        artifact_subjects = set()
        if attempt["role"] == "reviewer":
            required_files = required_artifact_evidence(state, attempt, files, resolution=resolution)
            if required_files <= {entry["path"] for entry in value["artifact_evidence"]}:
                # These conclusions concern the current documents, not source
                # implementation facts. Their anchors are artifact_evidence.
                artifact_subjects.update(DOCUMENT_CONCLUSIONS)
        if attempt["stage"] == "P6" and any(item["subject"] == "directory" and not item["evidence"]
                                             for item in value["conclusions"]):
            planned_files = {entry["id"] + ".yaml" for entry in state["selected_designs"]} | {"dispatch.md"}
            cited_files = {entry["path"] for entry in value["artifact_evidence"]}
            if planned_files == set(files) and planned_files <= cited_files:
                validate_index(semantic_artifacts, state["selected_designs"], parser_root=parser_root, final=True)
                artifact_subjects.add("directory")
        subjects = {item["subject"] for item in value["conclusions"] if item["text"].strip()
                    and (item["evidence"] or item["subject"] in artifact_subjects)}
        required_subjects = set(required_conclusions(attempt))
        if not required_subjects <= subjects:
            raise ReceiptFormatError("missing supported stage conclusions: " + ", ".join(sorted(required_subjects - subjects)))
        if attempt["stage"] == "P1" and attempt.get("mode", "normal") != "adjudicate":
            validate_comparisons(value, lock, semantic_artifacts, parser_root=parser_root)
    if attempt["role"] == "reviewer":
        required_artifacts = required_artifact_evidence(state, attempt, files, resolution=resolution)
        if not required_artifacts <= {item["path"] for item in value["artifact_evidence"]}:
            raise ReceiptFormatError("reviewer must cite all stage artifacts")
        if not value["source_evidence"]:
            raise ReceiptFormatError("reviewer needs actual source evidence")
        if final and attempt["stage"] == "P5" and not resolution.get("diagnostic"):
            resolved = resolve_workspace(semantic_artifacts, parser_root=parser_root, nodes=[attempt["node"]])
            _validate_derivation_anchors(resolved, [attempt["node"]], value["artifact_evidence"])
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
            provider_request_details(value, state, attempt, semantic_artifacts, parser_root=parser_root)
    elif value.get("reopen") is not None:
        raise ExtractError("only a reopen decision may carry an upstream request")
    if not value["checks"] and not value["checks_note"].strip():
        raise ReceiptFormatError("explain why no check command was run")
    if value["decision"] == "approve" and attempt["stage"] in {"P2", "P3", "P4"}:
        handoff = value["handoff"]
        if not handoff.strip() or len(handoff) > 1200 or len([line for line in handoff.splitlines() if line.strip()]) > 20:
            raise ReceiptFormatError("approval needs a short handoff (<=1200 characters, <=20 nonempty lines)")
    if attempt["stage"] in {"P2", "P4"} and final:
        selected = state.get("selected_designs") if attempt["stage"] == "P4" else None
        signature = validate_plan(attempt["stage"], value["plan"], parser_root=parser_root, selected_designs=selected)
        if not (attempt.get("mode") == "adjudicate" and value["adjudication"]["action"] == "patch"):
            validate_index(semantic_artifacts, selected if selected is not None else value["plan"], parser_root=parser_root)
        if attempt["role"] == "reviewer" and attempt.get("mode", "normal") != "adjudicate":
            draft_signature = validate_plan(attempt["stage"], state["plan_draft"], parser_root=parser_root, selected_designs=selected)
            if signature != draft_signature:
                raise ExtractError(attempt["stage"] + " reviewer plan differs from the worker's submitted plan")
    if attempt["stage"] == "P6":
        planned = {entry["id"] for entry in state["selected_designs"]}
        affected = set(value["affected_nodes"])
        if len(affected) != len(value["affected_nodes"]) or not affected <= planned:
            raise ExtractError("P6 affected_nodes must name unique planned nodes")
        validate_revision_scope(value["revision_scope"], state["selected_designs"])
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
