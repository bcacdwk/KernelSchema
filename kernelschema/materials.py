"""Explicit material role selection and fixed-request semantic checks.

Returned bytes are an allowlist, not a sandbox. Runtime owners must enforce
that allowlist across first calls, retries, resume and reopen.
"""
from __future__ import annotations

from .artifacts import Bundle, dependencies
from .contracts import PROTOCOL, ProtocolError, UnsupportedError, capabilities, validate
from .template_api import resolve_bundle


def for_role(bundle: Bundle, materials: dict, role: str) -> dict[str, bytes]:
    validate(materials)
    if materials["kind"] != "materials" or role not in ("planner", "coder", "reviewer"):
        raise ProtocolError("expected Materials and a supported role")
    return {entry["id"]: bundle.read(entry["file"]) for entry in materials["entries"] if role in entry["roles"]}


def validate_request(bundle: Bundle, request: dict) -> dict[str, dict]:
    """No model/GPU admission. Verify fixed objects, closure and cheap bindings."""
    validate(request)
    if "run_id" in request: raise ProtocolError("expected request, not result")
    bundle.closure(dependencies(request))
    objects = {key: bundle.document(request[key], key) for key in ("task", "rules", "materials", "scope")}
    task, scope = objects["task"], objects["scope"]
    profile = bundle.document(request["backend"]["profile"], "backend_profile")
    objects["profile"] = profile
    if profile["name"] != request["backend"]["name"]:
        raise ProtocolError("backend name differs from fixed profile")
    evaluator = bundle.document(task["evaluator"], "evaluator")
    objects["evaluator"] = evaluator
    private_hashes = {r["sha256"] for r in evaluator["files"]}
    for entry in objects["materials"]["entries"]:
        if entry["roles"] and entry["file"]["sha256"] in private_hashes:
            raise ProtocolError("private evaluator files cannot enter model role materials")
    bundle.source_files(task["baseline_source"])
    if task["starter_source"] is not None: bundle.source_files(task["starter_source"])
    if request["budget"]["parent"]:
        objects["budget_account"] = bundle.document(request["budget"]["parent"], "budget_account")
    if request["operation"] in ("design", "optimize"):
        objectives = [e for e in objects["materials"]["entries"] if e["id"] == "workflow_objective"]
        if len(objectives) != 1 or not {"planner", "reviewer"} <= set(objectives[0]["roles"]):
            raise ProtocolError("writing requires exactly one workflow_objective visible to planner/reviewer")
    if "schema" in request:
        schema = bundle.document(request["schema"], "schema")
        objects["schema"] = schema
        if schema["task_sha256"] != request["task"]["sha256"]:
            raise ProtocolError("Schema belongs to another Task")
        resolved = resolve_bundle(bundle, schema)
        for binding in scope["bindings"]:
            if binding["node"] not in resolved.raw_task:
                raise ProtocolError("scope binding names an absent actual Schema node")
    if request["operation"] == "implement":
        if not scope["bindings"] or {c for b in scope["bindings"] for c in b["coverage_ids"]} != set(scope["coverage_ids"]):
            raise ProtocolError("implement needs complete nonempty scope bindings")
        seed = request["seed"] or task["starter_source"]
        objects["seed"] = seed
        if seed is not None: bundle.source_files(seed)
        source_refs = [task["baseline_source"]] + ([seed] if seed else [])
        objects["source_formats"] = sorted({bundle.document(r, "source")["format"] for r in source_refs})
    # Coverage partition semantics and when/variant admissibility are Task and
    # writing/backend owner checks, not a generic expression evaluator here.
    return objects


def require_capabilities(request: dict, objects: dict, actual: dict):
    capabilities(actual)
    profile_caps = objects["profile"]["capabilities"]
    for caps in (profile_caps, actual):
        if PROTOCOL not in caps["protocols"] or request["operation"] not in caps["operations"]:
            raise UnsupportedError("backend does not support this protocol/operation")
        if request["operation"] == "implement":
            if request["mode"] not in caps["modes"] or not set(objects.get("source_formats", ["single_file"])) <= set(caps["source_formats"]):
                raise UnsupportedError("backend does not support this mode/source format")
            if objects["seed"] is None and not caps["empty_start"]:
                raise UnsupportedError("backend does not support empty_start")
        if request["budget"]["limits"]["tokens"] is not None and caps["token_enforcement"] != "hard":
            raise UnsupportedError("finite token budget requires qualified hard token enforcement")
    for obj in (request, *[v for v in objects.values() if isinstance(v, dict)]):
        unknown = set(obj.get("extensions", {})) - {"information"} - set(actual["extensions"])
        if unknown: raise UnsupportedError(f"unsupported execution/acceptance extensions: {sorted(unknown)}")
