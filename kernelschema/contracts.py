"""The ks.workflow.v1 wire contract and its single RFC 8785 byte codec.

The protocol restricts JSON numbers to safe integers. Fractional measurements
are decimal strings. UTF-16 property ordering is intentional (RFC 8785 §3.2.3).
Native reports are raw bytes and are not reserialized by this codec.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Literal, TypedDict

PROTOCOL = "ks.workflow.v1"
MAX_INTEGER = 2**53 - 1
OPERATIONS = ("design", "optimize", "implement")
ROLES = ("planner", "coder", "reviewer", "extract")
STATUSES = ("pass", "fail", "unknown", "not_run")


class ProtocolError(ValueError):
    """Malformed, incompatible, missing, or wrongly bound input."""


class UnsupportedError(ProtocolError):
    """An explicitly requested capability has not been implemented/qualified."""


class Ref(TypedDict):
    path: str
    sha256: str


class Check(TypedDict):
    status: Literal["pass", "fail", "unknown", "not_run"]
    receipt: Ref | None
    reason: str


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _scalar(value: str) -> str:
    if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ProtocolError("JSON contains a non-Unicode-scalar surrogate")
    return value


def jcs(value: Any) -> bytes:
    """Canonical JSON for the workflow's integer-only RFC 8785 domain."""
    def encode(item):
        if item is None:
            return "null"
        if type(item) is bool:
            return "true" if item else "false"
        if type(item) is int:
            if abs(item) > MAX_INTEGER:
                raise ProtocolError("JSON number is outside the safe integer range")
            return str(item)
        if isinstance(item, str):
            return json.dumps(_scalar(item), ensure_ascii=False, separators=(",", ":"))
        if isinstance(item, list):
            return "[" + ",".join(encode(x) for x in item) + "]"
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ProtocolError("JSON object keys must be strings")
            keys = sorted(item, key=lambda k: _scalar(k).encode("utf-16-be"))
            return "{" + ",".join(encode(k) + ":" + encode(item[k]) for k in keys) + "}"
        raise ProtocolError("JSON numbers must be safe integers; use decimal strings for fractions")
    try:
        return encode(value).encode("utf-8")
    except RecursionError as error:
        raise ProtocolError("JSON nesting exceeds capacity") from error


def loads(data: bytes | str) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ProtocolError(f"duplicate JSON key: {key!r}")
            out[key] = value
        return out
    def invalid(value):
        raise ProtocolError(f"unsupported JSON number: {value}; use a decimal string")
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="strict")
        if data.startswith("\ufeff"):
            raise ProtocolError("JSON BOM is forbidden")
        value = json.loads(data, object_pairs_hook=pairs, parse_float=invalid, parse_constant=invalid)
        jcs(value)  # Also rejects escaped surrogates and unsafe integers.
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ProtocolError(f"invalid JSON: {error}") from error


def digest(value: Any) -> str:
    return sha256(jcs(value))


def relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        raise ProtocolError(f"expected canonical POSIX relative path: {value!r}")
    _scalar(value)
    if any(part in ("", ".", "..") for part in value.split("/")) or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value):
        raise ProtocolError(f"unsafe relative path: {value!r}")
    return value


def _obj(value, required, optional=()):
    if not isinstance(value, dict):
        raise ProtocolError("expected an object")
    missing, extra = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ProtocolError(f"invalid fields; missing={sorted(missing)}, unknown={sorted(extra)}")
    return value


def _text(value, *, empty=False):
    if not isinstance(value, str) or (not empty and not value):
        raise ProtocolError("expected a nonempty string")
    return _scalar(value)


def _enum(value, choices):
    if value not in choices:
        raise ProtocolError(f"expected one of {tuple(choices)}, got {value!r}")


def _count(value, *, nullable=False):
    if nullable and value is None:
        return
    if type(value) is not int or not 0 <= value <= MAX_INTEGER:
        raise ProtocolError("expected a nonnegative safe integer")


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ProtocolError("expected sha256: followed by 64 lowercase hex digits")


def ref(value, *, nullable=False):
    if nullable and value is None:
        return
    _obj(value, ("path", "sha256"))
    relative_path(value["path"])
    _hash(value["sha256"])


def _list(value, validate=lambda x: None, *, key=None, sorted_set=False):
    if not isinstance(value, list):
        raise ProtocolError("expected an array")
    for item in value:
        validate(item)
    if key is not None or sorted_set:
        keys = [(key(x) if key else x) for x in value]
        if len(keys) != len(set(keys)):
            raise ProtocolError("duplicate set-array member")
        if sorted_set and keys != sorted(keys):
            raise ProtocolError("set array must be sorted by its contract key")


def _refs(value, *, sorted_set=False):
    _list(value, ref, key=(lambda x: x["path"]) if sorted_set else None, sorted_set=sorted_set)


def _time(value):
    if value is None:
        return
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z", value):
        raise ProtocolError("expected RFC3339 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ProtocolError("invalid timestamp") from error


def limits(value):
    _obj(value, ("calls", "evaluator_jobs", "active_ms", "tokens"), ("model_calls", "deadline_utc"))
    _obj(value["calls"], ROLES)
    for count in value["calls"].values():
        _count(count)
    for key in ("evaluator_jobs", "active_ms", "model_calls"):
        if key in value:
            _count(value[key])
    _count(value["tokens"], nullable=True)
    if "deadline_utc" in value:
        _time(value["deadline_utc"])


def check(value):
    _obj(value, ("status", "receipt", "reason"))
    _enum(value["status"], STATUSES)
    ref(value["receipt"], nullable=True)
    _text(value["reason"], empty=value["status"] != "unknown")
    if value["status"] in ("pass", "fail") and value["receipt"] is None:
        raise ProtocolError("pass/fail requires an evidence receipt")
    if value["status"] == "not_run" and value["receipt"] is not None:
        raise ProtocolError("not_run cannot have an execution receipt")


def _subject(value, keys):
    _obj(value, keys)
    for h in value.values():
        _hash(h)


def _extensions(value):
    if not isinstance(value, dict):
        raise ProtocolError("extensions must be an owner-keyed object")
    for owner, extension in value.items():
        _text(owner)
        if not isinstance(extension, dict):
            raise ProtocolError("extensions.<owner> must be an object")


def validate(value: Any) -> dict:
    """Validate one normative object (shape only; Bundle verifies actual bytes)."""
    jcs(value)
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL:
        raise ProtocolError("missing or unsupported protocol; expected ks.workflow.v1")
    if "extensions" in value:
        _extensions(value["extensions"])
    if "operation" in value:
        _enum(value["operation"], OPERATIONS)
        return _result(value) if "run_id" in value else _request(value)
    kind = value.get("kind")
    fields = {
        "task": "task_id contract acceptance evaluator environment workload baseline_source starter_source",
        "schema": "task_sha256 raw dispatch design_revision parent change change_report",
        "rules": "files tools",
        "materials": "entries",
        "scope": "coverage_ids bindings",
        "source": "format encoding entrypoint files",
        "budget_account": "id owner limits deadline_utc",
        "backend_profile": "name version runtime model parameters environment capabilities native_limits",
        "evaluator": "version entrypoint files inputs timing environment adapters",
        "fidelity": "operation_id subject method coverage fidelity limitations evidence usage_receipts",
        "evaluation": "operation_id subject build correctness performance case_ids samples raw_receipts usage_receipts environment",
        "writing_review": "operation_id subject checks evidence usage_receipts",
        "change_report": "operation_id subject changes raw evidence",
        "usage_index": "entries",
        "feedback": "subject reason evidence",
    }
    if not isinstance(kind, str) or kind not in fields:
        raise ProtocolError(f"unsupported kind: {kind!r}")
    _obj(value, ("protocol", "kind", *fields[kind].split()), ("extensions",))
    v = value
    if kind == "task":
        _text(v["task_id"])
        for k in ("contract", "acceptance", "evaluator", "environment", "workload", "baseline_source", "starter_source"):
            ref(v[k], nullable=k in ("starter_source", "baseline_source", "evaluator"))
    elif kind == "schema":
        _hash(v["task_sha256"])
        def raw(x):
            _obj(x, ("node", "file")); _text(x["node"]); ref(x["file"])
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", x["node"]):
                raise ProtocolError("invalid Schema node ID")
        _list(v["raw"], raw, key=lambda x: x["node"], sorted_set=True)
        if not v["raw"] or "root" not in [x["node"] for x in v["raw"]]:
            raise ProtocolError("Schema must retain root Contract raw")
        for k in ("dispatch", "parent", "change_report"):
            ref(v[k], nullable=True)
        _count(v["design_revision"])
        _enum(v["change"], ("new_task", "new_revision", "unchanged"))
        if v["change"] == "new_task" and v["design_revision"] != 0:
            raise ProtocolError("new_task starts at design_revision 0")
        if v["change"] == "new_revision" and (v["parent"] is None or v["change_report"] is None):
            raise ProtocolError("new_revision requires parent and change_report")
    elif kind == "rules":
        _refs(v["files"], sorted_set=True)
        def tool(x):
            _obj(x, ("name", "version", "source")); _text(x["name"]); _text(x["version"]); ref(x["source"])
        _list(v["tools"], tool, key=lambda x: x["name"])
    elif kind == "materials":
        def entry(x):
            _obj(x, ("id", "file", "roles", "origin", "license"))
            for k in ("id", "origin", "license"): _text(x[k])
            ref(x["file"])
            _list(x["roles"], lambda r: _enum(r, ROLES[:3]), key=lambda r: r)
        _list(v["entries"], entry, key=lambda x: x["id"], sorted_set=True)
    elif kind == "scope":
        _list(v["coverage_ids"], _text, sorted_set=True)
        def binding(x):
            _obj(x, ("coverage_ids", "node", "variant"))
            _list(x["coverage_ids"], _text, key=lambda a: a)
            if not x["coverage_ids"] or not set(x["coverage_ids"]) <= set(v["coverage_ids"]):
                raise ProtocolError("binding must select nonempty declared coverage IDs")
            _text(x["node"])
            if x["variant"] is not None: _text(x["variant"])
        _list(v["bindings"], binding)
        assigned = [c for b in v["bindings"] for c in b["coverage_ids"]]
        if len(assigned) != len(set(assigned)):
            raise ProtocolError("coverage ID bound more than once")
    elif kind == "source":
        for field in ("format", "encoding", "entrypoint"):
            _text(v[field])
        supported = {("single_file", "utf-8", "kernel.py"), ("python_tree", "utf-8", "solution.py")}
        if (v["format"], v["encoding"], v["entrypoint"]) not in supported:
            raise UnsupportedError("unsupported Source format/encoding/entrypoint")
        def file(x):
            _obj(x, ("name", "mode", "size", "sha256")); relative_path(x["name"]); _count(x["size"]); _hash(x["sha256"])
            if x["mode"] != "0644":
                raise UnsupportedError("Source supports regular mode 0644 files only")
            if v["format"] == "single_file" and x["name"] != "kernel.py":
                raise UnsupportedError("single_file requires only kernel.py with mode 0644")
            if v["format"] == "python_tree":
                # Finite source-only format: root Python modules, one level of
                # package modules, and root text provenance. No native assets,
                # nested trees, executable scripts or installation instructions.
                python = re.fullmatch(r"(?:[A-Za-z_][A-Za-z_0-9]*/)?[A-Za-z_][A-Za-z_0-9]*\.py", x["name"])
                text = re.fullmatch(r"[A-Za-z_][A-Za-z_0-9-]*\.(?:json|md|txt)", x["name"])
                if not (python or text or x["name"] == "LICENSE"):
                    raise UnsupportedError("python_tree supports flat Python packages and root text files only")
        _list(v["files"], file, key=lambda x: x["name"], sorted_set=True)
        if v["format"] == "single_file" and len(v["files"]) != 1:
            raise ProtocolError("Source is a complete single-file snapshot")
        if v["format"] == "python_tree":
            if not 1 <= len(v["files"]) <= 32 or sum(x["size"] for x in v["files"]) > 4 * 1024 * 1024:
                raise UnsupportedError("python_tree limit is 32 files / 4 MiB")
            if v["entrypoint"] not in {x["name"] for x in v["files"]}:
                raise ProtocolError("Source entrypoint is absent")
    elif kind == "budget_account":
        _text(v["id"]); _text(v["owner"]); limits(v["limits"]); _time(v["deadline_utc"])
    elif kind == "backend_profile":
        for k in ("name", "version", "runtime", "model"): _text(v[k])
        for k in ("parameters", "native_limits"):
            if not isinstance(v[k], dict): raise ProtocolError(f"{k} must be an object")
        ref(v["environment"])
        capabilities(v["capabilities"])
    elif kind == "evaluator":
        _text(v["version"]); ref(v["entrypoint"]); _refs(v["files"], sorted_set=True)
        for k in ("inputs", "timing", "environment"): ref(v[k])
        _list(v["adapters"], _text, key=lambda x: x)
        if v["entrypoint"] not in v["files"]:
            raise ProtocolError("evaluator entrypoint must belong to its complete files list")
    elif kind == "fidelity":
        _text(v["operation_id"])
        _subject(v["subject"], ("request_sha256", "task_sha256", "schema_sha256", "rules_sha256", "scope_sha256", "source_sha256"))
        _obj(v["method"], ("spec", "reviewer")); ref(v["method"]["spec"])
        reviewer = _obj(v["method"]["reviewer"], ("runtime", "requested_model", "reported_model", "implementation", "scope_id"))
        for k, x in reviewer.items():
            if k != "reported_model" or x is not None: _text(x)
        _obj(v["coverage"], ("status", "mappings", "task_uncovered")); _enum(v["coverage"]["status"], STATUSES)
        def mapping(x):
            _obj(x, ("coverage_id", "node", "variant", "source_locations")); _text(x["coverage_id"]); _text(x["node"])
            if x["variant"] is not None: _text(x["variant"])
            _list(x["source_locations"], _text)
        _list(v["coverage"]["mappings"], mapping); _list(v["coverage"]["task_uncovered"], _text)
        _obj(v["fidelity"], ("status", "items")); _enum(v["fidelity"]["status"], STATUSES)
        def obligation(x):
            _obj(x, ("id", "obligation", "schema_locations", "source_locations", "status", "reason"))
            for k in ("id", "obligation", "reason"): _text(x[k])
            for k in ("schema_locations", "source_locations"): _list(x[k], _text)
            _enum(x["status"], STATUSES)
        _list(v["fidelity"]["items"], obligation, key=lambda x: x["id"])
        if v["coverage"]["status"] == "pass" and (not v["coverage"]["mappings"] or any(not m["source_locations"] for m in v["coverage"]["mappings"])):
            raise ProtocolError("coverage pass requires actual mappings and source locations")
        if v["fidelity"]["status"] == "pass" and (not v["fidelity"]["items"] or any(i["status"] != "pass" or not i["schema_locations"] or not i["source_locations"] for i in v["fidelity"]["items"])):
            raise ProtocolError("fidelity pass requires explicit passing obligations and actual locations")
        _list(v["limitations"], _text); _refs(v["evidence"]); _refs(v["usage_receipts"])
    elif kind == "evaluation":
        _text(v["operation_id"])
        _subject(v["subject"], ("task_sha256", "source_sha256", "evaluator_sha256", "inputs_sha256", "environment_sha256", "timing_sha256"))
        for k in ("build", "correctness", "performance"): check(v[k])
        _list(v["case_ids"], _text, key=lambda x: x)
        ref(v["samples"], nullable=True); ref(v["environment"])
        _refs(v["raw_receipts"]); _refs(v["usage_receipts"])
        if v["environment"]["sha256"] != v["subject"]["environment_sha256"]:
            raise ProtocolError("evaluation environment Ref must bind actual subject environment")
    elif kind == "writing_review":
        _text(v["operation_id"])
        _subject(v["subject"], ("request_sha256", "task_sha256", "schema_sha256", "rules_sha256", "scope_sha256"))
        _checks(v["checks"]); _refs(v["evidence"]); _refs(v["usage_receipts"])
    elif kind == "change_report":
        _text(v["operation_id"])
        _subject(v["subject"], ("request_sha256", "task_sha256"))
        def change(x):
            _obj(x, ("location", "reason")); _text(x["location"]); _text(x["reason"])
        _list(v["changes"], change); _refs(v["raw"]); _refs(v["evidence"])
    elif kind == "usage_index":
        _list(v["entries"], _usage_row)
    elif kind == "feedback":
        _subject(v["subject"], ("request_sha256", "task_sha256", "schema_sha256", "source_sha256"))
        _text(v["reason"]); _refs(v["evidence"])
    return v


def capabilities(value):
    _obj(value, ("protocols", "operations", "modes", "source_formats", "empty_start", "recovery", "token_enforcement", "extensions"))
    for k in ("protocols", "operations", "modes", "source_formats", "recovery", "extensions"):
        _list(value[k], _text, key=lambda x: x)
    if type(value["empty_start"]) is not bool: raise ProtocolError("empty_start must be boolean")
    _enum(value["token_enforcement"], ("hard", "admission_only", "unsupported"))


def _request(v):
    op = v["operation"]
    extra = {"design": (), "implement": ("schema", "seed", "mode"), "optimize": ("schema", "feedback", "evidence")}[op]
    _obj(v, ("protocol", "operation", "request_id", "task", "rules", "materials", "scope", "budget", "backend", *extra), ("extensions",))
    _text(v["request_id"])
    for k in ("task", "rules", "materials", "scope", "schema", "seed"):
        if k in v: ref(v[k], nullable=k == "seed")
    _obj(v["backend"], ("name", "profile")); _text(v["backend"]["name"]); ref(v["backend"]["profile"])
    _obj(v["budget"], ("parent", "limits")); ref(v["budget"]["parent"], nullable=True); limits(v["budget"]["limits"])
    if op == "implement":
        _enum(v["mode"], ("code", "verify_only"))
        if v["mode"] == "verify_only" and v["seed"] is None:
            raise ProtocolError("verify_only requires an explicit nonempty seed")
    if op == "optimize":
        _refs(v["feedback"]); _refs(v["evidence"])
    return v


def _checks(v):
    _obj(v, ("structure", "permissions", "semantics", "publication"))
    for x in v.values(): check(x)


def _usage_row(v):
    _obj(v, ("producer", "operation_id", "role", "purpose", "receipt_id", "sequence", "receipt", "state", "calls", "evaluator_jobs", "tokens", "gpu_ms"))
    for k in ("producer", "operation_id", "purpose", "receipt_id"): _text(v[k])
    _enum(v["role"], (*ROLES, "evaluator")); _enum(v["state"], ("settled", "unknown"))
    _count(v["sequence"]); ref(v["receipt"], nullable=True)
    for k in ("calls", "evaluator_jobs", "tokens", "gpu_ms"): _count(v[k], nullable=True)


def _result(v):
    op = v["operation"]
    extra = ("code", "assessments") if op == "implement" else ("schema", "scope", "writing_checks")
    if op == "optimize": extra += ("decision", "reason", "next_action", "action_reason", "action_inputs", "action_evidence")
    _obj(v, ("protocol", "operation", "run_id", "request_id", "request_sha256", "status", "stop_reason", "usage", "feedback", "recovery", *extra), ("extensions",))
    for k in ("run_id", "request_id", "stop_reason"): _text(v[k])
    _hash(v["request_sha256"]); _enum(v["status"], ("completed", "partial", "blocked", "failed")); _enum(v["recovery"], ("none", "continue", "settle_only", "new_request")); _refs(v["feedback"])
    u = _obj(v["usage"], ("scope", "receipts", "calls", "evaluator_jobs", "tokens", "active_ms", "unsettled"))
    if u["scope"] != "request_cumulative": raise ProtocolError("usage.scope must be request_cumulative")
    ref(u["receipts"]); _obj(u["calls"], ROLES)
    for x in u["calls"].values(): _count(x, nullable=True)
    for k in ("evaluator_jobs", "tokens", "active_ms"): _count(u[k], nullable=True)
    _list(u["unsettled"], _text, key=lambda x: x)
    if u["unsettled"] and (v["status"], v["recovery"]) != ("blocked", "settle_only"):
        raise ProtocolError("unsettled operations require blocked/settle_only")
    if op == "implement":
        _obj(v["code"], ("latest", "qualified"))
        for x in v["code"].values(): ref(x, nullable=True)
        def assessment(x):
            _obj(x, ("source_sha256", "coverage", "task_uncovered", "build", "correctness", "fidelity", "performance", "performance_goal"))
            _hash(x["source_sha256"]); _list(x["task_uncovered"], _text)
            for k in ("coverage", "build", "correctness", "fidelity", "performance"): check(x[k])
            _enum(x["performance_goal"], ("met", "unmet", "unknown"))
        _list(v["assessments"], assessment, key=lambda x: x["source_sha256"])
        qualified = v["code"]["qualified"]
        if qualified:
            found = [a for a in v["assessments"] if a["source_sha256"] == qualified["sha256"]]
            if not found or any(found[0][k]["status"] != "pass" for k in ("coverage", "build", "correctness", "fidelity", "performance")):
                raise ProtocolError("qualified requires its own five passing checks")
        if v["status"] == "completed" and qualified is None:
            raise ProtocolError("completed implement requires qualified Source")
    else:
        ref(v["schema"], nullable=True); ref(v["scope"], nullable=True); _checks(v["writing_checks"])
        write_only = v.get("extensions", {}).get("information", {}).get("writing_mode") == "write_only"
        required_checks = ("structure", "publication") if write_only else tuple(v["writing_checks"])
        if v["status"] == "completed" and (v["schema"] is None or v["scope"] is None or any(v["writing_checks"][key]["status"] != "pass" for key in required_checks)):
            raise ProtocolError("completed writing requires Schema/scope and four passing checks")
    if op == "optimize":
        _enum(v["decision"], ("keep", "revise")); _text(v["reason"]); _enum(v["next_action"], ("none", "implement", "verify"))
        _refs(v["action_inputs"]); _refs(v["action_evidence"])
        if v["next_action"] != "none":
            _text(v["action_reason"])
            if not v["action_inputs"]: raise ProtocolError("action requires explicit inputs")
        elif v["action_reason"] is not None: _text(v["action_reason"])
    return v
