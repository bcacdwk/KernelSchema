"""Finite, current-Schema obligations plus an independent read-only review.

Host AST observations can reject narrowly proven violations; they cannot grant
overall fidelity. Unknown applicable semantics are never silently passed.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

from . import template_api
from .artifacts import Publication
from .contracts import PROTOCOL, ProtocolError, digest, jcs, sha256, validate

METHOD_PATH = Path(__file__).parent / "implement/fidelity-method.v1.md"
METHOD_VERSION = "finite-current-schema.v1"


def freeze_method(publication):
    """Return the Rules file/tool entries with actual checker and method bytes."""
    return {"file": publication.put_bytes("fidelity/method.md", METHOD_PATH.read_bytes()),
            "tool": {"name": "kernelschema.fidelity", "version": METHOD_VERSION,
                     "source": publication.put_bytes("fidelity/checker.py", Path(__file__).read_bytes())}}


def method_ref(context):
    expected = sha256(METHOD_PATH.read_bytes())
    matches = [r for r in context.objects["rules"]["files"] if r["sha256"] == expected]
    if len(matches) != 1 or context.bundle.read(matches[0]) != METHOD_PATH.read_bytes():
        raise ProtocolError("Rules must freeze exactly one current finite fidelity method")
    checkers = [t for t in context.objects["rules"]["tools"] if t["name"] == "kernelschema.fidelity"]
    if len(checkers) != 1 or checkers[0]["version"] != METHOD_VERSION or context.bundle.read(checkers[0]["source"]) != Path(__file__).read_bytes():
        raise ProtocolError("Rules must freeze the actual finite fidelity checker bytes/version")
    return matches[0]


def _text(value):
    return value if isinstance(value, str) else jcs(value).decode()


def obligations(context):
    """Enumerate selected current design sections; retain unrecognized semantics."""
    schema = context.objects["schema"]
    resolved = template_api.resolve_bundle(context.bundle, schema)
    output = []

    def add(node, path, value, category):
        if value in (None, {}, []):
            return
        location = node + "::" + path
        output.append({"id": digest({"location": location, "value": value})[7:23],
                       "obligation": _text(value), "schema_locations": [location], "category": category})

    for node in sorted({b["node"] for b in context.objects["scope"]["bindings"]}):
        doc = resolved.effective_document(node)
        contract = doc.get("CONTRACT", {})
        add(node, "CONTRACT.Algorithm_requirement", contract.get("Algorithm_requirement"), "math-and-abi")
        design = doc.get("DESIGN", {})
        for key, value in design.items():
            category = {"Math_reshape": "math-and-precision", "Work_schedule": "ownership-and-dispatch",
                        "Tile_pipeline": "memory-and-lifetime"}.get(key, "unrecognized-design")
            add(node, "DESIGN." + key, value, category)
        realization = doc.get("REALIZATION", {})
        for key, value in realization.items():
            if key in ("Prefer", "Feedback"):
                continue  # advisory, never a mandatory qualification gate
            if isinstance(value, list):
                for i, item in enumerate(value):
                    add(node, f"REALIZATION.{key}[{i}]", item, key.lower())
            else:
                add(node, "REALIZATION." + key, value, "unrecognized-realization")
        dispatch = doc.get("META_DATA", {}).get("Dispatch")
        if dispatch:
            add(node, "META_DATA.Dispatch", dispatch, "dispatch")
    if not output:
        raise ProtocolError("current Schema has no interpretable design obligations")
    return output


def review_input(context, source, scope_validation):
    return {"method": method_ref(context), "source_sha256": source["sha256"],
            "obligations": obligations(context), "coverage": scope_validation["mappings"],
            "task_uncovered": scope_validation["task_uncovered"]}


def _aggregate(statuses):
    values = list(statuses)
    return "fail" if "fail" in values else "unknown" if not values or any(v != "pass" for v in values) else "pass"


def _located(item, lines):
    locations = item.get("source_locations", [])
    if not isinstance(locations, list) or not locations:
        return False
    evidence = item.get("evidence", [])
    if not isinstance(evidence, list):
        return False
    for loc in locations:
        match = re.fullmatch(r"kernel\.py:L([1-9][0-9]*)(?:-L([1-9][0-9]*))?", loc) if isinstance(loc, str) else None
        if not match:
            return False
        first, last = int(match[1]), int(match[2] or match[1])
        if not 1 <= first <= last <= len(lines):
            return False
        quote = "\n".join(lines[first - 1:last])
        if not any(e == {"location": loc, "quote": quote} for e in evidence):
            return False
    return True


def _machine(obligation, code):
    """Only direct, explicit forbidden constructs, not an approximate prover."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # build owns syntax; it is not a design obstruction
    text = obligation["obligation"].lower()
    # A forbidden call in an unused helper or an unselected dispatch branch is
    # not proof of delegation. Only a direct forward return is mechanically
    # decisive here; the independent reviewer owns less obvious control flow.
    calls = [function.body[0].value for function in ast.walk(tree)
             if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and function.name == "forward"
             and len(function.body) == 1 and isinstance(function.body[0], ast.Return)
             and isinstance(function.body[0].value, ast.Call)]
    for node in calls:
        name = ast.unparse(node.func)
        if obligation["category"] == "forbid" and "torch.softmax" in text and name == "torch.softmax":
            return {"status": "fail", "source_locations": [f"kernel.py:L{node.lineno}-L{node.end_lineno}"],
                    "reason": "Direct call to torch.softmax violates this current Forbid; no numerical failure is inferred."}
        if "without fp16 or bf16" in text and name.endswith(".to") and any(ast.unparse(a) in ("tl.float16", "tl.bfloat16", "torch.float16", "torch.bfloat16") for a in node.args):
            return {"status": "fail", "source_locations": [f"kernel.py:L{node.lineno}-L{node.end_lineno}"],
                    "reason": "Explicit low-precision conversion contradicts this current precision obligation."}
    return None


def report(context, *, source, operation_id, scope_validation, receipt, publication: Publication):
    """Publish a report even when the reviewer failed; original fees remain."""
    spec = method_ref(context)
    code = context.bundle.source(source).decode("utf-8")
    lines = code.splitlines()
    inventory = obligations(context)
    content = receipt.output.content if receipt.ok and receipt.output is not None else None
    valid = isinstance(content, dict) and set(content) == {"items", "coverage"} and isinstance(content["items"], list) and isinstance(content["coverage"], list)
    answers = {}
    if valid:
        for item in content["items"]:
            if not isinstance(item, dict) or item.get("id") in answers:
                valid = False
                break
            answers[item.get("id")] = item
        valid = valid and set(answers) == {i["id"] for i in inventory}
    items = []
    for obligation in inventory:
        answer = answers.get(obligation["id"], {}) if valid else {}
        status = answer.get("status")
        understood = obligation["category"] not in ("unrecognized-design", "unrecognized-realization")
        accepted = understood and status in ("pass", "fail", "unknown") and bool(answer.get("reason")) and _located(answer, lines)
        result = {k: obligation[k] for k in ("id", "obligation", "schema_locations")}
        result.update(status=status if accepted else "unknown", source_locations=answer["source_locations"] if accepted else [],
                      reason=answer["reason"] if accepted else "Independent review missing, unlocated, inconsistent or method cannot interpret the applicable obligation.")
        machine = _machine(obligation, code)
        if machine:
            result.update(machine)
        items.append(result)
    mappings, coverage_statuses = [], []
    for mapping in scope_validation["mappings"]:
        candidates = [a for a in content["coverage"] if isinstance(a, dict) and all(a.get(k) == mapping.get(k) for k in ("coverage_id", "node", "variant"))] if valid else []
        answer = candidates[0] if len(candidates) == 1 else {}
        accepted = answer.get("status") in ("pass", "fail", "unknown") and _located(answer, lines) and bool(answer.get("reason"))
        coverage_statuses.append(answer["status"] if accepted else "unknown")
        mappings.append({**{k: mapping[k] for k in ("coverage_id", "node", "variant")},
                         "source_locations": answer["source_locations"] if accepted else []})
    facts = receipt.facts
    models = facts.get("reported_models", [])
    value = {"protocol": PROTOCOL, "kind": "fidelity", "operation_id": operation_id,
             "subject": {"request_sha256": context.request_sha256, **{k + "_sha256": context.request[k]["sha256"] for k in ("task", "schema", "rules", "scope")}, "source_sha256": source["sha256"]},
             "method": {"spec": spec, "reviewer": {"runtime": facts.get("runtime", "unknown"),
                 "requested_model": facts.get("requested_model", context.objects["profile"]["model"]),
                 "reported_model": models[0] if len(models) == 1 else None,
                 "implementation": digest(context.anchor["implementation"]), "scope_id": facts.get("scope_id", "unknown")}},
             "coverage": {"status": _aggregate([scope_validation["status"], *coverage_statuses]), "mappings": mappings,
                          "task_uncovered": scope_validation["task_uncovered"]},
             "fidelity": {"status": _aggregate(i["status"] for i in items), "items": items},
             "limitations": ["Finite current-Schema review, not full-domain equivalence or memory-safety proof.",
                             "Execution, numerical correctness, performance and transport qualification remain separate."],
             "evidence": [context.request_ref, source, *receipt.evidence_refs()], "usage_receipts": [receipt.receipt] if receipt.receipt else []}
    reference = publication.put_json("fidelity.json", validate(value))
    publication.commit([reference], subjects=[context.request_ref, source])
    return reference, value
