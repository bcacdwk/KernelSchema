"""Finite, current-Schema obligations plus an independent read-only review.

Host AST observations can reject narrowly proven violations; they cannot grant
overall fidelity. Unknown applicable semantics are never silently passed.
"""
from __future__ import annotations

import ast
from decimal import Decimal
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


def _wire(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _wire(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_wire(v) for v in value]
    return value


def obligations(context):
    """Enumerate selected current design sections; retain unrecognized semantics."""
    schema = context.objects["schema"]
    resolved = template_api.resolve_bundle(context.bundle, schema)
    output = []

    def add(node, path, value, category):
        value = _wire(value)
        if value in (None, {}, []):
            return
        location = node + "::" + path
        output.append({"id": digest({"location": location, "value": value})[7:23],
                       "obligation": _text(value), "schema_locations": [location], "category": category})

    for node in sorted({b["node"] for b in context.objects["scope"]["bindings"]}):
        doc = resolved.effective_document(node)
        contract = doc.get("CONTRACT", {})
        from .evaluation.fa4 import TASK_ID
        fa4 = context.objects["task"]["task_id"] == TASK_ID
        if fa4:
            add(node, "CONTRACT", contract, "math-and-abi")
        else:
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
        if fa4:
            for name, obligation in {
                "empty_parity": "N1: establish initialization and valid readback of every empty parity slot, including slots not advanced by a second iteration.",
                "tma_tail": "Establish TMA padded page and page-tail accesses for L64 with bk256; no invalid pointer or stale masked data use.",
                "tmem": "Establish TMEM allocation, fences, accumulator ownership, completion, return and deallocation lifetimes.",
                "output_staging": "Establish output staging lifetime and complete writes to original out storage before full-forward completion.",
                "runtime_when": "Establish native launcher, same six objects/signature/thread/device and noncapture before and after each steady full forward; cold/probe is separate.",
            }.items():
                location = "META_DATA.Dispatch.when" if name == "runtime_when" else "DESIGN.Math_reshape.Rewrite[2]" if name == "empty_parity" else "DESIGN.Tile_pipeline"
                add(node, location, obligation, "memory-and-lifetime")
    if not output:
        raise ProtocolError("current Schema has no interpretable design obligations")
    return output


def review_input(context, source, scope_validation):
    return {"method": method_ref(context), "source_sha256": source["sha256"],
            "source_files": context.bundle.document(source, "source")["files"],
            "obligations": obligations(context), "coverage": scope_validation["mappings"],
            "task_uncovered": scope_validation["task_uncovered"]}


def _aggregate(statuses):
    values = list(statuses)
    return "fail" if "fail" in values else "unknown" if not values or any(v != "pass" for v in values) else "pass"


def _located(item, files):
    locations = item.get("source_locations", [])
    if not isinstance(locations, list) or not locations:
        return False
    evidence = item.get("evidence", [])
    if not isinstance(evidence, list):
        return False
    for loc in locations:
        match = re.fullmatch(r"([^:]+\.py):L([1-9][0-9]*)(?:-L([1-9][0-9]*))?", loc) if isinstance(loc, str) else None
        if not match:
            return False
        if match[1] not in files:
            return False
        lines = files[match[1]]
        first, last = int(match[2]), int(match[3] or match[2])
        if not 1 <= first <= last <= len(lines):
            return False
        quote = "\n".join(lines[first - 1:last])
        if not any(e == {"location": loc, "quote": quote} for e in evidence):
            return False
    return True


def _machine(obligation, code, filename="kernel.py"):
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
            return {"status": "fail", "source_locations": [f"{filename}:L{node.lineno}-L{node.end_lineno}"],
                    "reason": "Direct call to torch.softmax violates this current Forbid; no numerical failure is inferred."}
        if "without fp16 or bf16" in text and name.endswith(".to") and any(ast.unparse(a) in ("tl.float16", "tl.bfloat16", "torch.float16", "torch.bfloat16") for a in node.args):
            return {"status": "fail", "source_locations": [f"{filename}:L{node.lineno}-L{node.end_lineno}"],
                    "reason": "Explicit low-precision conversion contradicts this current precision obligation."}
    return None


def report(context, *, source, operation_id, scope_validation, receipt, publication: Publication):
    """Publish a report even when the reviewer failed; original fees remain."""
    spec = method_ref(context)
    files = {name: file.data.decode("utf-8") for name, file in context.bundle.source_files(source).items() if name.endswith(".py")}
    lines = {name: code.splitlines() for name, code in files.items()}
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
        # The narrow row checker cannot determine reachability across a tree.
        machine = _machine(obligation, files["kernel.py"]) if set(files) == {"kernel.py"} else None
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
    located_quotes = []
    for item in items:
        evidence = []
        for loc in item["source_locations"]:
            name, positions = loc.split(":L", 1)
            first, _, last = positions.partition("-L")
            evidence.append({"location": loc, "quote": "\n".join(lines[name][int(first)-1:int(last or first)])})
        located_quotes.append({"id": item["id"], "evidence": evidence})
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
             "evidence": [context.request_ref, source, *receipt.evidence_refs()], "usage_receipts": [receipt.receipt] if receipt.receipt else [],
             "extensions": {"information": {"located_quotes": located_quotes,
                 "scope_limitations": scope_validation.get("limitations", [])}}}
    reference = publication.put_json("fidelity.json", validate(value))
    publication.commit([reference], subjects=[context.request_ref, source])
    return reference, value
