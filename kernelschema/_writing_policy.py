"""Finite writing policy using the original Template, never a second parser."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import re
import tempfile

from . import template_api
from .contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, sha256, validate
from .evaluation import RowSoftmaxTask

DRAFT_FORMAT = "ks.writing.draft.v1"
REVIEW_FORMAT = "ks.writing.review.v1"


def wire_value(value):
    """Preserve original YAML decimals in JSON role data without binary floats.

    This is a quotation encoding, not Schema normalization or a new parser.
    Published YAML and comparisons continue to use the original Template values.
    """
    if isinstance(value, Decimal):
        return {"yaml_decimal": str(value)}
    if isinstance(value, dict):
        return {k: wire_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [wire_value(v) for v in value]
    return value


def template_contract(bundle, task_ref):
    """A Task-derived Contract, with no R0 or private evaluator reads.

    The first finite writing method supports the same fixed public row-softmax
    Task as K1/K4. Unknown Task formats require an explicit future adapter.
    """
    adapter = RowSoftmaxTask(bundle, task_ref)
    task = adapter.task
    from .resources import resource
    environment = bundle.read(task["environment"])
    hardware = None
    if environment != resource("examples/workflows/row_softmax/task/environment.md").read_bytes():
        from .evaluation.environment import read_target
        hardware = read_target(environment)["hardware_specification"]
    public = template_api.tools().yaml_source.loads(bundle.read(task["contract"]).decode(), "task").value
    shape = [{"row": "m"}, {"column": "n"}]
    domain = public["domain"]
    formula = public["math"]["formula"]
    math = (f"m in {domain['M']}; n in {domain['N']}; 0<=i<m; 0<=j<n. "
            "x and out are CUDA FP32, shape (m,n), stride (n,1), arbitrary legal storage offsets, requires_grad=False and distinct storage. "
            "Model().forward(x,out) runs eager inference on one call stream; CUDA Graph and concurrent streams are unsupported. "
            + formula + ". Write FP32 approximations to g into the original out region and return the identical out object. "
            "Preserve x bytes, both original storage/view signatures and every backing byte outside out. "
            "Recompute changed legal x; no test-number, invocation-order or precomputed-output dependency. Evaluator files are not writable.")
    contract = {"Algorithm_requirement": {
        "Input": {"x": {"meaning": "Read-only logits on the selected CUDA device", "shape": shape, "dtype": "fp32", "range": public["tensors"]["x"]["range"]},
                  "out": {"meaning": "Supplied original output buffer", "shape": shape, "dtype": "fp32", "range": public["tensors"]["out"]["initial_contents"]}},
        "Output": {"y": {"meaning": "Identical out object with original region written", "shape": shape, "dtype": "fp32"}},
        "Compute": [{"Name": "Row softmax", "Math": math}],
        "Golden_reference": {"Source": "KernelSchema@2efc92479db497e0ecf681e276ecddc72cad0de6:examples/workflows/row_softmax/evaluator/golden.py::softmax_fp64",
                             "Tolerance": {"y": "All output elements finite; abs(fp64(actual)-g)<=2e-6+2e-5*abs(g)"}}},
        "Hardware_specification": {
            "Device": "Unselected NVIDIA CUDA device", "Architecture": "Unselected NVIDIA architecture",
            "Execution": {"cta": "Cooperative thread block on one SM; member threads own private registers", "warp": "SIMT threads in a CTA; concrete target details are unselected"},
            "Feature": {"register": "Thread-private register storage; concrete device capacity is unselected"},
            "Stack": {"Language": "Triton DSL (runtime version unselected)", "Toolchain": "Unselected Triton compiler and CUDA stack"},
            "Wiki": {"Hardware": "<task-hardware@pending:NVIDIA/selected-device.md>",
                     "Programming": {"Language": "<task-language@pending:Triton/selected-version/>", "Virtual_ISA": "<task-ptx@pending:PTX/selected-version/>", "Machine_ISA": "<task-sass@pending:NVIDIA/selected-architecture/>"},
                     "Microbench": "<task-microbench@pending:NVIDIA/selected-device/>",
                     "Sota": {"Kernel": "<task-kernel@pending:selected-device/index.md>", "Schema": "<task-schema@pending:selected-device/index.md>"},
                     "Profile": "<task-profile@pending:NVIDIA/selected-tools/>"}}}
    if hardware is not None:
        contract["Hardware_specification"] = deepcopy(hardware)
    return contract


def contract_for(context):
    if context.request["operation"] == "optimize":
        return template_api.effective_document(context.bundle, context.objects["schema"], "root")["CONTRACT"]
    return template_contract(context.bundle, context.request["task"])


def catalog(context):
    result = {}
    for category in ("feedback", "evidence"):
        for index, ref in enumerate(context.request.get(category, [])):
            result[f"{category}:{index}"] = ref
    if context.objects["task"]["starter_source"]:
        result["task/starter_source"] = context.objects["task"]["starter_source"]
    # This explicit Task marker means seed=null, including a legal empty start.
    result["task/default_start"] = context.request["task"]
    for entry in context.objects["materials"]["entries"]:
        if {"planner", "reviewer"} <= set(entry["roles"]):
            result["material:" + entry["id"]] = entry["file"]
    return result


def selection_catalog(context):
    """Exact planner grants, with the two other ID namespaces made explicit."""
    actions = catalog(context)
    return [{"id": entry["id"], "ref": entry["file"],
             "delivery_id": "material/" + entry["id"],
             "action_id": "material:" + entry["id"] if "material:" + entry["id"] in actions else None}
            for entry in context.objects["materials"]["entries"] if "planner" in entry["roles"]]


def text_policy(context):
    """Project the pinned original Template budgets, without another validator.

    Writing's Rules check binds this installed implementation to the frozen
    source. The summary carries that actual Ref; host-only source bodies stay
    host-only. Both writing roles receive the same small normative projection.
    """
    original = template_api.tools().text_policy
    source_sha256 = sha256(Path(original.__file__).read_bytes())
    sources = [ref for ref in context.objects["rules"]["files"] if ref["sha256"] == source_sha256]
    if len(sources) != 1 or context.bundle.read(sources[0]) != Path(original.__file__).read_bytes():
        raise ProtocolError("writing text policy requires exactly one frozen original Template source")
    return {
        "format": "ks.writing.text-policy.v1", "source": sources[0],
        "symbols": ["BUDGETS", "text_size", "check_values"],
        "path_syntax": "Path segments are literal field names; * selects each named mapping member; [] selects each list item. These are structural paths, not arbitrary matching field names.",
        "counting": {
            "value": "Decoded raw YAML string values only, after YAML folding/escapes; no trimming or Unicode normalization. Each matching list item or mapping value is counted separately.",
            "characters": "Python len(value): Unicode code points, including spaces and newline characters; not bytes or tokens. maxLength=null means no character limit for this entry.",
            "lines": "Python len(value.splitlines()): logical lines, including interior blank lines; a final line terminator adds no extra line. The empty string has zero lines.",
            "prefix": "A non-null prefix restricts this entry to strings starting with that exact prefix; count the entire value, including the prefix.",
        },
        "budgets": [{"path": ["[]" if part is int else "*" if part is None else part for part in budget.path],
                     "maxLength": budget.characters, "maxLines": budget.lines, "prefix": budget.prefix}
                    for budget in original.BUDGETS],
        "scope": "Required raw-field text limits, not a complete YAML schema. The original Template still checks structure, references, other field rules and semantics. Self-check every applicable value before returning raw; do not assume host truncation or repair.",
    }


def parse_draft(context, value):
    fields = {"format", "decision", "reason", "selection", "raw", "dispatch", "bindings", "next_action", "action_reason", "action_inputs", "action_evidence"}
    if not isinstance(value, dict) or set(value) != fields or value["format"] != DRAFT_FORMAT:
        raise ProtocolError("writing_draft must match ks.writing.draft.v1 exactly")
    if value["decision"] not in ("keep", "revise") or not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ProtocolError("draft requires keep/revise and a reason")
    if context.request["operation"] == "design" and value["decision"] != "revise":
        raise ProtocolError("design must propose its initial complete Schema")
    selected = value["selection"]
    allowed = {e["id"] for e in context.objects["materials"]["entries"] if "planner" in e["roles"]}
    if not isinstance(selected, list) or any(not isinstance(x, dict) or set(x) != {"id", "reason"} or x["id"] not in allowed or not isinstance(x["reason"], str) or not x["reason"].strip() for x in selected):
        raise ProtocolError("selection must explain actual planner-granted material IDs")
    if len({x["id"] for x in selected}) != len(selected) or "workflow_objective" not in {x["id"] for x in selected}:
        raise ProtocolError("selection must retain the unique current workflow_objective")
    raw = value["raw"]
    if not isinstance(raw, list) or len(raw) > 10:
        raise ProtocolError("raw must contain at most ten explicit Template nodes")
    for entry in raw:
        if not isinstance(entry, dict) or set(entry) != {"node", "text"} or not isinstance(entry["node"], str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", entry["node"]):
            raise ProtocolError("raw entries require a node ID and inline text, never a host path")
        if not isinstance(entry["text"], str) or not entry["text"] or "\x00" in entry["text"]:
            raise ProtocolError("raw needs complete UTF-8 YAML text")
    if len({r["node"] for r in raw}) != len(raw):
        raise ProtocolError("duplicate raw node")
    if value["dispatch"] is not None and (not isinstance(value["dispatch"], str) or not value["dispatch"].strip()):
        raise ProtocolError("dispatch must be complete inline Markdown or null")
    if value["decision"] == "keep":
        if raw or value["dispatch"] is not None or value["bindings"] is not None:
            raise ProtocolError("keep reuses exact original Schema and scope")
    elif not raw or "root" not in {r["node"] for r in raw} or not isinstance(value["bindings"], list):
        raise ProtocolError("revision needs the complete root/raw and scope bindings")
    if value["next_action"] not in ("none", "implement", "verify"):
        raise ProtocolError("invalid next_action")
    refs = catalog(context)
    for key in ("action_inputs", "action_evidence"):
        if not isinstance(value[key], list) or any(not isinstance(x, str) or x not in refs for x in value[key]) or len(set(value[key])) != len(value[key]):
            raise ProtocolError("action may select only explicit IDs from the fixed input catalog")
    if value["next_action"] == "none":
        if value["action_reason"] is not None or value["action_inputs"] or value["action_evidence"]:
            raise ProtocolError("no action has null reason and empty input/evidence lists")
    else:
        if context.request["operation"] != "optimize" or not isinstance(value["action_reason"], str) or not value["action_reason"].strip() or not value["action_inputs"] or not value["action_evidence"]:
            raise ProtocolError("action needs an optimize request, reason, explicit inputs and evidence")
        if any(not key.startswith(("evidence:", "feedback:", "material:")) for key in value["action_evidence"]):
            raise ProtocolError("action evidence must be authorized feedback, evidence or role-granted material")
        sources = [refs[key] for key in value["action_inputs"] if _is_source(context, refs[key])]
        default_start = "task/default_start" in value["action_inputs"]
        if value["next_action"] == "verify":
            if default_start or len(sources) != 1 or not context.bundle.source(sources[0]):
                raise ProtocolError("verify action needs one explicit nonempty original Source")
        elif not ((len(sources) == 1 and not default_start) or (not sources and default_start)):
            raise ProtocolError("implement action must explicitly select one Source or task/default_start (seed=null)")
    return deepcopy(value)


def _is_source(context, ref):
    try:
        return context.bundle.document(ref).get("kind") == "source"
    except ProtocolError:
        return False


def normalize(context, draft):
    engine = template_api.tools()
    if draft["decision"] == "keep":
        schema = context.objects["schema"]
        raw = [{"node": r["node"], "text": context.bundle.read(r["file"]).decode()} for r in schema["raw"]]
        dispatch = context.bundle.read(schema["dispatch"]).decode() if schema["dispatch"] else None
        bindings = context.objects["scope"]["bindings"]
    else:
        raw, dispatch, bindings = draft["raw"], draft["dispatch"], draft["bindings"]
    with tempfile.TemporaryDirectory(prefix="ks-writing-preview-") as name:
        directory = Path(name)
        for entry in raw:
            (directory / (entry["node"] + ".yaml")).write_text(entry["text"], encoding="utf-8")
        plan = template_api.normalize_directory(directory)
        resolved = engine.validate_documents(plan.files, plan.documents)
        if dispatch is not None:
            errors = engine.dispatch.validate(dispatch, plan.files)
            if errors:
                raise ProtocolError("; ".join(e.message for e in errors))
        elif len(raw) > 1:
            raise ProtocolError("multiple nodes require actual dispatch")
        normalized = [{"node": node, "text": text} for node, text in sorted(plan.candidates.items())]
    scope = validate({"protocol": PROTOCOL, "kind": "scope", "coverage_ids": context.objects["scope"]["coverage_ids"], "bindings": bindings})
    return normalized, dispatch, scope, resolved


def _diff(before, after, location=""):
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        return [p for k in sorted(before.keys() | after.keys()) for p in _diff(before.get(k), after.get(k), (location + "." if location else "") + k)]
    return [location]


def precheck(context, draft, raw, dispatch, scope, resolved):
    failures, unknowns = [], []
    if resolved.effective_document("root")["CONTRACT"] != contract_for(context):
        failures.append("root CONTRACT differs from the fixed Task-derived/current Contract")
    if {c for b in scope["bindings"] for c in b["coverage_ids"]} != set(scope["coverage_ids"]):
        failures.append("bindings do not cover the entire requested Task scope")
    previous = template_api.resolve_bundle(context.bundle, context.objects["schema"]) if "schema" in context.objects else None
    current_docs = resolved.raw_task
    old_docs = previous.raw_task if previous else {}
    changes = _diff(old_docs, current_docs)
    old_dispatch = context.bundle.read(context.objects["schema"]["dispatch"]).decode() if previous and context.objects["schema"]["dispatch"] else None
    if old_dispatch != dispatch:
        changes.append("dispatch")
    if context.objects["scope"] != scope:
        changes.append("scope.bindings")
    for node, doc in current_docs.items():
        old_feedback = old_docs.get(node, {}).get("REALIZATION", {}).get("Feedback", [])
        feedback = doc.get("REALIZATION", {}).get("Feedback", [])
        # Template already enforces Prefer targeting and immutable locator shape.
        # The finite host method additionally requires original measured evidence.
        if feedback != old_feedback:
            unknowns.append("new REALIZATION.Feedback measurement provenance cannot be authenticated by this finite method; keep ordinary feedback external")
    task = RowSoftmaxTask(context.bundle, context.request["task"])
    scope_facts = task.validate_scope_documents({"task_sha256": context.request["task"]["sha256"]}, scope, resolved=resolved)
    if scope_facts["status"] == "fail":
        failures.extend(scope_facts["reasons"])
    if previous:
        def substantive(docs):
            result = deepcopy(docs)
            for doc in result.values():
                doc.pop("META_DATA", None)
                doc.get("REALIZATION", {}).pop("Feedback", None)
            return result
        same = substantive(old_docs) == substantive(current_docs) and old_dispatch == dispatch
        # when is a design decision even though it lives in metadata.
        same &= {n: d.get("META_DATA", {}).get("Dispatch") for n, d in old_docs.items()} == {n: d.get("META_DATA", {}).get("Dispatch") for n, d in current_docs.items()}
        change = "unchanged" if same else "new_revision"
    else:
        change = "new_task"
    return {"structure": "pass", "permissions": "fail" if failures else "unknown" if unknowns else "pass", "scope": scope_facts,
            "failures": failures, "unknowns": unknowns, "changes": [{"location": loc, "reason": draft["reason"]} for loc in changes], "change": change}


def inventory(context, resolved, scope, draft, facts):
    items = []
    def add(location, text, category):
        text = text if isinstance(text, str) else jcs(wire_value(text)).decode()
        items.append({"id": digest({"location": location, "text": text})[7:23], "location": location, "quote": text, "category": category})
    for key in ("contract", "acceptance", "workload", "environment"):
        add("task/" + key, context.bundle.read(context.objects["task"][key]).decode(), "task-equivalence")
    add("scope", scope, "full-task-coverage")
    add("root::CONTRACT", resolved.effective_document("root")["CONTRACT"], "task-equivalence")
    for node in sorted({b["node"] for b in scope["bindings"]}):
        if node not in resolved.raw_task:
            continue
        doc = resolved.effective_document(node)
        for key, value in doc.get("DESIGN", {}).items():
            add(node + "::DESIGN." + key, value, key)
        if doc.get("REALIZATION"):
            add(node + "::REALIZATION", doc["REALIZATION"], "constraints")
    add("impact", {"changes": facts["changes"], "change": facts["change"]}, "design-impact")
    add("action", {k: draft[k] for k in ("next_action", "action_reason", "action_inputs", "action_evidence")}, "action-evidence")
    objective = next(e for e in context.objects["materials"]["entries"] if e["id"] == "workflow_objective")
    add("workflow_objective", context.bundle.read(objective["file"]).decode(), "bounded-objective")
    return items


def review_answers(content, expected_subject, obligations):
    valid = isinstance(content, dict) and set(content) == {"format", "subject", "items"} and content["format"] == REVIEW_FORMAT and content["subject"] == expected_subject and isinstance(content["items"], list)
    answers = {}
    if valid:
        for item in content["items"]:
            if not isinstance(item, dict) or set(item) != {"id", "status", "reason", "location", "quote"} or not isinstance(item["id"], str) or item["id"] in answers:
                valid = False
                break
            answers[item["id"]] = item
        valid &= set(answers) == {o["id"] for o in obligations}
    output = []
    for obligation in obligations:
        answer = answers.get(obligation["id"], {}) if valid else {}
        accepted = answer.get("status") in ("pass", "fail", "unknown") and isinstance(answer.get("reason"), str) and bool(answer["reason"].strip()) and all(answer.get(k) == obligation[k] for k in ("location", "quote"))
        output.append({**obligation, "status": answer["status"] if accepted else "unknown", "reason": answer["reason"] if accepted else "Independent review missing, unlocated, malformed or bound to another frozen output."})
    statuses = [o["status"] for o in output]
    return ("fail" if "fail" in statuses else "unknown" if "unknown" in statuses or not statuses else "pass"), output
