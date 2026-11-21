"""Finite Schema writing: one planner call and retained, readable file artifacts.

All external calls use K3 and the one K4 OwnerAccount. A completed publication
is not admission of a subsequent implementation request or code qualification.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
from uuid import uuid4

from . import _writing_policy as policy, template_api
from .artifacts import Publication, read_regular
from .budget import BudgetDenied, OwnerAccount, cost
from .contracts import PROTOCOL, ROLES, ProtocolError, UnsupportedError, jcs, loads, sha256, validate, limits as validate_limits
from .resources import freeze_rules as freeze_template_rules
from .runtime import AdmissionDenied, ClaudeProfile, ClaudeRuntime, RoleDelivery

VERSION = "writing.v1"


def mode(profile):
    value = profile["parameters"].get("writing_mode", "write_only")
    if value not in ("write_only", "strict"):
        raise ProtocolError("writing_mode must be write_only or strict")
    return value


def _finish_floor(profile):
    value = profile["parameters"].get("writing_finish_floor", cost())
    if (not isinstance(value, dict) or set(value) != set(cost())
            or any(type(n) is not int or n < 0 for n in value.values())
            or value["model_calls"] != sum(value[role] for role in ROLES)):
        raise ProtocolError("writing_finish_floor requires the complete nonnegative budget.cost vector with matching model_calls")
    return dict(value)


STRICT_METHOD = b"""Finite writing method v1
Planner: follow the frozen DESIGN_SCHEMA/OPTIMIZE_SCHEMA and Template documents.
The workflow_objective is the sole current instruction inside the fixed Task,
coverage and hard acceptance. Select only explicitly granted references, explain
selection, choose a concrete math/work/pipeline route, then produce complete
raw nodes. Preserve the supplied locked_contract exactly. Never invent device,
stack, measurement, citation, private evaluator contents or qualified code.
Use ks.writing.draft.v1, exactly: format, decision (keep|revise), reason,
selection [{id,reason}] including workflow_objective, raw [{node,text}],
dispatch (inline Markdown|null), bindings, next_action (none|implement|verify),
action_reason, action_inputs [catalog ID], action_evidence [catalog ID].
selection[].id must copy an exact bare id from writing_input.selection_catalog,
with a nonempty reason and no duplicates. Its delivery_id (material/<id>) locates
the delivered bytes; neither that delivery_id nor material:<id> is a selection ID.
action_inputs/action_evidence instead use exact action_catalog keys, including
material:<id> when listed. A null selection_catalog.action_id grants no action ID.
Initial design uses revise and next_action=none, action_reason=null and empty
action_inputs/action_evidence. Only optimize can propose implement or verify.
Keep has raw=[], dispatch=null, bindings=null.
A revision includes every raw node and root Contract. bindings is a JSON list
of {coverage_ids:[ID,...],node:node_ID,variant:null|ID}, not a Scope object.
Bind every fixed_assignment.scope.coverage_ids entry exactly once to a supplied
node; use variant=null unless an actual supported variant is selected.
Follow writing_input.text_policy for all decoded raw YAML strings: its numeric
maxLength/maxLines come from the frozen original Template BUDGETS. Count Unicode
code points and logical lines as specified there, including whitespace. Self-check
before returning; the host does not truncate text or strip ID prefixes to accept it.
An optimize action requires a reason, explicit inputs and nonempty authorized feedback/evidence or
role-granted materials (including the target/objective; measurements are not required).
No action uses null reason and empty input/evidence lists. Verify selects exactly
one nonempty Source. Implement explicitly selects one Source or task/default_start,
which means seed=null and uses Task.starter_source or a legal empty start. Never
implicitly select old C. A new Schema never inherits old C qualification. No action
is executed inside this writing call. Prefer keep/none when no useful action is
supported. Do not force a revision or treat ordinary implementation defects as
design changes. Ordinary feedback stays external; new REALIZATION.Feedback is
currently unknown unless the finite method can authenticate measured provenance.
Reviewer: independent readonly review of the frozen output/Q in review_subject.
The supplied review_subject.text_policy is the same original Template budget
projection used by the planner; review the frozen raw without rewriting it.
Return ks.writing.review.v1 with exactly format, subject (copy exact hashes),
items [{id,status,reason,location,quote}]. Cover every supplied obligation ID once,
quote its exact supplied location/text, and explain pass/fail/unknown concretely.
Quoted YAML decimals use {"yaml_decimal":"exact decimal"}; this is a role-data
quotation only. Published raw YAML is always authoritative.
Judge complete Task equivalence, bounded objective, coverage/dispatch, math and
precision, ownership, lifetimes, Require/Forbid, impact classification and action
evidence together. The quoted text identifies the obligation, not proof of its
truth. Missing facts or an obligation you cannot evaluate is unknown; structural
pass, planner self-review, hardware placeholders and reported hopes do not prove
semantic pass. Resource estimates may be conditional, but unmade design decisions
must not become coder choices. CPU structure does not qualify a model/GPU result.
"""


METHOD = b"""Schema writing, write-only mode
Write a concrete Schema for the supplied Task and current objective. Preserve
the actual Task constraints, acceptance, scope, target declarations and sources.
Use the supplied Template and documents as writing guidance. Make the math,
work scheduling and pipeline decisions understandable to the downstream coder.
References are context, not proof that this design works. Do not invent measured
benefits, hardware facts, verification or acceptance. State uncertainty plainly.
For optimize, read the complete current Schema and supplied feedback; keep the
original when no useful revision follows. Explain what changed and why, citing
the supplied feedback where relevant. Do not silently relax the task to fix code.
Return content as {"raw":[{"node":"root","text":"complete YAML text"}],
"reason":"brief rationale / changes and remaining uncertainties"}.
Additional nodes use safe node IDs and complete text; dispatch may contain the
directory's Markdown dispatch. Optional bindings may map coverage IDs to nodes.
To keep the current Schema, return {"decision":"keep","reason":"why"}.
There is no selection catalog to reproduce, locked Contract wrapper, review
answer, performance claim or next-action contract to complete. Existing Template
text limits are guidance; do not discard useful design content to satisfy them.
The host saves the exact output, checks only file safety and basic YAML
readability, and reports diagnostics. No independent reviewer or evaluator runs.
Writing completed means files were saved, not that their effect was evaluated.
"""


def freeze_rules(publication, *, writing_mode="write_only", guidance=None):
    """Freeze selected writing guidance; the full old checker is opt-in."""
    if writing_mode not in ("write_only", "strict"):
        raise ProtocolError("writing_mode must be write_only or strict")
    if writing_mode == "strict":
        original = freeze_template_rules(publication)
        rules = loads(read_regular(publication.stage, original["path"])[0])
    else:
        from .resources import resource
        paths = guidance if guidance is not None else (
            "DESIGN_SCHEMA.md", "OPTIMIZE_SCHEMA.md", "schema_docs/0.common.md", "schema_template/starter.yaml")
        rules = {"protocol": PROTOCOL, "kind": "rules", "files": [
            publication.put_bytes("rules/" + path, resource(path).read_bytes()) for path in paths], "tools": []}
    method = publication.put_bytes("writing/method.md", STRICT_METHOD if writing_mode == "strict" else METHOD)
    checker = publication.put_bytes("writing/checker.py", Path(__file__).read_bytes())
    helper = publication.put_bytes("writing/policy.py", Path(policy.__file__).read_bytes())
    from .evaluation import environment
    adapter = publication.put_bytes("writing/environment.py", Path(environment.__file__).read_bytes())
    rules = {**rules, "files": sorted([*rules["files"], method], key=lambda r: r["path"]),
             "tools": [*rules["tools"], {"name": "kernelschema.writing", "version": VERSION, "source": checker},
                       {"name": "kernelschema.writing.policy", "version": VERSION, "source": helper},
                       {"name": "kernelschema.writing.environment", "version": VERSION, "source": adapter}]}
    return publication.put_json("writing/rules.json", rules)


def backend_profile(runtime_profile, environment, *, writing_mode="write_only"):
    """Narrow a fixed K3 ClaudeProfile to the two actually installed workflows."""
    value = runtime_profile.backend_profile(environment, name="claude")
    value["parameters"]["writing_mode"] = writing_mode
    mode(value)
    value["capabilities"] = {"protocols": [PROTOCOL], "operations": ["design", "optimize"],
        "modes": [], "source_formats": [], "empty_start": False,
        "recovery": ["local-original-operation-settlement"], "token_enforcement": "unsupported", "extensions": []}
    return validate(value)


def template_contract(bundle, task):
    """Preview the public Task-derived locked Contract, without a model call."""
    return policy.template_contract(bundle, task)


def freeze_schema(publication, directory, *, task):
    """Snapshot an explicit writing input directory without Template acceptance."""
    import re
    directory = Path(directory).absolute()
    files = [directory] if directory.is_file() else sorted(path for path in directory.iterdir() if path.suffix in (".yaml", ".yml"))
    source_root = directory.parent if directory.is_file() else directory
    raw = []
    for path in files:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", path.stem):
            raise ProtocolError("invalid Schema node filename")
        data, _ = read_regular(source_root, path.name)
        data.decode("utf-8")
        raw.append({"node": path.stem, "file": publication.put_bytes("schema/" + path.name, data)})
    dispatch = None
    if (source_root / "dispatch.md").exists():
        data, _ = read_regular(source_root, "dispatch.md")
        dispatch = publication.put_bytes("schema/dispatch.md", data)
    return publication.put_json("schema/manifest.json", {"protocol": PROTOCOL, "kind": "schema",
        "task_sha256": task["sha256"], "raw": raw, "dispatch": dispatch, "design_revision": 0,
        "parent": None, "change": "new_task", "change_report": None,
        "extensions": {"information": {"writing_mode": "write_only"}}})


def write_task(operation, *, task, out, schema=None, references=(), feedback=(),
               runtime_profile=None, objective=None, environment=None):
    """File-oriented public entry point; no evaluator or handwritten Ref bundle.

    ``task`` is a UTF-8 task-description file. Optional files are copied in full.
    The existing finite runtime owns one planner call and its original receipts.
    """
    import tempfile
    from . import api
    from .artifacts import durable_write
    from .resources import resource

    if operation not in ("design", "optimize"):
        raise ProtocolError("writing operation must be design or optimize")
    if operation == "optimize" and schema is None:
        raise ProtocolError("optimize requires the current Schema path")
    if operation == "design" and (schema is not None or feedback):
        raise ProtocolError("design uses references for neighboring Schemas; feedback belongs to optimize")
    if runtime_profile is None:
        defaults = loads(resource("kernelschema/runtime/profiles/claude-code-2.1.316-qwen3.8-max-no-tools.v1.json").read_bytes())
        runtime_profile = ClaudeProfile.inspect(requested_model=defaults["requested_model"], effort="high",
            timeout_ms=600000, max_budget_usd="2.00", auth_mode="existing_cli_auth",
            reported_models=tuple(defaults["reported_models"]), builtin_plugins=tuple(defaults["builtin_plugins"]))
    elif not isinstance(runtime_profile, ClaudeProfile):
        settings = loads(Path(runtime_profile).read_bytes()) if isinstance(runtime_profile, (str, Path)) else dict(runtime_profile)
        for key in ("reported_models", "builtin_plugins"):
            settings[key] = tuple(settings.get(key, ()))
        runtime_profile = ClaudeProfile(**settings)
    task_path = Path(task).absolute()
    task_text = read_regular(task_path.parent, task_path.name)[0]
    task_text.decode("utf-8")
    with tempfile.TemporaryDirectory(prefix="ks-writing-input-") as temporary:
        root = Path(temporary)
        p = Publication(root)
        text_ref = p.put_bytes("task.md", task_text)
        env_ref = text_ref if environment is None else p.put_bytes("environment.md", Path(environment).read_bytes())
        task_ref = p.put_json("task.json", {"protocol": PROTOCOL, "kind": "task", "task_id": task_path.stem,
            "contract": text_ref, "acceptance": text_ref, "workload": text_ref, "environment": env_ref,
            "evaluator": None, "baseline_source": None, "starter_source": None})
        objective_ref = text_ref if objective is None else p.put_bytes("objective.md", Path(objective).read_bytes())
        entries = [{"id": "workflow_objective", "file": objective_ref, "roles": ["planner"],
            "origin": str(Path(objective).absolute()) if objective else str(task_path), "license": "caller supplied"}]
        for index, filename in enumerate(references):
            path = Path(filename).absolute()
            data, _ = read_regular(path.parent, path.name)
            data.decode("utf-8")
            entries.append({"id": f"reference_{index:04d}", "file": p.put_bytes(f"references/{index}-{path.name}", data),
                "roles": ["planner"], "origin": str(path), "license": "caller supplied"})
        materials_ref = p.put_json("materials.json", {"protocol": PROTOCOL, "kind": "materials", "entries": sorted(entries, key=lambda e: e["id"])})
        scope_ref = p.put_json("scope.json", {"protocol": PROTOCOL, "kind": "scope", "coverage_ids": ["task"],
            "bindings": [{"coverage_ids": ["task"], "node": "root", "variant": None}] if schema else []})
        schema_ref = freeze_schema(p, schema, task=task_ref) if schema else None
        feedback_refs = []
        for index, filename in enumerate(feedback):
            path = Path(filename).absolute()
            data, _ = read_regular(path.parent, path.name)
            data.decode("utf-8")
            feedback_refs.append(p.put_bytes(f"feedback/{index}-{path.name}", data))
        profile_ref = p.put_json("profile.json", backend_profile(runtime_profile, env_ref))
        value = request(operation=operation, request_id=operation + "-" + uuid4().hex, task=task_ref,
            rules=freeze_rules(p), materials=materials_ref, scope=scope_ref, profile=profile_ref, schema=schema_ref,
            feedback=feedback_refs, budget={"parent": None, "limits": {"calls": {"planner": 1, "reviewer": 0, "coder": 0, "extract": 0},
                "model_calls": 1, "evaluator_jobs": 0, "active_ms": runtime_profile.timeout_ms + 10000, "tokens": None}})
        request_ref = p.put_json("request.json", value)
        p.commit([request_ref])
        durable_write(root / "request.json", jcs(value))
        return api._invoke(operation, request=root / "request.json", out=out)


def request(*, operation, request_id, task, rules, materials, scope, budget, profile,
            schema=None, feedback=(), evidence=()):
    """Construct the same wire request consumed by kernelschema.api/CLI.

    All arguments are already-frozen Refs except the literal fields and budget;
    this helper does not scan, publish, launch or allocate a new owner.
    """
    if operation not in ("design", "optimize"):
        raise ProtocolError("writing request operation must be design or optimize")
    value = {"protocol": PROTOCOL, "operation": operation, "request_id": request_id,
             "task": task, "rules": rules, "materials": materials, "scope": scope,
             "budget": budget, "backend": {"name": "claude", "profile": profile}}
    if operation == "optimize":
        value.update(schema=schema, feedback=list(feedback), evidence=list(evidence))
    elif schema is not None or feedback or evidence:
        raise ProtocolError("design accepts no Schema/feedback/evidence request fields")
    return validate(value)


def _runtime(profile):
    settings = dict(profile["parameters"].get("claude", {}))
    if not settings:
        raise UnsupportedError("writing requires the explicitly frozen Claude no-tools profile")
    for name in ("reported_models", "builtin_plugins"):
        settings[name] = tuple(settings.get(name, ()))
    actual = ClaudeProfile(**settings)
    if actual.requested_model != profile["model"]:
        raise ProtocolError("writing model differs from fixed profile")
    return ClaudeRuntime(actual)


def _check(status="not_run", receipt=None, reason="not yet run"):
    return {"status": status, "receipt": receipt, "reason": reason}


class WritingBackend:
    def __init__(self, operation, *, runtime_factory=None, clock=None, checkpoint=None):
        if operation not in ("design", "optimize"):
            raise ProtocolError("invalid writing operation")
        self.operation = operation
        self.runtime_factory = runtime_factory or _runtime
        self.clock = clock
        self.checkpoint = checkpoint or (lambda label, context: None)

    def capabilities(self, profile):
        return {"protocols": [PROTOCOL], "operations": [self.operation], "modes": [],
                "source_formats": [], "empty_start": False, "recovery": ["local-original-operation-settlement"],
                "token_enforcement": "unsupported", "extensions": []}

    def identity(self, profile):
        return {"backend": "claude", "operation": self.operation, "version": VERSION,
                "runtime": self.runtime_factory(profile).identity(),
                "implementation": {"writing": sha256(Path(__file__).read_bytes()), "policy": sha256(Path(policy.__file__).read_bytes()),
                                   "budget": sha256(Path(__file__).with_name("budget.py").read_bytes()),
                                   "task_adapter": sha256((Path(__file__).parent / "evaluation/task.py").read_bytes()),
                                   "environment_adapter": sha256((Path(__file__).parent / "evaluation/environment.py").read_bytes()),
                                   "template_api": sha256(Path(template_api.__file__).read_bytes()), "method": sha256(STRICT_METHOD if mode(profile) == "strict" else METHOD)}}

    def _account(self, context, *, create=False):
        return OwnerAccount.for_context(context, create=create, **({"clock": self.clock} if self.clock else {}))

    def _rules(self, context):
        rules = context.objects["rules"]
        method = STRICT_METHOD if mode(context.objects["profile"]) == "strict" else METHOD
        matches = [ref for ref in rules["files"] if ref["sha256"] == sha256(method)]
        if len(matches) != 1 or context.bundle.read(matches[0]) != method:
            raise ProtocolError("Rules must freeze exactly one current finite writing method")
        for name, path in (("kernelschema.writing", Path(__file__)), ("kernelschema.writing.policy", Path(policy.__file__)),
                           ("kernelschema.writing.environment", Path(__file__).parent / "evaluation/environment.py")):
            found = [t for t in rules["tools"] if t["name"] == name]
            if len(found) != 1 or found[0]["version"] != VERSION or context.bundle.read(found[0]["source"]) != path.read_bytes():
                raise ProtocolError("Rules must pin the actual writing checker and policy")
        if mode(context.objects["profile"]) == "write_only":
            return matches[0]
        from .resources import resource, rule_files
        available = {r["sha256"] for r in rules["files"]}
        implementation = (path for path in resource("schema_template/tools/schema.py").parent.rglob("*.py") if "tests" not in path.parts)
        if not all(sha256(path.read_bytes()) in available for path in (*rule_files(), *implementation)):
            raise ProtocolError("writing Rules must retain current original docs and all Template implementation bytes")
        template = [t for t in rules["tools"] if t["name"] == "template"]
        if len(template) != 1 or context.bundle.read(template[0]["source"]) != resource("schema_template/tools/schema.py").read_bytes():
            raise ProtocolError("writing Rules must retain the original Template tool")
        return matches[0]

    def _save(self, context, state, label):
        context.save_state("writing", state)
        self.checkpoint(label, context)

    def _validate(self, context, account, state):
        if state.get("version") != VERSION or state.get("operation") != self.operation:
            raise ProtocolError("writing state identity/version mismatch")
        operations = account.operations(context.run_id)
        declared = {op["id"]: op for op in state["operations"]}
        if not set(operations) <= declared.keys():
            raise ProtocolError("sole owner has an operation absent from writing state")
        for op in declared.values():
            if op["reservation"] is not None and operations.get(op["id"], {}).get("reservation") != op["reservation"]:
                raise ProtocolError("writing reservation differs from original sole owner")
        for intent in context.intents():
            op = declared.get(intent["operation_id"])
            if op is None or operations.get(op["id"], {}).get("reservation") != intent["reservation"]:
                raise ProtocolError("original writing intent absent from owner/state")
        for key, kind in (("schema", "schema"), ("scope", "scope"), ("change_report", "change_report"), ("review", "writing_review")):
            if state.get(key):
                context.bundle.document(state[key], kind)
        for key in ("planner_input", "draft_ref", "facts_ref", "review_input"):
            if state.get(key):
                context.bundle.read(state[key])
        if state.get("draft"):
            retained = loads(context.bundle.read(state["draft_ref"]))
            if state["draft"] != retained["content"]:
                raise ProtocolError("writing draft differs from its original retained output")
        if state.get("facts_ref"):
            if state["facts"] != loads(context.bundle.read(state["facts_ref"])):
                raise ProtocolError("writing precheck facts differ from their frozen original")
            expected = {entry["node"]: entry["text"].encode() for entry in state["raw"]}
            retained = {Path(ref["path"]).stem: context.bundle.read(ref) for ref in state["preview_raw"]}
            if expected != retained:
                raise ProtocolError("writing raw differs from its frozen normalized preview")
            if mode(context.objects["profile"]) == "write_only":
                expected = policy.write_only_files(context, state["draft"])
                if expected != (state["raw"], state["dispatch"], state["output_scope"], state["facts"]):
                    raise ProtocolError("writing files differ from the original retained output")
                return
            raw, dispatch, scope, resolved = policy.normalize(context, state["draft"])
            if (raw, dispatch, scope) != (state["raw"], state["dispatch"], state["output_scope"]):
                raise ProtocolError("writing normalized output differs from the original draft")
            obligations = policy.inventory(context, resolved, scope, state["draft"], state["facts"])
            if obligations != state["obligations"]:
                raise ProtocolError("writing obligations differ from the original output")
        if state.get("review_input"):
            original = loads(context.bundle.read(state["review_input"]))
            expected = {"request_sha256": context.request_sha256, "task_sha256": context.request["task"]["sha256"],
                        "schema_sha256": state["schema"]["sha256"], "scope_sha256": state["scope"]["sha256"], "rules_sha256": context.request["rules"]["sha256"]}
            if state["subject"] != expected or original["subject"] != expected or original["obligations"] != state["obligations"]:
                raise ProtocolError("writing review input differs from its original frozen subjects")

    def validate_resume(self, context):
        account = self._account(context)
        if not (context.root / ".runtime/state/writing.json").exists():
            if account.operations(context.run_id) or context.intents():
                raise ProtocolError("lost writing state cannot restart original charges")
            return
        state = context.load_state("writing")
        self._validate(context, account, state)
        runtime = self.runtime_factory(context.objects["profile"])
        for op in state["operations"]:
            if op["admitted"]:
                runtime.reopen(context, op["id"])

    def _initial(self, context, account):
        if (context.root / ".runtime/state/writing.json").exists():
            return context.load_state("writing")
        if context.intents() or account.operations(context.run_id):
            raise ProtocolError("lost writing state with charges cannot be recreated")
        state = {"version": VERSION, "operation": self.operation, "phase": "prepare", "operations": [],
                 "schema": None, "scope": None, "change_report": None, "review": None,
                 "writing_checks": {k: _check() for k in ("structure", "permissions", "semantics", "publication")},
                 "status": "partial", "recovery": "none", "stop_reason": "writing not complete", "diagnostics": [],
                 "ids": {key: uuid4().hex for key in ("input", "draft", "preview", "change", "candidate", "review_input", "review", "review_facts")}}
        self._save(context, state, "writing_intent")
        return state

    def _prepare(self, context, state):
        if state["phase"] != "prepare":
            return
        if mode(context.objects["profile"]) == "write_only":
            body = {"mode": "write_only", "output": {"raw": [{"node": "root", "text": "complete YAML"}],
                    "reason": "brief changes, rationale and uncertainties"},
                    "note": "Read the supplied Task, current Schema and explicit references. No review or evaluation is requested."}
            p = Publication(context.root, state["ids"]["input"])
            state["planner_input"] = p.put_bytes("writing-input.json", jcs(body))
            p.commit([state["planner_input"]])
            state["phase"] = "planner"
            self._save(context, state, "materials_fixed")
            return
        # Full Task support is checked before any charge; no evaluator execution.
        from .evaluation import RowSoftmaxTask
        RowSoftmaxTask(context.bundle, context.request["task"]).require_evaluation_scope(context.request["scope"])
        body = {"format": policy.DRAFT_FORMAT, "locked_contract": policy.wire_value(policy.contract_for(context)),
                "action_catalog": policy.catalog(context), "objective_id": "workflow_objective",
                "selection_catalog": policy.selection_catalog(context), "text_policy": policy.text_policy(context),
                "note": "All permitted reference bytes arrive via role Materials or explicit request evidence. No repository search is authorized."}
        p = Publication(context.root, state["ids"]["input"])
        state["planner_input"] = p.put_bytes("writing-input.json", jcs(body))
        p.commit([state["planner_input"]])
        state["phase"] = "planner"
        self._save(context, state, "materials_fixed")

    def _deliveries(self, context, state, role):
        result = [RoleDelivery("writing_input" if role == "planner" else "review_subject",
                               state["planner_input"] if role == "planner" else state["review_input"], (role,))]
        for key in ("workload", "environment"):
            result.append(RoleDelivery("task_" + key, context.objects["task"][key], (role,)))
        for key in ("feedback", "evidence"):
            for index, ref in enumerate(context.request.get(key, [])):
                kind = "source" if policy._is_source(context, ref) else "text"
                result.append(RoleDelivery(f"{key}_{index}", ref, (role,), kind))
        return tuple(result)

    def _operate(self, context, state, account, runtime, role):
        op = next((o for o in state["operations"] if o["role"] == role), None)
        if op is None:
            op = {"id": f"{context.run_id}_{role}", "role": role, "purpose": self.operation + " finite writing " + role,
                  "reservation": None, "admitted": False, "finished": False}
            state["operations"].append(op)
            self._save(context, state, role + "_planned")
        if op["reservation"] is None:
            # Recover a reserve->state crash using the original owner record.
            record = account.operations(context.run_id).get(op["id"])
            if record:
                op["reservation"] = record["reservation"]
            else:
                minimum = context.objects["profile"]["parameters"].get("writing_minimum_active_ms", {"planner": 1, "reviewer": 1000})
                write_only = mode(context.objects["profile"]) == "write_only"
                allowed = ({"planner"}, {"planner", "reviewer"}) if write_only else ({"planner", "reviewer"},)
                if not isinstance(minimum, dict) or set(minimum) not in allowed or any(type(v) is not int or v <= 0 for v in minimum.values()):
                    raise ProtocolError("writing_minimum_active_ms requires positive planner/reviewer intervals")
                review_floor = cost(role="reviewer", calls=1, active_ms=minimum["reviewer"]) if role == "planner" and not write_only else cost()
                finish_floor = _finish_floor(context.objects["profile"])
                floor = {key: finish_floor[key] + review_floor[key] for key in review_floor}
                op["reservation"] = account.reserve(context.run_id, op["id"], role=role, purpose=op["purpose"],
                    amount=cost(role=role, calls=1, active_ms=minimum[role]), floor=floor,
                    native_limits=context.objects["profile"]["native_limits"].get("workflow_limits"))
            self._save(context, state, role + "_reserved")
        handle = runtime.admit(context, operation_id=op["id"], role=role, purpose=op["purpose"], source=None,
                               reservation=op["reservation"], deliveries=self._deliveries(context, state, role))
        op["admitted"] = True
        self._save(context, state, role + "_admitted")
        original = account.operations(context.run_id)[op["id"]]
        receipt = runtime.settle(context, handle) if original["launched"] else runtime.invoke(context, handle, authorize=lambda _: account.authorize(op["reservation"]))
        self.checkpoint(role + "_returned", context)
        if not receipt.terminated:
            account.mark_unknown(op["id"], "original writing operation has no known terminal")
            return None
        account.absorb(context.run_id, receipt.usage_row(), terminated=True,
                       lower_bounds={"calls": receipt.facts.get("calls_lower_bound", receipt.facts.get("calls") or 0)})
        op["receipt"] = receipt.receipt
        self._save(context, state, role + "_settled")
        return receipt

    def _draft(self, context, state, receipt):
        op = next(o for o in state["operations"] if o["role"] == "planner")
        content = receipt.output.content if receipt.ok and receipt.output else None
        state["call_artifacts"] = receipt.evidence_refs()
        p = Publication(context.root, state["ids"]["draft"])
        state["draft_ref"] = p.put_bytes("draft.json", jcs({"format": "ks.writing.retained-draft.v1", "content": content,
                                                           "receipt": receipt.receipt, "ok": receipt.ok}))
        state["retained_files"] = []
        if isinstance(content, dict) and isinstance(content.get("raw"), list):
            for index, entry in enumerate(content["raw"]):
                if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                    state["retained_files"].append(p.put_bytes(f"original/{index}.txt", entry["text"].encode("utf-8", errors="backslashreplace")))
        p.commit([state["draft_ref"], *state["retained_files"], *receipt.evidence_refs()], subjects=[context.request_ref])
        self._save(context, state, "draft_retained")
        try:
            if content is None:
                raise ProtocolError("planner returned no complete valid writing proposal")
            state["draft"] = content if mode(context.objects["profile"]) == "write_only" else policy.parse_draft(context, content)
        except ProtocolError as error:
            self._fail(state, str(error))
        else:
            state["phase"] = "normalize"
        op["finished"] = True
        op["draft"] = state["draft_ref"]
        self._save(context, state, "planner_finished")

    def _fail(self, state, reason):
        state.update(phase="done", status="failed", recovery="new_request", stop_reason=reason)
        state["diagnostics"].append(reason)
        state["writing_checks"] = {k: _check("unknown", reason="No legally publishable Schema: " + reason) for k in state["writing_checks"]}

    def _normalize(self, context, state):
        if mode(context.objects["profile"]) == "write_only":
            try:
                raw, dispatch, scope, facts = policy.write_only_files(context, state["draft"])
            except (ProtocolError, UnicodeError) as error:
                self._fail(state, str(error))
                self._save(context, state, "normalization_failed")
                return
            preview = Publication(context.root, state["ids"]["preview"])
            references = [preview.put_bytes(r["node"] + ".yaml", r["text"].encode()) for r in raw]
            state["facts_ref"] = preview.put_bytes("diagnostics.json", jcs(facts))
            preview.commit([state["facts_ref"], *references])
            state.update(raw=raw, dispatch=dispatch, output_scope=scope, facts=facts,
                         preview_raw=references, phase="change")
            state["diagnostics"].extend(facts["diagnostics"])
            self._save(context, state, "precheck_frozen")
            return
        try:
            raw, dispatch, scope, resolved = policy.normalize(context, state["draft"])
            facts = policy.precheck(context, state["draft"], raw, dispatch, scope, resolved)
        except (ProtocolError, template_api.tools().SchemaError, template_api.tools().ToolError, UnicodeError) as error:
            self._fail(state, str(error))
            self._save(context, state, "normalization_failed")
            return
        preview = Publication(context.root, state["ids"]["preview"])
        references = [preview.put_bytes(r["node"] + ".yaml", r["text"].encode()) for r in raw]
        state["facts_ref"] = preview.put_bytes("precheck.json", jcs(facts))
        preview.commit([state["facts_ref"], *references])
        state.update(raw=raw, dispatch=dispatch, output_scope=scope, facts=facts,
                     obligations=policy.inventory(context, resolved, scope, state["draft"], facts), preview_raw=references, phase="change")
        self._save(context, state, "precheck_frozen")

    def _publish(self, context, state):
        if state["phase"] == "change":
            p = Publication(context.root, state["ids"]["change"])
            if mode(context.objects["profile"]) == "write_only":
                state["rationale"] = p.put_bytes("rationale.md", (state["facts"]["changes"][0]["reason"] + "\n\nEffects have not been evaluated.\n").encode())
            state["change_report"] = p.put_json("change_report.json", {
                "protocol": PROTOCOL, "kind": "change_report", "operation_id": context.run_id + "_planner",
                "subject": {"request_sha256": context.request_sha256, "task_sha256": context.request["task"]["sha256"]},
                "changes": state["facts"]["changes"], "raw": state["preview_raw"],
                "evidence": [context.request_ref, state["facts_ref"], state["draft_ref"]]})
            p.commit([state["change_report"], *([state["rationale"]] if state.get("rationale") else [])], subjects=[context.request_ref])
            state["phase"] = "publish"
            self._save(context, state, "change_report_frozen")
        if state["phase"] == "publish":
            p = Publication(context.root, state["ids"]["candidate"])
            previous = context.objects.get("schema")
            exact = previous is not None and state["output_scope"] == context.objects["scope"]
            if exact:
                old_raw = [{"node": r["node"], "text": context.bundle.read(r["file"]).decode()} for r in previous["raw"]]
                old_dispatch = context.bundle.read(previous["dispatch"]).decode() if previous["dispatch"] else None
                exact = old_raw == state["raw"] and old_dispatch == state["dispatch"]
            if state["draft"].get("decision") == "keep" or exact:
                state["schema"], state["scope"] = context.request["schema"], context.request["scope"]
            else:
                raw = [{"node": r["node"], "file": p.put_bytes("raw/" + r["node"] + ".yaml", r["text"].encode())} for r in state["raw"]]
                dispatch = p.put_bytes("raw/dispatch.md", state["dispatch"].encode()) if state["dispatch"] is not None else None
                change = state["facts"]["change"]
                previous = context.objects.get("schema")
                state["schema"] = p.put_json("schema.json", {"protocol": PROTOCOL, "kind": "schema", "task_sha256": context.request["task"]["sha256"],
                    "raw": raw, "dispatch": dispatch, "change": change, "design_revision": previous["design_revision"] + (change == "new_revision") if previous else 0,
                    "parent": context.request.get("schema"), "change_report": state["change_report"],
                    **({"extensions": {"information": {"writing_mode": "write_only"}}} if mode(context.objects["profile"]) == "write_only" else {})})
                state["scope"] = p.put_json("scope.json", state["output_scope"])
            p.commit([state["schema"], state["scope"], state["change_report"]], subjects=[context.request_ref])
            if mode(context.objects["profile"]) == "write_only":
                state["writing_checks"] = {
                    "structure": _check("pass" if state["facts"]["readable"] else "fail", state["facts_ref"], reason="Basic YAML readability only; Template and design semantics not evaluated."),
                    "permissions": _check("not_run", reason="No semantic or Contract-equivalence review requested."),
                    "semantics": _check("not_run", reason="Write-only; correctness and performance not evaluated."),
                    "publication": _check("pass", state["change_report"], reason="Exact Schema text, rationale and call records saved.")}
                readable = state["facts"]["readable"]
                state.update(phase="done", status="completed" if readable else "partial", recovery="none" if readable else "new_request",
                             stop_reason="Schema files written; effects not evaluated" if readable else "Schema text saved with YAML readability diagnostics")
                self._save(context, state, "schema_frozen")
                return
            state["phase"] = "review_prepare"
            self._save(context, state, "schema_frozen")
        if state["phase"] == "review_prepare":
            subject = {"request_sha256": context.request_sha256, "task_sha256": context.request["task"]["sha256"],
                       "schema_sha256": state["schema"]["sha256"], "scope_sha256": state["scope"]["sha256"], "rules_sha256": context.request["rules"]["sha256"]}
            p = Publication(context.root, state["ids"]["review_input"])
            frozen = context.bundle.document(state["schema"], "schema")
            output_raw = [{"node": r["node"], "text": context.bundle.read(r["file"]).decode()} for r in frozen["raw"]]
            output_dispatch = context.bundle.read(frozen["dispatch"]).decode() if frozen["dispatch"] else None
            body = {"format": policy.REVIEW_FORMAT, "subject": subject, "schema": state["schema"], "scope": state["scope"],
                    "raw": output_raw, "dispatch": output_dispatch, "obligations": state["obligations"], "precheck": state["facts"],
                    "draft": state["draft"], "action_catalog": policy.catalog(context),
                    "text_policy": policy.text_policy(context)}
            state["review_input"] = p.put_bytes("review-input.json", jcs(body))
            p.commit([state["review_input"], state["schema"], state["scope"], state["change_report"]], subjects=[context.request_ref])
            state["subject"] = subject
            state["phase"] = "reviewer"
            self._save(context, state, "review_input_frozen")

    def _review(self, context, state, receipt=None):
        content = receipt.output.content if receipt and receipt.ok and receipt.output else None
        semantics, items = policy.review_answers(content, state["subject"], state["obligations"])
        if state["facts"]["scope"]["status"] != "pass":
            semantics = "fail" if state["facts"]["scope"]["status"] == "fail" else "unknown"
        p = Publication(context.root, state["ids"]["review_facts"])
        evidence = p.put_bytes("review-facts.json", jcs({"items": items, "finite_scope": state["facts"]["scope"],
            "model_facts": receipt.facts if receipt else None, "limitations": ["Finite writing review; no GPU/model/Task qualification is implied by CPU tests.",
            "No subsequent request is admitted and no C/fidelity qualification is inherited."]}))
        p.commit([evidence, *receipt.evidence_refs()] if receipt else [evidence], subjects=[context.request_ref])
        statuses = {"structure": state["facts"]["structure"], "permissions": state["facts"]["permissions"], "semantics": semantics, "publication": "pass"}
        checks = {k: _check(v, evidence, "host checks and independently located frozen-output review; " + "; ".join(state["facts"]["failures"] + state["facts"]["unknowns"])) for k, v in statuses.items()}
        p = Publication(context.root, state["ids"]["review"])
        state["review"] = p.put_json("writing_review.json", {"protocol": PROTOCOL, "kind": "writing_review", "operation_id": context.run_id + "_reviewer",
            "subject": state["subject"], "checks": checks, "evidence": [context.request_ref, state["schema"], state["scope"], state["change_report"], state["review_input"], evidence],
            "usage_receipts": [receipt.receipt] if receipt and receipt.receipt else []})
        p.commit([state["review"]], subjects=[context.request_ref])
        state["writing_checks"] = {k: _check(v, state["review"], checks[k]["reason"]) for k, v in statuses.items()}
        ok = all(v == "pass" for v in statuses.values())
        state.update(phase="done", status="completed" if ok else "partial", recovery="none" if ok else "new_request",
                     stop_reason="writing completed; output is not subsequent-request admission" if ok else "immutable candidate retained with failed/unknown writing checks")
        op = next((o for o in state["operations"] if o["role"] == "reviewer"), None)
        if receipt and op:
            op["finished"] = True
            op["review"] = state["review"]
        self._save(context, state, "review_frozen")

    def execute(self, context, *, continuing=False):
        fd = os.open(context.root / ".runtime/writing-loop.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return self._execute(context, continuing=continuing)
        finally:
            os.close(fd)

    def _execute(self, context, *, continuing):
        _finish_floor(context.objects["profile"])
        self._rules(context)
        account = self._account(context, create=not continuing)
        native = context.objects["profile"]["native_limits"].get("workflow_limits")
        if native is not None:
            validate_limits(native)
        if account.descriptor["limits"]["tokens"] is not None or native is not None and native["tokens"] is not None:
            raise UnsupportedError("writing runtime cannot enforce finite parent/native token budgets")
        state = self._initial(context, account)
        self._validate(context, account, state)
        if state["phase"] == "done":
            return self._result(context, state, account)
        runtime = self.runtime_factory(context.objects["profile"])
        account.begin(context.run_id)
        try:
            self._prepare(context, state)
            while state["phase"] != "done":
                phase = state["phase"]
                if phase in ("planner", "reviewer"):
                    receipt = self._operate(context, state, account, runtime, phase)
                    if receipt is None:
                        break
                    if phase == "planner":
                        self._draft(context, state, receipt)
                    else:
                        self._review(context, state, receipt)
                elif phase == "normalize":
                    self._normalize(context, state)
                else:
                    self._publish(context, state)
        except (BudgetDenied, AdmissionDenied) as error:
            for op in state["operations"]:
                owned = account.operations(context.run_id).get(op["id"])
                if owned and not owned["launched"]:
                    account.release_unstarted(op["id"])
            if state.get("subject"):
                self._review(context, state)
            state.update(phase="done", status="partial", recovery="none", stop_reason=str(error))
        finally:
            account.end(context.run_id)
        return self._result(context, state, account)

    def _result(self, context, state, account):
        if state["phase"] == "done" and state.get("result") is not None and not account.fences():
            return state["result"]
        usage = account.usage(context.run_id, Publication(context.root))
        if usage["unsettled"]:
            state.update(status="blocked", recovery="settle_only", stop_reason="original operation or constrained owner balance unresolved")
        result = {"protocol": PROTOCOL, "operation": self.operation, "run_id": context.run_id, "request_id": context.request["request_id"],
                  "request_sha256": context.request_sha256, **{k: state[k] for k in ("status", "recovery", "stop_reason", "schema", "scope", "writing_checks")},
                  "usage": usage, "feedback": [], "extensions": {"information": {"backend": VERSION, "diagnostics": state["diagnostics"],
                    "writing_mode": mode(context.objects["profile"]),
                    "retained_files": state.get("retained_files", []), "rationale": state.get("rationale"),
                    "call_artifacts": state.get("call_artifacts", []),
                    "candidate_draft": state.get("draft_ref"), "change_report": state.get("change_report"), "admitted_request": None,
                    "qualification": "Schema writing only; effects not evaluated." if mode(context.objects["profile"]) == "write_only" else "Writing checks only; no code or execution qualification."}}}
        if self.operation == "optimize":
            draft = state.get("draft", {})
            authorized = mode(context.objects["profile"]) == "strict" and state["status"] == "completed" and not usage["unsettled"]
            action = draft.get("next_action", "none") if authorized else "none"
            refs = policy.catalog(context)
            result.update(decision="revise" if state.get("facts", {}).get("change") == "new_revision" else "keep",
                          reason=(state.get("facts", {}).get("changes") or [{"reason": state["stop_reason"]}])[0]["reason"] if mode(context.objects["profile"]) == "write_only" else draft.get("reason", state["stop_reason"]), next_action=action,
                          action_reason=draft.get("action_reason") if action != "none" else None,
                          action_inputs=[state["schema"], state["scope"], *[refs[k] for k in draft["action_inputs"]]] if action != "none" else [],
                          action_evidence=[refs[k] for k in draft["action_evidence"]] if action != "none" else [])
        if mode(context.objects["profile"]) == "write_only" and state["schema"]:
            manifest = context.bundle.document(state["schema"], "schema")
            result["extensions"]["information"]["schema_directory"] = str(Path(manifest["raw"][0]["file"]["path"]).parent)
        if state["phase"] == "done" and not usage["unsettled"]:
            state["result"] = result
        self._save(context, state, "result_ready")
        return result
