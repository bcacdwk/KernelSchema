"""One finite design/optimize kernel: Task -> draft -> freeze -> readonly review.

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
from .contracts import PROTOCOL, ProtocolError, UnsupportedError, jcs, loads, sha256, validate, limits as validate_limits
from .resources import freeze_rules as freeze_template_rules
from .runtime import AdmissionDenied, ClaudeProfile, ClaudeRuntime, RoleDelivery

VERSION = "writing.v1"
METHOD = b"""Finite writing method v1
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
Initial design uses revise. Keep has raw=[], dispatch=null, bindings=null.
A revision includes every raw node, root Contract and current coverage bindings.
An action requires a reason, explicit inputs and authorized feedback/evidence or
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


def freeze_rules(publication):
    """Freeze original docs/Template and this method/checker into one Rules Ref."""
    original = freeze_template_rules(publication)
    rules = loads(read_regular(publication.stage, original["path"])[0])
    method = publication.put_bytes("writing/method.md", METHOD)
    checker = publication.put_bytes("writing/checker.py", Path(__file__).read_bytes())
    helper = publication.put_bytes("writing/policy.py", Path(policy.__file__).read_bytes())
    from .evaluation import environment
    adapter = publication.put_bytes("writing/environment.py", Path(environment.__file__).read_bytes())
    rules = {**rules, "files": sorted([*rules["files"], method], key=lambda r: r["path"]),
             "tools": [*rules["tools"], {"name": "kernelschema.writing", "version": VERSION, "source": checker},
                       {"name": "kernelschema.writing.policy", "version": VERSION, "source": helper},
                       {"name": "kernelschema.writing.environment", "version": VERSION, "source": adapter}]}
    return publication.put_json("writing/rules.json", rules)


def backend_profile(runtime_profile, environment):
    """Narrow a fixed K3 ClaudeProfile to the two actually installed workflows."""
    value = runtime_profile.backend_profile(environment, name="claude")
    value["capabilities"] = {"protocols": [PROTOCOL], "operations": ["design", "optimize"],
        "modes": [], "source_formats": [], "empty_start": False,
        "recovery": ["local-original-operation-settlement"], "token_enforcement": "unsupported", "extensions": []}
    return validate(value)


def template_contract(bundle, task):
    """Preview the public Task-derived locked Contract, without a model call."""
    return policy.template_contract(bundle, task)


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
                                   "template_api": sha256(Path(template_api.__file__).read_bytes()), "method": sha256(METHOD)}}

    def _account(self, context, *, create=False):
        return OwnerAccount.for_context(context, create=create, **({"clock": self.clock} if self.clock else {}))

    def _rules(self, context):
        rules = context.objects["rules"]
        matches = [ref for ref in rules["files"] if ref["sha256"] == sha256(METHOD)]
        if len(matches) != 1 or context.bundle.read(matches[0]) != METHOD:
            raise ProtocolError("Rules must freeze exactly one current finite writing method")
        for name, path in (("kernelschema.writing", Path(__file__)), ("kernelschema.writing.policy", Path(policy.__file__)),
                           ("kernelschema.writing.environment", Path(__file__).parent / "evaluation/environment.py")):
            found = [t for t in rules["tools"] if t["name"] == name]
            if len(found) != 1 or found[0]["version"] != VERSION or context.bundle.read(found[0]["source"]) != path.read_bytes():
                raise ProtocolError("Rules must pin the actual writing checker and policy")
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
        # Full Task support is checked before any charge; no evaluator execution.
        from .evaluation import RowSoftmaxTask
        RowSoftmaxTask(context.bundle, context.request["task"]).require_evaluation_scope(context.request["scope"])
        body = {"format": policy.DRAFT_FORMAT, "locked_contract": policy.wire_value(policy.contract_for(context)),
                "action_catalog": policy.catalog(context), "objective_id": "workflow_objective",
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
                if not isinstance(minimum, dict) or set(minimum) != {"planner", "reviewer"} or any(type(v) is not int or v <= 0 for v in minimum.values()):
                    raise ProtocolError("writing_minimum_active_ms requires positive planner/reviewer intervals")
                floor = cost(role="reviewer", calls=1, active_ms=minimum["reviewer"]) if role == "planner" else cost()
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
        p = Publication(context.root, state["ids"]["draft"])
        state["draft_ref"] = p.put_bytes("draft.json", jcs({"format": "ks.writing.retained-draft.v1", "content": content,
                                                           "receipt": receipt.receipt, "ok": receipt.ok}))
        p.commit([state["draft_ref"], *receipt.evidence_refs()], subjects=[context.request_ref])
        self._save(context, state, "draft_retained")
        try:
            if content is None:
                raise ProtocolError("planner returned no complete valid writing proposal")
            state["draft"] = policy.parse_draft(context, content)
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
            state["change_report"] = p.put_json("change_report.json", {
                "protocol": PROTOCOL, "kind": "change_report", "operation_id": context.run_id + "_planner",
                "subject": {"request_sha256": context.request_sha256, "task_sha256": context.request["task"]["sha256"]},
                "changes": state["facts"]["changes"], "raw": state["preview_raw"],
                "evidence": [context.request_ref, state["facts_ref"], state["draft_ref"]]})
            p.commit([state["change_report"]], subjects=[context.request_ref])
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
            if state["draft"]["decision"] == "keep" or exact:
                state["schema"], state["scope"] = context.request["schema"], context.request["scope"]
            else:
                raw = [{"node": r["node"], "file": p.put_bytes("raw/" + r["node"] + ".yaml", r["text"].encode())} for r in state["raw"]]
                dispatch = p.put_bytes("raw/dispatch.md", state["dispatch"].encode()) if state["dispatch"] is not None else None
                change = state["facts"]["change"]
                previous = context.objects.get("schema")
                state["schema"] = p.put_json("schema.json", {"protocol": PROTOCOL, "kind": "schema", "task_sha256": context.request["task"]["sha256"],
                    "raw": raw, "dispatch": dispatch, "change": change, "design_revision": previous["design_revision"] + (change == "new_revision") if previous else 0,
                    "parent": context.request.get("schema"), "change_report": state["change_report"]})
                state["scope"] = p.put_json("scope.json", state["output_scope"])
            p.commit([state["schema"], state["scope"], state["change_report"]], subjects=[context.request_ref])
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
                    "draft": state["draft"], "action_catalog": policy.catalog(context)}
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
                    "candidate_draft": state.get("draft_ref"), "change_report": state.get("change_report"), "admitted_request": None,
                    "qualification": "Writing checks only; no code or execution qualification."}}}
        if self.operation == "optimize":
            draft = state.get("draft", {})
            authorized = state["status"] == "completed" and not usage["unsettled"]
            action = draft.get("next_action", "none") if authorized else "none"
            refs = policy.catalog(context)
            result.update(decision="revise" if state.get("facts", {}).get("change") == "new_revision" else "keep",
                          reason=draft.get("reason", state["stop_reason"]), next_action=action,
                          action_reason=draft.get("action_reason") if action != "none" else None,
                          action_inputs=[state["schema"], state["scope"], *[refs[k] for k in draft["action_inputs"]]] if action != "none" else [],
                          action_evidence=[refs[k] for k in draft["action_evidence"]] if action != "none" else [])
        if state["phase"] == "done" and not usage["unsettled"]:
            state["result"] = result
        self._save(context, state, "result_ready")
        return result
