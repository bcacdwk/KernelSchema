"""Bounded fixed-Q coder -> exact C -> full Task -> independent review loop.

The default backend uses only the requested K3 profile. An installed, explicitly
pinned evaluation channel is required for real execution; there is no pretend
CPU/CUDA success path. Tests inject synthetic transports into this same loop.
"""
from __future__ import annotations

import importlib
import fcntl
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from .. import fidelity
from ..artifacts import Publication, _rename_new, durable_mkdir, fsync_directory, install_source, read_regular, verify_source_directory
from ..budget import BudgetDenied, OwnerAccount, cost
from ..contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, loads, sha256, limits as validate_limits
from ..evaluation.dispatch import task_adapter, stage, verify_stage
from ..runtime import AdmissionDenied, ClaudeProfile, ClaudeRuntime, RoleDelivery

VERSION = "reference.v1"


def _runtime(profile):
    settings = dict(profile["parameters"].get("claude", {}))
    if not settings:
        raise UnsupportedError("reference requires the explicit requested Claude no-tools profile")
    for key in ("reported_models", "builtin_plugins"):
        settings[key] = tuple(settings.get(key, ()))
    actual = ClaudeProfile(**settings)
    if actual.requested_model != profile["model"]:
        raise ProtocolError("runtime model differs from fixed profile")
    return ClaudeRuntime(actual)


def _evaluation_factory(profile):
    selection = profile["parameters"].get("evaluation")
    if selection is None:
        return None, None
    if not isinstance(selection, dict) or set(selection) != {"factory", "implementation_sha256", "parameters"}:
        raise ProtocolError("evaluation needs an explicit installed factory, source hash and parameters")
    module_name, separator, name = selection["factory"].partition(":")
    if not separator or not module_name or not name.isidentifier():
        raise ProtocolError("evaluation factory must be module:function")
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise UnsupportedError("configured evaluation factory has no installed implementation file")
    actual = sha256(Path(spec.origin).read_bytes())
    if actual != selection["implementation_sha256"]:
        raise ProtocolError("installed evaluation channel differs from pinned profile")
    module = importlib.import_module(module_name)
    return getattr(module, name), actual


def _check(status="not_run", receipt=None, reason="not yet run"):
    return {"status": status, "receipt": receipt, "reason": reason}


class ReferenceBackend:
    def __init__(self, *, runtime_factory=None, evaluation_factory=None, clock=None, checkpoint=None):
        self.runtime_factory = runtime_factory or _runtime
        self.evaluation_factory = evaluation_factory
        self.clock = clock
        self.checkpoint = checkpoint or (lambda label, context: None)

    def capabilities(self, profile):
        return {"protocols": [PROTOCOL], "operations": ["implement"], "modes": ["code", "verify_only"],
                "source_formats": ["single_file", "python_tree"], "empty_start": True,
                "recovery": ["local-original-operation-settlement", "new-session-from-explicit-source"],
                "token_enforcement": "unsupported", "extensions": []}

    def identity(self, profile):
        runtime = self.runtime_factory(profile)
        _, evaluator_hash = _evaluation_factory(profile) if self.evaluation_factory is None else (None, "explicit-test-injection")
        return {"backend": "reference", "version": VERSION, "runtime": runtime.identity(), "evaluation": evaluator_hash,
                "implementation": {name: sha256(path.read_bytes()) for name, path in (
                    ("reference.py", Path(__file__)), ("fidelity.py", Path(fidelity.__file__)),
                    ("task-dispatch.py", Path(__file__).parents[1] / "evaluation/dispatch.py"),
                    ("fa4.py", Path(__file__).parents[1] / "evaluation/fa4.py"),
                    ("budget.py", Path(__file__).parents[1] / "budget.py"), ("method", fidelity.METHOD_PATH))}}

    def _account(self, context, *, create=False):
        return OwnerAccount.for_context(context, create=create, **({"clock": self.clock} if self.clock else {}))

    def _channel(self, context):
        factory = self.evaluation_factory
        if factory is None:
            factory, _ = _evaluation_factory(context.objects["profile"])
        return None if factory is None else factory(context.objects["profile"], context.root / ".runtime/reference/evaluation")

    def validate_resume(self, context):
        account = self._account(context)
        state_path = context.root / ".runtime/state/reference.json"
        if not state_path.exists():
            if context.intents() or account.operations(context.run_id):
                raise ProtocolError("reference state missing with original operations; cannot restart")
            return
        state = context.load_state("reference")
        self._validate_state(context, account, state)
        channel = self._channel(context)
        runtime = self.runtime_factory(context.objects["profile"])
        for op in state["operations"]:
            if op.get("admitted"):
                if op["role"] == "evaluator":
                    if channel is None:
                        raise ProtocolError("original evaluator channel missing")
                    channel.validate(context, op)
                else:
                    runtime.reopen(context, op["id"])

    def _validate_state(self, context, account, state):
        if state.get("version") != VERSION:
            raise ProtocolError("reference state version mismatch")
        for source in (state["latest"], state["qualified"]):
            if source:
                context.bundle.source_files(source)
        records = account.operations(context.run_id)
        declared = {op["id"] for op in state["operations"]}
        if not set(records) <= declared:
            raise ProtocolError("owner has an operation absent from original loop state")
        for op in state["operations"]:
            if op.get("reservation") and records.get(op["id"], {}).get("reservation") != op["reservation"]:
                raise ProtocolError("loop reservation differs from sole owner")
        for intent in context.intents():
            if intent["operation_id"] not in declared or records.get(intent["operation_id"], {}).get("reservation") != intent["reservation"]:
                raise ProtocolError("original runtime intent missing from owner/loop state")

    def _save(self, context, state, boundary):
        context.save_state("reference", state)
        self.checkpoint(boundary, context)

    def _initial(self, context, account):
        if (context.root / ".runtime/state/reference.json").exists():
            return context.load_state("reference")
        if account.operations(context.run_id) or context.intents():
            raise ProtocolError("cannot recreate lost loop state after an operation")
        seed = context.request["seed"] or context.objects["task"]["starter_source"]
        if seed is not None:
            data = context.bundle.source_files(seed)
            entrypoint = context.bundle.document(seed, "source")["entrypoint"]
            if context.request["mode"] == "verify_only" and not data[entrypoint].data:
                raise ProtocolError("verify_only seed must be nonempty")
        state = {"version": VERSION, "phase": "seed_intent", "seed": seed, "latest": None, "qualified": None,
                 "assessments": [], "operations": [], "current": None, "feedback": [], "episode": 0,
                 "stop_reason": "initializing", "status": "partial", "recovery": "none", "diagnostics": []}
        self._save(context, state, "seed_intent")
        return state

    def _seed(self, context, state):
        if state["phase"] != "seed_intent":
            return
        if state["seed"]:
            path = context.root / ".runtime/reference/seed"
            if path.exists():
                # Only the original pre-admission installation is idempotent.
                try:
                    verify_source_directory(context.bundle, state["seed"], path)
                except ProtocolError as error:
                    raise ProtocolError("seed ready installation changed; never overwrite a changed draft") from error
            else:
                install_source(context.bundle, state["seed"], path)
            state["latest"] = state["seed"]
            self._assessment(state, state["latest"], context)
        state["phase"] = "coder" if context.request["mode"] == "code" else "evaluator"
        self._save(context, state, "seed_ready")

    def _assessment(self, state, source, context):
        found = next((a for a in state["assessments"] if a["source_sha256"] == source["sha256"]), None)
        if found is None:
            uncovered = sorted(set(task_adapter(context.bundle, context.request["task"]).partitions) - set(context.objects["scope"]["coverage_ids"]))
            found = {"source_sha256": source["sha256"], "task_uncovered": uncovered, "performance_goal": "unknown",
                     **{k: _check() for k in ("coverage", "build", "correctness", "fidelity", "performance")}}
            state["assessments"].append(found)
        return found

    def _new_op(self, context, state, role):
        op = {"id": f"{context.run_id}_{len(state['operations']) + 1}_{role}", "role": role,
              "source": state["latest"], "purpose": "fixed Schema " + role,
              "publication_id": uuid4().hex, "reservation": None, "admitted": False, "finished": False}
        state["operations"].append(op)
        state["current"] = op["id"]
        self._save(context, state, "operation_planned")
        return op

    def _prepare_stage(self, context, op):
        root = context.root / ".runtime/reference/stages" / op["id"]
        if root.exists():
            manifest = loads(read_regular(root, "stage.json")[0])
            staged = {"root": str(root), "manifest": manifest, "sha256": digest(manifest)}
            verify_stage(staged)
            if manifest["source"] != op["source"] or manifest["task"] != context.request["task"]:
                raise ProtocolError("original stage subject changed")
            if "schema" in manifest and any(manifest[k] != context.request[k] for k in ("schema", "scope")):
                raise ProtocolError("original FA4 stage Schema/scope changed")
            return staged
        durable_mkdir(root.parent)
        temporary = Path(tempfile.mkdtemp(prefix=".preparing-", dir=root.parent))
        temporary.rmdir()
        try:
            staged = stage(context, op["source"], temporary)
            _rename_new(temporary, root)
            fsync_directory(root.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return {**staged, "root": str(root)}

    def _operate(self, context, state, account, runtime, channel, scope_validation):
        role = state["phase"]
        op = next((o for o in state["operations"] if o["id"] == state["current"]), None)
        if op is None:
            op = self._new_op(context, state, role)
        if op["reservation"] is None:
            minimum = context.objects["profile"]["parameters"].get("minimum_active_ms", {"coder": 1, "evaluator": 1000, "reviewer": 1000})
            if set(minimum) != {"coder", "evaluator", "reviewer"} or any(type(n) is not int or n <= 0 for n in minimum.values()):
                raise ProtocolError("minimum_active_ms requires positive per-role intervals")
            floor = cost(role="reviewer", calls=1, evaluator_jobs=1, active_ms=minimum["evaluator"] + minimum["reviewer"]) if role == "coder" else cost(role="reviewer", calls=1, active_ms=minimum["reviewer"]) if role == "evaluator" else cost()
            op["reservation"] = account.reserve(context.run_id, op["id"], role=role, purpose=op["purpose"],
                amount=cost(role=role, calls=int(role != "evaluator"), evaluator_jobs=int(role == "evaluator"), active_ms=minimum[role]),
                floor=floor, native_limits=context.objects["profile"]["native_limits"].get("workflow_limits"))
            self._save(context, state, "operation_reserved")
        reservation = op["reservation"]
        if role == "evaluator":
            if not op["admitted"]:
                op["stage"] = self._prepare_stage(context, op)
                op["locator"] = channel.locator(op["id"])
                context.record_intent(operation_id=op["id"], role=role, purpose=op["purpose"], source=op["source"],
                                      reservation=reservation, submission_key=op["id"], native_locator=op["locator"], episode=state["episode"])
                self._save(context, state, "evaluation_ready")
                channel.prepare(context, op)
                op["admitted"] = True
                self._save(context, state, "evaluation_admitted")
            owner_op = account.operations(context.run_id)[op["id"]]
            if not owner_op["launched"]:
                channel.submit(context, op, authorize=lambda: account.authorize(reservation))
            result = channel.settle(context, op, Publication(context.root, op["publication_id"]))
            if result.get("state") != "terminal":
                account.mark_unknown(op["id"], "original evaluator has no known terminal")
                return False
            account.absorb(context.run_id, result["usage_row"], terminated=True)
            op["evaluation"] = result.get("report")
            if op["evaluation"]:
                report = context.bundle.document(op["evaluation"], "evaluation")
                expected = {"task_sha256": context.request["task"]["sha256"], "source_sha256": op["source"]["sha256"],
                            "evaluator_sha256": context.objects["task"]["evaluator"]["sha256"]}
                if report["operation_id"] != op["id"] or any(report["subject"].get(k) != v for k, v in expected.items()):
                    raise ProtocolError("actual evaluation report subject differs from admitted operation")
                assessment = self._assessment(state, op["source"], context)
                for key in ("build", "correctness", "performance"):
                    check = report[key]
                    assessment[key] = _check(check["status"], op["evaluation"] if check["status"] != "not_run" else None, check["reason"])
                goal = report.get("extensions", {}).get("information", {}).get("performance_goal", "unknown")
                assessment["performance_goal"] = goal if goal in ("met", "unmet", "unknown") else "unknown"
            else:
                assessment = self._assessment(state, op["source"], context)
                for key in ("build", "correctness", "performance"):
                    assessment[key] = _check("unknown", reason="original job terminated without a complete evaluation report")
            context.record_terminal(op["id"], {"report": op["evaluation"], "usage": result["usage_row"], "terminated": True})
            state["phase"] = "reviewer"
        else:
            deliveries = [RoleDelivery(f"feedback_{i}", ref, ("coder",)) for i, ref in enumerate(state["feedback"][-1:])]
            if role == "coder" and state["feedback"]:
                feedback = context.bundle.document(state["feedback"][-1], "feedback")
                deliveries.extend(RoleDelivery(f"feedback_evidence_{i}", ref, ("coder",))
                                  for i, ref in enumerate(feedback["evidence"]))
            if role == "reviewer":
                publication = Publication(context.root, op["publication_id"])
                review = fidelity.review_input(context, op["source"], scope_validation)
                op["review_input"] = publication.put_bytes("review-input.json", jcs(review))
                publication.commit([op["review_input"]])
                deliveries = [RoleDelivery("finite_obligations", op["review_input"], ("reviewer",))]
            handle = runtime.admit(context, operation_id=op["id"], role=role, purpose=op["purpose"],
                                   source=op["source"], reservation=reservation, deliveries=tuple(deliveries))
            op["admitted"] = True
            self._save(context, state, role + "_admitted")
            owner_op = account.operations(context.run_id)[op["id"]]
            receipt = runtime.settle(context, handle) if owner_op["launched"] else runtime.invoke(context, handle, authorize=lambda _: account.authorize(reservation))
            if not receipt.terminated:
                account.mark_unknown(op["id"], "original runtime has no known terminal")
                return False
            account.absorb(context.run_id, receipt.usage_row(), terminated=True,
                           lower_bounds={"calls": receipt.facts.get("calls_lower_bound", receipt.facts.get("calls") or 0)})
            op["receipt"] = receipt.receipt
            self._save(context, state, role + "_settled")
            if role == "coder":
                state["episode"] += 1
                if receipt.ok and receipt.output is not None:
                    try:
                        publication = Publication(context.root, op["publication_id"])
                        required = context.bundle.document(context.objects["task"]["baseline_source"], "source")["format"]
                        if required == "python_tree":
                            files = receipt.output.candidate_files
                            if files is None:
                                raise ProtocolError("python_tree requires a complete typed file snapshot")
                            candidate = publication.put_source_files("candidate", files)
                        else:
                            source_bytes = receipt.output.candidate_source
                            if not source_bytes:
                                raise ProtocolError("single_file requires complete candidate_source text")
                            candidate = publication.put_source("candidate", source_bytes)
                        publication.commit([candidate])
                    except (ProtocolError, UnicodeError) as error:
                        state["diagnostics"].append(str(error))
                        state["phase"] = "coder"
                    else:
                        state["latest"] = candidate
                        self._assessment(state, candidate, context)
                        state["phase"] = "evaluator"
                else:
                    state["diagnostics"].append("coder returned no complete valid source; stable C retained")
                    state["phase"] = "coder"
            else:
                # Input and output have separate immutable publication closures.
                if "fidelity_publication" not in op:
                    op["fidelity_publication"] = uuid4().hex
                    self._save(context, state, "fidelity_intent")
                reference, report = fidelity.report(context, source=op["source"], operation_id=op["id"],
                    scope_validation=scope_validation, receipt=receipt, publication=Publication(context.root, op["fidelity_publication"]))
                op["fidelity"] = reference
                assessment = self._assessment(state, op["source"], context)
                for key in ("coverage", "fidelity"):
                    assessment[key] = _check(report[key]["status"], reference, "current fixed Schema finite independent review")
                assessment["task_uncovered"] = report["coverage"]["task_uncovered"]
                if all(assessment[k]["status"] == "pass" for k in ("coverage", "build", "correctness", "fidelity", "performance")):
                    state["qualified"] = op["source"]
                    state["phase"] = "done"
                    state.update(status="completed", recovery="none", stop_reason="current source qualified; soft performance goal " + assessment["performance_goal"])
                else:
                    state["phase"] = "coder" if context.request["mode"] == "code" else "done"
                    state.update(status="partial", recovery="none", stop_reason="current source has failed or unknown required checks")
                publication = Publication(context.root)
                feedback = publication.put_json("feedback.json", {"protocol": PROTOCOL, "kind": "feedback",
                    "subject": {"request_sha256": context.request_sha256, "task_sha256": context.request["task"]["sha256"],
                                "schema_sha256": context.request["schema"]["sha256"], "source_sha256": op["source"]["sha256"]},
                    "reason": state["stop_reason"], "evidence": [reference] + [a[k]["receipt"] for a in [assessment] for k in ("build",) if a[k]["receipt"]]})
                publication.commit([feedback], subjects=[context.request_ref, op["source"]])
                state["feedback"].append(feedback)
        op["finished"] = True
        state["current"] = None
        self._save(context, state, role + "_finished")
        return True

    def execute(self, context, *, continuing=False):
        lock_path = context.root / ".runtime/reference-loop.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return self._execute(context, continuing=continuing)
        finally:
            os.close(fd)

    def _execute(self, context, *, continuing=False):
        task = task_adapter(context.bundle, context.request["task"])
        task.require_evaluation_scope(context.request["scope"])  # unsupported subset costs nothing
        required_format = context.bundle.document(context.objects["task"]["baseline_source"], "source")["format"]
        if context.objects["seed"] is not None and context.bundle.document(context.objects["seed"], "source")["format"] != required_format:
            raise UnsupportedError("selected Task and explicit seed Source formats differ")
        scope_validation = task.validate_scope(context.request["schema"], context.request["scope"])
        fidelity.method_ref(context)
        account = self._account(context, create=not continuing)
        native_limits = context.objects["profile"]["native_limits"].get("workflow_limits")
        if native_limits is not None:
            validate_limits(native_limits)
        if account.descriptor["limits"]["tokens"] is not None or (native_limits is not None and native_limits["tokens"] is not None):
            raise UnsupportedError("reference runtime cannot enforce a finite parent/native token budget")
        state = self._initial(context, account)
        self._validate_state(context, account, state)
        self._seed(context, state)
        for assessment in state["assessments"]:
            if assessment["coverage"]["status"] == "not_run":
                assessment["task_uncovered"] = scope_validation["task_uncovered"]
        if state["phase"] == "done":
            return self._result(context, state, account)
        channel = self._channel(context)
        if scope_validation["status"] != "pass":
            state.update(status="blocked", recovery="new_request", stop_reason="scope/dispatch admission: " + "; ".join(scope_validation.get("reasons", [scope_validation.get("reason", "unknown")])) )
            return self._result(context, state, account)
        if channel is None:
            state.update(status="blocked", recovery="new_request", stop_reason="no configured qualified full-Task evaluation channel")
            return self._result(context, state, account)
        admission = channel.admission(context)
        if admission["status"] != "pass":
            state.update(status="blocked", recovery="new_request", stop_reason=admission["reason"])
            return self._result(context, state, account)
        runtime = self.runtime_factory(context.objects["profile"])
        account.begin(context.run_id)
        try:
            while state["phase"] != "done":
                if account.fences() and state["current"] is None:
                    break
                try:
                    if not self._operate(context, state, account, runtime, channel, scope_validation):
                        break
                except (BudgetDenied, AdmissionDenied) as error:
                    op = next((o for o in state["operations"] if o["id"] == state["current"]), None)
                    if op and op["reservation"]:
                        owned = account.operations(context.run_id)[op["id"]]
                        if not owned["launched"]:
                            account.release_unstarted(op["id"])
                    state.update(status="partial", recovery="none", stop_reason=str(error), phase="done")
                    break
        finally:
            account.end(context.run_id)
        return self._result(context, state, account)

    def _result(self, context, state, account):
        if state["phase"] == "done" and state.get("result") is not None and not account.fences():
            return state["result"]
        if not state["operations"] and not account.operations(context.run_id):
            # Recheck the owner/admission on every blocked replay, while keeping
            # the unchanged empty receipt index at its original publication.
            if "empty_usage_publication" not in state:
                state["empty_usage_publication"] = uuid4().hex
                self._save(context, state, "empty_usage_intent")
            publication = Publication(context.root, state["empty_usage_publication"])
        else:
            publication = Publication(context.root)
        usage = account.usage(context.run_id, publication)
        if usage["unsettled"]:
            state.update(status="blocked", recovery="settle_only", stop_reason="original operation or limited balance unresolved")
        result = {"protocol": PROTOCOL, "operation": "implement", "run_id": context.run_id,
                "request_id": context.request["request_id"], "request_sha256": context.request_sha256,
                **{k: state[k] for k in ("status", "stop_reason", "recovery", "feedback", "assessments")},
                "usage": usage, "code": {"latest": state["latest"], "qualified": state["qualified"]},
                "extensions": {"information": {"backend": VERSION, "coder_episodes": state["episode"],
                    "promotions": 0, "diagnostics": state["diagnostics"], "fixed_schema": context.request["schema"]}}}
        if state["phase"] == "done" and not usage["unsettled"]:
            state["result"] = result
        self._save(context, state, "result_ready")
        return result


def workflow_backend():
    """The single CLI/API factory; no runtime, model or GPU call on import."""
    return ReferenceBackend()
