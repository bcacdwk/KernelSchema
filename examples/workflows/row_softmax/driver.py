"""Finite standalone KS driver: one owner, initial writing/fixed work, one optimize.

This module performs no work on import. Public workflow calls all go through
kernelschema.api. Persisted state only locates original runs/publications; budget
and native operation recovery remain with their existing owners.
"""
from copy import deepcopy
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from uuid import uuid4

from kernelschema import api, fidelity, writing
from kernelschema.artifacts import Bundle, Publication, durable_write, import_bundle, read_regular
from kernelschema.budget import OwnerAccount, use_authority
from kernelschema.contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from kernelschema.evaluation import freeze_task
from kernelschema.evaluation.channel import selection, qualify
from kernelschema.evaluation.environment import read_target
from kernelschema.extract_bridge import consume_raw, implement_request
from kernelschema.implement.reference import ReferenceBackend
from kernelschema.materials import validate_request
from kernelschema.resources import resource
from kernelschema.runtime import ClaudeProfile

VERSION = "ks.row-softmax.driver.v1"
DEFAULT_LIMITS = {"calls": {"planner": 2, "coder": 4, "reviewer": 6, "extract": 0},
                  "model_calls": 10, "evaluator_jobs": 8, "active_ms": 3600000, "tokens": None}


def load_runtime(path=None):
    value = loads((Path(path) if path else resource("kernelschema/runtime/profiles/claude-code-2.1.316-qwen3.8-max-no-tools.v1.json")).read_bytes())
    for key in ("reported_models", "builtin_plugins"): value[key] = tuple(value.get(key, ()))
    return ClaudeProfile(**value)


def _limits(operation, *, verify=False):
    return {"calls": {"planner": int(operation != "implement"), "coder": 0 if operation != "implement" or verify else 2,
                       "reviewer": 2 if operation == "implement" else 1, "extract": 0},
            "evaluator_jobs": 2 if operation == "implement" else 0, "active_ms": 1200000, "tokens": None}


def _put_request(root, publication, name, value):
    reference = publication.put_json(name + ".json", value)
    publication.commit([reference])
    durable_write(root / (name + ".request.json"), jcs(value))
    validate_request(Bundle(root), value)
    return reference


class Demo:
    def __init__(self, root, *, checkpoint=None):
        self.root = Path(root).absolute()
        self.checkpoint = checkpoint or (lambda label, demo: None)
        self.anchor = loads(read_regular(self.root, ".runtime/demo-anchor.json")[0])
        if self.anchor["version"] != VERSION or self.anchor["implementation"] != sha256(Path(__file__).read_bytes()):
            raise ProtocolError("original driver implementation changed")
        self.config = loads(Bundle(self.root).read(self.anchor["config"]))
        authority = self.anchor.get("authority")
        if not isinstance(authority, str) or not Path(authority).is_absolute():
            raise ProtocolError("original demo authority locator missing or invalid")
        self.owner = OwnerAccount(authority)
        if self.owner.descriptor != Bundle(self.root).document(self.config["parent"], "budget_account"):
            raise ProtocolError("original demo parent changed")
        self.state = loads(read_regular(self.root, ".runtime/demo-state.json")[0])
        if self.state["config"] != self.anchor["config"]:
            raise ProtocolError("original demo state/config changed")

    @classmethod
    def create(cls, root, *, start="design", runtime=None, environment=None, objective=None,
               schema_directory=None, receipt=None, limits=None, remote=None, checkpoint=None,
               parent=None, parent_bundle=None, authority=None):
        """Freeze requests with a new owner, or explicitly reuse one existing owner.

        Reuse requires the original public Ref, its Bundle and the private
        authority path together. Only the public Ref closure is imported.
        """
        root = Path(root).absolute()
        if root.exists(): raise ProtocolError("demo exists; reopen the original directory")
        if start not in ("design", "fixed", "partial", "verify"):
            raise ProtocolError("start must be design/fixed/partial/verify")
        external = any(value is not None for value in (parent, parent_bundle, authority))
        if external:
            if any(value is None for value in (parent, parent_bundle, authority)):
                raise ProtocolError("existing parent requires parent, parent_bundle and authority together")
            parent_bundle = parent_bundle if isinstance(parent_bundle, Bundle) else Bundle(parent_bundle)
            parent = deepcopy(parent)
            descriptor = parent_bundle.document(parent, "budget_account")
            owner = OwnerAccount(authority)
            if owner.descriptor != descriptor:
                raise ProtocolError("parent descriptor differs from existing owner authority")
            if limits is not None and jcs(limits) != jcs(descriptor["limits"]):
                raise ProtocolError("limits conflict with existing parent; cannot replace original limits")
        runtime = runtime or load_runtime()
        if environment is not None: read_target(environment)
        if external: import_bundle(parent_bundle, root, [parent])
        p = Publication(root)
        if not external:
            owner = OwnerAccount.create(root / ".runtime/owner.json", owner=uuid4().hex, limits=limits or DEFAULT_LIMITS)
            parent = p.put_json("parent.json", owner.descriptor)
        env = p.put_bytes("environment.json" if environment else "environment.md",
                          environment or resource("examples/workflows/row_softmax/task/environment.md").read_bytes())
        task = freeze_task(p, environment=env)["task"]
        w_rules = writing.freeze_rules(p)
        rules = loads(read_regular(p.stage, w_rules["path"])[0])
        method = fidelity.freeze_method(p)
        host_tools = []
        for relative in ("kernelschema/implement/reference.py", "kernelschema/budget.py", "kernelschema/evaluation/channel.py",
                         "kernelschema/evaluation/agate.py", "kernelschema/evaluation/task.py",
                         "examples/workflows/row_softmax/driver.py", "examples/workflows/row_softmax/run_demo.py"):
            body = p.put_bytes("host-implementation/" + relative, resource(relative).read_bytes())
            host_tools.append({"name": "standalone:" + relative, "version": body["sha256"], "source": body})
        rules = p.put_json("rules.json", {**rules, "files": sorted([*rules["files"], method["file"]], key=lambda r: r["path"]),
                                        "tools": [*rules["tools"], method["tool"], *host_tools]})
        metadata = loads(resource("examples/workflows/row_softmax/task/materials.json").read_bytes())
        entries = []
        for item in metadata["entries"]:
            data = resource(item["repository_path"]).read_bytes()
            if sha256(data) != item["sha256"] or len(data) != item["size"]:
                raise ProtocolError("pinned public material changed: " + item["id"])
            ref = env if item["id"] == "task_environment" else p.put_bytes("materials/" + item["id"],
                objective if item["id"] == "workflow_objective" and objective is not None else data)
            entries.append({"id": item["id"], "file": ref, "roles": item["roles"], "origin": "caller-selected current objective" if objective is not None and item["id"] == "workflow_objective" else item["origin"], "license": item["license"]})
        materials = p.put_json("materials.json", {"protocol": PROTOCOL, "kind": "materials", "entries": entries})
        scope_data = loads(resource("examples/workflows/row_softmax/task/scope.json").read_bytes())
        scope = p.put_json("scope.json", {"protocol": PROTOCOL, "kind": "scope", "coverage_ids": scope_data["coverage_ids"], "bindings": scope_data["fixed_r0_bindings"]})
        draft_scope = p.put_json("draft-scope.json", {"protocol": PROTOCOL, "kind": "scope", "coverage_ids": scope_data["coverage_ids"], "bindings": []})
        consumed = consume_raw(p, schema_directory or resource("examples/workflows/row_softmax/schema/root.yaml").parent,
                               task=task, origin="extract" if receipt else "handwritten", receipt=receipt)
        seed = p.put_source("manual-partial-seed", resource("examples/workflows/row_softmax/partial-seed/kernel.py").read_bytes())
        seed_origin = p.put_bytes("manual-partial-origin.md", resource("examples/workflows/row_softmax/partial-seed/ORIGIN.md").read_bytes())
        wp = p.put_json("writing-profile.json", writing.backend_profile(runtime, env))
        rp = runtime.backend_profile(env, name="reference")
        rp["capabilities"] = ReferenceBackend().capabilities(rp)
        rp["parameters"]["claude"] = runtime.as_dict()
        if remote: rp["parameters"]["evaluation"] = selection(remote=remote)
        rp_ref = p.put_json("reference-profile.json", rp)
        config = {"start": start, "task": task, "rules": rules, "materials": materials, "scope": scope,
                  "draft_scope": draft_scope, "schema": consumed["schema"], "seed": seed, "seed_origin": seed_origin,
                  "receipt": consumed["receipt"], "parent": parent, "writing_profile": wp, "reference_profile": rp_ref,
                  "remote": remote}
        base = {"task": task, "rules": rules, "materials": materials, "budget": {"parent": parent, "limits": _limits("design")}}
        requests = {}
        q = writing.request(operation="design", request_id="row-softmax-design", scope=draft_scope, profile=wp, **base)
        requests["design"] = p.put_json("design.request.json", q)
        values = {"design": q}
        for name in ("fixed", "partial", "verify"):
            q = implement_request(request_id="row-softmax-" + name, task=task, rules=rules, materials=materials, scope=scope,
                budget={"parent": parent, "limits": _limits("implement", verify=name == "verify")},
                backend={"name": "reference", "profile": rp_ref}, schema=consumed["schema"],
                seed=None if name == "fixed" else seed, mode="verify_only" if name == "verify" else "code")
            requests[name] = p.put_json(name + ".request.json", q); values[name] = q
        q = writing.request(operation="optimize", request_id="row-softmax-optimize", scope=scope, profile=wp,
                            schema=consumed["schema"], **base)
        requests["optimize"] = p.put_json("optimize.request.json", q); values["optimize"] = q
        config["requests"] = requests
        config_ref = p.put_bytes("demo-config.json", jcs(config))
        p.commit([*requests.values(), seed_origin, config_ref, *([consumed["receipt"]] if consumed["receipt"] else [])])
        for name, value in values.items():
            durable_write(root / (name + ".request.json"), jcs(value))
            validate_request(Bundle(root), value)
        durable_write(root / ".runtime/demo-anchor.json", jcs({"version": VERSION, "implementation": sha256(Path(__file__).read_bytes()),
            "config": config_ref, "authority": str(owner.path)}), mode=0o600)
        durable_write(root / ".runtime/demo-state.json", jcs({"config": config_ref, "phase": "initial", "current": None,
            "history": [], "schema": consumed["schema"] if start != "design" else None, "scope": scope if start != "design" else draft_scope,
            "qualification": None, "optimize_count": 0, "status": "prepared", "recovery": "continue", "reason": "five concrete request examples frozen"}))
        return cls(root, checkpoint=checkpoint)

    def _save(self, boundary):
        durable_write(self.root / ".runtime/demo-state.json", jcs(self.state), replace=True)
        self.checkpoint(boundary, self)

    def _request(self, phase):
        if phase == "initial": return Bundle(self.root).document(self.config["requests"][self.config["start"]])
        verify = phase == "followup" and self.state["action"]["mode"] == "verify_only"
        base = {k: self.config[k] for k in ("task", "rules", "materials")}
        base.update(scope=self.state["scope"], budget={"parent": self.config["parent"], "limits": _limits("optimize" if phase == "optimize" else "implement", verify=verify)})
        if phase == "optimize":
            previous = Bundle(self.root).document(self.state["history"][-1]["result"])
            evidence = [previous["code"]["latest"]] if previous["code"]["latest"] else []
            for assessment in previous["assessments"]:
                for key in ("build", "correctness", "fidelity", "performance", "coverage"):
                    ref = assessment[key]["receipt"]
                    if ref and ref not in evidence: evidence.append(ref)
            return writing.request(operation="optimize", request_id="row-softmax-optimize-after-fixed", profile=self.config["writing_profile"],
                schema=self.state["schema"], feedback=previous["feedback"], evidence=evidence, **base)
        action = self.state.get("action", {"mode": "code", "seed": None}) if phase == "followup" else {"mode": "code", "seed": None}
        return implement_request(request_id="row-softmax-" + phase, schema=self.state["schema"],
            backend={"name": "reference", "profile": self.config["reference_profile"]}, **action, **base)

    def _profile(self, request, p):
        if request["operation"] == "implement" and self.state["qualification"]:
            profile = deepcopy(Bundle(self.root).document(self.config["reference_profile"]))
            evidence = self.state["qualification"]
            remote = loads(evidence["execute"]["intent.json"].encode())["profile"]
            profile["parameters"]["evaluation"] = selection(remote=remote, qualification=evidence)
            request = {**request, "backend": {"name": "reference", "profile": p.put_json("qualified-profile.json", profile)}}
        return request

    @contextmanager
    def _locked(self):
        lock = os.open(self.root / ".runtime/demo.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.state = loads(read_regular(self.root, ".runtime/demo-state.json")[0])
            if self.state["config"] != self.anchor["config"]:
                raise ProtocolError("original demo state/config changed")
            yield
        finally: os.close(lock)

    def run(self):
        with self._locked():
            with use_authority(self.owner.path):
                return self._run()

    def _run(self):
        while self.state["phase"] != "done":
            phase = self.state["phase"]
            current = self.state["current"]
            if current is None:
                if self.owner.fences():
                    self.state.update(status="blocked", recovery="settle_only", reason="original owner operation unresolved")
                    self._save("owner_fenced"); break
                request = self._request(phase)
                p = Publication(self.root)
                request = self._profile(request, p)
                name = "step-" + str(len(self.state["history"]))
                ref = _put_request(self.root, p, name, request)
                current = {"request": ref, "path": name + ".request.json", "operation": request["operation"], "out": "runs/" + name}
                self.state["current"] = current
                self._save("new_schema_unadmitted" if phase == "followup" else "request_ready")
            request = Bundle(self.root).document(current["request"])
            if request["budget"]["parent"] != self.config["parent"]: raise ProtocolError("step changed original parent")
            out = self.root / current["out"]
            operation = getattr(api, current["operation"])
            self.checkpoint("before_api", self)
            result = operation(continue_run=out) if out.exists() else operation(request=self.root / current["path"], out=out)
            self.checkpoint("child_result_before_driver", self)
            if result["recovery"] == "settle_only" or result["usage"]["unsettled"]:
                self.state.update(status="blocked", recovery="settle_only", reason=result["stop_reason"])
                self._save("original_operation_unknown"); break
            context = api.open_run(current["operation"], out)
            p = Publication(out)
            result_ref = p.put_json("driver-selected-result.json", result)
            p.commit([result_ref], subjects=[context.request_ref])
            import_bundle(Bundle(out), self.root, [result_ref], subjects=[context.request_ref])
            self.state["history"].append({"result": result_ref, "request": context.request_ref, "out": current["out"]})
            self.state["current"] = None
            if current["operation"] in ("design", "optimize") and result["schema"]:
                self.state["schema"], self.state["scope"] = result["schema"], result["scope"]
            if result["status"] != "completed":
                self.state.update(phase="done", status=result["status"], recovery=result["recovery"], reason=result["stop_reason"])
            elif current["operation"] == "design":
                self.state.update(phase="implement", status="writing_published", recovery="continue", reason="design published; fixed implementation not admitted")
            elif current["operation"] == "optimize":
                self.state["optimize_count"] += 1
                if self.state["optimize_count"] != 1: raise ProtocolError("driver allows at most one optimize")
                if result["next_action"] == "none":
                    self.state.update(phase="done", status="stopped", recovery="none", reason="optimize returned no action")
                else:
                    self.state["action"] = self._action(result)
                    self.state.update(phase="followup", status="writing_published", recovery="continue", reason="writing output published; next fixed request not admitted")
            elif phase == "followup" or self.config["start"] == "verify":
                self.state.update(phase="done", status=result["status"], recovery=result["recovery"], reason=result["stop_reason"])
            else:
                self.state.update(phase="optimize", status="fixed_settled", recovery="continue", reason="original fixed work settled; one optimize remains")
            self._save("writing_published" if current["operation"] in ("design", "optimize") else "child_selected")
        return self.summary()

    def _action(self, result):
        if result["action_inputs"][:2] != [result["schema"], result["scope"]]:
            raise ProtocolError("action lacks exact output Schema/Scope")
        sources = []
        for ref in result["action_inputs"][2:]:
            try: document = Bundle(self.root).document(ref)
            except ProtocolError: continue
            if document.get("kind") == "source": sources.append(ref)
        default = self.config["task"] in result["action_inputs"][2:]
        if result["next_action"] == "verify":
            if default or len(sources) != 1 or not Bundle(self.root).source(sources[0]):
                raise ProtocolError("verify requires exact nonempty Source")
        elif not ((len(sources) == 1 and not default) or (not sources and default)):
            raise ProtocolError("implement requires explicit Source or default Task start")
        return {"mode": "verify_only" if result["next_action"] == "verify" else "code", "seed": sources[0] if sources else None}

    def prepare_agate(self, *, transport=None):
        with self._locked():
            return self._prepare_agate(transport=transport)

    def _prepare_agate(self, *, transport=None):
        if self.state["history"] or self.state["current"]:
            raise ProtocolError("prepare Agate before the business chain; never replace an admitted profile")
        remote = self.config["remote"]
        if remote is None: raise UnsupportedError("create with an explicit remote GPU/tool profile")
        identity = {"qualification": sha256(resource("kernelschema/evaluation/channel.py").read_bytes()), "agate": sha256(resource("kernelschema/evaluation/agate.py").read_bytes())}
        out = self.root / "qualification"
        with use_authority(self.owner.path):
            if out.exists():
                context = api.open_run("implement", out, backend_identity=identity)
            else:
                q = deepcopy(Bundle(self.root).document(self.config["requests"]["fixed"]))
                q["request_id"] = "row-softmax-agate-qualification"
                q["budget"]["limits"] = {"calls": {"planner": 0, "coder": 0, "reviewer": 0, "extract": 0},
                                          "evaluator_jobs": 6, "active_ms": 1200000, "tokens": None}
                _put_request(self.root, Publication(self.root), "qualification", q)
                context = api.prepare_run("implement", request=self.root / "qualification.request.json", out=out, backend_identity=identity)
            result = qualify(context, remote=remote, transport=transport)
        if "usage" in result:
            import_bundle(context.bundle, self.root, [result["usage"]["receipts"], *([result["evidence"]] if result.get("evidence") else [])])
            self.state["qualification_usage"] = result["usage"]
            self.state["qualification_evidence"] = result.get("evidence")
        self.state["qualification"] = result["qualification"]
        self._save("qualification_settled")
        durable_write(self.root / "qualification-result.json", jcs(result), replace=True)
        return result

    def summary(self):
        current_qualified = None
        history = []
        for entry in self.state["history"]:
            result = Bundle(self.root).document(entry["result"])
            request = Bundle(self.root).document(entry["request"])
            history.append({"operation": result["operation"], "status": result["status"], "result": entry["result"],
                            "request": entry["request"], "usage": result["usage"], "schema": request.get("schema", result.get("schema"))})
            if result["operation"] == "implement" and request["schema"] == self.state["schema"]:
                current_qualified = result["code"]["qualified"]
        if self.state["current"] is not None and self.state["current"]["operation"] == "implement":
            current_qualified = None  # pending new Q, including a same-R verify
        return {"format": VERSION, **{k: self.state[k] for k in ("status", "recovery", "reason", "phase", "schema", "scope", "optimize_count")},
                "current_qualified": current_qualified, "current_run": self.state["current"], "history": history,
                "owner": self.owner.descriptor, "fences": self.owner.fences(),
                "qualification_usage": self.state.get("qualification_usage"),
                "qualification": "original Agate capture supplied" if self.state["qualification"] else "GPU unqualified"}

    def export(self, destination):
        roots = list(self.config["requests"].values()) + [self.config["seed_origin"]]
        subjects = []
        if self.config["receipt"]: roots.append(self.config["receipt"])
        if self.state.get("qualification_usage"): roots.append(self.state["qualification_usage"]["receipts"])
        if self.state.get("qualification_evidence"): roots.append(self.state["qualification_evidence"])
        for entry in self.state["history"]:
            roots.append(entry["result"]); subjects.append(entry["request"])
        import_bundle(Bundle(self.root), destination, roots, subjects=subjects)
        return str(Path(destination).absolute())
