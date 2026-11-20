"""Synthetic only: transports use actual K2 inputs, intents, ledger and staging.

Nothing here executes/qualifies a kernel, model or GPU. Production never imports
this fixture. Fake terminals are durable so a new backend object can recover.
"""
from dataclasses import asdict
from pathlib import Path

from kernelschema import fidelity
from kernelschema.artifacts import Publication, durable_write
from kernelschema.contracts import PROTOCOL, digest, jcs, loads
from kernelschema.evaluation import freeze_task
from kernelschema.evaluation.task import verify_stage
from kernelschema.extract_bridge import consume_raw
from kernelschema.resources import resource
from kernelschema.runtime import CallReceipt, RoleOutput, prepare_role_input

CODE = "# synthetic source: no kernel execution\nclass Model:\n    def forward(self, x, out):\n        return out\n"


def prepare_fixture(f, *, backend, mode="code", limit=2, seed=None, schema_directory=None, parent=None):
    p = Publication(f["root"])
    task = freeze_task(p, environment=f["originals"]["task/environment.md"])["task"]
    schema = consume_raw(p, schema_directory or resource("examples/workflows/row_softmax/schema/root.yaml").parent, task=task)["schema"]
    method = fidelity.freeze_method(p)
    rules = p.put_json("rules.json", {"protocol": PROTOCOL, "kind": "rules", "files": [method["file"]], "tools": [method["tool"]]})
    profile = {"protocol": PROTOCOL, "kind": "backend_profile", "name": "reference", "version": "synthetic-only",
               "runtime": "synthetic", "model": "synthetic", "parameters": {}, "native_limits": {},
               "environment": f["originals"]["task/environment.md"], "capabilities": backend.capabilities({})}
    request = {**f["request"], "task": task, "schema": schema, "rules": rules, "mode": mode,
               "seed": seed, "backend": {"name": "reference", "profile": p.put_json("profile.json", profile)}}
    request["budget"] = {"parent": parent, "limits": {"calls": {"planner": 0, "coder": limit, "reviewer": limit, "extract": 0},
                         "evaluator_jobs": limit, "active_ms": 60000, "tokens": None}}
    r = p.put_json("request.json", request)
    p.commit([r])
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    return request


class FakeRuntime:
    def __init__(self, *, candidates=None, reviews=None, unknown=False):
        self.candidates = candidates or [CODE]
        self.reviews = reviews or ["pass"]
        self.unknown = unknown
        self.calls = []
        self.inputs = []

    def identity(self):
        return {"runtime": "synthetic-reference-transport-v1"}

    def admit(self, context, **kw):
        op = kw["operation_id"]
        namespace = "fake_" + op
        path = context.root / f".runtime/state/{namespace}.json"
        delivery = prepare_role_input(context, kw["role"], source=kw["source"], deliveries=kw.get("deliveries", ()))
        fixed = {"operation_id": op, "role": kw["role"], "source": kw["source"], "purpose": kw["purpose"],
                 "reservation": kw["reservation"], "delivery": delivery}
        if path.exists():
            assert context.load_state(namespace)["fixed"] == fixed
        else:
            context.save_state(namespace, {"fixed": fixed, "publication_id": __import__("uuid").uuid4().hex, "terminal": None})
        context.record_intent(operation_id=op, role=kw["role"], purpose=kw["purpose"], source=kw["source"],
                              reservation=kw["reservation"], submission_key=op, native_locator={"synthetic_state": namespace})
        self.inputs.append(delivery)
        return op

    def reopen(self, context, op):
        context.load_state("fake_" + op)
        return op

    def invoke(self, context, handle, *, authorize):
        state = context.load_state("fake_" + handle)
        assert authorize(state["fixed"])
        self.calls.append(state["fixed"]["role"])
        if self.unknown:
            return self.settle(context, handle)
        return self.complete(context, handle)

    def complete(self, context, handle):
        state = context.load_state("fake_" + handle)
        role = state["fixed"]["role"]
        role_number = sum(i["role"] == role for i in context.intents()) - 1
        if role == "coder":
            content = self.candidates[min(role_number, len(self.candidates) - 1)]
        else:
            source = state["fixed"]["source"]
            code = context.bundle.source(source).decode()
            location = f"kernel.py:L1-L{len(code.splitlines())}"
            evidence = [{"location": location, "quote": "\n".join(code.splitlines())}]
            status = self.reviews[min(role_number, len(self.reviews) - 1)]
            inventory = fidelity.obligations(context)
            items = []
            for obligation in inventory:
                selected = status
                if "MULTI_ROW" in code and obligation["category"] == "ownership-and-dispatch":
                    selected = "fail" if '"count":1' in obligation["obligation"] else "pass"
                if "OMIT_MAX" in code and obligation["category"] == "math-and-precision":
                    selected = "fail"
                items.append({"id": obligation["id"], "status": selected, "reason": "synthetic scripted evidence",
                              "source_locations": [location], "evidence": evidence})
            scope = context.objects["scope"]
            coverage = [{"coverage_id": cid, "node": b["node"], "variant": b["variant"], "status": "pass",
                         "reason": "synthetic routing evidence", "source_locations": [location], "evidence": evidence}
                        for b in scope["bindings"] for cid in b["coverage_ids"]]
            content = {"items": items, "coverage": coverage}
        output = RoleOutput(role, "sha256:" + "0" * 64, "ok", "candidate_source" if role == "coder" else "review", content, "synthetic only")
        p = Publication(context.root, state["publication_id"])
        facts = {"runtime": "synthetic-reference-v1", "producer": "synthetic:" + context.run_id, "operation_id": handle,
                 "receipt_id": handle + ":terminal", "role": role, "purpose": state["fixed"]["purpose"],
                 "terminated": True, "status": "succeeded", "calls": 1, "tokens": 10, "raw": {},
                 "scope_id": handle + "_scope", "requested_model": "synthetic", "reported_models": ["synthetic"]}
        receipt = p.put_bytes("receipt.json", jcs({"facts": facts, "output": asdict(output), "synthetic": True}))
        p.commit([receipt])
        state["terminal"] = {"facts": facts, "output": asdict(output), "receipt": receipt}
        context.save_state("fake_" + handle, state)
        return CallReceipt(facts, output, receipt)

    def settle(self, context, handle):
        state = context.load_state("fake_" + handle)
        terminal = state["terminal"]
        if terminal:
            return CallReceipt(terminal["facts"], RoleOutput(**terminal["output"]), terminal["receipt"])
        return CallReceipt({"terminated": False, "status": "in_flight", "raw": {}}, None)


class FakeEvaluation:
    def __init__(self, root, *, statuses=None, unknown=False, goal="unmet"):
        self.root = Path(root)
        self.statuses = statuses or ["pass"]
        self.unknown = unknown
        self.goal = goal
        self.calls = []

    def admission(self, context):
        return {"status": "pass", "reason": "synthetic CPU only; not qualification"}

    def locator(self, operation_id):
        return {"synthetic": True, "path": str(self.root / operation_id)}

    def prepare(self, context, op):
        verify_stage(op["stage"])
        context.save_state("eval_" + op["id"], {"source": op["source"], "submitted": False})

    def validate(self, context, op):
        verify_stage(op["stage"])
        assert context.load_state("eval_" + op["id"])["source"] == op["source"]

    def submit(self, context, op, *, authorize):
        assert authorize()
        self.calls.append(op["id"])
        context.save_state("eval_" + op["id"], {"source": op["source"], "submitted": True})

    def settle(self, context, op, publication):
        self.validate(context, op)
        if self.unknown:
            return {"state": "unknown"}
        manifest = op["stage"]["manifest"]
        index = sum(i["role"] == "evaluator" for i in context.intents()) - 1
        status = self.statuses[min(index, len(self.statuses) - 1)]
        raw = publication.put_bytes("raw.json", jcs({"synthetic": True, "source": op["source"], "operation": op["id"]}))
        env = publication.put_bytes("env.json", b'{"synthetic":true}')
        check = {"status": status, "receipt": raw if status in ("pass", "fail") else None, "reason": "synthetic only"}
        report = {"protocol": PROTOCOL, "kind": "evaluation", "operation_id": op["id"],
            "subject": {"task_sha256": manifest["task"]["sha256"], "source_sha256": manifest["source"]["sha256"],
                "evaluator_sha256": manifest["evaluator"]["sha256"], "inputs_sha256": manifest["inputs"]["sha256"],
                "timing_sha256": manifest["timing"]["sha256"], "environment_sha256": env["sha256"]},
            "build": check, "correctness": check, "performance": check, "case_ids": [], "samples": None,
            "raw_receipts": [raw], "usage_receipts": [], "environment": env,
            "extensions": {"information": {"synthetic": True, "performance_goal": self.goal}}}
        ref = publication.put_json("evaluation.json", report)
        publication.commit([ref], subjects=[context.request_ref, op["source"]])
        row = {"producer": "synthetic-eval:" + context.run_id, "operation_id": op["id"], "role": "evaluator",
               "purpose": op["purpose"], "receipt_id": op["id"] + ":terminal", "sequence": 1,
               "receipt": raw, "state": "settled", "calls": 0, "evaluator_jobs": 1, "tokens": 0, "gpu_ms": None}
        return {"state": "terminal", "report": ref, "usage_row": row}
