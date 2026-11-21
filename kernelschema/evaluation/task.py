"""Fixed row-softmax snapshots, Task partitions and finite dispatch admission."""
from __future__ import annotations

import json
from pathlib import Path
import re

from ..artifacts import Bundle, Publication, durable_mkdir, durable_write, read_regular
from ..contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, sha256, validate
from ..resources import resource
from .. import template_api

VERSION = "row-softmax.evaluator.v1"
COMPONENTS = ("baseline.py", "check.py", "entry.py", "golden.py", "input.py", "timing.py")
EXECUTORS = ("_worker.py", "_sandbox.py", "_controller.py", "_remote.py", "_stage_ro.py")
EXAMPLE = "examples/workflows/row_softmax/"


def freeze_task(publication: Publication, *, environment: dict) -> dict:
    """Freeze implementation bytes and real Refs; environment is caller-selected.

    This creates executable evaluator materials, never an execution qualification.
    Callers must freeze an actual qualified profile separately for CUDA admission.
    """
    from ..contracts import ref
    ref(environment)
    def put(name):
        return publication.put_bytes("row-softmax/" + name, resource(EXAMPLE + name).read_bytes())
    files = {name: put("evaluator/" + name) for name in COMPONENTS}
    for name in EXECUTORS:
        files[name] = publication.put_bytes("row-softmax/executor/" + name, Path(__file__).with_name(name).read_bytes())
    originals = {name: put("task/" + name) for name in ("inputs.json", "timing.md", "contract.yaml", "acceptance.md", "workload.json")}
    evaluator = publication.put_json("row-softmax/evaluator.json", {
        "protocol": PROTOCOL, "kind": "evaluator", "version": VERSION,
        "entrypoint": files["entry.py"], "files": sorted(files.values(), key=lambda r: r["path"]),
        "inputs": originals["inputs.json"], "timing": originals["timing.md"], "environment": environment,
        "adapters": ["agate-dev-bwrap-v1", "agate-dev-stage-ro-v1", "local-cpu-logic-seccomp-v1"],
        "extensions": {"information": {"CUDA_qualification": "none", "CPU": "logic only"}}})
    baseline = publication.put_source("row-softmax/baseline", resource(EXAMPLE + "evaluator/baseline.py").read_bytes())
    starter = publication.put_source("row-softmax/starter", resource(EXAMPLE + "starter/kernel.py").read_bytes())
    task = publication.put_json("row-softmax/task.json", {
        "protocol": PROTOCOL, "kind": "task", "task_id": "row_softmax_fp32_v1", "evaluator": evaluator,
        "contract": originals["contract.yaml"], "acceptance": originals["acceptance.md"],
        "workload": originals["workload.json"], "environment": environment,
        "baseline_source": baseline, "starter_source": starter})
    return {"task": task, "evaluator": evaluator, "baseline": baseline, "starter": starter}


class RowSoftmaxTask:
    def __init__(self, bundle: Bundle, task: dict):
        self.bundle, self.reference = bundle, task
        self.task = bundle.document(task, kind="task")
        if self.task["task_id"] != "row_softmax_fp32_v1":
            raise UnsupportedError("only row_softmax_fp32_v1 is implemented")
        for key, name in (("contract", "contract.yaml"), ("acceptance", "acceptance.md"), ("workload", "workload.json")):
            if bundle.read(self.task[key]) != resource(EXAMPLE + "task/" + name).read_bytes():
                raise UnsupportedError("Task " + key + " differs from this implementation's fixed criteria")
        self.workload = json.loads(bundle.read(self.task["workload"]))
        self.partitions = {p["coverage_id"]: p["shape"] for p in self.workload["coverage_partitions"]}

    def evaluation_capabilities(self) -> dict:
        """The implemented evaluator always measures the entire fixed Task."""
        return {"scope_mode": "full_task_only", "coverage_ids": sorted(self.partitions),
                "subset_only": False}

    def require_evaluation_scope(self, scope: dict) -> dict:
        """Reject a partial evaluation request before staging or execution."""
        scope_doc = self.bundle.document(scope, kind="scope")
        return self.require_evaluation_scope_document(scope_doc)

    def require_evaluation_scope_document(self, scope_doc: dict) -> dict:
        """Apply the same Task coverage boundary before a Scope is published."""
        validate(scope_doc)
        if scope_doc["kind"] != "scope":
            raise ProtocolError("expected a Scope document")
        selected = set(scope_doc["coverage_ids"])
        if not selected <= set(self.partitions):
            raise ProtocolError("scope uses case IDs or unknown Task coverage partitions")
        if selected != set(self.partitions):
            raise UnsupportedError("row-softmax evaluator supports full Task scope only; subset-only evaluation is unsupported")
        return self.evaluation_capabilities()

    def validate_scope(self, schema: dict, scope: dict, *, require_hardware_binding=False) -> dict:
        """Finite Task dispatch checking for K4; this is NOT a fidelity verdict.

        Recognized when grammar: absent (root only), or conjunctions separated by
        semicolons of m/n = integer and m/n in {integers}. Unknown prose/variants
        returns unknown, never an inferred all-domain pass. Root Contract is
        inherited through the actual Template resolver for every selected node.
        Partial evaluation scopes are explicitly unsupported. The optional
        hardware check detects missing fields/placeholders only; its success
        is not evidence of actual device, documentation or profile qualification.
        """
        schema_doc = self.bundle.document(schema, kind="schema")
        scope_doc = self.bundle.document(scope, kind="scope")
        return self.validate_scope_documents(schema_doc, scope_doc,
            require_hardware_binding=require_hardware_binding)

    def validate_scope_documents(self, schema_doc: dict, scope_doc: dict, *,
                                 resolved=None, require_hardware_binding=False) -> dict:
        """Check host draft documents with the same finite Task semantics.

        Draft callers supply the actual normalized Template ResolvedTask; this
        does not publish or certify a complete Schema manifest. Publication and
        writing checks remain the caller's responsibility.
        """
        self.require_evaluation_scope_document(scope_doc)
        if schema_doc["task_sha256"] != self.reference["sha256"]:
            raise ProtocolError("Schema belongs to a different Task")
        if resolved is None:
            resolved = template_api.resolve_bundle(self.bundle, schema_doc)
        mappings, reasons, failures = [], [], []
        for binding in scope_doc["bindings"]:
            node = binding["node"]
            if node not in resolved.raw_task:
                raise ProtocolError("scope binds unknown Schema node")
            document = resolved.effective_document(node)
            when = document.get("META_DATA", {}).get("Dispatch", {}).get("when")
            if binding["variant"] is not None:
                reasons.append("variant semantics unsupported: " + binding["variant"])
                continue
            for coverage_id in binding["coverage_ids"]:
                eligible = _when(when, self.partitions[coverage_id], root=node == "root")
                if eligible is None:
                    reasons.append("uninterpreted when at " + node)
                elif not eligible:
                    failures.append(coverage_id + " is outside " + node + " when")
                else:
                    mappings.append({"coverage_id": coverage_id, "node": node, "variant": None,
                                     "shape": self.partitions[coverage_id]})
            if require_hardware_binding:
                hardware = document.get("CONTRACT", {}).get("Hardware_specification", {})
                if (not isinstance(hardware, dict) or
                    any(not hardware.get(key) for key in ("Device", "Architecture", "Stack", "Wiki")) or
                    re.search(r"unselected|@pending|unknown", json.dumps(hardware), re.I)):
                    reasons.append("Schema hardware/stack documentation is not bound: " + node)
        covered = {m["coverage_id"] for m in mappings}
        requested_uncovered = sorted(set(scope_doc["coverage_ids"]) - covered)
        if requested_uncovered and not reasons and not failures:
            failures.append("requested partitions lack a binding")
        return {"status": "fail" if failures else "unknown" if reasons else "pass", "mappings": mappings,
                "task_uncovered": sorted(set(self.partitions) - covered), "requested_uncovered": requested_uncovered,
                "reasons": sorted(set(failures + reasons)),
                "limitations": ["finite Task shape dispatch only; current Schema fidelity belongs to K4",
                                "no natural-language when or variant inference",
                                "hardware binding check does not verify referenced documents or an execution profile",
                                "CPU checking does not qualify CUDA"]}


def _when(text, shape, *, root):
    if text is None:
        return True if root else None
    if not isinstance(text, str):
        return None
    accepted = True
    for clause in text.split(";"):
        clause = clause.strip()
        match = re.fullmatch(r"([mMnN])\s*(?:=|==)\s*([0-9]+)", clause)
        group = re.fullmatch(r"([mMnN])\s+in\s*\{\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*\}", clause)
        if match:
            accepted &= shape[0 if match[1].lower() == "m" else 1] == int(match[2])
        elif group:
            accepted &= shape[0 if group[1].lower() == "m" else 1] in [int(x) for x in group[2].split(",")]
        else:
            return None
    return bool(accepted)


def stage(bundle: Bundle, task: dict, source: dict, destination: str | Path, *, scope: dict | None = None) -> dict:
    """Stage full-Task evaluation; an explicit partial scope is unsupported.

    Omitted scope means full Task evaluation, never an inferred subset. K4 must
    pass its selected scope. Never follow an import cache.
    """
    adapter = RowSoftmaxTask(bundle, task)
    if scope is not None:
        adapter.require_evaluation_scope(scope)
    evaluator_ref = adapter.task["evaluator"]
    evaluator = bundle.document(evaluator_ref, kind="evaluator")
    if evaluator["version"] != VERSION:
        raise UnsupportedError("evaluator implementation/version is not supported")
    components = {Path(r["path"]).name: r for r in evaluator["files"]}
    if set(components) != set(COMPONENTS + EXECUTORS) or len(evaluator["files"]) != len(COMPONENTS + EXECUTORS) or evaluator["entrypoint"] != components["entry.py"]:
        raise ProtocolError("evaluator component map/entrypoint differs")
    payload = {}
    for name, reference in components.items():
        data = bundle.read(reference)
        expected = resource(EXAMPLE + "evaluator/" + name).read_bytes() if name in COMPONENTS else Path(__file__).with_name(name).read_bytes()
        if data != expected:
            raise UnsupportedError("evaluator bytes differ from installed trusted implementation: " + name)
        payload[("evaluator/" if name in COMPONENTS else "") + name] = data
    for key, name in (("inputs", "inputs.json"), ("timing", "timing.md")):
        data = bundle.read(evaluator[key])
        if data != resource(EXAMPLE + "task/" + name).read_bytes():
            raise UnsupportedError("evaluator " + key + " criteria differ")
        payload[name] = data
    payload["kernel.py"] = bundle.source(source)
    payload["baseline.py"] = bundle.source(adapter.task["baseline_source"])
    if payload["baseline.py"] != payload["evaluator/baseline.py"]:
        raise ProtocolError("Task baseline differs from frozen evaluator baseline")
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError("stage must be a new directory")
    durable_mkdir(destination)
    for name, data in payload.items():
        durable_write(destination / name, data)
    manifest = {"version": VERSION, "task": task, "source": source, "evaluator": evaluator_ref,
                "inputs": evaluator["inputs"], "timing": evaluator["timing"],
                "files": {name: sha256(data) for name, data in payload.items()}}
    durable_write(destination / "stage.json", jcs(manifest))
    return {"root": str(destination), "sha256": digest(manifest), "manifest": manifest}


def verify_stage(staged):
    root = Path(staged["root"])
    data, _ = read_regular(root, "stage.json")
    if sha256(data) != staged["sha256"] or json.loads(data) != staged["manifest"]:
        raise ProtocolError("staged manifest changed")
    for name, expected in staged["manifest"]["files"].items():
        if sha256(read_regular(root, name)[0]) != expected:
            raise ProtocolError("staged bytes changed: " + name)
    return staged["manifest"]
