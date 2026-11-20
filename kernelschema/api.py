"""One implementation for CLI/Python entry points and durable run boundaries.

K2 supplies identity, immutable inputs, intent/state persistence and publication.
The selected backend owns admission, its one budget ledger and business recovery.
No built-in execution backend is implemented by this interface layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import shutil
import tempfile
from typing import Protocol
from uuid import uuid4

from .artifacts import Bundle, Publication, _rename_new, dependencies, durable_mkdir, durable_write, fsync_directory, import_bundle, read_regular
from .contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, loads, validate
from .materials import require_capabilities, validate_request


class Backend(Protocol):
    """Narrow owner seam; actual implementations are explicit installed modules."""
    def capabilities(self, profile: dict) -> dict: ...
    def identity(self, profile: dict) -> dict: ...
    def validate_resume(self, context: "RunContext") -> None: ...
    def execute(self, context: "RunContext", *, continuing: bool) -> dict: ...


def _backend(name: str, operation: str) -> Backend:
    modules = {("reference", "implement"): "kernelschema.implement.reference",
               ("claude", "design"): "kernelschema.design",
               ("claude", "optimize"): "kernelschema.optimize"}
    module_name = modules.get((name, operation))
    if module_name is None:
        raise UnsupportedError(f"backend {name!r} for {operation} is unsupported/not yet implemented")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name and (module_name == error.name or module_name.startswith(error.name + ".")):
            raise UnsupportedError(f"{module_name} is not yet implemented") from error
        raise UnsupportedError(f"backend dependency unavailable: {error.name}") from error
    factory = getattr(module, "workflow_backend", None)
    if factory is None:
        raise UnsupportedError(f"{module_name}.workflow_backend is not yet implemented")
    return factory()


@dataclass(frozen=True)
class RunContext:
    root: Path
    anchor: dict
    request: dict
    objects: dict

    @property
    def run_id(self): return self.anchor["run_id"]

    @property
    def request_sha256(self): return self.anchor["request_sha256"]

    @property
    def bundle(self): return Bundle(self.root)

    @property
    def request_ref(self): return self.anchor["request"]

    @property
    def owner(self): return self.anchor["owner"]

    def save_state(self, namespace: str, value: dict):
        """Durable private owner/runtime state. No public recovery token."""
        _name(namespace)
        if not isinstance(value, dict): raise ProtocolError("private state must be an object")
        payload = {"run_id": self.run_id, "request_sha256": self.request_sha256, "owner": self.owner, "state": value}
        durable_write(self.root / f".runtime/state/{namespace}.json", jcs(payload), replace=True)

    def load_state(self, namespace: str) -> dict:
        _name(namespace)
        value = loads(read_regular(self.root, f".runtime/state/{namespace}.json")[0])
        if any(value.get(k) != expected for k, expected in (("run_id", self.run_id), ("request_sha256", self.request_sha256), ("owner", self.owner))):
            raise ProtocolError("private owner/runtime state identity mismatch; cannot reset recovery")
        return value["state"]

    def record_intent(self, *, operation_id: str, role: str, purpose: str, source: dict | None,
                      reservation: dict, submission_key: str | None, native_locator: dict | None,
                      episode: int | None = None):
        """Before charging/submitting: persist owner authorization and locator.

        Reservation semantics, balance and unknown/fence handling belong to the
        sole owner. This persistence helper does not grant any authorization.
        """
        from .contracts import ROLES, _count, _text, ref
        _name(operation_id)
        if role not in (*ROLES, "evaluator"): raise ProtocolError("invalid intent role")
        _text(purpose); ref(source, nullable=True); _count(episode, nullable=True)
        if source: self.bundle.source(source)
        if not isinstance(reservation, dict) or not reservation:
            raise ProtocolError("intent needs explicit owner reservation facts")
        if submission_key is not None: _text(submission_key)
        if native_locator is not None and not isinstance(native_locator, dict): raise ProtocolError("native locator must be an object")
        value = {"run_id": self.run_id, "request_sha256": self.request_sha256, "owner": self.owner,
                 "operation_id": operation_id, "role": role, "purpose": purpose, "source": source,
                 "reservation": reservation, "submission_key": submission_key, "native_locator": native_locator, "episode": episode}
        _immutable_json(self.root / f".runtime/intents/{operation_id}.json", value)
        return value

    def intents(self) -> list[dict]:
        directory = self.root / ".runtime/intents"
        if not directory.exists(): return []
        values = []
        for path in sorted(directory.iterdir()):
            if path.suffix != ".json": raise ProtocolError("unexpected run intent file")
            value = loads(read_regular(self.root, str(path.relative_to(self.root)))[0])
            if (value.get("run_id"), value.get("request_sha256"), value.get("owner"), value.get("operation_id")) != (self.run_id, self.request_sha256, self.owner, path.stem):
                raise ProtocolError("operation intent identity mismatch")
            values.append(value)
        return values

    def record_terminal(self, operation_id: str, facts: dict):
        _name(operation_id)
        if operation_id not in {i["operation_id"] for i in self.intents()}:
            raise ProtocolError("terminal facts require the original durable intent")
        _immutable_json(self.root / f".runtime/terminals/{operation_id}.json", {"run_id": self.run_id, "operation_id": operation_id, "facts": facts})

    def publish_result(self, result: dict) -> dict:
        validate_result(self, result)
        # Reuse an interrupted publication ID for exactly the same result.
        key = digest(result).split(":", 1)[1]
        path = self.root / f".runtime/result-publications/{key}.json"
        if path.exists():
            publication_id = loads(read_regular(self.root, str(path.relative_to(self.root)))[0])["publication_id"]
        else:
            publication_id = uuid4().hex
            durable_write(path, jcs({"publication_id": publication_id}))
        publication = Publication(self.root, publication_id)
        result_ref = publication.put_json("result.json", result)
        publication.commit([result_ref], subjects=[self.request_ref])
        _immutable_json(self.root / f".runtime/results/{key}.json", {"result": result_ref, "request": self.request_ref})
        durable_write(self.root / "result.json", jcs(result), replace=True)
        return result


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ProtocolError("private record name must contain only letters, digits, underscore or dash")


def _immutable_json(path: Path, value: dict):
    data = jcs(value)
    if path.exists() or path.is_symlink():
        existing, mode = read_regular(path.parent, path.name)
        if existing != data or mode != 0o644: raise ProtocolError(f"immutable private identity conflict: {path.name}")
    else: durable_write(path, data)


def prepare_run(operation: str, *, request: str | Path, out: str | Path, backend_identity: dict) -> RunContext:
    """Freeze inputs and anchor before a backend may create its first operation."""
    from .contracts import OPERATIONS
    if operation not in OPERATIONS: raise ProtocolError("unsupported operation")
    request_path = Path(request).absolute()
    raw, _ = read_regular(request_path.parent, request_path.name)
    fixed = validate(loads(raw))
    if fixed.get("operation") != operation or "run_id" in fixed: raise ProtocolError("request operation mismatch")
    source = Bundle(request_path.parent)
    objects = validate_request(source, fixed)
    root = Path(out).absolute()
    if root.exists() or root.is_symlink(): raise ProtocolError("run directory already exists; use --continue with its original operation")
    if not isinstance(backend_identity, dict) or not backend_identity:
        raise ProtocolError("backend implementation identity must be explicit")
    durable_mkdir(root.parent)
    stage = Path(tempfile.mkdtemp(prefix=".ks-run-", dir=root.parent))
    run_id = uuid4().hex
    owner = {"id": objects["budget_account"]["owner"] if "budget_account" in objects else run_id,
             "account": fixed["budget"]["parent"]}
    try:
        initial = {"run_id": run_id, "operation": operation, "request_sha256": digest(fixed), "backend": fixed["backend"], "implementation": backend_identity, "owner": owner}
        durable_write(stage / ".runtime/run-intent.json", jcs(initial))
        import_bundle(source, stage, dependencies(fixed))
        publication = Publication(stage)
        request_ref = publication.put_json("request.json", fixed)
        publication.commit([request_ref])
        anchor = {**initial, "request": request_ref}
        durable_write(stage / ".runtime/run.json", jcs(anchor))
        durable_write(stage / ".runtime/owner-binding.json", jcs(owner))
        durable_write(stage / "request.json", jcs(fixed))
        fsync_directory(stage)
        _rename_new(stage, root); fsync_directory(root.parent)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return open_run(operation, root, backend_identity=backend_identity)


def open_run(operation: str, root: str | Path, *, backend_identity: dict | None = None) -> RunContext:
    """Validate original run identities even before the first result exists."""
    root = Path(root).absolute()
    bundle = Bundle(root)
    anchor = loads(read_regular(root, ".runtime/run.json")[0])
    required = {"run_id", "operation", "request_sha256", "backend", "implementation", "owner", "request"}
    if not isinstance(anchor, dict) or set(anchor) != required or anchor["operation"] != operation:
        raise ProtocolError("run anchor missing/corrupt or operation mismatch")
    _name(anchor["run_id"])
    original = bundle.document(anchor["request"])
    projection = loads(read_regular(root, "request.json")[0])
    if original != projection or digest(original) != anchor["request_sha256"] or original["operation"] != operation:
        raise ProtocolError("original request/Q changed; continue cannot override inputs")
    if original["backend"] != anchor["backend"]:
        raise ProtocolError("backend/profile changed")
    if backend_identity is not None and backend_identity != anchor["implementation"]:
        raise ProtocolError("backend implementation changed; cannot resume")
    owner = loads(read_regular(root, ".runtime/owner-binding.json")[0])
    if owner != anchor["owner"]: raise ProtocolError("owner binding missing/corrupt; cannot allocate a new owner")
    objects = validate_request(bundle, original)
    expected_owner = {"id": objects["budget_account"]["owner"] if "budget_account" in objects else anchor["run_id"], "account": original["budget"]["parent"]}
    if owner != expected_owner: raise ProtocolError("owner differs from frozen budget account")
    context = RunContext(root, anchor, original, objects)
    context.intents()
    return context


def validate_result(context: RunContext, result: dict):
    validate(result)
    expected = {"operation": context.request["operation"], "run_id": context.run_id,
                "request_id": context.request["request_id"], "request_sha256": context.request_sha256}
    if any(result.get(k) != v for k, v in expected.items()): raise ProtocolError("result belongs to a different run/request")
    bundle = context.bundle
    bundle.closure(dependencies(result), subjects=[context.request_ref])
    base = {"request_sha256": context.request_sha256, **{key + "_sha256": context.request[key]["sha256"] for key in ("task", "rules", "scope")}}
    if result["operation"] == "implement":
        base["schema_sha256"] = context.request["schema"]["sha256"]
        for assessment in result["assessments"]:
            for name in ("coverage", "build", "correctness", "fidelity", "performance"):
                item = assessment[name]
                if item["receipt"] is None: continue
                kind = "fidelity" if name in ("coverage", "fidelity") else "evaluation"
                report = bundle.document(item["receipt"], kind)
                expected_subject = {**base, "source_sha256": assessment["source_sha256"]}
                if kind == "evaluation":
                    expected_subject = {"task_sha256": base["task_sha256"], "source_sha256": assessment["source_sha256"], "evaluator_sha256": context.objects["task"]["evaluator"]["sha256"]}
                if any(report["subject"].get(k) != v for k, v in expected_subject.items()):
                    raise ProtocolError(f"{name} evidence binds another actual subject")
                if report[name]["status"] != item["status"]:
                    raise ProtocolError(f"{name} summary contradicts its original report")
    else:
        if result["schema"] is not None:
            schema = bundle.document(result["schema"], "schema")
            if schema["task_sha256"] != base["task_sha256"]: raise ProtocolError("writing output Task mismatch")
        if result["scope"] is not None:
            scope = bundle.document(result["scope"], "scope")
            if result["status"] == "completed" and not scope["bindings"]: raise ProtocolError("completed writing must have nonempty scope bindings")
        for name, item in result["writing_checks"].items():
            if item["receipt"] is None: continue
            report = bundle.document(item["receipt"], "writing_review")
            expected_subject = {**base, "schema_sha256": result["schema"]["sha256"], "scope_sha256": result["scope"]["sha256"]}
            if report["subject"] != expected_subject or report["checks"][name]["status"] != item["status"]:
                raise ProtocolError("writing review binds another output or contradicts summary")
    return result


def _invoke(operation, *, request=None, out=None, continue_run=None):
    if continue_run is not None:
        if request is not None or out is not None: raise ProtocolError("request/out and continue_run are mutually exclusive")
        context = open_run(operation, continue_run)
        backend = _backend(context.request["backend"]["name"], operation)
        identity = backend.identity(context.objects["profile"])
        context = open_run(operation, continue_run, backend_identity=identity)
        require_capabilities(context.request, context.objects, backend.capabilities(context.objects["profile"]))
        backend.validate_resume(context)  # Owner verifies its actual ledger, fences and native handles.
    else:
        if request is None or out is None: raise ProtocolError("provide request and out, or continue_run")
        path = Path(request).absolute()
        fixed = validate(loads(read_regular(path.parent, path.name)[0]))
        if fixed.get("operation") != operation or "run_id" in fixed: raise ProtocolError("request operation mismatch")
        objects = validate_request(Bundle(path.parent), fixed)
        backend = _backend(fixed["backend"]["name"], operation)
        require_capabilities(fixed, objects, backend.capabilities(objects["profile"]))
        context = prepare_run(operation, request=request, out=out, backend_identity=backend.identity(objects["profile"]))
    result = backend.execute(context, continuing=continue_run is not None)
    return context.publish_result(result)


def design(*, request=None, out=None, continue_run=None):
    return _invoke("design", request=request, out=out, continue_run=continue_run)


def optimize(*, request=None, out=None, continue_run=None):
    return _invoke("optimize", request=request, out=out, continue_run=continue_run)


def implement(*, request=None, out=None, continue_run=None):
    return _invoke("implement", request=request, out=out, continue_run=continue_run)
