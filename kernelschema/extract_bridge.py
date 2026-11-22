"""Consume existing raw/receipt bytes without changing the extraction lifecycle."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from .artifacts import Bundle, Publication, _metadata, read_regular
from .contracts import PROTOCOL, ProtocolError, digest, loads, sha256, validate
from .materials import validate_request
from .resources import resource
from .template_api import tools


def extract(argv: list[str]) -> int:
    # A separate interpreter preserves the original two tools namespaces and all
    # original argument, P1-P6, recovery, publication and exit-code semantics.
    result = subprocess.run([sys.executable, "-B", str(resource("schema_extract/extract_yaml.py")), *argv], check=False)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


def consume_raw(publication: Publication, directory: str | Path, *, task: dict,
                origin: str = "handwritten", receipt: str | Path | None = None,
                name="schema") -> dict:
    """Snapshot actual complete raw Template files and optional native receipt.

    The return contains a Schema Ref and a separate provenance/receipt Ref. It
    is not workflow qualification and never edits or reopens the extraction.
    All sources (handwritten, curated, design, extract) use the same raw path.
    """
    from .contracts import ref, relative_path
    ref(task); relative_path(name)
    if origin not in ("handwritten", "curated", "design", "extract"):
        raise ProtocolError("unsupported provenance label")
    directory = Path(directory).absolute()
    engine = tools()
    try:
        files, documents = engine.read_directory(directory)
        engine.resolve_documents(files, documents)
    except (engine.SchemaError, engine.ToolError) as error:
        raise ProtocolError(f"cannot consume incomplete raw Schema: {error}") from error
    refs = []
    for node, path in sorted(files.items()):
        data, mode = read_regular(directory, path.name)
        # Reuse the parser's original source identity rather than a text rewrite.
        if sha256(data).split(":", 1)[1] != documents[node].source_sha256:
            raise ProtocolError("raw source changed during extraction consumption")
        refs.append({"node": node, "file": publication.put_bytes(f"{name}/{path.name}", data, mode=mode)})
    dispatch = None
    if (directory / "dispatch.md").exists():
        data, mode = read_regular(directory, "dispatch.md")
        errors = engine.dispatch.validate(data.decode("utf-8"), files)
        if errors: raise ProtocolError("invalid actual dispatch: " + "; ".join(e.message for e in errors))
        dispatch = publication.put_bytes(f"{name}/dispatch.md", data, mode=mode)
    elif len(refs) > 1: raise ProtocolError("complete multi-node Schema requires dispatch")
    receipt_ref = None
    if receipt is not None:
        receipt_path = Path(receipt).absolute()
        raw, mode = read_regular(receipt_path.parent, receipt_path.name)
        proof = _metadata(raw)
        if not isinstance(proof, dict) or proof.get("format") != "schema_extract.delivery.v2" or not isinstance(proof.get("files"), dict):
            raise ProtocolError("expected original schema_extract.delivery.v2 receipt")
        covered = {entry["node"]: Path(entry["file"]["path"]).name for entry in refs}
        required = list(covered.values()) + (["dispatch.md"] if dispatch else [])
        if any(filename not in proof["files"] for filename in required):
            raise ProtocolError("native receipt does not cover the actual raw/dispatch being consumed")
        # Native receipt edges are explicitly specified by its files manifest;
        # copy each original dependency, not the run's unrelated private history.
        for filename, identity in proof["files"].items():
            data, file_mode = read_regular(directory, filename)
            if not isinstance(identity, str): raise ProtocolError("native receipt file digest must be a string")
            expected = identity if identity.startswith("sha256:") else "sha256:" + identity
            if sha256(data) != expected: raise ProtocolError(f"native receipt dependency mismatch: {filename}")
            publication.put_bytes(f"{name}/{filename}", data, mode=file_mode)
        receipt_ref = publication.put_bytes(f"{name}/.extraction/receipt.json", raw, mode=mode)
    schema = {"protocol": PROTOCOL, "kind": "schema", "task_sha256": task["sha256"], "raw": refs,
              "dispatch": dispatch, "design_revision": 0, "parent": None, "change": "new_task", "change_report": None,
              "extensions": {"information": {"origin": origin, "historical_usage": "unknown; not current request usage"}}}
    schema_ref = publication.put_json(f"{name}/manifest.json", schema)
    return {"schema": schema_ref, "receipt": receipt_ref, "historical_usage": None,
            "historical_usage_reason": "Native historical initialization costs are separate; no zero-cost claim"}


def implement_request(*, request_id: str, task: dict, rules: dict, materials: dict, scope: dict,
                      budget: dict, backend: dict, schema: dict, seed: dict | None, mode="code") -> dict:
    """Build the common fixed request after the caller supplies the full Task."""
    request = {"protocol": PROTOCOL, "operation": "implement", "request_id": request_id,
               "task": task, "rules": rules, "materials": materials, "scope": scope, "budget": budget,
               "backend": backend, "schema": schema, "seed": seed, "mode": mode}
    return validate(request)
