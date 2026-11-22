"""Adapter to the existing Template implementation, under one private name.

No copied parser, global sys.path changes, or bare `tools` modules. Reading a
non-root node uses the real effective_document (including root Contract).
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import threading

from .contracts import ProtocolError
from .resources import resource

_LOCK = threading.RLock()
_PACKAGE = "kernelschema._template_tools"


def tools():
    with _LOCK:
        if _PACKAGE not in sys.modules:
            directory = resource("schema_template/tools/__init__.py").parent
            spec = importlib.util.spec_from_file_location(_PACKAGE, directory / "__init__.py", submodule_search_locations=[str(directory)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[_PACKAGE] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(_PACKAGE, None)
                raise
        return importlib.import_module(_PACKAGE + ".schema")


def resolve_bundle(bundle, schema_manifest, *, validate_structure=False, skip_hardware=False):
    """Parse only explicit raw Refs and resolve real from chains; never glob input."""
    from .contracts import validate
    validate(schema_manifest)
    if schema_manifest["kind"] != "schema": raise ProtocolError("expected Schema")
    engine = tools()
    files, documents = {}, {}
    try:
        for entry in schema_manifest["raw"]:
            node, reference = entry["node"], entry["file"]
            path = Path(reference["path"])
            if path.stem != node or path.suffix not in (".yaml", ".yml"):
                raise ProtocolError("raw filename must retain actual Template node identity")
            files[node] = bundle.root / path
            text = bundle.read(reference).decode("utf-8", errors="strict")
            documents[node] = engine.yaml_source.loads(text, node, path=files[node])
        resolved = engine.validate_documents(files, documents, skip_hardware=skip_hardware) if validate_structure else engine.resolve_documents(files, documents)
        if schema_manifest["dispatch"] is not None:
            content = bundle.read(schema_manifest["dispatch"]).decode("utf-8", errors="strict")
            errors = engine.dispatch.validate(content, files)
            if errors: raise ProtocolError("; ".join(e.message for e in errors))
        elif len(files) > 1:
            raise ProtocolError("multi-node Schema requires its actual dispatch")
        return resolved
    except (engine.SchemaError, engine.ToolError, UnicodeError) as error:
        raise ProtocolError(f"Template: {error}") from error


def effective_document(bundle, schema_manifest, node: str) -> dict:
    resolved = resolve_bundle(bundle, schema_manifest)
    if node not in resolved.raw_task: raise ProtocolError(f"unknown Schema node {node}")
    return resolved.effective_document(node)


def normalize_directory(directory: str | Path, *, skip_hardware=False):
    """Return the original normalization plan; caller freezes new candidate bytes."""
    return tools().normalize_directory(directory, skip_hardware=skip_hardware)


def check_directory(directory: str | Path, *, skip_hardware=False):
    return tools().validate_directory(directory, skip_hardware=skip_hardware)
