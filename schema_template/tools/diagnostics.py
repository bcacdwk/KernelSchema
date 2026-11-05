"""Diagnostic records, unambiguous field paths and author-facing rendering."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
DOCS_ROOT = Path(__file__).resolve().parents[2] / "schema_docs"


@dataclass(frozen=True)
class Diagnostic:
    path: str
    message: str
    missing: bool = False
    hint: str = ""

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def field_path(path: str, key: str) -> str:
    return f"{path}.{key}" if IDENTIFIER.fullmatch(key) else f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def encode_value(value: Any) -> str:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("JSON numbers must be finite")
        return str(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return "{" + ",".join(json.dumps(key, ensure_ascii=False) + ":" +
                              encode_value(child) for key, child in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(encode_value(item) for item in value) + "]"
    if value is None or type(value) in (str, int, float, bool):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise TypeError(f"not a JSON value: {type(value).__name__}")


def format_diagnostics(errors, files, documents, *, resolved=None):
    sections = {"meta_data": "0.common.md", "contract.algorithm_requirement": "1.contract.algorithm_requirement.md",
                "contract.hardware_specification": "2.contract.hardware_specification.md",
                "design.math_reshape": "3.design.math_reshape.md", "design.work_schedule": "4.design.work_schedule.md",
                "design.tile_pipeline": "5.design.tile_pipeline.md", "realization": "6.realization.md"}
    messages = []
    for error in errors:
        node = next(name for name in sorted(files, key=len, reverse=True)
                    if error.path == name or error.path.startswith(name + ".") or error.path.startswith(name + "["))
        provider, source_path, origin = (resolved.diagnostic_source(node, error.path) if resolved else
                                         (node, error.path, None))
        document = documents[provider]
        field = error.path[len(node):].lstrip(".") or "<file>"
        position = document.positions.get(source_path) if not error.missing else None
        location = str(files[provider]) + (f":{position[0]}:{position[1]}" if position else "")
        lines = [location, f"  {field}: {error.message}"]
        if provider != node:
            region = error.path[len(node) + len(".design."):].split(".", 1)[0].split("[", 1)[0]
            use_path = f"{node}.design.{region}.from_parent"
            use_position = documents[node].positions.get(use_path)
            use_location = str(files[node]) + (f":{use_position[0]}:{use_position[1]}" if use_position else "")
            lines.append(f"  Consumer: {error.path}")
            lines.append(f"  Inherited at: {use_location} ({use_path})")
            lines.append("  Source chain: " + " -> ".join(origin.chain))
        if not error.missing and source_path in document.values:
            actual = encode_value(document.values[source_path])
            lines.append("  Actual: " + (actual if len(actual) <= 160 else actual[:157] + "..."))
        if error.hint:
            lines.append("  Fix: " + error.hint)
        for prefix, filename in sections.items():
            if field == prefix or field.startswith(prefix + "."):
                lines.append(f"  See: {DOCS_ROOT / filename}")
                break
        if node != "root" and ("Contract" in error.message or "shared task" in error.message or "hardware" in error.message):
            lines.append(f"  Shared root: {files['root']}")
        messages.append("\n".join(lines))
    return "\n\n".join(messages)
