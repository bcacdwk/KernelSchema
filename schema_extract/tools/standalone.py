"""Export validated local Design and Contract with the complete raw source closure."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from . import __version__
from .artifacts import schema_for, write_yaml
from .common import ExtractError, fsync_directory, manifest, read_json, sha256, sync_tree, write_json
from .publication import rename_new


def _frozen_view(source: Path, output: Path, repo: Path) -> dict:
    """Keep old inline exports on their own complete frozen implementation."""
    entry = repo / "schema_extract/extract.py"
    exporter = repo / "schema_extract/tools/standalone.py"
    if not entry.is_file() or not exporter.is_file() or exporter.resolve() == Path(__file__).resolve():
        raise ExtractError("selected rules lack a matching frozen standalone exporter")
    # A copied current exporter with an older schema is an inconsistent rules
    # tree, not a reason to recurse or interpret old rules with the new resolver.
    marker = "KERNEL_SCHEMA_STANDALONE_RULES"
    if os.environ.get(marker) == str(repo):
        raise ExtractError("selected frozen standalone exporter and schema API are inconsistent")
    environment = {**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1", marker: str(repo)}
    try:
        result = subprocess.run([sys.executable, "-B", str(entry), "standalone", str(source),
                                 "--output", str(output), "--repo", str(repo)],
                                cwd=repo, env=environment, capture_output=True, text=True)
    except OSError as exc:
        raise ExtractError("cannot run the selected frozen standalone exporter: " + str(exc)) from exc
    if result.returncode:
        raise ExtractError(result.stderr.strip() or result.stdout.strip() or "frozen standalone export failed")
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ExtractError("frozen standalone exporter returned invalid JSON") from exc
    if not isinstance(receipt, dict):
        raise ExtractError("frozen standalone exporter returned an invalid result")
    return receipt


def _verify_sources(directory: Path, inputs: dict[str, str]) -> None:
    for name, checksum in inputs.items():
        try:
            current = sha256((directory / name).read_bytes())
        except OSError as exc:
            raise ExtractError("source changed while generating standalone view: " + name) from exc
        if current != checksum:
            raise ExtractError("source changed while generating standalone view: " + name)


def create_view(source: Path, output: Path, repo: Path) -> dict:
    source, output, repo = source.expanduser().absolute(), output.expanduser().absolute(), repo.resolve()
    if output.resolve().is_relative_to(source.parent.resolve()):
        raise ExtractError("standalone output must be outside the source task directory")
    schema = schema_for(repo)
    if not hasattr(schema, "validate_resolved_file"):
        return _frozen_view(source, output, repo)
    identity = source.stem
    try:
        directory, resolved = schema.validate_resolved_file(source, skip_hardware=True)
        value = resolved.effective_document(identity)
        inputs = resolved.source_manifest(identity)
        origins = resolved.origins_dict(identity)
        effective_identity = resolved.effective_identity(identity)
    except (OSError, schema.SchemaError, schema.ToolError) as exc:
        raise ExtractError(str(exc)) from exc
    _verify_sources(directory, inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".standalone-", dir=output.parent))
    try:
        destination = temporary / source.name
        write_yaml(destination, value, source=source, parser_root=repo)
        try:
            schema.validate_standalone_file(destination, skip_hardware=True)
        except (OSError, schema.SchemaError, schema.ToolError) as exc:
            raise ExtractError(str(exc)) from exc
        receipt = {
            "format": "schema_extract.standalone.v2", "tool_version": __version__,
            "node": identity, "source_directory": str(directory),
            "source": resolved.source_files[identity].name,
            "root": resolved.source_files["root"].name,
            "source_files": inputs, "origins": origins, "effective_identity": effective_identity,
            "view": {source.name: sha256(destination.read_bytes())},
            "check_policy": "skip-hardware",
        }
        write_json(temporary / ".source.json", receipt)
        sync_tree(temporary)
        # Include receipt generation and syncing in the guarded interval. Both
        # reuse and a new atomic publication must still match every raw source.
        _verify_sources(directory, inputs)
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.is_dir() or manifest(output) != manifest(temporary):
                raise ExtractError("standalone destination already contains different content")
            if read_json(output / ".source.json") != receipt:
                raise ExtractError("standalone source receipt differs")
            _verify_sources(directory, inputs)
            return {"result": "reused", "output": str(output), **receipt}
        try:
            rename_new(temporary, output)
            fsync_directory(output.parent)
        except FileExistsError:
            # A concurrent writer may have published the identical view.
            if (output.is_symlink() or not output.is_dir() or manifest(output) != manifest(temporary)
                    or read_json(output / ".source.json") != receipt):
                raise
            _verify_sources(directory, inputs)
        return {"result": "created", "output": str(output), **receipt}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
