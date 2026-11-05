"""Materialize a current node and its shared Contract without changing either."""

from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path
import shutil
import tempfile

from . import __version__
from .artifacts import parser_for, write_yaml
from .common import ExtractError, fsync_directory, manifest, read_json, sha256, sync_tree, write_json
from .publication import rename_new


def create_view(source: Path, output: Path, repo: Path) -> dict:
    source, output, repo = source.expanduser().absolute(), output.expanduser().absolute(), repo.resolve()
    if output.resolve().is_relative_to(source.parent.resolve()):
        raise ExtractError("standalone output must be outside the source task directory")
    parser = parser_for(str(repo))
    schema = importlib.import_module(parser.__package__ + ".schema")
    hashes = {}
    try:
        directory, task = schema.validate_file(source, source_hashes=hashes, skip_hardware=True)
    except (schema.SchemaError, schema.ToolError) as exc:
        raise ExtractError(str(exc)) from exc
    identity = source.stem
    value = deepcopy(task[identity])
    value["contract"] = deepcopy(task["root"]["contract"])
    inputs = {name + ".yaml": hashes[name] for name in {"root", identity}}
    # Locate the actual accepted suffix; .yml is also a task node.
    inputs = {schema.task_files(directory)[Path(name).stem].name: checksum
              for name, checksum in inputs.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".standalone-", dir=output.parent))
    try:
        destination = temporary / source.name
        write_yaml(destination, value, source=source, parser_root=repo)
        try:
            schema.validate_standalone_file(destination, skip_hardware=True)
        except (schema.SchemaError, schema.ToolError) as exc:
            raise ExtractError(str(exc)) from exc
        for name, checksum in inputs.items():
            if sha256((directory / name).read_bytes()) != checksum:
                raise ExtractError("source changed while generating standalone view: " + name)
        receipt = {
            "format": "schema_extract.standalone.v1", "tool_version": __version__,
            "node": identity, "source_directory": str(directory),
            "source_files": inputs, "view": {source.name: sha256(destination.read_bytes())},
            "check_policy": "skip-hardware",
        }
        write_json(temporary / ".source.json", receipt)
        sync_tree(temporary)
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.is_dir() or manifest(output) != manifest(temporary):
                raise ExtractError("standalone destination already contains different content")
            if read_json(output / ".source.json") != receipt:
                raise ExtractError("standalone source receipt differs")
            return {"result": "reused", "output": str(output), **receipt}
        try:
            rename_new(temporary, output)
            fsync_directory(output.parent)
        except FileExistsError:
            # A concurrent writer may have published the identical view.
            if (output.is_symlink() or not output.is_dir() or manifest(output) != manifest(temporary)
                    or read_json(output / ".source.json") != receipt):
                raise
        return {"result": "created", "output": str(output), **receipt}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
