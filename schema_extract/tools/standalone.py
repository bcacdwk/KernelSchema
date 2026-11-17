"""Export validated local Design and Contract with the complete raw source closure."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from . import __version__
from .artifacts import schema_for, write_yaml
from .common import ExtractError, digest, fsync_directory, manifest, read_json, sha256, sync_tree, write_json
from .publication import rename_new


def _verify_sources(directory: Path, inputs: dict[str, str]) -> None:
    current = manifest(directory)
    if current != inputs:
        changed = sorted(name for name in current.keys() | inputs.keys() if current.get(name) != inputs.get(name))
        raise ExtractError("source changed while generating standalone view: " + ", ".join(changed[:12]))


def create_view(source: Path, output: Path, repo: Path) -> dict:
    source, output, repo = source.expanduser().absolute(), output.expanduser().absolute(), repo.resolve()
    if output.resolve().is_relative_to(source.parent.resolve()):
        raise ExtractError("standalone output must be outside the source task directory")
    # The complete authoring directory is the validation subject. Actual
    # content dependencies remain the resolver's smaller source_manifest.
    original = manifest(source.parent)
    schema = schema_for(repo)
    if not all(callable(getattr(schema, name, None)) for name in ("validate_resolved_file", "validate_standalone_file")):
        raise ExtractError("standalone requires the current Template validation API")
    identity = source.stem
    template_files = {path.relative_to(repo).as_posix(): sha256(path.read_bytes())
                      for path in (repo / "schema_template/tools").rglob("*.py") if "tests" not in path.parts}
    try:
        directory, resolved = schema.validate_resolved_file(source, skip_hardware=True)
        value = resolved.effective_document(identity)
        inputs = resolved.source_manifest(identity)
        origins = resolved.origins_dict(identity)
        effective_identity = resolved.effective_identity(identity)
    except (OSError, schema.SchemaError, schema.ToolError) as exc:
        raise ExtractError(str(exc)) from exc
    _verify_sources(directory, original)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".standalone-", dir=output.parent))
    try:
        destination = temporary / source.name
        write_yaml(destination, value, source=source, parser_root=repo)
        try:
            _, generated = schema.validate_standalone_file(destination, skip_hardware=True)
            if generated != {identity: value}:
                raise ExtractError("standalone bytes differ from the validated effective document")
        except (OSError, schema.SchemaError, schema.ToolError) as exc:
            raise ExtractError(str(exc)) from exc
        receipt = {
            "format": "schema_extract.standalone.v2", "tool_version": __version__,
            "node": identity, "source_directory": str(directory),
            "source": resolved.source_files[identity].name,
            "root": resolved.source_files["root"].name,
            "source_files": inputs, "origins": origins, "effective_identity": effective_identity,
            "validated_directory_files": original,
            "template": {"files": template_files, "identity": digest(template_files)},
            "view": {source.name: sha256(destination.read_bytes())},
            "check_policy": "skip-hardware",
        }
        write_json(temporary / ".source.json", receipt)
        sync_tree(temporary)
        # Include receipt generation and syncing in the guarded interval. Both
        # reuse and a new atomic publication must still match every raw source.
        _verify_sources(directory, original)
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.is_dir() or manifest(output) != manifest(temporary):
                raise ExtractError("standalone destination already contains different content")
            if read_json(output / ".source.json") != receipt:
                raise ExtractError("standalone source receipt differs")
            _verify_sources(directory, original)
            return {"result": "reused", "output": str(output), **receipt}
        try:
            rename_new(temporary, output)
            fsync_directory(output.parent)
        except FileExistsError:
            # A concurrent writer may have published the identical view.
            if (output.is_symlink() or not output.is_dir() or manifest(output) != manifest(temporary)
                    or read_json(output / ".source.json") != receipt):
                raise ExtractError("standalone destination already contains different content")
            _verify_sources(directory, original)
        return {"result": "created", "output": str(output), **receipt}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
