"""Read-only whole-task checking. Markdown defines the schema; checks help authors."""
from __future__ import annotations

from pathlib import Path

from . import check_policy, rules, yaml_source
from .diagnostics import format_diagnostics
from .errors import ToolError, bounded
from .rules.values import node_id

class SchemaError(Exception):
    """Invalid YAML or task relations (exit 1)."""


def task_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise SchemaError(f"{directory}: pass a YAML task directory.")
    files = {}
    for file in sorted(directory.iterdir()):
        if file.suffix not in {".yaml", ".yml"}:
            continue
        if not file.is_file():
            raise SchemaError(f"{file}: expected a YAML file.")
        identity_errors = []
        node_id(file.stem, "<file>", identity_errors)
        if identity_errors:
            raise SchemaError(f"{file}: {identity_errors[0].message}")
        if file.stem in files:
            raise SchemaError(f"{file}: duplicate node ID {file.stem}; also defined by {files[file.stem]}.")
        files[file.stem] = file
    if "root" not in files:
        raise SchemaError(f"{directory}: missing root.yaml (or root.yml); a task needs one shared Contract.")
    return files


@bounded
def validate_directory(directory: str | Path, *, notices: list[str] | None = None,
                       source_hashes: dict[str, str] | None = None) -> dict:
    directory = Path(directory).resolve()
    files = task_files(directory)
    try:
        policy = check_policy.for_directory(directory)
    except check_policy.PolicyError as error:
        raise ToolError(str(error)) from error
    if policy.allow_missing_hardware and notices is not None:
        notices.append(policy.notice())
    documents = {}
    for name, path in files.items():
        try:
            documents[name] = yaml_source.load(path, name)
        except yaml_source.ReadError as error:
            raise SchemaError(str(error)) from error
        except yaml_source.DependencyError as error:
            raise ToolError(str(error)) from error
    task = {name: document.value for name, document in documents.items()}
    errors = rules.validate_structure(task, allow_missing_hardware=policy.allow_missing_hardware)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents) +
                          "\nFix the reported fields and rerun the same command. "
                          "Named relations will be checked automatically after field structure passes.")
    errors = rules.validate_relations(task, allow_missing_hardware=policy.allow_missing_hardware)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents))
    if source_hashes is not None:
        source_hashes.clear()
        source_hashes.update({name: document.source_sha256 for name, document in documents.items()})
    return task


def validate_file(filename: str | Path, *, notices: list[str] | None = None,
                  source_hashes: dict[str, str] | None = None) -> tuple[Path, dict]:
    # A file symlink keeps its node name and siblings, not its target's directory.
    entry = Path(filename).expanduser()
    directory = entry.parent.resolve()
    entry = directory / entry.name
    if entry.suffix not in {".yaml", ".yml"} or not entry.is_file():
        raise ToolError(f"{entry}: pass one existing .yaml or .yml file (root or a derived node).")
    return directory, validate_directory(directory, notices=notices, source_hashes=source_hashes)
