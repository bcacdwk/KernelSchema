"""Read-only whole-task checking. Markdown defines the language; checks help authors."""
from __future__ import annotations

from pathlib import Path

from . import rules, yaml_source
from .diagnostics import format_diagnostics
from .rules.values import node_id

class SchemaError(Exception):
    """Invalid YAML or task relations (exit 1)."""


class ToolError(Exception):
    """The check could not be completed (exit 2)."""


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


def validate_directory(directory: str | Path) -> dict:
    directory = Path(directory).resolve()
    files = task_files(directory)
    documents = {}
    for name, path in files.items():
        try:
            documents[name] = yaml_source.load(path, name)
        except yaml_source.ReadError as error:
            raise SchemaError(str(error)) from error
        except yaml_source.DependencyError as error:
            raise ToolError(str(error)) from error
    task = {name: document.value for name, document in documents.items()}
    errors = rules.validate_structure(task)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents) + "\nFix field structure before checking named relations.")
    errors = rules.validate_relations(task)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents))
    return task


def validate_file(filename: str | Path) -> tuple[Path, dict]:
    # A file symlink keeps its node name and siblings, not its target's directory.
    entry = Path(filename).expanduser()
    directory = entry.parent.resolve()
    entry = directory / entry.name
    if entry.suffix not in {".yaml", ".yml"} or not entry.is_file():
        raise ToolError(f"{entry}: pass one existing .yaml or .yml file (root or a derived node).")
    return directory, validate_directory(directory)
