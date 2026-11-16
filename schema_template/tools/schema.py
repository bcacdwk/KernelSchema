"""Read-only checks and normalization plans; the CLI owns source writes."""
from __future__ import annotations

from pathlib import Path

from . import rules, text_policy, yaml_source
from .design_resolution import (RegionOrigin, ResolvedTask, effective_yaml, read_directory,
                                resolve_directory, resolve_documents, task_files)
from .diagnostics import format_diagnostics
from .errors import SchemaError, ToolError, bounded
from .normalization import (LoadedDocument, NormalizedTask, load_for_normalization,
                            loads_for_normalization, normalize_documents)
from .rules import dispatch, values


@bounded
def normalize_directory(directory: str | Path, *, skip_hardware: bool = False) -> NormalizedTask:
    """Read a complete task and plan its canonical snapshot without writing.

    Candidates carry their own parsed source marks and hashes. ``source_hashes``
    separately identifies the original disk bytes for compare-before-write.
    Call ``validate_documents`` on the resulting documents to check that snapshot;
    filesystem checks remain separate. Existing ``validate_*`` APIs stay strict.
    """
    files = task_files(Path(directory).expanduser().resolve())
    documents = {}
    for name, path in files.items():
        try:
            documents[name] = load_for_normalization(path, name, skip_hardware=skip_hardware)
        except yaml_source.ReadError as error:
            raise SchemaError(str(error)) from error
        except (yaml_source.DependencyError, OSError) as error:
            raise ToolError(str(error)) from error
    return normalize_documents(files, documents, skip_hardware=skip_hardware)


def _hardware_notice(notices, skip_hardware):
    if skip_hardware and notices is not None:
        notices.append("Hardware skipped explicitly: CONTRACT.Hardware_specification and hardware-dependent relations were not validated")


def _text_diagnostics(files, documents, notices, skip_hardware):
    diagnostics = [item for document in documents.values()
                   for item in text_policy.check_document(document, skip_hardware=skip_hardware)]
    warnings = [item for item in diagnostics if item.severity == "warning"]
    if warnings and notices is not None:
        notices.append(format_diagnostics(warnings, files, documents))
    return [item for item in diagnostics if item.severity != "warning"]


def validate_dispatch(files: dict[str, Path]) -> None:
    """Read the task index once; never interpret its prose as YAML changes."""
    directory = files["root"].parent
    path = directory / "dispatch.md"
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise SchemaError(f"{path}: required task index is missing") from error
    except UnicodeError as error:
        raise SchemaError(f"{path}: expected UTF-8 Markdown") from error
    except OSError as error:
        raise ToolError(f"{path}: cannot read task index: {error}") from error
    errors = dispatch.validate(content, files)
    if errors:
        raise SchemaError("\n".join(f"{directory / item.path}: {item.message}" for item in errors))


def validate_local_repositories(files: dict[str, Path], documents: dict, *, skip_hardware=False) -> None:
    """Check locators against identifiable local revisions without reloading YAML."""
    errors = values.local_repository_errors(files, documents, skip_hardware=skip_hardware)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents))


@bounded
def validate_documents(files: dict, documents: dict, *, notices: list[str] | None = None,
                       source_hashes: dict[str, str] | None = None,
                       skip_hardware: bool = False) -> ResolvedTask:
    """Validate an already-read canonical snapshot without reading or writing.

    No implicit recovery or normalization occurs. Opt-in callers can first use
    ``normalize_documents`` and label those candidates as previews. The CLI uses
    actual reread documents after writing. Text budgets count raw definitions
    once; dispatch and local repositories are separate file checks.
    """
    if files.keys() != documents.keys():
        raise ToolError("source files and parsed documents must name the same snapshot nodes")
    _hardware_notice(notices, skip_hardware)
    task = {name: document.value for name, document in documents.items()}
    errors = rules.validate_structure(task, skip_hardware=skip_hardware, allow_from=True)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents) +
                          "\nFix the reported fields and rerun the same command. "
                          "Named relations are checked after field structure passes.")
    text_errors = _text_diagnostics(files, documents, notices, skip_hardware)
    try:
        resolved = resolve_documents(files, documents)
    except SchemaError as error:
        prefix = format_diagnostics(text_errors, files, documents)
        raise SchemaError((prefix + "\n\n" if prefix else "") + str(error)) from error
    errors = rules.validate_relations(resolved.effective_task, skip_hardware=skip_hardware)
    messages = [format_diagnostics(text_errors, files, documents),
                format_diagnostics(errors, files, documents, resolved=resolved)]
    if any(messages):
        raise SchemaError("\n\n".join(message for message in messages if message))
    if source_hashes is not None:
        source_hashes.clear()
        source_hashes.update(resolved.source_hashes)
    return resolved


def _validate_resolved_directory(directory: Path, *, notices=None, source_hashes=None,
                                 skip_hardware=False) -> ResolvedTask:
    files, documents = read_directory(directory)
    resolved = validate_documents(files, documents, notices=notices, source_hashes=source_hashes,
                                  skip_hardware=skip_hardware)
    validate_local_repositories(files, documents, skip_hardware=skip_hardware)
    validate_dispatch(files)
    return resolved


@bounded
def validate_directory(directory: str | Path, *, notices: list[str] | None = None,
                       source_hashes: dict[str, str] | None = None, skip_hardware: bool = False) -> dict:
    return _validate_resolved_directory(Path(directory).expanduser().resolve(), notices=notices,
                                       source_hashes=source_hashes, skip_hardware=skip_hardware).effective_task


def validate_file(filename: str | Path, *, notices: list[str] | None = None,
                  source_hashes: dict[str, str] | None = None, skip_hardware: bool = False) -> tuple[Path, dict]:
    directory, resolved = validate_resolved_file(filename, notices=notices, source_hashes=source_hashes,
                                                skip_hardware=skip_hardware)
    return directory, resolved.effective_task


@bounded
def validate_resolved_file(filename: str | Path, *, notices: list[str] | None = None,
                           source_hashes: dict[str, str] | None = None,
                           skip_hardware: bool = False) -> tuple[Path, ResolvedTask]:
    # A file symlink retains its node filename and directory of siblings.
    entry = Path(filename).expanduser()
    directory = entry.parent.resolve()
    entry = directory / entry.name
    if entry.suffix not in {".yaml", ".yml"} or not entry.is_file():
        raise ToolError(f"{entry}: pass one existing .yaml or .yml file")
    return directory, _validate_resolved_directory(directory, notices=notices, source_hashes=source_hashes,
                                                   skip_hardware=skip_hardware)


@bounded
def validate_standalone_file(filename: str | Path, *, notices: list[str] | None = None,
                             source_hashes: dict[str, str] | None = None,
                             skip_hardware: bool = False) -> tuple[Path, dict]:
    """Check a self-contained delivery view, retaining its ID and reading no siblings."""
    entry = Path(filename).expanduser().absolute()
    if entry.suffix not in {".yaml", ".yml"} or not entry.is_file():
        raise ToolError(f"{entry}: pass one existing standalone .yaml or .yml file")
    label = entry.stem
    try:
        document = yaml_source.load(entry, label)
    except yaml_source.ReadError as error:
        raise SchemaError(str(error)) from error
    except (yaml_source.DependencyError, OSError) as error:
        raise ToolError(str(error)) from error
    _hardware_notice(notices, skip_hardware)
    original = document.value
    files, documents, task = {label: entry}, {label: document}, {label: original}
    errors = rules.validate_structure(task, skip_hardware=skip_hardware, standalone=True)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents))
    text_errors = _text_diagnostics(files, documents, notices, skip_hardware)
    errors = rules.validate_relations(task, skip_hardware=skip_hardware, standalone=True)
    if text_errors or errors:
        raise SchemaError(format_diagnostics(text_errors + errors, files, documents))
    validate_local_repositories(files, documents, skip_hardware=skip_hardware)
    identity = original["META_DATA"].get("Dispatch", {}).get("id", "root")
    if source_hashes is not None:
        source_hashes.clear()
        source_hashes[identity] = document.source_sha256
    return entry.parent, {identity: original}
