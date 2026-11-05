"""Read-only whole-task checking. Markdown defines the schema; checks help authors."""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

from . import check_policy, rules, yaml_source
from .design_resolution import (RegionOrigin, ResolvedTask, effective_yaml, read_directory,
                                resolve_directory, resolve_documents, task_files)
from .diagnostics import format_diagnostics
from .errors import SchemaError, ToolError, bounded


def _validate_resolved_directory(directory: Path, *, notices=None, source_hashes=None,
                                 skip_hardware=False) -> ResolvedTask:
    files, documents = read_directory(directory)
    try:
        policy = check_policy.CheckPolicy(allow_missing_hardware=True) if skip_hardware else check_policy.for_directory(directory)
    except check_policy.PolicyError as error:
        raise ToolError(str(error)) from error
    if skip_hardware and notices is not None:
        notices.append("Hardware skipped explicitly: hardware_specification and hardware-dependent checks were not validated.")
    elif policy.allow_missing_hardware and notices is not None:
        notices.append(policy.notice())
    task = {name: document.value for name, document in documents.items()}
    checked = _checking_view(task, skip_hardware=skip_hardware)
    errors = rules.validate_structure(checked, allow_missing_hardware=policy.allow_missing_hardware,
                                      allow_parent_regions=True)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents) +
                          "\nFix the reported fields and rerun the same command. "
                          "Named relations will be checked automatically after field structure passes.")
    resolved = resolve_documents(files, documents)
    checked = _checking_view(resolved.effective_task, skip_hardware=skip_hardware)
    errors = rules.validate_relations(checked, allow_missing_hardware=policy.allow_missing_hardware)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents, resolved=resolved))
    if source_hashes is not None:
        source_hashes.clear()
        source_hashes.update(resolved.source_hashes)
    return resolved


def _checking_view(task, *, skip_hardware):
    if not skip_hardware:
        return task
    checked = deepcopy(task)
    for name, plan in checked.items():
        contract = plan.get("contract")
        if isinstance(contract, dict):
            contract.pop("hardware_specification", None)
            if name != "root" and not contract:
                plan.pop("contract")
    return checked


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
    # A file symlink keeps its node name and siblings, not its target's directory.
    entry = Path(filename).expanduser()
    directory = entry.parent.resolve()
    entry = directory / entry.name
    if entry.suffix not in {".yaml", ".yml"} or not entry.is_file():
        raise ToolError(f"{entry}: pass one existing .yaml or .yml file (root or a derived node).")
    return directory, _validate_resolved_directory(directory, notices=notices, source_hashes=source_hashes,
                                                   skip_hardware=skip_hardware)


@bounded
def validate_standalone_file(filename: str | Path, *, notices: list[str] | None = None,
                             source_hashes: dict[str, str] | None = None,
                             skip_hardware: bool = False) -> tuple[Path, dict]:
    """Check one explicit delivery view with its own Contract; read no siblings."""
    entry = Path(filename).expanduser().absolute()
    if entry.suffix not in {".yaml", ".yml"} or not entry.is_file():
        raise ToolError(f"{entry}: pass one existing standalone .yaml or .yml file.")
    try:
        document = yaml_source.load(entry, "root")
        policy = (check_policy.CheckPolicy(allow_missing_hardware=True) if skip_hardware
                  else check_policy.for_directory(entry.parent))
    except yaml_source.ReadError as error:
        raise SchemaError(str(error)) from error
    except (yaml_source.DependencyError, check_policy.PolicyError) as error:
        raise ToolError(str(error)) from error
    if skip_hardware and notices is not None:
        notices.append("Hardware skipped explicitly: hardware_specification and hardware-dependent checks were not validated.")
    original = document.value
    checked = deepcopy(original)
    if skip_hardware and isinstance(checked.get("contract"), dict):
        checked["contract"].pop("hardware_specification", None)
    files, documents = {"root": entry}, {"root": document}
    errors = rules.validate_structure({"root": checked},
                                      allow_missing_hardware=policy.allow_missing_hardware)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents))
    dispatch = checked["meta_data"].get("dispatch", {})
    identity = dispatch.get("id", "root")
    if "parent" in dispatch and (identity == "root" or dispatch["parent"] == identity):
        raise SchemaError(f"{entry}: standalone root has no parent; a derived node cannot parent itself.")
    # Identity and parent remain in the returned view as provenance. Relations
    # are local to this document: neither imports another node's Design.
    dispatch.pop("id", None)
    dispatch.pop("parent", None)
    errors = rules.validate_relations({"root": checked},
                                      allow_missing_hardware=policy.allow_missing_hardware)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents))
    if source_hashes is not None:
        source_hashes.clear()
        source_hashes[identity] = document.source_sha256
    return entry.parent, {identity: original}
