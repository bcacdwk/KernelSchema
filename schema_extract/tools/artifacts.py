"""Current artifact scopes, Template snapshot checks and candidate transactions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from urllib.parse import unquote, urlsplit

from .common import (ExtractError, digest, fsync_directory, manifest, regular_file,
                     rules_root, sha256, sync_tree)
from .inputs import validate_source_reference


DESIGN_REGIONS = ("Math_reshape", "Work_schedule", "Tile_pipeline")
SHARED_METADATA = ("Name", "Describe")


class UpstreamRequired(ExtractError):
    """A prerequisite belongs to a different stage."""

    def __init__(self, target: str, detail: str):
        self.target = target
        super().__init__(f"{detail}; request upstream reopen target={target}")


class CandidateRejected(ExtractError):
    """A rejected transaction retains the original/candidate evidence record."""

    def __init__(self, detail: str, validation: dict):
        self.validation = validation
        super().__init__(detail)


@lru_cache(maxsize=16)
def parser_for(root: str):
    """Load the parser from the invocation's frozen Template package."""
    source = Path(root) / "schema_template/tools/__init__.py"
    name = "_schema_extract_checker_" + hashlib.sha256(root.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, source, submodule_search_locations=[str(source.parent)])
    if spec is None or spec.loader is None:
        raise ExtractError(f"cannot load frozen Template: {source}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(package)
        return importlib.import_module(name + ".yaml_source")
    finally:
        sys.dont_write_bytecode = previous


def _module(parser_root: Path, name: str):
    parser = parser_for(str(parser_root))
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        return importlib.import_module(parser.__package__ + "." + name)
    finally:
        sys.dont_write_bytecode = previous


def schema_for(parser_root: Path):
    return _module(parser_root, "schema")


def load_yaml(directory: Path, filename: str, *, parser_root: Path | None = None) -> dict:
    parser = parser_for(str(parser_root or rules_root()))
    try:
        value = parser.load(regular_file(directory, filename), Path(filename).stem).value
    except (parser.ReadError, parser.DependencyError, parser.CapacityError) as exc:
        raise ExtractError(f"{filename} is not parseable YAML: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("META_DATA"), dict):
        raise ExtractError(f"{filename} requires a META_DATA mapping")
    return value


def load_root(directory: Path, filename: str = "root.yaml", *, parser_root: Path | None = None) -> dict:
    value = load_yaml(directory, filename, parser_root=parser_root)
    if not isinstance(value.get("CONTRACT"), dict):
        raise ExtractError("root.yaml requires META_DATA and CONTRACT mappings")
    return value


def write_yaml(path: Path, value: dict, *, source: Path | None = None,
               parser_root: Path | None = None) -> None:
    """Preserve unchanged source sections; verify all emitted scalar types."""
    import yaml

    parser = parser_for(str(parser_root or rules_root()))

    class Dumper(yaml.SafeDumper):
        pass

    # SafeDumper uses YAML 1.1 resolution, unlike the frozen core parser.
    # Explicit strings keep numeric-looking prose such as "1e3" and "0o12".
    Dumper.add_representer(str, lambda dumper, item: dumper.represent_scalar(
        "tag:yaml.org,2002:str", item, style="|" if "\n" in item else '"'))
    Dumper.add_representer(Decimal, lambda dumper, number:
                          dumper.represent_scalar("tag:yaml.org,2002:float", str(number)))

    def dump(item):
        return yaml.dump(item, Dumper=Dumper, sort_keys=False, allow_unicode=True)

    rendered = None
    if source is not None:
        original = source.read_text(encoding="utf-8")
        old = parser.loads(original, source.stem, path=source).value
        node = yaml.compose(original, Loader=yaml.SafeLoader)
        # A top-level merge or flow mapping has no independent field spans.
        # Aliases within unchanged Design sections retain their original text.
        if (isinstance(node, yaml.MappingNode) and not node.flow_style and
                all(key.tag == "tag:yaml.org,2002:str" and key.start_mark.column == 0
                    for key, _ in node.value)):
            pieces, seen = [original[:node.value[0][0].start_mark.index]] if node.value else [], set()
            for index, (key, _) in enumerate(node.value):
                name, start = key.value, key.start_mark.index
                end = node.value[index + 1][0].start_mark.index if index + 1 < len(node.value) else len(original)
                seen.add(name)
                if name in value:
                    pieces.append(original[start:end] if _same(old[name], value[name]) else dump({name: value[name]}))
            pieces.extend(dump({name: item}) for name, item in value.items() if name not in seen)
            candidate = "".join(pieces)
            try:
                restored = parser.loads(candidate, path.stem, path=path).value
            except (parser.ReadError, parser.CapacityError):
                restored = None  # A removed/rewritten field may own an alias used elsewhere.
            if _same(restored, value):
                rendered = candidate
    if rendered is None:
        rendered = dump(value)
    try:
        restored = parser.loads(rendered, path.stem, path=path).value
    except (parser.ReadError, parser.DependencyError, parser.CapacityError) as exc:
        raise ExtractError(f"YAML rewrite is not readable by the frozen parser: {exc}") from exc
    if not _same(restored, value):
        raise ExtractError("YAML rewrite changed a frozen parser value or scalar type")
    from .common import write_text
    write_text(path, rendered)


def _same(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _semantic_digest(value) -> str:
    """Typed semantic identity, independent of formatting and mapping order."""
    def encode(item):
        if isinstance(item, dict):
            return ["mapping", sorted((key, encode(value)) for key, value in item.items())]
        if isinstance(item, list):
            return ["sequence", [encode(value) for value in item]]
        return [type(item).__name__, str(item) if isinstance(item, Decimal) else item]
    return digest(encode(value))


def validate_plan(stage: str, plan: list, *, parser_root: Path,
                  selected_designs: list | None = None) -> str:
    """Validate one receipt plan and return its order-independent decision digest."""
    values = _module(parser_root, "rules.values")
    if stage not in {"P2", "P4"} or not isinstance(plan, list):
        raise ExtractError("plan belongs to P2 or P4 and must be a list")
    entries = {}
    for entry in plan:
        fields = {"id", "instance"} if stage == "P2" else {"id", "wait_for"}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ExtractError(f"{stage} plan entries require exactly " + "/".join(sorted(fields)))
        errors = []
        values.node_id(entry["id"], "plan.id", errors)
        if errors:
            raise ExtractError(str(errors[0]))
        if entry["id"] in entries:
            raise ExtractError("plan IDs must be unique")
        entries[entry["id"]] = entry
    if stage == "P2":
        if "root" not in entries or not 1 <= len(entries) <= 10:
            raise ExtractError("P2 selection requires root and at most 10 designs including root")
        instances = []
        for entry in plan:
            instance = entry["instance"]
            if not isinstance(instance, str) or not instance.strip():
                raise ExtractError("P2 instance must locate a concrete call or template instance")
            instances.append(instance.strip())
        if len(set(instances)) != len(instances):
            raise ExtractError("P2 concrete instance identities must be unique")
        if selected_designs is not None and validate_plan("P2", selected_designs, parser_root=parser_root) != digest(entries):
            raise ExtractError("selected design identities or instances changed")
        return digest(entries)
    validate_plan("P2", selected_designs, parser_root=parser_root)
    selected = {entry["id"] for entry in selected_designs}
    if set(entries) != selected - {"root"}:
        raise ExtractError("P4 must arrange each selected non-root design exactly once")
    for node, entry in entries.items():
        waits = entry["wait_for"]
        if (not isinstance(waits, list) or any(not isinstance(item, str) for item in waits)
                or len(set(waits)) != len(waits) or not set(waits) <= selected or node in waits):
            raise ExtractError("P4 wait_for requires unique selected IDs without self-dependency")
    pending = {node: set(entry["wait_for"]) - {"root"} for node, entry in entries.items()}
    while pending:
        ready = {node for node, waits in pending.items() if not waits}
        if not ready:
            raise ExtractError("P4 wait_for contains a cycle")
        pending = {node: waits - ready for node, waits in pending.items() if node not in ready}
    return digest({node: sorted(entry["wait_for"]) for node, entry in entries.items()})


def artifact_inventory(directory: Path, *, expected=None, additions=()) -> set[str]:
    """Check the entire directory against trusted names before reading any bytes."""
    if not directory.is_dir() or directory.is_symlink():
        raise ExtractError("invalid artifact directory: " + str(directory))
    names = set()
    for path in directory.iterdir():
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            raise ExtractError("artifact files must be directly in the task directory: " + path.name)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ExtractError("artifact must be an ordinary file without links: " + path.name)
        names.add(path.name)
    unexpected = names - {"dispatch.md"} - {name for name in names if name.endswith(".yaml")}
    if unexpected:
        raise ExtractError("unexpected artifact files: " + ", ".join(sorted(unexpected)))
    if "root.yaml" not in names:
        raise ExtractError("artifact directory requires root.yaml")
    if expected is not None:
        unexpected = names - set(expected) - set(additions)
        missing = set(expected) - names
        if unexpected or missing:
            raise ExtractError("artifact inventory differs from the prepared scope: " +
                               "; ".join(part for part in (
                                   "unexpected " + ", ".join(sorted(unexpected)) if unexpected else "",
                                   "missing " + ", ".join(sorted(missing)) if missing else "") if part))
    return names


def artifact_manifest(directory: Path, *, expected=None, additions=()) -> dict[str, str]:
    names = artifact_inventory(directory, expected=expected, additions=additions)
    return {name: sha256(regular_file(directory, name).read_bytes()) for name in sorted(names)}


def _yaml_files(directory: Path) -> dict[str, Path]:
    files = artifact_inventory(directory)
    return {Path(name).stem: directory / name for name in files if name.endswith(".yaml")}


def _references(document, *, parser_root: Path, normalizing=False) -> dict[str, str]:
    if not normalizing:
        design = document.value.get("DESIGN", {}) if isinstance(document.value, dict) else {}
        return {region: body["from"] for region in DESIGN_REGIONS
                if isinstance(body := design.get(region), dict) and "from" in body} if isinstance(design, dict) else {}
    parser = parser_for(str(parser_root))
    formatter = _module(parser_root, "yaml_format")
    fields = _module(parser_root, "rules.structure").CANONICAL_FIELDS
    result = {}
    for context in parser.scalar_contexts(document):
        parts = formatter.canonical_parts(context.parts, fields)
        if not context.key and len(parts) == 3 and parts[0] == "DESIGN" and parts[1] in DESIGN_REGIONS and parts[2] == "from":
            result[parts[1]] = context.node.value
    return result


def _snapshot(directory: Path, *, parser_root: Path, nodes=None, normalizing=False, baseline_materials=None):
    """Load root and every region of each needed provider, including cross-region edges."""
    schema, parser = schema_for(parser_root), parser_for(str(parser_root))
    available = _yaml_files(directory)
    if baseline_materials is not None:
        # The carried target raw is compared in the historical definition
        # context that made it meaningful. Current same-name providers must
        # never reinterpret an old chain during this baseline-only operation.
        available = {**_yaml_files(baseline_materials),
                     **{node: path for node, path in available.items() if node in set(nodes or ()) - {"root"}}}
    pending, files, documents = set(available if nodes is None else nodes) | {"root"}, {}, {}
    while pending:
        node = pending.pop()
        if node in documents:
            continue
        if not isinstance(node, str) or node not in available:
            raise ExtractError(f"missing material required to resolve from: {node!r}")
        path = regular_file(available[node].parent, available[node].name)
        try:
            loaded = (schema.load_for_normalization(path, node, skip_hardware=True) if normalizing
                      else parser.load(path, node))
        except (OSError, parser.ReadError, parser.DependencyError, parser.CapacityError) as exc:
            raise ExtractError(f"cannot read required material {path.name}: {exc}") from exc
        document = loaded.document if normalizing else loaded
        if normalizing and not isinstance(document.value, dict):
            raise ExtractError(f"{path.name} requires one task mapping")
        documents[node], files[node] = loaded, path
        for target in _references(document, parser_root=parser_root, normalizing=normalizing).values():
            if not isinstance(target, str):
                raise ExtractError(f"invalid from source in {path.name}: {target!r}")
            pending.add(target)
    return files, documents


def resolve_workspace(directory: Path, *, parser_root: Path, nodes=None):
    schema = schema_for(parser_root)
    try:
        files, documents = _snapshot(directory, parser_root=parser_root, nodes=nodes)
        return schema.resolve_documents(files, documents)
    except (OSError, schema.SchemaError, schema.ToolError) as exc:
        raise ExtractError(f"cannot resolve Design sources in {directory}: {exc}") from exc


def region_evidence_lines(resolved, node: str, region: str) -> set[int]:
    prefix = f"{node}.DESIGN.{region}"
    return {position[0] for field, position in resolved.documents[node].positions.items()
            if field == prefix or field.startswith(prefix + ".") or field.startswith(prefix + "[")}


def _resolved_context(resolved, *, strict=False) -> dict:
    result = {"validation": "strict" if strict else "resolved", "nodes": {}}
    for node in resolved.raw_task:
        pending, closure = [node, "root"], set()
        while pending:
            current = pending.pop()
            if current in closure:
                continue
            closure.add(current)
            for region in DESIGN_REGIONS:
                body = resolved.raw_task[current]["DESIGN"][region]
                if "from" in body:
                    pending.append(body["from"])
        result["nodes"][node] = {
            "validation": result["validation"], "file": resolved.source_files[node].name,
            "raw_sha256": resolved.source_hashes[node],
            "origins": resolved.origins_dict(node),
            "source_manifest": resolved.source_manifest(node),
            "effective_identity": resolved.effective_identity(node),
            "resolution_nodes": sorted(closure),
            "resolution_manifest": {resolved.source_files[item].name: resolved.source_hashes[item] for item in sorted(closure)},
            "contract_sha256": _semantic_digest(resolved.raw_task["root"].get("CONTRACT")),
            "region_sha256": {region: _semantic_digest(resolved.effective_task[node]["DESIGN"][region]) for region in DESIGN_REGIONS},
            "use_context_sha256": _semantic_digest(resolved.effective_document(node)),
        }
    return result


def source_context(directory: Path, *, parser_root: Path, nodes=None) -> dict:
    return _resolved_context(resolve_workspace(directory, parser_root=parser_root, nodes=nodes))


def align_source_identity(previous: dict, current: dict) -> dict:
    """Record a new raw provenance identity only with exact unchanged-use evidence."""
    if current.get("validation") != "strict":
        raise ExtractError("source identity alignment requires a strictly validated current snapshot")
    fields = ("file", "raw_sha256", "origins", "contract_sha256", "region_sha256", "use_context_sha256")
    if any(field not in previous or field not in current for field in (*fields, "effective_identity", "source_manifest")):
        raise ExtractError("source identity alignment requires complete previous and current provenance")
    changed = sorted(name for name in previous["source_manifest"].keys() | current["source_manifest"].keys()
                     if previous["source_manifest"].get(name) != current["source_manifest"].get(name))
    equal = all(_same(previous[field], current[field]) for field in fields)
    return {"equivalent": equal, "previous_identity": previous["effective_identity"],
            "current_identity": current["effective_identity"], "changed_sources": changed,
            "compared": {field: {"previous": previous[field], "current": current[field]} for field in fields},
            "current_source_manifest": current["source_manifest"]}


def design_context(state: dict, attempt: dict, directory: Path, *, parser_root: Path) -> dict:
    """Authoring context and actual definition owners eligible for scoped return."""
    result = {"design_resolution": {"validation": "not_run", "nodes": {}}, "provider_reopen_targets": {}}
    node = attempt.get("node", "root")
    try:
        graph = source_context(directory, parser_root=parser_root, nodes=None if attempt["stage"] == "P6" else [node])
        result["design_resolution"] = graph
        if attempt["stage"] not in {"P5", "P6"} or attempt.get("purpose") == "maintenance" or attempt.get("mode", "normal") != "normal":
            return result
        provided = attempt.get("source_identities", {})
        dependencies = attempt["dependency_digests"]
        consumers = list(graph["nodes"]) if attempt["stage"] == "P6" else [node]
        for consumer in consumers:
            for region, origin in graph["nodes"][consumer]["origins"].items():
                provider = origin["provider"]
                if provider == "root" or attempt["stage"] == "P5" and provider == node:
                    continue
                chain = origin["chain"] if attempt["stage"] == "P6" else origin["chain"][1:]
                if any(graph["nodes"][source]["effective_identity"] != provided.get(source)
                       or state.get("nodes", {}).get(source, {}).get("status") != "approved"
                       or (state["nodes"][source].get("source_context") or state["nodes"][source].get("approval") or {}).get("effective_identity") != provided.get(source)
                       or any(dependencies.get(name) != checksum for name, checksum in graph["nodes"][source]["source_manifest"].items())
                       for source in chain):
                    continue
                target = result["provider_reopen_targets"].setdefault("P5:" + provider, {
                    "provider": provider, "regions": {}, "consumers": [],
                    "effective_identity": graph["nodes"][provider]["effective_identity"],
                    "source_manifest": graph["nodes"][provider]["source_manifest"]})
                target["regions"][region] = origin["chain"]
                if consumer not in target["consumers"]:
                    target["consumers"].append(consumer)
        return result
    except (ExtractError, KeyError, TypeError) as exc:
        result["design_resolution"]["diagnostic"] = str(exc)
        result["design_resolution"]["available_files"] = sorted(attempt.get("dependency_digests", {}))
        result["provider_reopen_targets"] = {}
        return result


def validate_artifact_sources(directory: Path, lock: dict, *, decision: str,
                              parser_root: Path | None = None, nodes=None, completed_nodes=None) -> None:
    parser_root = parser_root or rules_root(lock)
    files = _yaml_files(directory)
    root = load_root(directory, parser_root=parser_root)
    algorithm = root["CONTRACT"].get("Algorithm_requirement")
    golden = algorithm.get("Golden_reference") if isinstance(algorithm, dict) else None
    reference = golden.get("Source") if isinstance(golden, dict) else None
    if reference is not None or decision in {"complete", "approve"}:
        validate_source_reference(reference, lock, role="reference")
    for node in files if nodes is None else nodes:
        value = load_yaml(directory, files[node].name, parser_root=parser_root)["META_DATA"].get("Source")
        if value is not None or decision in {"complete", "approve"} and (completed_nodes is None or node in completed_nodes):
            validate_source_reference(value, lock, role="implementation")


def validate_index(directory: Path, selected_designs: list, *, parser_root: Path, final=False) -> dict:
    """Check the real inventory independently of the completed Design snapshot."""
    validate_plan("P2", selected_designs, parser_root=parser_root)
    files = _yaml_files(directory)
    selected = {entry["id"] for entry in selected_designs}
    if not set(files) <= selected or final and set(files) != selected:
        raise ExtractError("YAML inventory differs from the selected designs")
    dispatch = _module(parser_root, "rules.dispatch")
    text = regular_file(directory, "dispatch.md").read_text(encoding="utf-8")
    lines = list(dispatch._visible(text))
    registered, planned = set(), set()
    names = {path.name: node for node, path in files.items()}
    for _, target in dispatch._links(lines):
        parsed = urlsplit(target)
        if not parsed.scheme and not parsed.netloc:
            name = str(Path(unquote(parsed.path)))
            if name in names:
                registered.add(names[name])
    design_lines, in_design = [], False
    for number, line in lines:
        heading = dispatch._HEADING.fullmatch(line)
        if heading and len(heading[1]) <= 2:
            in_design = len(heading[1]) == 2 and heading[2] == "完整设计"
        elif in_design:
            design_lines.append((number, line))
    for number, cell in dispatch._design_rows(design_lines)[1]:
        if "仅计划" not in cell:
            continue
        mentions = set(re.findall(r"(?<![\w./-])([a-z_][a-z0-9_]*)\.yaml(?![\w./-])", cell))
        if len(mentions) != 1 or not mentions <= selected or mentions & planned:
            raise ExtractError(f"dispatch.md:{number}: planned row needs one unique selected <id>.yaml")
        planned.update(mentions)
    if "root" not in registered or planned & registered or registered | planned != selected:
        raise ExtractError("dispatch must link root and register every selected design, with future files marked 仅计划 and no links")
    if final and planned:
        raise ExtractError("final dispatch still contains planned designs")
    checked_files = files if final else {node: path for node, path in files.items() if node in registered}
    errors = dispatch.validate(text, checked_files)
    if errors:
        raise ExtractError("; ".join(map(str, errors)))
    return {"kind": "final_directory" if final else "stage_directory", "yaml_files": sorted(path.name for path in files.values()),
            "registered_nodes": sorted(registered), "planned_nodes": sorted(planned), "missing_nodes": sorted(selected - set(files))}


def validate_revision_scope(scopes: list, selected_designs: list) -> None:
    files = {entry["id"] + ".yaml" for entry in selected_designs}
    if not isinstance(scopes, list):
        raise ExtractError("P6 revision_scope must be a list")
    seen = set()
    for item in scopes:
        if not isinstance(item, dict) or set(item) != {"path", "fields"}:
            raise ExtractError("P6 revision scope needs path and fields")
        path, fields = item["path"], item["fields"]
        if (not isinstance(path, str) or path in seen or path not in files | {"dispatch.md"}
                or not isinstance(fields, list) or not fields or any(not isinstance(field, str) for field in fields)
                or len(set(fields)) != len(fields)):
            raise ExtractError("P6 revision scope requires unique selected files and explicit fields")
        seen.add(path)
        if path == "dispatch.md":
            if fields != ["*"]:
                raise ExtractError("dispatch.md revision scope uses fields ['*']")
            continue
        if path == "root.yaml":
            raise UpstreamRequired("P3", "P6 root changes belong to their defining stage")
        for field in fields:
            if (not any(field == allowed or field.startswith(allowed + ".") for allowed in
                        ("DESIGN", "REALIZATION", "META_DATA.Source", "META_DATA.Dispatch.when"))
                    or any(not part or part in {"*", ".."} for part in field.split("."))):
                raise ExtractError("P6 field exceeds the allowed node scope: " + field)


def _erase_fields(value: dict, fields: list[str], *, version=True) -> dict:
    value = deepcopy(value)
    for field in [*fields, *(["META_DATA.Version"] if version else [])]:
        parts = field.split(".")
        current = value
        parents = []
        for part in parts[:-1]:
            if not isinstance(current, dict):
                break
            parents.append((current, part))
            current = current.get(part)
        else:
            if isinstance(current, dict):
                current.pop(parts[-1], None)
                for owner, key in reversed(parents):
                    if owner.get(key) == {}:
                        owner.pop(key)
                    else:
                        break
    return value


def _versions(old: dict, new: dict, *, first_draft=False) -> None:
    left, right = old["META_DATA"].get("Version"), new["META_DATA"].get("Version")
    if type(left) is not int or type(right) is not int or left < 1 or right < 1:
        raise ExtractError("META_DATA.Version must be a positive integer")
    # A prepared starter may already have a P1 metadata revision before its
    # first Design authoring call. It then follows the ordinary revision rule.
    first_draft = first_draft and left == 1
    if right < left or first_draft and right != 1:
        raise ExtractError("node versions cannot decrease and a first draft starts at 1")
    if not first_draft and not _same(_erase_fields(old, []), _erase_fields(new, [])) and right <= left:
        raise ExtractError("node content changed without increasing Version")


def _file_scope(stage, role, old_files, new_files, *, node_entry=None, revision_scope=None, synchronized=()):
    writes = {"P1": {"root.yaml"}, "P2": {"root.yaml", "dispatch.md"}, "P3": {"root.yaml"}, "P4": {"dispatch.md"},
              "P5": {node_entry["id"] + ".yaml"} if node_entry else set(),
              "P6": {entry["path"] for entry in revision_scope or []}}
    if stage not in writes or role not in {"worker", "reviewer", "runner"}:
        raise ExtractError("unknown stage or role")
    changed = sorted(name for name in old_files.keys() | new_files.keys() if old_files.get(name) != new_files.get(name))
    allowed = writes[stage] if role in {"worker", "runner"} else set()
    if synchronized:
        if stage != "P1" or role not in {"worker", "runner"}:
            raise ExtractError("shared metadata synchronization belongs to the P1 transaction")
        allowed = allowed | set(synchronized)
    if set(changed) - allowed or not set(old_files) <= set(new_files):
        raise ExtractError("write outside stage scope or deleted artifact: " + ", ".join(changed))
    additions = set(new_files) - set(old_files)
    if additions - ({"dispatch.md"} if stage == "P2" else set()):
        raise ExtractError("Runner must create artifact files before the invocation")
    return changed


def _validate_scope(stage, role, old_files, new_files, old, new, template, *, decision, node_entry=None,
                    revision_scope=None, first_draft=False, maintenance_fields=None, synchronized=()):
    changed = _file_scope(stage, role, old_files, new_files, node_entry=node_entry,
                          revision_scope=revision_scope, synchronized=synchronized)
    for documents in (old, new):
        if any(not isinstance(value, dict) or not isinstance(value.get("META_DATA"), dict) for value in documents.values()):
            raise ExtractError("each artifact requires a META_DATA mapping")
        if not isinstance(documents.get("root", {}).get("CONTRACT"), dict):
            raise ExtractError("root requires the shared CONTRACT mapping")
    hardware_path = "CONTRACT.Hardware_specification.Execution"
    hardware = new["root"]["CONTRACT"].get("Hardware_specification")
    expected = template["CONTRACT"].get("Hardware_specification")
    if (not isinstance(hardware, dict) or not isinstance(expected, dict)
            or not _same({key: value for key, value in hardware.items() if key != "Execution"},
                         {key: value for key, value in expected.items() if key != "Execution"})):
        raise ExtractError("hardware outside Execution must remain the current starter")
    fields = {"P1": ["META_DATA.Name", "META_DATA.Describe", "META_DATA.Source", "CONTRACT.Algorithm_requirement", hardware_path],
              "P2": ["META_DATA.Source", "META_DATA.Dispatch"],
              "P3": ["DESIGN", "REALIZATION", "META_DATA.Dispatch.when"]}
    if stage in fields:
        if not _same(_erase_fields(old["root"], fields[stage]), _erase_fields(new["root"], fields[stage])):
            raise ExtractError(stage + " changed protected fields")
        _versions(old["root"], new["root"], first_draft=first_draft or _same(old["root"], template))
    for name in synchronized:
        node = Path(name).stem
        left, right = old[node], new[node]
        if (node == "root" or not _same(_erase_fields(left, ["META_DATA." + field for field in SHARED_METADATA]),
                                       _erase_fields(right, ["META_DATA." + field for field in SHARED_METADATA]))
                or right["META_DATA"].get("Version") != left["META_DATA"].get("Version") + 1):
            raise ExtractError("P1 synchronization changed protected node content or Version: " + name)
        _versions(left, right)
    if stage == "P5":
        if not node_entry or node_entry["id"] == "root":
            raise ExtractError("P5 requires one selected non-root node")
        node = node_entry["id"]
        left, right = old[node], new[node]
        if not _same(_erase_fields(left, ["DESIGN", "REALIZATION", "META_DATA.Source", "META_DATA.Dispatch.when"]),
                     _erase_fields(right, ["DESIGN", "REALIZATION", "META_DATA.Source", "META_DATA.Dispatch.when"])):
            raise ExtractError("P5 changed protected task or node identity")
        _versions(left, right, first_draft=first_draft)
    if stage == "P6":
        validate_revision_scope(revision_scope or [], [{"id": node} for node in new])
        scopes = {item["path"]: item["fields"] for item in revision_scope or []}
        for name in changed:
            if name == "dispatch.md":
                continue
            node = Path(name).stem
            if not _same(_erase_fields(old[node], scopes[name]), _erase_fields(new[node], scopes[name])):
                raise ExtractError("P6 changed fields outside the explicit revision scope: " + name)
            _versions(old[node], new[node])
    if maintenance_fields is not None:
        scopes = {item["path"]: item["fields"] for item in maintenance_fields}
        for name in changed:
            if name in synchronized:
                continue  # Derived only from the already scoped root fields.
            if name not in scopes:
                raise ExtractError("maintenance changed a file outside its scope: " + name)
            if name.endswith(".yaml") and scopes[name] != ["*"]:
                node = Path(name).stem
                if not _same(_erase_fields(old[node], scopes[name]), _erase_fields(new[node], scopes[name])):
                    raise ExtractError("maintenance changed fields outside its scope: " + name)
    for node, value in new.items():
        metadata = value.get("META_DATA", {})
        dispatch = metadata.get("Dispatch", {})
        if not isinstance(dispatch, dict) or dispatch.get("id", "root") != node:
            raise ExtractError("META_DATA.Dispatch.id must equal the filename: " + node)
        if node != "root" and ("CONTRACT" in value or any(not _same(metadata.get(key), new["root"]["META_DATA"].get(key)) for key in SHARED_METADATA)):
            raise ExtractError("non-root nodes omit CONTRACT and keep the task Name/Describe")
    return {"changed_files": changed, "artifact_digests": new_files,
            "check": {"kind": "stage_scope", "stage": stage, "exit_code": 0,
                      "hardware": "only_P1_Execution_is_writable"}}


def validate_artifacts(stage: str, role: str, before: Path, after: Path, starter: Path, *, decision: str,
                       node_entry: dict | None = None, revision_scope: list | None = None,
                       first_draft=False, maintenance_fields: list | None = None) -> dict:
    """Scope gate for canonical raw bytes; normalization is a separate transaction."""
    parser_root = starter.parent.parent
    if role not in {"worker", "reviewer"}:
        raise ExtractError("artifact invocation must be worker or reviewer")
    old_files = artifact_manifest(before)
    new_files = artifact_manifest(after, expected=old_files,
                                  additions={"dispatch.md"} if stage == "P2" and role == "worker" else ())
    changed = _file_scope(stage, role, old_files, new_files, node_entry=node_entry, revision_scope=revision_scope)
    nodes = {"root"} | {Path(name).stem for name in changed if name.endswith(".yaml")}
    if stage == "P1":
        nodes.update(_yaml_files(after))
    if node_entry:
        nodes.add(node_entry["id"])
    old = {node: load_yaml(before, node + ".yaml", parser_root=parser_root) for node in nodes}
    new = {node: load_yaml(after, node + ".yaml", parser_root=parser_root) for node in nodes}
    template = load_root(starter.parent, starter.name, parser_root=parser_root)
    return _validate_scope(stage, role, old_files, new_files, old, new, template, decision=decision,
                           node_entry=node_entry, revision_scope=revision_scope, first_draft=first_draft,
                           maintenance_fields=maintenance_fields)


def _early_check(root, *, parser_root: Path, stage: str) -> None:
    rules = _module(parser_root, "rules")
    errors = rules.validate_structure({"root": root}, skip_hardware=True, allow_from=True)
    errors = [error for error in errors if not (error.path == "root.DESIGN" or error.path.startswith("root.DESIGN."))
              and not (stage == "P2" and error.path == "root.META_DATA.Dispatch.when")]
    errors.extend(rules.validate_algorithm(root.get("CONTRACT", {}).get("Algorithm_requirement")))
    if stage == "P1":
        values = _module(parser_root, "rules.values")
        values.named(values.known_text, key=values.hardware_identity)(
            root.get("CONTRACT", {}).get("Hardware_specification", {}).get("Execution"),
            "root.CONTRACT.Hardware_specification.Execution", errors)
    if errors:
        raise ExtractError("; ".join(map(str, errors)))


def check_stage_snapshot(directory: Path, *, parser_root: Path, stage: str,
                         selected_designs: list | None = None, node="root", checked_nodes=None) -> dict:
    """Strict completed-material checks plus independent stage/final directory checks."""
    schema = schema_for(parser_root)
    result = {"kind": "stage_snapshot", "stage": stage, "exit_code": 1, "diagnostics": [], "checked_nodes": [], "excluded_nodes": []}
    try:
        if stage not in {"P1", "P2", "P3", "P4", "P5", "P6"}:
            raise ExtractError("unknown stage")
        available = _yaml_files(directory)
        if node not in available or stage in {"P1", "P2", "P3", "P4"} and node != "root":
            raise ExtractError("checked node is outside the stage snapshot")
        if stage != "P1":
            result["directory"] = validate_index(directory, selected_designs, parser_root=parser_root, final=stage == "P6")
        if stage in {"P3", "P4", "P5"} and len(selected_designs) > 1:
            root = load_root(directory, parser_root=parser_root)
            if not root["META_DATA"].get("Dispatch", {}).get("when"):
                raise UpstreamRequired("P3", "multiple selected designs require root's actual Dispatch.when")
        if stage in {"P1", "P2", "P4"}:
            root = load_root(directory, parser_root=parser_root)
            if stage == "P1":
                for current in available.keys() - {"root"}:
                    value = load_yaml(directory, available[current].name, parser_root=parser_root)
                    if "CONTRACT" in value or any(not _same(value["META_DATA"].get(field), root["META_DATA"].get(field))
                                                  for field in SHARED_METADATA):
                        raise ExtractError("non-root nodes omit CONTRACT and keep the task Name/Describe")
                result["shared_metadata_nodes"] = sorted(available)
            if stage == "P4":
                files, documents = _snapshot(directory, parser_root=parser_root, nodes=["root"])
                resolved = schema.validate_documents(files, documents, skip_hardware=True)
                schema.validate_local_repositories(files, documents, skip_hardware=True)
                result["design_resolution"] = _resolved_context(resolved, strict=True)
            else:
                _early_check(root, parser_root=parser_root, stage=stage)
            result["checked_nodes"] = ["root"]
        elif stage == "P6":
            notices = []
            _, resolved = schema.validate_resolved_file(directory / (node + ".yaml"), notices=notices, skip_hardware=True)
            result.update(checked_nodes=sorted(resolved.raw_task), notices=notices,
                          design_resolution=_resolved_context(resolved, strict=True))
        else:
            wanted = set(checked_nodes or ()) | {"root", node}
            files, documents = _snapshot(directory, parser_root=parser_root, nodes=wanted)
            notices = []
            resolved = schema.validate_documents(files, documents, notices=notices, skip_hardware=True)
            schema.validate_local_repositories(files, documents, skip_hardware=True)
            result.update(checked_nodes=sorted(files), notices=notices,
                          design_resolution=_resolved_context(resolved, strict=True))
        result["excluded_nodes"] = sorted(set(available) - set(result["checked_nodes"]))
        result["exit_code"] = 0
    except (ExtractError, schema.SchemaError, schema.ToolError, OSError, UnicodeError) as exc:
        result["diagnostics"].append(str(exc))
        result["exit_code"] = 2 if isinstance(exc, (OSError, schema.ToolError)) else 1
    return result


def check_complete(directory: Path, rules_root: Path, *, selected_designs: list, filename="root.yaml") -> dict:
    """Final file API; never run the CLI's normalization-writing mode."""
    return check_stage_snapshot(directory, parser_root=rules_root, stage="P6", selected_designs=selected_designs, node=Path(filename).stem)


def approval_projection(stage: str, directory: Path, *, parser_root: Path, plan=None,
                        selected_designs=None, node="root") -> dict:
    """Stable stage subject alongside, rather than in place of, original raw evidence."""
    files = artifact_manifest(directory)
    root = load_root(directory, parser_root=parser_root)
    graph = None
    if stage == "P1":
        subject = {"Name": root["META_DATA"]["Name"], "Describe": root["META_DATA"]["Describe"], "CONTRACT": root["CONTRACT"]}
    elif stage == "P2":
        subject = {"selection": validate_plan("P2", plan, parser_root=parser_root),
                   "Source": root["META_DATA"].get("Source"), "id": root["META_DATA"].get("Dispatch", {}).get("id", "root")}
    elif stage == "P4":
        subject = {"selection": validate_plan("P2", selected_designs, parser_root=parser_root),
                   "writing": validate_plan("P4", plan, parser_root=parser_root, selected_designs=selected_designs)}
    elif stage == "P3":
        subject = {"DESIGN": root["DESIGN"], "REALIZATION": root.get("REALIZATION"),
                   "when": root["META_DATA"].get("Dispatch", {}).get("when")}
    elif stage == "P5":
        graph = source_context(directory, parser_root=parser_root, nodes=[node])["nodes"][node]
        subject = {"effective_identity": graph["effective_identity"], "use_context_sha256": graph["use_context_sha256"]}
    elif stage == "P6":
        subject = files
    else:
        raise ExtractError("unknown approval stage")
    return {"stage": stage, "node": node, "subject_sha256": _semantic_digest(subject),
            "raw_manifest": files, "source_context": graph}


def _normalization_snapshot(directory, *, parser_root, nodes, baseline_materials=None):
    schema = schema_for(parser_root)
    files, documents = _snapshot(directory, parser_root=parser_root, nodes=nodes, normalizing=True, baseline_materials=baseline_materials)
    return schema.normalize_documents(files, documents, skip_hardware=True)


def _normalization_consumers(directory, initial, old, *, parser_root, stage):
    """Expand a changed definition to all actual same-region users before planning writes."""
    changes = set()
    contract_changed = stage != "P1" and not _same(
        old.documents["root"].value.get("CONTRACT"), initial.documents["root"].value.get("CONTRACT"))
    for node, document in initial.documents.items():
        before = old.documents.get(node)
        if before is None:
            continue
        for region in DESIGN_REGIONS:
            if not _same(before.value.get("DESIGN", {}).get(region), document.value.get("DESIGN", {}).get(region)):
                changes.add((node, region))
    for correction in initial.corrections:
        # P1's semantic Contract edits need later body reviews. Only Template's
        # binding edits require all consumers in this normalization transaction.
        if correction.path.startswith("root.CONTRACT.") and (
                stage != "P1" or correction.message.startswith("identifier+reference:")):
            contract_changed = True
        for node in initial.documents:
            for region in DESIGN_REGIONS:
                prefix = f"{node}.DESIGN.{region}"
                if correction.path == prefix or correction.path.startswith(prefix + ".") or correction.path.startswith(prefix + "["):
                    changes.add((node, region))
    if not changes and not contract_changed:
        return set(initial.documents)
    available = _yaml_files(directory)
    if contract_changed:
        return set(available)
    references = {}
    schema, parser = schema_for(parser_root), parser_for(str(parser_root))
    for node, path in available.items():
        try:
            loaded = schema.load_for_normalization(path, node, skip_hardware=True)
            references[node] = _references(loaded.document, parser_root=parser_root, normalizing=True)
        except (OSError, parser.ReadError, parser.DependencyError, parser.CapacityError) as exc:
            raise ExtractError(f"cannot establish consumers of a changed definition: {path.name}: {exc}") from exc
    result = set(initial.documents)
    for node in references:
        for region in DESIGN_REGIONS:
            current, seen = node, set()
            while current not in seen:
                seen.add(current)
                if (current, region) in changes:
                    result.add(node)
                    break
                current = references.get(current, {}).get(region)
                if current is None:
                    break
    return result


def _verify_adjudication_patch(before: Path, original: Path, receipt: dict) -> None:
    if (not isinstance(receipt, dict) or receipt.get("format") != "schema_extract.receipt.v2"
            or receipt.get("identity", {}).get("role") != "reviewer"
            or receipt.get("identity", {}).get("mode") != "adjudicate" or receipt.get("decision") != "approve"):
        raise ExtractError("Runner normalization requires the validated approving adjudication receipt")
    action = receipt.get("adjudication", {})
    if action.get("action") != "patch" or not action.get("checkpoint") or not action.get("patches"):
        raise ExtractError("Runner normalization requires an explicit checkpoint and adjudication patch")
    expected = {name: regular_file(before, name).read_bytes() for name in manifest(before)}
    for item in action["patches"]:
        if not isinstance(item, dict) or set(item) != {"path", "old", "new"}:
            raise ExtractError("adjudication patch requires path/old/new")
        path, old, new = item["path"], item["old"], item["new"]
        if path not in expected or not isinstance(old, str) or not old or not isinstance(new, str):
            raise ExtractError("adjudication patch is outside the candidate or has invalid text")
        text = expected[path].decode("utf-8")
        if text.count(old) != 1:
            raise ExtractError("adjudication old text must match exactly once: " + path)
        expected[path] = text.replace(old, new, 1).encode("utf-8")
    if manifest(original) != {name: sha256(data) for name, data in expected.items()}:
        raise ExtractError("Runner candidate differs from the exact adjudication patch")


def _synchronize_metadata(before: Path, landing: Path, old: dict, new: dict, *, parser_root: Path, nodes=None) -> dict:
    """Derive only the two shared fields, inside the P1 candidate transaction."""
    if any(not isinstance(documents["root"].get("META_DATA"), dict) for documents in (old, new)):
        raise ExtractError("shared metadata synchronization requires a META_DATA mapping")
    changed = {field: {"before": old["root"]["META_DATA"].get(field), "after": new["root"]["META_DATA"].get(field)}
               for field in SHARED_METADATA
               if not _same(old["root"]["META_DATA"].get(field), new["root"]["META_DATA"].get(field))}
    record = {"origin": "runner_shared_metadata", "fields": changed, "nodes": {}}
    for node, path in _yaml_files(landing).items():
        if node == "root" or nodes is not None and node not in nodes:
            continue
        previous = old.setdefault(node, load_yaml(before, path.name, parser_root=parser_root))
        current = new.setdefault(node, load_yaml(landing, path.name, parser_root=parser_root))
        fields = {field: {"before": previous["META_DATA"].get(field), "after": new["root"]["META_DATA"].get(field)}
                  for field in SHARED_METADATA
                  if not _same(previous["META_DATA"].get(field), new["root"]["META_DATA"].get(field))}
        if not fields:
            continue
        version = previous["META_DATA"].get("Version")
        if type(version) is not int or version < 1:
            raise ExtractError("P1 synchronization requires an existing positive Version: " + path.name)
        synchronized = deepcopy(current)
        for field in SHARED_METADATA:
            synchronized["META_DATA"][field] = deepcopy(new["root"]["META_DATA"].get(field))
        synchronized["META_DATA"]["Version"] = version + 1
        before_hash = sha256((before / path.name).read_bytes())
        normalized_hash = sha256(path.read_bytes())
        write_yaml(path, synchronized, source=path, parser_root=parser_root)
        new[node] = load_yaml(landing, path.name, parser_root=parser_root)
        record["nodes"][path.name] = {"before_sha256": before_hash, "normalized_sha256": normalized_hash,
                                      "final_sha256": sha256(path.read_bytes()), "version_before": version,
                                      "version_after": version + 1, "fields": fields,
                                      "protected_sha256": _semantic_digest(_erase_fields(previous,
                                          ["META_DATA." + field for field in SHARED_METADATA]))}
    return record


def normalize_candidate(before: Path, original: Path, destination: Path, starter: Path, *, stage: str,
                        role: str, decision: str, selected_designs=None, node_entry=None,
                        revision_scope=None, first_draft=False, checked_nodes=None,
                        maintenance_fields=None, adjudication=None, baseline_materials=None) -> dict:
    """Normalize into one new controlled directory; reject the entire group on any failure.

    The caller retains ``original`` and its worker receipt as immutable evidence.
    Returned final identities refer only to strict rereads of ``destination``;
    candidate marks are never turned into evidence anchors. Reviewers call the
    read-only scope/check APIs. ``role='runner'`` requires exact replay of an
    approving adjudication patch and retains that receipt's distinct authorship.
    """
    validation = {"kind": "normalization_transaction", "stage": stage, "status": "rejected",
                  "original_directory": str(original), "final_directory": str(destination), "corrections": []}
    temporary = None
    parser_root, schema = starter.parent.parent, schema_for(starter.parent.parent)
    try:
        before_hashes = artifact_manifest(before)
        originals = artifact_manifest(original, expected=before_hashes,
                                      additions={"dispatch.md"} if stage == "P2" and role in {"worker", "runner"} else ())
        validation.update(original_artifact_digests=originals, baseline_artifact_digests=before_hashes)
        if role == "runner":
            _verify_adjudication_patch(before, original, adjudication)
            if (adjudication["identity"].get("stage") != stage or decision != "approve"
                    or stage == "P5" and adjudication["identity"].get("node") != (node_entry or {}).get("id")):
                raise ExtractError("adjudication stage/node/decision differs from the normalization scope")
            validation.update(origin="runner_adjudication", adjudication_receipt_sha256=digest(adjudication))
        elif role == "worker" and adjudication is None and decision in {"complete", "continue", "blocked", "reopen"}:
            validation["origin"] = "worker_candidate"
        else:
            raise ExtractError("reviewer artifacts are read-only; normalization requires a worker or verified Runner patch")
        if (destination.exists() or destination.is_symlink()
                or destination.resolve().is_relative_to(original.resolve()) or destination.resolve().is_relative_to(before.resolve())):
            raise ExtractError("normalization destination must be a new Runner-controlled sibling directory")
        changed = _file_scope(stage, role, before_hashes, originals, node_entry=node_entry, revision_scope=revision_scope)
        validation["original_changed_files"] = changed
        wanted = (set(checked_nodes or ()) if stage != "P1" else set()) | {"root"}
        wanted.update(Path(name).stem for name in changed if name.endswith(".yaml"))
        if node_entry:
            wanted.add(node_entry["id"])
        if stage == "P6":
            wanted = set(_yaml_files(original))
            validate_revision_scope(revision_scope or [], selected_designs)
        preview = _normalization_snapshot(original, parser_root=parser_root, nodes=wanted)
        baseline = _normalization_snapshot(before, parser_root=parser_root, nodes=wanted, baseline_materials=baseline_materials)
        expanded = _normalization_consumers(original, preview, baseline, parser_root=parser_root, stage=stage)
        if expanded != set(preview.documents):
            preview = _normalization_snapshot(original, parser_root=parser_root, nodes=expanded)
            baseline = _normalization_snapshot(before, parser_root=parser_root, nodes=expanded, baseline_materials=baseline_materials)
        historical_hashes = manifest(baseline_materials) if baseline_materials is not None else {}
        baseline_hashes = {path.name: (before_hashes if path.parent == before else historical_hashes)[path.name]
                           for path in baseline.files.values()}
        for normalized, expected in ((preview, originals), (baseline, baseline_hashes)):
            if any(expected.get(normalized.files[node].name) != checksum for node, checksum in normalized.source_hashes.items()):
                raise ExtractError("normalization source_hashes differ from the original captured bytes")
        validation["baseline_source_hashes"] = {baseline.files[node].name: checksum for node, checksum in baseline.source_hashes.items()}
        validation.update(source_hashes={preview.files[node].name: checksum for node, checksum in preview.source_hashes.items()},
                          candidate_digests={preview.files[node].name: document.source_sha256 for node, document in preview.documents.items()},
                          corrections=[asdict(item) for item in preview.corrections],
                          normalized_files=sorted(preview.files[node].name for node in preview.candidates))
        final_hashes = {**originals, **validation["candidate_digests"]}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".normalize-", dir=destination.parent))
        for name in originals:
            (temporary / name).write_bytes(regular_file(original, name).read_bytes())
        for node, text in preview.candidates.items():
            (temporary / preview.files[node].name).write_text(text, encoding="utf-8")
        old = {node: doc.value for node, doc in baseline.documents.items()}
        new = {node: doc.value for node, doc in preview.documents.items()}
        synchronized = ()
        if stage == "P1":
            synchronization = _synchronize_metadata(before, temporary, old, new, parser_root=parser_root)
            if synchronization["fields"] or synchronization["nodes"]:
                validation["shared_metadata_sync"] = synchronization
                synchronized = synchronization["nodes"]
                final_hashes.update({name: item["final_sha256"] for name, item in synchronized.items()})
        template = load_root(starter.parent, starter.name, parser_root=parser_root)
        scope = _validate_scope(stage, role, before_hashes, final_hashes, old, new, template,
                               decision=decision, node_entry=node_entry, revision_scope=revision_scope,
                               first_draft=first_draft, maintenance_fields=maintenance_fields, synchronized=synchronized)
        validation["scope"] = scope
        if artifact_manifest(original, expected=originals) != originals or artifact_manifest(before, expected=before_hashes) != before_hashes:
            raise ExtractError("candidate input changed during normalization")
        if artifact_manifest(temporary, expected=final_hashes) != final_hashes:
            raise ExtractError("candidate landing differs from the planned bytes")
        if decision in {"complete", "approve"}:
            check = check_stage_snapshot(temporary, parser_root=parser_root, stage=stage,
                                        selected_designs=selected_designs, node=node_entry["id"] if node_entry else "root",
                                        checked_nodes=checked_nodes)
            validation["check"] = check
            if check["exit_code"]:
                raise ExtractError("strict reread failed: " + "; ".join(check["diagnostics"]))
        else:
            # Incomplete drafts keep authoring access, but every landed YAML is
            # reread and all scope/identity gates above remain mandatory.
            for node in preview.documents:
                load_yaml(temporary, preview.files[node].name, parser_root=parser_root)
            validation["check"] = {"kind": "incomplete_draft", "exit_code": 0, "full_check": "not_requested"}
        if (artifact_manifest(original, expected=originals) != originals or
                artifact_manifest(before, expected=before_hashes) != before_hashes or destination.exists()):
            raise ExtractError("candidate input or destination changed before commit")
        sync_tree(temporary)
        os.replace(temporary, destination)
        temporary = None
        fsync_directory(destination.parent)
        validation.update(status="landed", artifact_digests=manifest(destination), changed_files=scope["changed_files"])
        return validation
    except (ExtractError, schema.SchemaError, schema.ToolError, OSError, UnicodeError) as exc:
        validation["diagnostic"] = str(exc)
        raise CandidateRejected(str(exc), validation) from exc
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
