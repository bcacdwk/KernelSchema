"""Artifact scope gates and calls to the frozen schema parser/checker."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import sys

from .common import ExtractError, manifest, regular_file, relative_name, rules_root
from .inputs import validate_source_reference
from .stages import stage_files


class UpstreamRequired(ExtractError):
    """The current stage cannot repair an already reviewed root prerequisite."""

    def __init__(self, target: str, detail: str):
        self.target = target
        super().__init__(f"{detail}; request upstream reopen target={target} before P5 (P4/P5 root is read-only)")


@lru_cache(maxsize=16)
def parser_for(root: str):
    """Use the frozen checker's scalar, Decimal, alias, and duplicate-key rules."""
    source = Path(root) / "schema_template/tools/__init__.py"
    name = "_schema_extract_v2_checker_" + hashlib.sha256(root.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, source, submodule_search_locations=[str(source.parent)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(package)
        return importlib.import_module(name + ".yaml_source")
    finally:
        sys.dont_write_bytecode = previous


def load_yaml(directory: Path, filename: str, *, parser_root: Path | None = None) -> dict:
    parser = parser_for(str(parser_root or rules_root()))
    try:
        value = parser.load(regular_file(directory, filename), Path(filename).stem).value
    except (parser.ReadError, parser.DependencyError, parser.CapacityError) as exc:
        raise ExtractError(f"{filename} is not parseable YAML: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("meta_data"), dict):
        raise ExtractError(f"{filename} requires a meta_data mapping")
    return value


def load_root(directory: Path, filename: str = "root.yaml", *, parser_root: Path | None = None) -> dict:
    value = load_yaml(directory, filename, parser_root=parser_root)
    if not isinstance(value.get("contract"), dict):
        raise ExtractError("root.yaml requires meta_data and contract mappings")
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


def _without_p1_fields(value: dict) -> dict:
    result = deepcopy(value)
    metadata = result.pop("meta_data")
    result["protected_metadata"] = {key: metadata[key] for key in ("dispatch", "id", "parent") if key in metadata}
    result["contract"].pop("algorithm_requirement", None)
    return result


def _without_p2_fields(value: dict) -> dict:
    result = deepcopy(value)
    result["meta_data"].pop("dispatch", None)
    result["meta_data"].pop("version", None)
    return result


def _without_design(value: dict, *, root_revision=False) -> dict:
    result = deepcopy(value)
    result.pop("design", None)
    result.pop("realization", None)
    result["meta_data"].pop("version", None)
    if root_revision and isinstance(result["meta_data"].get("dispatch"), dict):
        result["meta_data"]["dispatch"].pop("when", None)
        if not result["meta_data"]["dispatch"]:
            result["meta_data"].pop("dispatch")
    return result


def _version(value: dict) -> int:
    version = value["meta_data"].get("version")
    if type(version) is not int or version < 1:
        raise ExtractError("meta_data.version must be a positive integer")
    return version


def _versions(old: dict, new: dict, *, first_draft: bool = False) -> None:
    old_version, new_version = _version(old), _version(new)
    if new_version < old_version:
        raise ExtractError("node version decreased")
    left, right = deepcopy(old), deepcopy(new)
    left["meta_data"].pop("version", None)
    right["meta_data"].pop("version", None)
    if first_draft and new_version != 1:
        raise ExtractError("new node starts at version 1")
    if not first_draft and not _same(left, right) and new_version <= old_version:
        raise ExtractError("node content changed without increasing version")


def validate_artifact_sources(directory: Path, lock: dict, *, decision: str,
                              parser_root: Path | None = None) -> None:
    parser_root = parser_root or rules_root(lock)
    root = load_root(directory, parser_root=parser_root)
    algorithm = root["contract"].get("algorithm_requirement")
    golden = algorithm.get("golden_reference") if isinstance(algorithm, dict) else None
    reference = golden.get("source") if isinstance(golden, dict) else None
    if reference is not None or decision in {"complete", "approve"}:
        validate_source_reference(reference, lock, role="reference")
    for name in manifest(directory):
        if name.endswith((".yaml", ".yml")):
            value = load_yaml(directory, name, parser_root=parser_root)["meta_data"].get("source")
            if value is not None or decision in {"complete", "approve"}:
                validate_source_reference(value, lock, role="implementation")


def validate_plan(plan: list, directory: Path, parser_root: Path, *, require_files: bool = False) -> None:
    """Validate the scheduling graph, reusing the checker's existing ID rule."""
    if not isinstance(plan, list) or not plan:
        raise ExtractError("P4 requires a nonempty structured plan")
    parser = parser_for(str(parser_root))
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        node_id = importlib.import_module(parser.__package__ + ".rules.values").node_id
    finally:
        sys.dont_write_bytecode = previous
    nodes, files = {}, set()
    for entry in plan:
        expected = {"id", "file", "parent", "workload", "difference", "selection", "evidence"}
        if not isinstance(entry, dict) or set(entry) != expected:
            raise ExtractError("plan node fields must be id/file/parent/workload/difference/selection/evidence")
        errors = []
        node_id(entry["id"], "plan.id", errors)
        if errors:
            raise ExtractError(errors[0].message)
        if entry["id"] in nodes or entry["file"] in files:
            raise ExtractError("plan node IDs and files must be unique")
        if relative_name(entry["file"]) != entry["id"] + ".yaml":
            raise ExtractError("plan filename must be <id>.yaml")
        if any(not isinstance(entry[key], str) or not entry[key].strip()
               for key in ("workload", "difference", "selection")):
            raise ExtractError("plan node needs workload, difference, and selection evidence")
        nodes[entry["id"]] = entry
        files.add(entry["file"])
    roots = [entry["id"] for entry in plan if entry["parent"] == ""]
    if roots != ["root"] or nodes["root"]["file"] != "root.yaml":
        raise ExtractError("plan must contain exactly one root, named root, with an empty parent")
    for entry in plan:
        if entry["id"] != "root" and (not isinstance(entry["parent"], str) or entry["parent"] not in nodes):
            raise ExtractError(f"plan parent does not exist: node {entry['id']!r}, parent {entry['parent']!r}")
    # Establish every direct edge before following ancestors: input order must
    # not turn an invalid parent in a later entry into an uncaught KeyError.
    for entry in plan:
        visited, current = set(), entry["id"]
        while current != "root":
            if current in visited:
                raise ExtractError("plan has a cycle or an unreachable node: " + entry["id"])
            visited.add(current)
            current = nodes[current]["parent"]
    root = load_root(directory, parser_root=parser_root)
    dispatch = root["meta_data"].get("dispatch", {})
    if (not isinstance(dispatch, dict) or dispatch.get("id", "root") != "root" or "parent" in dispatch):
        raise UpstreamRequired("P2", "root.yaml dispatch identity differs from the approved plan")
    if len(plan) > 1 or dispatch:
        if not isinstance(dispatch.get("when"), str) or not dispatch["when"].strip():
            raise UpstreamRequired("P3", "root.yaml requires a nonempty meta_data.dispatch.when for the planned paths")
    validate_index(directory, plan)
    if require_files:
        actual = {name for name in manifest(directory) if name.endswith((".yaml", ".yml"))}
        if actual != files:
            raise ExtractError("final YAML file set differs from the approved plan")
        for entry in plan:
            value = load_yaml(directory, entry["file"], parser_root=parser_root)
            dispatch = value["meta_data"].get("dispatch", {})
            if not isinstance(dispatch, dict):
                raise ExtractError("node dispatch must be a mapping")
            if dispatch.get("id", "root") != entry["id"] or dispatch.get("parent", "" if entry["id"] == "root" else "root") != entry["parent"]:
                raise ExtractError("YAML identity/parent differs from the approved plan")


def index_prefix(plan: list) -> str:
    """One deterministic tree projection; selection details remain in YAML."""
    def cell(value):
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")

    rows = ["# 路径索引", "", "| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |",
            "| --- | --- | --- | --- | --- |"]
    for entry in sorted(plan, key=lambda item: (item["id"] != "root", item["id"])):
        rows.append("| " + " | ".join([
            f"[{entry['id']}]({entry['file']})", entry["parent"] or "无",
            *(cell(entry[key]) for key in ("workload", "difference", "selection"))]) + " |")
    return "\n".join([*rows, "", "## 覆盖缺口", "", ""])


def _index_prose(text: str) -> str:
    """Do not count commented or fenced examples as coverage sections."""
    text = re.sub(r"<!--.*?(?:-->|$)", "", text, flags=re.S)
    lines, fence = [], ""
    for line in text.splitlines():
        marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = ""
        elif marker:
            fence = marker[1]
        else:
            lines.append(line)
    return "\n".join(lines)


def validate_index(directory: Path, plan: list) -> None:
    text = regular_file(directory, "dispatch.md").read_text(encoding="utf-8")
    table = index_prefix(plan).removesuffix("## 覆盖缺口\n\n")
    if not text.startswith(table):
        position = next((i for i, (actual, expected) in enumerate(zip(text, table)) if actual != expected),
                        min(len(text), len(table)))
        line = table[:position].count("\n") + 1
        column = position - table.rfind("\n", 0, position)
        raise ExtractError("dispatch.md tree, parents, workload, differences and selection must match the canonical "
                           f"approved plan table; first difference at line {line}, column {column}")
    # The table carries plan identity; explanatory paragraph placement does not.
    # Require the coverage section itself, rather than treating preceding prose
    # or the next section as evidence that its body was completed.
    body = _index_prose(text[len(table):])
    sections = re.split(r"(?m)^## 覆盖缺口[ \t]*$", body)
    if len(sections) != 2:
        raise ExtractError("dispatch.md requires one '## 覆盖缺口' section after the plan table")
    coverage = re.split(r"(?m)^#{1,2}(?:[ \t]|$)", sections[1], maxsplit=1)[0].strip()
    if not coverage:
        raise ExtractError("dispatch.md requires actual coverage gaps, or an explicit statement that none are known")
    links = {target.removeprefix("./").split("#", 1)[0]
             for target in re.findall(r"\]\(([^)\s]+\.(?:yaml|yml)(?:#[^)\s]*)?)\)", text)}
    if links != {entry["file"] for entry in plan}:
        raise ExtractError("dispatch.md YAML links differ from the approved node plan")


def validate_revision_scope(scopes: list, plan: list) -> None:
    files = {entry["file"] for entry in plan}
    seen = set()
    for item in scopes:
        path, fields = item["path"], item["fields"]
        if path in seen or path not in files | {"dispatch.md"} or not fields or len(set(fields)) != len(fields):
            raise ExtractError("P6 revision scope requires unique planned files and explicit fields")
        seen.add(path)
        if path == "dispatch.md":
            if fields != ["*"]:
                raise ExtractError("dispatch.md revision scope uses fields ['*']")
            continue
        # Root/Contract/tree changes enter the same reviewed upstream route.
        if path == "root.yaml":
            raise ExtractError("P6 root changes require upstream stage reopening")
        for field in fields:
            if not any(field == allowed or field.startswith(allowed + ".")
                       for allowed in ("design", "realization", "meta_data.source", "meta_data.dispatch.when")):
                raise ExtractError("P6 field requires upstream reopening or exceeds the allowed node scope: " + field)
            if any(not part or part in {"*", ".."} for part in field.split(".")):
                raise ExtractError("invalid P6 field path")


def _erase_fields(value: dict, fields: list[str]) -> dict:
    value = deepcopy(value)
    for field in [*fields, "meta_data.version"]:
        parts = field.split(".")
        current = value
        for part in parts[:-1]:
            if not isinstance(current, dict):
                break
            current = current.get(part)
        else:
            if isinstance(current, dict):
                current.pop(parts[-1], None)
    return value


def validate_artifacts(stage: str, role: str, before: Path, after: Path, starter: Path,
                       *, decision: str, node_entry: dict | None = None, revision_scope: list | None = None,
                       first_draft: bool = False, root_revision: bool = False,
                       maintenance_fields: list | None = None, candidate_review: bool = False) -> dict:
    old_files, new_files = manifest(before), manifest(after)
    changed = sorted(name for name in old_files.keys() | new_files.keys() if old_files.get(name) != new_files.get(name))
    allowed = set(stage_files(stage, node_entry=node_entry, revision_scope=revision_scope))
    if role == "reviewer" and changed:
        raise ExtractError("reviewer changed its read-only checkpoint copy: " + ", ".join(changed))
    if set(changed) - (allowed if role == "worker" else set()) or not set(old_files) <= set(new_files):
        raise ExtractError("write outside stage scope or deleted artifact: " + ", ".join(changed))
    if stage in {"P1", "P2", "P3", "P4"} and set(new_files) - {"root.yaml", "dispatch.md"}:
        raise ExtractError("unexpected artifacts for this stage")
    if stage in {"P3", "P4", "P5", "P6"} and set(new_files) != set(old_files):
        raise ExtractError("this stage cannot add or delete artifacts")
    parser_root = starter.parent.parent
    if maintenance_fields is not None and role == "worker":
        boundaries = {item["path"]: item["fields"] for item in maintenance_fields}
        for name in changed:
            if name not in boundaries:
                raise ExtractError("maintenance changed a file outside its evidence-bound scope: " + name)
            fields = boundaries[name]
            if name.endswith((".yaml", ".yml")) and fields != ["*"]:
                left = load_yaml(before, name, parser_root=parser_root)
                right = load_yaml(after, name, parser_root=parser_root)
                if not _same(_erase_fields(left, fields), _erase_fields(right, fields)):
                    raise ExtractError("maintenance changed fields outside its evidence-bound scope: " + name)
    old, new = load_root(before, parser_root=parser_root), load_root(after, parser_root=parser_root)
    template = load_root(starter.parent, starter.name, parser_root=parser_root)
    if not _same(new.get("contract", {}).get("hardware_specification"), template["contract"]["hardware_specification"]):
        raise ExtractError("hardware placeholder changed")
    if stage == "P1" and not _same(_without_p1_fields(old), _without_p1_fields(new)):
        raise ExtractError("P1 changed protected Design, hardware, dispatch, or other fields")
    if stage == "P2" and not _same(_without_p2_fields(old), _without_p2_fields(new)):
        raise ExtractError("P2 changed Contract, Design, source, or other protected fields")
    if stage == "P3" and not _same(_without_design(old, root_revision=root_revision), _without_design(new, root_revision=root_revision)):
        raise ExtractError("P3 changed Contract, selection, source, or other protected fields")
    if stage in {"P1", "P2", "P3"}:
        _versions(old, new, first_draft=_same(old, template))
    if stage == "P5":
        if not node_entry:
            raise ExtractError("P5 requires the approved node entry")
        name = node_entry["file"]
        left, right = load_yaml(before, name, parser_root=parser_root), load_yaml(after, name, parser_root=parser_root)
        metadata = right["meta_data"]
        dispatch = metadata.get("dispatch", {})
        inspect_incomplete = candidate_review and role == "reviewer" and decision in {"revise", "blocked"}
        if "contract" in right and not inspect_incomplete:
            raise ExtractError("derived nodes must omit the whole Contract")
        if not inspect_incomplete and (not isinstance(dispatch, dict) or dispatch.get("id") != node_entry["id"] or dispatch.get("parent") != node_entry["parent"]):
            raise ExtractError("derived ID/parent must match the approved direct-parent plan")
        if not inspect_incomplete and any(not _same(metadata.get(key), new["meta_data"].get(key)) for key in ("name", "describe")):
            raise ExtractError("derived name/describe must equal root")
        _versions(left, right, first_draft=first_draft)
        if decision in {"complete", "approve"} and (not isinstance(dispatch.get("when"), str) or not dispatch["when"].strip()):
            raise ExtractError("derived node needs its own nonempty when")
    if stage == "P6":
        scope_map = {item["path"]: item["fields"] for item in revision_scope or []}
        for name in changed:
            if name == "dispatch.md":
                continue
            left, right = load_yaml(before, name, parser_root=parser_root), load_yaml(after, name, parser_root=parser_root)
            if not _same(_erase_fields(left, scope_map[name]), _erase_fields(right, scope_map[name])):
                raise ExtractError("P6 changed fields outside the reviewer's explicit scope: " + name)
            _versions(left, right)
    if decision in {"complete", "approve"}:
        for field in ("name", "describe", "source"):
            if not isinstance(new["meta_data"].get(field), str) or not new["meta_data"][field].strip():
                raise ExtractError(f"completion requires meta_data.{field}")
        if not isinstance(new["contract"].get("algorithm_requirement"), dict):
            raise ExtractError("completion requires algorithm_requirement")
        if stage == "P1":
            parser = parser_for(str(parser_root))
            previous = sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode = True
                rules = importlib.import_module(parser.__package__ + ".rules")
                errors = rules.validate_algorithm(new["contract"]["algorithm_requirement"])
            finally:
                sys.dont_write_bytecode = previous
            if errors:
                raise ExtractError("P1 Contract check failed: " + "; ".join(map(str, errors)))
        if stage == "P2":
            dispatch = regular_file(after, "dispatch.md").read_text(encoding="utf-8")
            if not dispatch.strip() or not re.search(r"\]\((?:\./)?root\.yaml(?:#[^)]*)?\)", dispatch):
                raise ExtractError("P2 dispatch.md must link root.yaml")
    return {"changed_files": changed, "artifact_digests": new_files,
            "check": {"kind": "stage_gate", "stage": stage, "exit_code": 0,
                      "full_yaml_check": "not_run_in_P1_P2_P4" if stage in {"P1", "P2", "P4"} else "pending",
                      "hardware": "placeholder_unchanged"}}


def check_complete(directory: Path, rules_root: Path, *, filename="root.yaml", timeout=120.0) -> dict:
    """Run the frozen executable; never replace schema checks with local rules."""
    directory, rules_root = directory.resolve(), rules_root.resolve()
    command = [sys.executable, "-B", str(rules_root / "schema_template/check_yaml.py"),
               "--skip-hardware", str(directory / filename)]
    try:
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=max(0.01, timeout))
        code = result.returncode if result.returncode in {0, 1, 2} else 2
        return {"kind": "frozen_checker", "command": command, "exit_code": code,
                "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"kind": "frozen_checker", "command": command, "exit_code": 2, "returncode": None,
                "stdout": "", "stderr": str(exc)}
