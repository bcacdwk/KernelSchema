"""Explicit whole-region derivation, with raw sources and bounded effective views."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from . import yaml_source
from .diagnostics import Diagnostic, format_diagnostics
from .errors import CapacityError, SchemaError, ToolError, bounded
from .rules.values import node_id

REGIONS = ("math_reshape", "work_schedule", "tile_pipeline")
# Across one task, allow 100 times the parser's unchanged per-file limit.
# Copies, origin chain entries and dependency records all consume this budget.
MAX_RESOLVED_UNITS = 10_000_000


@dataclass(frozen=True)
class RegionOrigin:
    provider: str
    chain: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"provider": self.provider, "chain": list(self.chain)}


@dataclass
class ResolvedTask:
    raw_task: dict
    effective_task: dict
    origins: dict[str, dict[str, RegionOrigin]]
    dependency_nodes: dict[str, frozenset[str]]
    source_files: dict[str, Path]
    source_hashes: dict[str, str]
    documents: dict[str, yaml_source.Document]

    def source_manifest(self, node: str) -> dict[str, str]:
        return dict(sorted((self.source_files[name].name, self.source_hashes[name])
                           for name in self.dependency_nodes[node]))

    def origins_dict(self, node: str) -> dict:
        return {region: self.origins[node][region].as_dict() for region in sorted(REGIONS)}

    def effective_identity(self, node: str) -> str:
        identity = {"format": "kernel_schema.design_resolution.v1", "node": node,
                    "source_manifest": self.source_manifest(node), "origins": self.origins_dict(node)}
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def effective_document(self, node: str) -> dict:
        """Independent view with the root Contract, suitable for reading/export."""
        plan = dict(self.effective_task[node])
        if "contract" in self.effective_task["root"]:
            plan["contract"] = self.effective_task["root"]["contract"]
        budget = _Budget()
        budget.charge(_units(plan))
        return _clone(plan)

    def diagnostic_source(self, node: str, path: str) -> tuple[str, str, RegionOrigin | None]:
        for region, origin in self.origins[node].items():
            prefix = f"{node}.design.{region}"
            if path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "["):
                return origin.provider, origin.provider + path[len(node):], origin
        return node, path, None


class _Budget:
    def __init__(self):
        self.used = 0

    def charge(self, units: int):
        if units > MAX_RESOLVED_UNITS - self.used:
            raise CapacityError(f"Design resolution exceeds the tool's capacity ({MAX_RESOLVED_UNITS} resolved units)")
        self.used += units


def _units(value) -> int:
    """Count before allocating a copy; raw YAML has already rejected cycles."""
    count, pending = 0, [value]
    while pending:
        item = pending.pop()
        count += 1
        if count > MAX_RESOLVED_UNITS:
            raise CapacityError(f"Design resolution exceeds the tool's capacity ({MAX_RESOLVED_UNITS} resolved units)")
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return count


def _clone(value):
    """Iterative, alias-free copy of the ordinary values accepted by yaml_source."""
    result = [None]
    pending = [(result, 0, value)]
    while pending:
        parent, key, item = pending.pop()
        if isinstance(item, dict):
            child = {}
            parent[key] = child
            # Preserve source insertion order as well as scalar types.
            for name in item:
                child[name] = None
            pending.extend((child, name, part) for name, part in item.items())
        elif isinstance(item, list):
            child = [None] * len(item)
            parent[key] = child
            pending.extend((child, index, part) for index, part in enumerate(item))
        else:
            parent[key] = item
    return result[0]


def task_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise SchemaError(f"{directory}: pass a YAML task directory.")
    files = {}
    for file in sorted(directory.iterdir()):
        if file.suffix not in {".yaml", ".yml"}:
            continue
        if not file.is_file():
            raise SchemaError(f"{file}: expected a YAML file.")
        errors = []
        node_id(file.stem, "<file>", errors)
        if errors:
            raise SchemaError(f"{file}: {errors[0].message}")
        if file.stem in files:
            raise SchemaError(f"{file}: duplicate node ID {file.stem}; also defined by {files[file.stem]}.")
        files[file.stem] = file
    if "root" not in files:
        raise SchemaError(f"{directory}: missing root.yaml (or root.yml); a task needs one shared Contract.")
    return files


def read_directory(directory: Path) -> tuple[dict, dict]:
    files = task_files(directory)
    documents = {}
    for name, path in files.items():
        try:
            documents[name] = yaml_source.load(path, name)
        except yaml_source.ReadError as error:
            raise SchemaError(str(error)) from error
        except yaml_source.DependencyError as error:
            raise ToolError(str(error)) from error
    return files, documents


def resolve_documents(files: dict, documents: dict) -> ResolvedTask:
    """Resolve a single read snapshot; full validation checks raw shapes first."""
    raw = {name: document.value for name, document in documents.items()}
    errors, parents, references = [], {}, {}

    def fail(path, message, *, missing=False):
        errors.append(Diagnostic(path, message, missing))

    for name, plan in raw.items():
        meta = plan.get("meta_data")
        if not isinstance(meta, dict):
            fail(name + ".meta_data", "expected a metadata mapping", missing=meta is None)
            meta = {}
        dispatch = meta.get("dispatch", {})
        if not isinstance(dispatch, dict):
            fail(name + ".meta_data.dispatch", "expected a dispatch mapping")
            dispatch = {}
        identity = dispatch.get("id", "root")
        node_id(identity, name + ".meta_data.dispatch.id", errors)
        if identity != name:
            fail(name + ".meta_data.dispatch.id", "dispatch.id must match the YAML filename",
                 missing="id" not in dispatch)
        parent = dispatch.get("parent", "root")
        if name == "root":
            if "parent" in dispatch:
                fail(name + ".meta_data.dispatch.parent", "root has no parent")
        else:
            node_id(parent, name + ".meta_data.dispatch.parent", errors)
            if isinstance(parent, str):
                if parent not in raw:
                    fail(name + ".meta_data.dispatch.parent", f"unknown parent {parent}")
                else:
                    parents[name] = parent
        design = plan.get("design")
        if not isinstance(design, dict):
            fail(name + ".design", "expected a Design mapping", missing="design" not in plan)
            continue
        for region in REGIONS:
            value, path = design.get(region), f"{name}.design.{region}"
            if not isinstance(value, dict):
                fail(path, "expected a local region mapping or {from_parent: <direct parent ID>}",
                     missing=region not in design)
                continue
            if "from_parent" not in value:
                continue
            if set(value) != {"from_parent"}:
                fail(path, "from_parent must be the only field; references and local fields cannot be mixed")
            target = value["from_parent"]
            node_id(target, path + ".from_parent", errors)
            if name == "root":
                fail(path + ".from_parent", "root Design regions must be local; root cannot use from_parent")
            elif target != parent:
                fail(path + ".from_parent", "from_parent must name this node's direct parent " + repr(parent))
            references[name, region] = target
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents))

    # Parent order and cycle detection are iterative even for very deep trees.
    order, done = ["root"], {"root"}
    for start in raw:
        trail, active, current = [], {}, start
        while current not in done:
            if current in active:
                cycle = trail[active[current]:] + [current]
                fail(current + ".meta_data.dispatch.parent", "cycle: " + " -> ".join(cycle))
                raise SchemaError(format_diagnostics(errors, files, documents))
            active[current] = len(trail)
            trail.append(current)
            current = parents[current]
        order.extend(reversed(trail))
        done.update(trail)

    budget, origins, dependencies, effective = _Budget(), {}, {}, {}
    region_sizes = {}
    for name in order:
        budget.charge(1)
        origins[name] = {}
        required = set()
        for member in ("root", name):
            if member not in required:
                budget.charge(1)
                required.add(member)
        design = {}
        for region in REGIONS:
            parent = references.get((name, region))
            inherited = origins[parent][region] if parent else None
            length = len(inherited.chain) + 1 if inherited else 1
            budget.charge(1 + length)
            origin = RegionOrigin(inherited.provider, (name,) + inherited.chain) if inherited else RegionOrigin(name, (name,))
            origins[name][region] = origin
            for member in origin.chain:
                if member not in required:
                    budget.charge(1)
                    required.add(member)
            design[region] = raw[origin.provider]["design"][region]
            key = (origin.provider, region)
            if key not in region_sizes:
                region_sizes[key] = _units(design[region])
        # Retain unknown raw Design fields in read-only draft views; full raw
        # shape validation rejects them before resolution is reached.
        design.update((key, value) for key, value in raw[name]["design"].items() if key not in REGIONS)
        local = {key: value for key, value in raw[name].items() if key != "design"}
        size = _units(local) + 1 + sum(region_sizes[origin.provider, region]
                                      for region, origin in origins[name].items())
        size += sum(_units(value) for key, value in design.items() if key not in REGIONS)
        budget.charge(size)
        local["design"] = design
        effective[name] = _clone(local)
        dependencies[name] = frozenset(required)
    # Preserve the old directory iteration order for callers of validate_*.
    return ResolvedTask(raw, {name: effective[name] for name in raw}, origins, dependencies,
                        files, {name: doc.source_sha256 for name, doc in documents.items()}, documents)


@bounded
def resolve_directory(directory: str | Path) -> ResolvedTask:
    """Read and expand references without approving incomplete local content."""
    files, documents = read_directory(Path(directory).expanduser().resolve())
    return resolve_documents(files, documents)


@bounded
def effective_yaml(resolved: ResolvedTask, node: str) -> str:
    """Emit readable YAML and verify Decimal values and scalar types survive."""
    import yaml

    class Dumper(yaml.SafeDumper):
        pass

    def decimal_value(dumper, value):
        text = str(value)
        if "." not in text and "e" not in text.lower():
            text += ".0"
        return dumper.represent_scalar("tag:yaml.org,2002:float", text)

    Dumper.add_representer(Decimal, decimal_value)
    # PyYAML's default YAML 1.1 resolver disagrees with our core scalar reader
    # on spellings such as "1e3". Quoting strings avoids changing their type.
    Dumper.add_representer(str, lambda dumper, value: dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style='"'))
    value = resolved.effective_document(node)
    text = yaml.dump(value, Dumper=Dumper, sort_keys=False, allow_unicode=True)
    reread = yaml_source.loads(text, node).value
    pending = [(value, reread)]
    while pending:
        before, after = pending.pop()
        if type(before) is not type(after) or before != after:
            raise ToolError("effective YAML output did not preserve raw scalar values and types")
        if isinstance(before, dict):
            pending.extend((item, after[key]) for key, item in before.items())
        elif isinstance(before, list):
            pending.extend(zip(before, after))
    return text
