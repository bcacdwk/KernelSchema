"""Whole-region from resolution, source chains and independent effective views."""
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

REGIONS = ("Math_reshape", "Work_schedule", "Tile_pipeline")
MAX_DESIGNS = 10
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
        """Root, self and actual region chains, not all reachable provider regions."""
        return dict(sorted((self.source_files[name].name, self.source_hashes[name])
                           for name in self.dependency_nodes[node]))

    def origins_dict(self, node: str) -> dict:
        return {region: self.origins[node][region].as_dict() for region in sorted(REGIONS)}

    def effective_identity(self, node: str) -> str:
        identity = {"node": node, "source_manifest": self.source_manifest(node),
                    "origins": self.origins_dict(node)}
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def effective_document(self, node: str) -> dict:
        """Independent delivery view with root Contract and this node's metadata."""
        plan = dict(self.effective_task[node])
        if "CONTRACT" in self.effective_task["root"]:
            plan["CONTRACT"] = self.effective_task["root"]["CONTRACT"]
        _Budget().charge(_units(plan))
        return _clone(plan)

    def diagnostic_source(self, node: str, path: str) -> tuple[str, str, RegionOrigin | None]:
        contract = f"{node}.CONTRACT"
        if node != "root" and (path == contract or path.startswith(contract + ".") or path.startswith(contract + "[")):
            return "root", "root" + path[len(node):], None
        for region, origin in self.origins[node].items():
            prefix = f"{node}.DESIGN.{region}"
            if path == prefix + ".from":
                return node, path, None
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
    """Count before allocating; yaml_source has already rejected recursive aliases."""
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
    """Iterative alias-free copy retaining scalar types and mapping order."""
    result = [None]
    pending = [(result, 0, value)]
    while pending:
        parent, key, item = pending.pop()
        if isinstance(item, dict):
            child = dict.fromkeys(item)
            parent[key] = child
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
        raise SchemaError(f"{directory}: pass a YAML task directory")
    files = {}
    for file in sorted(directory.iterdir()):
        if file.suffix not in {".yaml", ".yml"}:
            continue
        if not file.is_file():
            raise SchemaError(f"{file}: expected a YAML file")
        errors = []
        node_id(file.stem, "<file>", errors)
        if errors:
            raise SchemaError(f"{file}: {errors[0].message}")
        if file.stem in files:
            raise SchemaError(f"{file}: duplicate node ID {file.stem}; also defined by {files[file.stem]}")
        files[file.stem] = file
    if "root" not in files:
        raise SchemaError(f"{directory}: missing root.yaml (or root.yml); a task needs one shared Contract")
    if len(files) > MAX_DESIGNS:
        raise SchemaError(f"{directory}: at most {MAX_DESIGNS} complete YAML designs, including root, are allowed; found {len(files)}")
    return files


def read_directory(directory: Path) -> tuple[dict, dict]:
    """Read each current task file once; callers can reuse the final snapshot."""
    files = task_files(directory)
    documents = {}
    for name, path in files.items():
        try:
            documents[name] = yaml_source.load(path, name)
        except yaml_source.ReadError as error:
            raise SchemaError(str(error)) from error
        except (yaml_source.DependencyError, OSError) as error:
            raise ToolError(str(error)) from error
    return files, documents


@bounded
def resolve_documents(files: dict, documents: dict) -> ResolvedTask:
    """Resolve one read snapshot without approving unfinished local region content.

    Reference edges from all three regions form one node DAG. A region imports
    only its same-named content, so references inside that content bind later in
    each consumer's full effective context.
    """
    if files.keys() != documents.keys():
        raise ToolError("source files and parsed documents must name the same snapshot nodes")
    if "root" not in documents:
        raise SchemaError("a task needs root with the shared Contract")
    if len(documents) > MAX_DESIGNS:
        raise SchemaError(f"at most {MAX_DESIGNS} complete YAML designs, including root, are allowed; found {len(documents)}")
    raw = {name: document.value for name, document in documents.items()}
    errors, references, edges = [], {}, {name: [] for name in raw}

    def fail(path, message, *, missing=False):
        errors.append(Diagnostic(path, message, missing))

    for name, plan in raw.items():
        node_id(name, name, errors)
        if not isinstance(plan, dict):
            fail(name, "expected one plan mapping")
            continue
        meta = plan.get("META_DATA")
        if not isinstance(meta, dict):
            fail(name + ".META_DATA", "expected a metadata mapping", missing="META_DATA" not in plan)
            meta = {}
        dispatch = meta.get("Dispatch", {})
        if not isinstance(dispatch, dict):
            fail(name + ".META_DATA.Dispatch", "expected a Dispatch mapping")
            dispatch = {}
        identity = dispatch.get("id", "root")
        node_id(identity, name + ".META_DATA.Dispatch.id", errors)
        if identity != name:
            fail(name + ".META_DATA.Dispatch.id", "Dispatch.id must match the YAML filename",
                 missing="id" not in dispatch)
        design = plan.get("DESIGN")
        if not isinstance(design, dict):
            fail(name + ".DESIGN", "expected a DESIGN mapping", missing="DESIGN" not in plan)
            continue
        for region in REGIONS:
            value, path = design.get(region), f"{name}.DESIGN.{region}"
            if not isinstance(value, dict):
                fail(path, "expected a local region mapping or {from: id}", missing=region not in design)
                continue
            if "from" not in value:
                continue
            if set(value) != {"from"}:
                fail(path, "from must be the only field; a local definition and a reference cannot be mixed")
            target = value["from"]
            node_id(target, path + ".from", errors)
            if name == "root":
                fail(path + ".from", "root DESIGN regions must be local")
            if not isinstance(target, str) or target not in raw:
                fail(path + ".from", f"unknown source design {target!r}")
                continue
            references[name, region] = target
            edges[name].append((target, path + ".from"))
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents))

    order, active, done = [], [], set()

    def visit(name):
        active.append(name)
        for target, path in edges[name]:
            if target in active:
                cycle = active[active.index(target):] + [target]
                fail(path, "cross-region source cycle: " + " -> ".join(cycle))
            elif target not in done:
                visit(target)
        active.pop()
        done.add(name)
        order.append(name)

    for name in raw:
        if name not in done:
            visit(name)
    if errors:
        raise SchemaError(format_diagnostics(errors, files, documents))

    budget, origins, dependencies, effective = _Budget(), {}, {}, {}
    region_sizes = {}
    for name in order:
        origins[name] = {}
        required = {"root", name}
        design = {}
        for region in REGIONS:
            source_node = references.get((name, region))
            inherited = origins[source_node][region] if source_node is not None else None
            origin = (RegionOrigin(inherited.provider, (name,) + inherited.chain)
                      if inherited else RegionOrigin(name, (name,)))
            origins[name][region] = origin
            budget.charge(1 + len(origin.chain))
            required.update(origin.chain)
            design[region] = raw[origin.provider]["DESIGN"][region]
            key = (origin.provider, region)
            if key not in region_sizes:
                region_sizes[key] = _units(design[region])
        # A draft preview retains unknown fields for the author to correct;
        # validation checks raw structure before calling this resolver.
        design.update((key, value) for key, value in raw[name]["DESIGN"].items() if key not in REGIONS)
        local = {key: value for key, value in raw[name].items() if key != "DESIGN"}
        size = _units(local) + 1 + sum(region_sizes[origin.provider, region]
                                      for region, origin in origins[name].items())
        size += sum(_units(value) for key, value in design.items() if key not in REGIONS)
        budget.charge(size + len(required))
        local["DESIGN"] = design
        effective[name] = _clone(local)
        dependencies[name] = frozenset(required)
    return ResolvedTask(raw, {name: effective[name] for name in raw}, origins, dependencies,
                        dict(files), {name: doc.source_sha256 for name, doc in documents.items()}, dict(documents))


@bounded
def resolve_directory(directory: str | Path) -> ResolvedTask:
    """Read and expand references without approving incomplete local content."""
    files, documents = read_directory(Path(directory).expanduser().resolve())
    return resolve_documents(files, documents)


@bounded
def effective_yaml(resolved: ResolvedTask, node: str) -> str:
    """Emit an independent view and verify exact scalar values survive."""
    import yaml

    class Dumper(yaml.SafeDumper):
        pass

    def decimal_value(dumper, value):
        text = str(value)
        if "." not in text and "e" not in text.lower():
            text += ".0"
        return dumper.represent_scalar("tag:yaml.org,2002:float", text)

    Dumper.add_representer(Decimal, decimal_value)
    # PyYAML's own resolver differs from yaml_source for e.g. 1e3.
    Dumper.add_representer(str, lambda dumper, value: dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style='"'))
    value = resolved.effective_document(node)
    text = yaml.dump(value, Dumper=Dumper, sort_keys=False, allow_unicode=True)
    reread = yaml_source.loads(text, node).value
    pending = [(value, reread)]
    while pending:
        before, after = pending.pop()
        if type(before) is not type(after) or before != after:
            raise ToolError("effective YAML output did not preserve scalar values and types")
        if isinstance(before, dict):
            pending.extend((item, after[key]) for key, item in before.items())
        elif isinstance(before, list):
            pending.extend(zip(before, after))
    return text
