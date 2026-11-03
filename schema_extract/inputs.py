"""Resolve catalog, local, and historical sources to explicit input roles."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess

from .common import ExtractError, read_json, regular_file, relative_name


SOURCE_FORMAT = "schema_extract.source.v2"
IGNORED = {".git", ".schema_extract", "__pycache__", ".DS_Store"}


@dataclass
class SourceTree:
    root: Path
    commit: str | None = None

    def git(self, *args: str) -> bytes:
        result = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True)
        if result.returncode:
            raise ExtractError(f"Git source lookup failed: {' '.join(args[:2])}: "
                               + result.stderr.decode(errors="replace").strip())
        return result.stdout

    def read(self, name: str) -> bytes:
        relative_name(name)
        if self.commit:
            return self.git("show", f"{self.commit}:{name}")
        return regular_file(self.root, name).read_bytes()

    def exists(self, name: str) -> bool:
        try:
            self.read(name)
            return True
        except ExtractError:
            return False

    def files(self, directory: str) -> list[str]:
        if directory != ".":
            relative_name(directory)
        if self.commit:
            output = self.git("ls-tree", "-rz", self.commit, "--", directory)
            names = []
            for row in output.split(b"\0"):
                if not row:
                    continue
                head, raw_name = row.split(b"\t", 1)
                mode, kind, _ = head.decode().split()
                name = raw_name.decode("utf-8")
                if set(Path(name).parts) & IGNORED:
                    continue
                if kind != "blob" or mode not in {"100644", "100755"}:
                    raise ExtractError(f"Git source is a link/submodule: {name}")
                names.append(relative_name(name))
        else:
            base = self.root / directory
            if not base.is_dir() or base.is_symlink():
                raise ExtractError(f"missing source directory: {base}")
            names = []
            for path in sorted(base.rglob("*")):
                name = path.relative_to(self.root).as_posix()
                if set(Path(name).parts) & IGNORED:
                    continue
                if path.is_symlink():
                    raise ExtractError(f"source symlink is not supported: {name}")
                if path.is_file():
                    regular_file(self.root, name)
                    names.append(name)
        if not names:
            raise ExtractError(f"source directory contains no files: {directory}")
        return sorted(names)


def _commit(root: Path, ref: str) -> str:
    if not isinstance(ref, str) or not ref or ref.startswith("-"):
        raise ExtractError("revision must be an explicit Git ref")
    return SourceTree(root).git("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()


def _locator(value, name: str, prefix: str, tree: SourceTree) -> dict:
    if isinstance(value, str):
        path, _, symbol = value.partition("::")
    elif isinstance(value, dict) and set(value) <= {"path", "symbol"}:
        path, symbol = value.get("path"), value.get("symbol", "")
    else:
        raise ExtractError(f"missing/invalid {name}: expected a path or {{path, symbol}}")
    relative_name(path)
    if not isinstance(symbol, str):
        raise ExtractError(f"invalid {name} symbol")
    resolved = path if prefix == "." else f"{prefix}/{path}"
    try:
        data = tree.read(resolved)
    except ExtractError as exc:
        raise ExtractError(f"missing {name} input: {resolved}") from exc
    if symbol:
        # Resolve Python qualified names statically; never import GPU source.
        if resolved.endswith(".py"):
            try:
                module = ast.parse(data.decode("utf-8"))
            except (SyntaxError, UnicodeError) as exc:
                raise ExtractError(f"cannot locate {name} in Python file {resolved}") from exc
            found = []
            def visit(node, parents=()):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        qualified = (*parents, child.name)
                        found.append(".".join(qualified))
                        visit(child, qualified)
                    else:
                        visit(child, parents)
            visit(module)
            matches = [item for item in found if item == symbol or item.endswith("." + symbol)]
            if len(matches) != 1:
                raise ExtractError(f"{name} symbol must resolve uniquely: {resolved}::{symbol}")
            symbol = matches[0]
        elif symbol not in data.decode("utf-8", errors="replace"):
            raise ExtractError(f"missing {name} symbol: {resolved}::{symbol}")
    return {"path": resolved, "symbol": symbol}


def validate_source_reference(value: str, lock: dict, *, role: str) -> dict:
    """Resolve a YAML source citation in this run's actual frozen source version."""
    source = lock["source"]
    matches = [entry for entry in [source, *source.get("aliases", [])]
               if isinstance(value, str) and value.startswith(f"{entry['repository_id']}@{entry['ref']}:")]
    if not matches:
        raise ExtractError(f"{role} source must cite a fixed repository and ref: {source['repository_id']}@{source['ref']}:")
    origin = matches[0]
    prefix = f"{origin['repository_id']}@{origin['ref']}:"
    locator = _locator(value[len(prefix):], role, ".", SourceTree(Path(lock["snapshot"]) / "source"))
    if locator["path"] not in origin["files"] or origin["files"][locator["path"]] != source["files"].get(locator["path"]):
        raise ExtractError(f"{role} source is not in the fixed manifest")
    if role == "reference":
        expected = source["bindings"]["reference"]
        if locator["path"] != expected["path"] or (expected["symbol"] and locator["symbol"] != expected["symbol"]):
            raise ExtractError("golden_reference.source must resolve to the bound reference entry")
    return locator


def _normalize(spec: dict, tree: SourceTree, directory: str) -> dict:
    if directory != ".":
        relative_name(directory)
    operator_id = spec.get("id")
    if not isinstance(operator_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", operator_id):
        raise ExtractError("source requires an explicit id (letters, numbers, '_' or '-')")
    missing = [key for key in ("entrypoint", "io", "input", "reference", "acceptance", "scope")
               if not spec.get(key)]
    if missing:
        raise ExtractError("missing required input bindings: " + ", ".join(missing))
    result = {"id": operator_id, "directory": directory}
    for key in ("entrypoint", "host", "input", "reference"):
        result[key] = _locator(spec.get(key, spec.get("entrypoint")), key, directory, tree)
    for key in ("io", "acceptance", "scope"):
        values = spec[key]
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or not values:
            raise ExtractError(f"{key} must identify one or more files")
        result[key] = [_locator(value, key, directory, tree) for value in values]
    if result["reference"] == result["entrypoint"] or (
            result["reference"]["path"] == result["entrypoint"]["path"] and
            (not result["reference"]["symbol"] or not result["entrypoint"]["symbol"])):
        raise ExtractError("reference must identify a distinct reference entry, not the selected implementation")
    return result


def _catalog_spec(entry: dict) -> dict:
    paths = entry["paths"]
    def path(key):
        value = paths.get(key)
        if not value:
            raise ExtractError(f"catalog {entry['id']} is missing {key}")
        return "kernel_wiki/" + relative_name(value)
    return {"id": entry["id"],
            "entrypoint": {"path": path("entrypoint"), "symbol": entry.get("interface", {}).get("forward_target", "")},
            "host": {"path": path("entrypoint"), "symbol": entry.get("interface", {}).get("forward_target", "")},
            "io": [path("metadata")], "input": {"path": path("input"), "symbol": "_make_inputs"},
            "reference": {"path": path("reference"), "symbol": "Model.forward"},
            "acceptance": [path(key) for key in ("acceptance", "precision", "user_acceptance") if paths.get(key)],
            "scope": [path("shape_range"), path("shape_cases")],
            "directory": str(Path(path("readme")).parent)}


def resolve_request(repo: Path, *, operator_id: str | None = None,
                    source: str | None = None, bindings: Path | None = None) -> tuple[SourceTree, dict, list[str]]:
    """Return fixed source roles plus selected repository-relative file names.

    Arbitrary sources require an explicit descriptor; filename resemblance never
    decides which solution is the design or reference.
    """
    repo = repo.resolve()
    if bool(operator_id) == bool(source):
        raise ExtractError("select exactly one of --id and --source")
    catalog_path = "kernel_wiki/catalog.json"
    if operator_id:
        tree = SourceTree(repo)
        catalog = read_json(repo / catalog_path)
        entries = [entry for entry in catalog["operators"] if entry["id"] == operator_id]
        if len(entries) != 1:
            raise ExtractError(f"unknown or duplicate catalog ID: {operator_id}")
        spec = _catalog_spec(entries[0])
        normalized = _normalize(spec, tree, ".")
        normalized["directory"] = spec["directory"]
        names = tree.files(spec["directory"])
        names += [catalog_path]
        # Batch-wide conventions are part of the task evidence when present.
        batch = Path(spec["directory"]).parts[:2]
        for directory in ("kernel_wiki", "/".join(batch)):
            for name in ("README.md", "CONVENTIONS.md", "PROBLEM_NOTES.md", "EVIDENCE.md"):
                if tree.exists(f"{directory}/{name}"):
                    names.append(f"{directory}/{name}")
        normalized["selection"] = {"kind": "catalog", "catalog": catalog_path, "id": operator_id}
    else:
        source_path = Path(source).expanduser()
        if source.startswith("git:"):
            parts = source[4:].split(":", 1)
            if len(parts) != 2 or not bindings:
                raise ExtractError("git:<ref>:<directory> requires --bindings <source.json>")
            ref, directory = parts
            tree = SourceTree(repo, _commit(repo, ref))
            spec = read_json(bindings.resolve())
        else:
            source_path = source_path.resolve()
            descriptor = bindings.resolve() if bindings else (
                source_path / "schema-inputs.json" if source_path.is_dir() else source_path)
            if not descriptor.is_file():
                # A catalog match is explicit metadata, not conventional-name inference.
                if source_path.is_dir() and (repo / catalog_path).is_file():
                    for entry in read_json(repo / catalog_path)["operators"]:
                        if (repo / "kernel_wiki" / entry["paths"]["readme"]).parent.resolve() == source_path:
                            return resolve_request(repo, operator_id=entry["id"])
                raise ExtractError(f"missing input descriptor: {descriptor}; bind entrypoint, io, input, reference, acceptance, scope")
            spec = read_json(descriptor)
            repository = Path(spec.get("repository", ".")).expanduser()
            root = (descriptor.parent / repository).resolve()
            directory = spec.get("directory", ".")
            tree = SourceTree(root, _commit(root, spec["revision"]) if spec.get("revision") else None)
        if spec.get("format") != SOURCE_FORMAT:
            raise ExtractError(f"source descriptor format must be {SOURCE_FORMAT}")
        normalized = _normalize(spec, tree, directory)
        names = tree.files(directory)
        normalized["selection"] = {"kind": "git" if tree.commit else "local", "descriptor": spec}
    # Every binding is included even when catalog evidence lies outside its directory.
    for key in ("entrypoint", "host", "io", "input", "reference", "acceptance", "scope"):
        values = normalized[key] if isinstance(normalized[key], list) else [normalized[key]]
        names.extend(value["path"] for value in values)
    return tree, normalized, sorted(set(names))
