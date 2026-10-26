#!/usr/bin/env python3
"""Author CUE, validate the whole task, and generate/check derived YAML.

Python 3.10+ standard library and the pinned CUE binary are the only runtime
dependencies. YAML is never read as authoring input.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from validate_relations import validate_task
import compact_yaml
import json_value

TEMPLATE_ROOT = Path(__file__).resolve().parents[1]
ROOT = TEMPLATE_ROOT.parent
VERSION = ROOT.joinpath(".cue-version").read_text().strip()
MANIFEST = ".schema-generated.json"


class SchemaError(Exception):
    pass


@lru_cache(maxsize=1)
def cue_binary() -> str:
    candidates = [os.environ.get("CUE"), str(TEMPLATE_ROOT / ".tools" / "cue" / VERSION / "cue"), shutil.which("cue")]
    binary = next((p for p in candidates if p and Path(p).is_file()), None)
    if binary is None:
        raise SchemaError("CUE is missing. Run make setup.")
    try:
        result = subprocess.run([binary, "version"], capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as e:
        raise SchemaError(f"Cannot run CUE: {e}") from e
    if result.stdout.splitlines()[0] != f"cue version {VERSION}":
        raise SchemaError(f"Expected CUE {VERSION}; got {result.stdout.splitlines()[0]}.")
    return str(Path(binary).resolve())


def cue(*args: str, data: str | None = None) -> str:
    try:
        result = subprocess.run([cue_binary(), *args], cwd=ROOT, input=data,
                                capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise SchemaError(f"CUE invocation failed: {e}") from e
    if result.returncode:
        raise SchemaError(result.stderr.strip() or result.stdout.strip() or "CUE validation failed.")
    return result.stdout


def task_directory(path: str | Path, *, exists: bool = True) -> Path:
    directory = Path(path).resolve()
    try:
        directory.relative_to(ROOT)
    except ValueError as e:
        raise SchemaError("Task sources must be inside this repository's CUE module.") from e
    if directory == ROOT or (exists and not directory.is_dir()):
        raise SchemaError("Pass a task directory, not a YAML/CUE file or the repository root.")
    if exists and not list(directory.glob("*.cue")):
        raise SchemaError(f"{directory}: no CUE source; legacy YAML is not an authoring input.")
    return directory


def validate_data(task: dict) -> None:
    cue("vet", "-c", "./schema_template:kernel", "-d", "#Task", "json:", "-",
        data=json_value.dumps(task))
    errors = validate_task(task)
    if errors:
        raise SchemaError("\n".join(map(str, errors)))


def compile_task(directory: Path) -> dict:
    # Export the complete public value first: an omitted schema import or an
    # accidental extra public field must not bypass the generation boundary.
    package = "./" + directory.relative_to(ROOT).as_posix() + ":plan"
    data = json_value.loads(cue("export", package, "--out", "json"))
    if not isinstance(data, dict) or set(data) != {"plans"} or not isinstance(data["plans"], dict):
        raise SchemaError("A task package exports exactly 'plans'; use let/# definitions for CUE-only helpers.")
    validate_data(data["plans"])
    return data["plans"]


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def input_hashes(directory: Path) -> dict[str, str]:
    definitions = [p for p in TEMPLATE_ROOT.glob("*.cue") if p.name != "starter.cue"]
    paths = [*directory.glob("*.cue"), *definitions,
             ROOT / "cue.mod/module.cue", ROOT / ".cue-version",
             TEMPLATE_ROOT / "tools/schema.py", TEMPLATE_ROOT / "tools/validate_relations.py",
             TEMPLATE_ROOT / "tools/json_value.py", TEMPLATE_ROOT / "tools/compact_yaml.py"]
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()) for p in sorted(paths)}


def expected_files(directory: Path) -> dict[str, bytes]:
    inputs = input_hashes(directory)
    task = compile_task(directory)
    files = {}
    for identity in sorted(task):
        plan = task[identity]
        # Presentation is separate from CUE constraints. Verify that compact
        # YAML decodes to exactly the same data before publishing any output.
        plan = {key: plan[key] for key in ("meta_data", "contract", "design", "realization") if key in plan}
        payload = compact_yaml.dumps(plan)
        decoded = json_value.loads(cue("export", "yaml:", "-", "--out", "json", data=payload))
        if json_value.dumps(decoded, sort_keys=True) != json_value.dumps(plan, sort_keys=True):
            raise SchemaError(f"{identity}: YAML formatting changed the data; refusing to write outputs.")
        files[identity + ".yaml"] = payload.encode("utf-8")
    manifest = {
        "format": 1,
        "cue": VERSION,
        "inputs": inputs,
        "outputs": {name: sha256(data) for name, data in sorted(files.items())},
        "revisions": {name + ".yaml": task[name]["meta_data"]["version"] for name in sorted(task)},
        "values": {name + ".yaml": sha256(json_value.dumps(task[name], sort_keys=True).encode())
                   for name in sorted(task)},
    }
    if inputs != input_hashes(directory):
        raise SchemaError("CUE inputs changed during generation; retry with stable sources.")
    files[MANIFEST] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return files


def read_manifest(directory: Path) -> dict:
    file = directory / MANIFEST
    if not file.exists() and not file.is_symlink():
        return {}
    if file.is_symlink():
        raise SchemaError("Generated manifests cannot be symlinks.")
    try:
        manifest = json.loads(file.read_text())
        if manifest["format"] != 1 or not isinstance(manifest["outputs"], dict):
            raise ValueError("unsupported manifest")
        for name, digest in manifest["outputs"].items():
            if not isinstance(name, str) or not re_output_name(name) or not isinstance(digest, str):
                raise ValueError("invalid output identity")
        if set(manifest["outputs"]) != set(manifest["revisions"]) or set(manifest["outputs"]) != set(manifest["values"]):
            raise ValueError("inconsistent manifest identities")
        if any(type(v) is not int or v < 1 for v in manifest["revisions"].values()):
            raise ValueError("invalid plan revision")
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SchemaError(f"{file}: invalid generation manifest ({e})") from e
    return manifest


def re_output_name(name: str) -> bool:
    import re
    return re.fullmatch(r"[a-z_][a-z0-9_]*\.yaml", name) is not None


def write_file(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".schema-", delete=False) as output:
        temporary = Path(output.name)
        output.write(payload)
    try:
        temporary.chmod(0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(directory: Path, *, repair: bool = False) -> list[str]:
    # Evaluate, validate and render ALL nodes before changing any output.
    expected = expected_files(directory)
    previous = read_manifest(directory)
    old = previous.get("outputs", {})
    next_manifest = json.loads(expected[MANIFEST])
    yaml_files = {p.name for ext in ("*.yaml", "*.yml") for p in directory.glob(ext)}
    foreign = yaml_files - set(old)
    if foreign:
        raise SchemaError("Refusing to overwrite/adopt handwritten YAML: " + ", ".join(sorted(foreign)))
    for name in set(old) & set(next_manifest["outputs"]):
        if previous["values"][name] != next_manifest["values"][name]:
            if next_manifest["revisions"][name] <= previous["revisions"][name]:
                raise SchemaError(f"{name}: plan content changed; increment meta_data.version in CUE.")
    for name, digest in old.items():
        target = directory / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise SchemaError(f"{target}: generated outputs must be regular files.")
        if target.exists() and sha256(target.read_bytes()) != digest and not repair:
            raise SchemaError(f"{target}: generated YAML was edited. Change CUE; use build --repair to discard this edit.")
    for name in expected:
        target = directory / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise SchemaError(f"{target}: generated outputs must be regular files.")
    for name, payload in expected.items():
        write_file(directory / name, payload)
    # Remove only outputs owned by the previous manifest, never arbitrary files.
    for name in sorted(set(old) - set(expected)):
        (directory / name).unlink(missing_ok=True)
    return sorted(name for name in expected if name.endswith(".yaml"))


def check_generated(directory: Path) -> None:
    expected = expected_files(directory)
    actual_yaml = {p.name for ext in ("*.yaml", "*.yml") for p in directory.glob(ext)}
    failures = []
    for name, payload in expected.items():
        target = directory / name
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            failures.append(name + " is missing, stale or edited")
    for name in sorted(actual_yaml - set(expected)):
        failures.append(name + " has no CUE plan")
    if failures:
        raise SchemaError("\n".join(failures) + "\nRegenerate from CUE with schema_template/tools/schema.py build.")


def discover() -> list[Path]:
    return sorted({p.parent for base in (ROOT / "schema_wiki", TEMPLATE_ROOT / "examples")
                   for pattern in ("*.cue", MANIFEST)
                   for p in base.rglob(pattern) if "archived" not in p.relative_to(ROOT).parts})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init_parser = commands.add_parser("init", help="create a CUE task with non-exportable typed holes")
    init_parser.add_argument("directory")
    for name in ("validate", "build", "check"):
        sub = commands.add_parser(name)
        sub.add_argument("directory", nargs="?")
        sub.add_argument("--all", action="store_true", help="all CUE-authored tasks in schema_wiki and schema_template/examples")
        if name == "build":
            sub.add_argument("--repair", action="store_true", help="discard edits to manifest-owned generated YAML")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            directory = task_directory(args.directory, exists=False)
            target = directory / "task.cue"
            if directory.exists() and any(list(directory.glob(pattern)) for pattern in ("*.cue", "*.yaml", "*.yml")):
                raise SchemaError("init requires a new task; existing sources and legacy YAML are preserved.")
            directory.mkdir(parents=True, exist_ok=True)
            with target.open("x") as output:
                source = TEMPLATE_ROOT.joinpath("starter.cue").read_text()
                output.write(source.replace("package starter\n", "package plan\n", 1))
            print(f"Created {target}; fill typed holes, then run build.")
            return 0
        if bool(args.directory) == args.all:
            parser.error("choose one task directory or --all")
        directories = discover() if args.all else [task_directory(args.directory)]
        for directory in directories:
            if args.command == "validate":
                compile_task(directory)
                print(f"Valid: {directory.relative_to(ROOT)}")
            elif args.command == "build":
                outputs = build(directory, repair=args.repair)
                print(f"Generated {directory.relative_to(ROOT)}: {', '.join(outputs)}")
            else:
                check_generated(directory)
                print(f"Current: {directory.relative_to(ROOT)}")
        if not directories:
            print("No CUE-authored tasks found; legacy YAML was not migrated.")
        return 0
    except (SchemaError, OSError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
