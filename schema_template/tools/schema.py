#!/usr/bin/env python3
"""Initialize editable YAML plans or validate complete YAML task directories.

Python 3.10+ standard library and pinned CUE are the only dependencies. CUE
parses YAML and owns the structural constraints; validation never writes plans.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import sys

from validate_relations import validate_task
import json_value

TEMPLATE_ROOT = Path(__file__).resolve().parents[1]
ROOT = TEMPLATE_ROOT.parent
VERSION = ROOT.joinpath(".cue-version").read_text().strip()
YAML_ID = "strings.TrimSuffix(path.Base(filename), path.Ext(filename))"


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
    version = next(iter(result.stdout.splitlines()), "no version reported")
    if version != f"cue version {VERSION}":
        raise SchemaError(f"Expected CUE {VERSION}; got {version}.")
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


def task_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise SchemaError(f"{directory}: pass a YAML task directory.")
    files = {}
    for file in sorted(directory.iterdir()):
        if file.suffix not in {".yaml", ".yml"}:
            continue
        if not file.is_file():
            raise SchemaError(f"{file}: expected a YAML file.")
        if file.stem in files:
            raise SchemaError(f"{file}: duplicate node ID {file.stem}; also defined by {files[file.stem]}.")
        files[file.stem] = file
    if "root" not in files:
        raise SchemaError(f"{directory}: missing root.yaml (or root.yml); a task needs one shared Contract.")
    return files


def validate_directory(directory: str | Path) -> dict:
    """Read all direct YAML children; return their original parsed values.

    Decode as lists first so that CUE cannot silently unify multiple documents
    into one plan. Then vet the original files together, retaining CUE's source
    locations and enforcing #Task across the complete task. No defaults, shared
    Contract copies, formatting changes or validation metadata are written back.
    """
    directory = Path(directory).resolve()
    files = task_files(directory)
    paths = [str(file) for file in files.values()]
    placement = ("--with-context", "-l", YAML_ID)
    documents = json_value.loads(cue("export", "--list", *placement, "yaml:", *paths, "--out", "json"))
    for name, records in documents.items():
        if len(records) != 1 or not isinstance(records[0], dict):
            raise SchemaError(f"{files[name]}: expected exactly one YAML mapping document per node.")
    task = {name: records[0] for name, records in documents.items()}
    try:
        cue("vet", "-c", "-E", str(TEMPLATE_ROOT / "tools/validate_task.cue"),
            "--merge", *placement, "yaml:", *paths)
    except SchemaError as e:
        # Missing required fields may only carry the schema's source location.
        # Prefix their field paths with the responsible YAML file as well.
        lines = []
        for line in str(e).splitlines():
            node, dot, rest = line.partition(".")
            lines.append(f"{files[node]}: {rest}" if dot and node in files else line)
        if not any(str(file) in "\n".join(lines) or file.name in str(e) for file in files.values()):
            # A comprehension that reads a missing field can report only its
            # template location. Isolate structural failures to their source.
            isolated = []
            for node, file in files.items():
                try:
                    cue("vet", "-c", "./schema_template:kernel", "-d",
                        "#RootPlan" if node == "root" else "#DerivedPlan", str(file))
                except SchemaError as local:
                    isolated.append(f"{file}:\n{local}")
            lines = isolated or [f"{directory}: task relations:", *lines]
        raise SchemaError("\n".join(lines)) from e
    errors = validate_task(task)
    if errors:
        messages = []
        for error in errors:
            node, _, field = error.path.partition(".")
            messages.append(f"{files[node]}: {field}: {error.message}")
        raise SchemaError("\n".join(messages))
    return task


def init(directory: str | Path) -> Path:
    directory = Path(directory).resolve()
    if directory.exists() and (not directory.is_dir() or any(
            p.suffix in {".yaml", ".yml"} for p in directory.iterdir())):
        raise SchemaError(f"{directory}: init needs a new task; existing plan files are preserved.")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "root.yaml"
    with target.open("xb") as output:
        output.write(TEMPLATE_ROOT.joinpath("starter.yaml").read_bytes())
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init_parser = commands.add_parser("init", help="copy an editable YAML skeleton; fill its null holes before validation")
    init_parser.add_argument("directory")
    validate_parser = commands.add_parser("validate", help="read-only structural and relation checks on entire YAML tasks")
    validate_parser.add_argument("directories", nargs="+", help="task directories containing root.yaml and optional derived nodes")
    args = parser.parse_args(argv)
    if args.command == "init":
        try:
            target = init(args.directory)
            print(f"Created {target}; fill the YAML skeleton, then run validate.")
            return 0
        except (SchemaError, OSError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 1
    failed = False
    for directory in args.directories:
        try:
            validate_directory(directory)
            print(f"Valid: {Path(directory).resolve()}")
        except (SchemaError, OSError, ValueError) as e:
            print(str(e), file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
