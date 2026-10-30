#!/usr/bin/env python3
"""Check and visualize representative tile pipelines, one figure per YAML.

Usage: python3 schema_template/tools/pipeline_graph.py path/to/root.yaml
Adjacent pipeline_samples.json provides review sizes, not kernel parameters.
Use --all for sibling nodes; --set NAME=INT overrides sample parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import schema
from tools.pipeline import GraphError, build_phase
from tools.pipeline.model import ExpansionError
from tools.pipeline.render import RenderError, dot_text, gallery, graphviz, viewer


def read_sample(path: Path) -> dict:
    if not path.exists():
        return {}
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise GraphError(f"{path}: duplicate sample key {key!r}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    except json.JSONDecodeError as error:
        raise GraphError(f"{path}:{error.lineno}:{error.colno}: {error.msg}") from error
    if not isinstance(value, dict) or set(value) - {"parameters", "worktiles", "nodes", "notes"}:
        raise GraphError(f"{path}: expected parameters / worktiles / nodes / notes")
    if not isinstance(value.get("nodes", {}), dict):
        raise GraphError(f"{path}: nodes must be a mapping")
    notes = value.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise GraphError(f"{path}: notes must be a list of strings")
    for config in [value, *value.get("nodes", {}).values()]:
        if not isinstance(config, dict) or not isinstance(config.get("parameters", {}), dict):
            raise GraphError(f"{path}: sample parameters must be a mapping")
    for config in value.get("nodes", {}).values():
        if set(config) - {"parameters", "worktiles"}:
            raise GraphError(f"{path}: node samples accept only parameters / worktiles")
    return value


def configuration(sample: dict, node: str, overrides: dict, worktiles: int | None) -> dict:
    selected = sample.get("nodes", {}).get(node, {})
    parameters = {**sample.get("parameters", {}), **selected.get("parameters", {}), **overrides}
    if any(not isinstance(k, str) or type(v) is not int for k, v in parameters.items()):
        raise ExpansionError("sample parameters must map expression names to integers")
    count = worktiles if worktiles is not None else selected.get("worktiles", sample.get("worktiles", 2))
    if type(count) is not int or not 0 <= count <= 64:
        raise ExpansionError("sample worktiles must be an integer in 0..64")
    return {"parameters": parameters, "worktiles": count}


def generate(source: Path, plan: dict, sample: dict, config: dict, output: Path, format: str) -> dict:
    graphs, samples = [], {}
    for pid, phase in plan["design"]["tile_pipeline"]["phase"].items():
        samples[pid] = config
        worker = plan["design"]["work_schedule"]["phase"][pid]["mapping"]["worker"]
        graphs.append(build_phase(pid, phase, worker, **config))
    report = {"schema": "pipeline-graph-trial-1", "source": str(source),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "scope": "representative phase-local event dependencies",
              "acyclic": all(not g.cycle for g in graphs), "samples": samples,
              "notes": sample.get("notes", []),
              "phases": [g.report() for g in graphs]}
    output.mkdir(parents=True, exist_ok=True)
    base = output / source.stem
    title = source.parent.name + " / " + source.stem
    base.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    base.with_suffix(".dot").write_text(dot_text(title, graphs, samples), encoding="utf-8")
    # A DOT-only refresh must not leave an SVG/HTML from an older input.
    base.with_suffix(".svg").unlink(missing_ok=True)
    base.with_suffix(".html").unlink(missing_ok=True)
    if format == "svg":
        graphviz(base.with_suffix(".dot"), base.with_suffix(".svg"))
        base.with_suffix(".html").write_text(viewer(title, base.with_suffix(".svg").name, report), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("yaml", type=Path)
    parser.add_argument("--out", type=Path, help="output directory; defaults to <task>/pipeline_graphs")
    parser.add_argument("--sample", type=Path, help="sample JSON; defaults to <task>/pipeline_samples.json")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=INT")
    parser.add_argument("--worktiles", type=int, help="number of representative works for each persistent worker")
    parser.add_argument("--all", action="store_true", help="generate one figure for each sibling YAML")
    parser.add_argument("--format", choices=("svg", "dot"), default="svg")
    args = parser.parse_args(argv)
    notices = []
    try:
        directory, task = schema.validate_file(args.yaml, notices=notices)
        files = schema.task_files(directory)
        if args.sample and not args.sample.is_file():
            raise GraphError(f"sample file not found: {args.sample}")
        sample = read_sample(args.sample or directory / "pipeline_samples.json")
        if notices:
            sample = {**sample, "notes": [*sample.get("notes", []), *notices]}
        unknown = sample.get("nodes", {}).keys() - task.keys()
        if unknown:
            raise GraphError(f"sample refers to unknown YAML nodes: {', '.join(sorted(unknown))}")
        overrides = {}
        for item in args.set:
            name, value = item.split("=", 1)
            if not name.strip():
                raise ExpansionError("--set needs a nonempty parameter name, for example N=4")
            overrides[name] = int(value)
        output = (args.out or directory / "pipeline_graphs").expanduser().resolve()
        result = 0
        entries = []
        for node in (task if args.all else [args.yaml.stem]):
            try:
                config = configuration(sample, node, overrides, args.worktiles)
                report = generate(files[node], task[node], sample, config, output, args.format)
                count = sum(len(p["operations"]) for p in report["phases"])
                status = "Acyclic sample" if report["acyclic"] else "DEPENDENCY CYCLE"
                print(f"{status}: {node} ({len(report['phases'])} phases, {count} actions) -> {output / (node + '.' + args.format)}")
                for phase in report["phases"]:
                    if phase["cycle"]:
                        print("  " + " -> ".join(phase["cycle"]), file=sys.stderr)
                        pairs = set(zip(phase["cycle"], phase["cycle"][1:]))
                        reasons = list(dict.fromkeys(f"{edge['kind']}: {reason}"
                            for edge in phase["edges"] if (edge["from"], edge["to"]) in pairs and edge["kind"] != "execute"
                            for reason in edge["reasons"]))
                        for reason in reasons[:6]:
                            print("    " + reason, file=sys.stderr)
                result = max(result, 0 if report["acyclic"] else 1)
                entries.append({"name": node, "href": node + (".html" if args.format == "svg" else ".dot"),
                                "summary": f"{len(report['phases'])} phases · {count} actions · {status}"})
            except RenderError as error:
                for suffix in (".svg", ".html"):
                    (output / (node + suffix)).unlink(missing_ok=True)
                print(f"Render not completed: {error}; DOT and audit JSON are retained in {output}", file=sys.stderr)
                result = 2
            except GraphError as error:
                # Do not leave a successful picture from an earlier run for an
                # input that no longer expands. These are generated files only.
                for suffix in (".svg", ".html", ".dot", ".json"):
                    (output / (node + suffix)).unlink(missing_ok=True)
                label = "Graph not completed: " if isinstance(error, ExpansionError) else ""
                print(f"{label}{files[node]}: {error}", file=sys.stderr)
                result = max(result, 2 if isinstance(error, ExpansionError) else 1)
        if args.all:
            output.mkdir(parents=True, exist_ok=True)
            (output / "index.html").write_text(gallery(directory.name + " · Pipeline review", entries), encoding="utf-8")
        return result
    except schema.SchemaError as error:
        print(error, file=sys.stderr)
        return 1
    except (schema.ToolError, GraphError, OSError, ValueError) as error:
        print(f"Graph not completed: {error}", file=sys.stderr)
        return 2
    finally:
        for notice in notices:
            print(notice, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
