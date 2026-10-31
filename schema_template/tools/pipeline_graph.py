#!/usr/bin/env python3
"""Check finite pipeline dependencies and write <node>.png beside its YAML.

Use --set NAME=INT for illustration parameters and --worktiles for the sample
work sequence. --all processes sibling nodes; --check-only writes no files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import schema
from tools.pipeline import GraphError, build_phase
from tools.pipeline.model import ExpansionError
from tools.pipeline.output import GraphOutput
from tools.pipeline.render import RenderError


def configuration(parameters: dict, worktiles: int = 1) -> dict:
    if any(not isinstance(name, str) or not name.strip() or type(value) is not int
           for name, value in parameters.items()):
        raise ExpansionError("sample parameters must map nonempty expression names to integers")
    if type(worktiles) is not int or not 0 <= worktiles <= 64:
        raise ExpansionError("sample worktiles must be an integer in 0..64")
    return {"parameters": parameters, "worktiles": worktiles}


def analyze(plan: dict, config: dict) -> list:
    graphs = []
    for pid, phase in plan["design"]["tile_pipeline"]["phase"].items():
        worker = plan["design"]["work_schedule"]["phase"][pid]["mapping"]["worker"]
        graphs.append(build_phase(pid, phase, worker, **config))
    return graphs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("yaml", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=INT")
    parser.add_argument("--worktiles", type=int, default=1, help="representative works per worker (default: 1)")
    parser.add_argument("--all", action="store_true", help="process every sibling YAML")
    parser.add_argument("--check-only", action="store_true", help="check dependencies without rendering or changing files")
    args = parser.parse_args(argv)
    entry = args.yaml.expanduser()
    source = entry.parent.resolve() / entry.name
    notices = []
    try:
        if source.suffix not in {".yaml", ".yml"} or not source.is_file():
            raise schema.ToolError(f"{source}: pass one existing .yaml or .yml file")
        output = GraphOutput()
        if not args.check_only:
            output.begin(source, all_nodes=args.all)
        directory, task = schema.validate_file(source, notices=notices)
        files = schema.task_files(directory)
        parameters = {}
        for item in args.set:
            name, value = item.split("=", 1)
            if name in parameters:
                raise ExpansionError(f"duplicate --set parameter {name!r}")
            parameters[name] = int(value)
        config = configuration(parameters, args.worktiles)
        notes = ["Hardware exemption applied; missing hardware specifications and dependent references are deferred."] if notices else []
        result = 0
        for node in (task if args.all else [source.stem]):
            try:
                graphs = analyze(task[node], config)
                cyclic = any(graph.cycle for graph in graphs)
                status = "DEPENDENCY CYCLE" if cyclic else "Acyclic sample"
                count = sum(len(graph.operations) for graph in graphs)
                result = max(result, int(cyclic))
                for graph in graphs:
                    if graph.cycle:
                        print("  " + " -> ".join(graph.cycle), file=sys.stderr)
                        pairs = set(zip(graph.cycle, graph.cycle[1:]))
                        reasons = list(dict.fromkeys(f"{kind}: {reason}"
                            for (a, b, kind), explanations in graph.edges.items()
                            if (a, b) in pairs and kind != "execute" for reason in sorted(explanations)))
                        for reason in reasons[:6]:
                            print("    " + reason, file=sys.stderr)
                destination = ""
                if not args.check_only:
                    samples = {graph.phase: config for graph in graphs}
                    destination = " -> " + str(output.write(files[node], graphs, samples, notes))
                print(f"{status}: {node} ({len(graphs)} phases, {count} actions){destination}")
            except RenderError as error:
                print(f"Render not completed: {files[node]}: {error}", file=sys.stderr)
                result = 2
            except GraphError as error:
                incomplete = isinstance(error, ExpansionError)
                print(f"{'Graph not completed: ' if incomplete else ''}{files[node]}: {error}", file=sys.stderr)
                result = max(result, 2 if incomplete else 1)
        return result
    except schema.SchemaError as error:
        print(error, file=sys.stderr)
        return 1
    except (schema.ToolError, OSError, ValueError) as error:
        print(f"Graph not completed: {error}", file=sys.stderr)
        return 2
    finally:
        for notice in notices:
            print(notice, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
