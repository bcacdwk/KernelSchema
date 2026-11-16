#!/usr/bin/env python3
"""Normalize and check a complete YAML task; graph analysis is optional."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

sys.dont_write_bytecode = True


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml", help="one existing YAML in a complete task directory")
    parser.add_argument("--all", action="store_true", help="analyze every YAML when --graph is requested; normalization always covers the task")
    parser.add_argument("--read-only", action="store_true", help="preview normalization and check its candidates without writing")
    parser.add_argument("--skip-hardware", action="store_true", help="skip hardware rules, budgets and dependent relations")
    parser.add_argument("--graph", choices=("png", "check"), help="also analyze a finite sample; png publishes sibling images")
    parser.add_argument("--set", dest="parameters", action="append", default=[], metavar="NAME=INT",
                        help="explicit sample value; repeat for additional parameters")
    parser.add_argument("--worktiles", type=int, help="sample worktile count (default: 1)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--standalone", action="store_true", help="read-only check of one self-contained delivery view")
    mode.add_argument("--show-effective", action="store_true", help="read-only JSON draft view with validation: not_run")
    args = parser.parse_args(argv)
    if args.read_only and args.graph == "png":
        parser.error("--read-only and --graph png cannot be combined")
    if (args.standalone or args.show_effective) and (args.all or args.graph or args.parameters or args.worktiles is not None):
        parser.error("--standalone and --show-effective do not accept formatting or graph requests")
    if not args.graph and (args.parameters or args.worktiles is not None):
        parser.error("--set and --worktiles require --graph")
    parameters = {}
    for item in args.parameters:
        name, separator, value = item.rpartition("=")
        name = "".join(name.split())
        if not separator or not name or name in parameters:
            parser.error(f"expected a unique NAME=INT sample parameter: {item!r}")
        try:
            parameters[name] = int(value, 10)
        except ValueError:
            parser.error(f"expected an integer sample value: {item!r}")
    args.parameters = parameters
    args.worktiles = 1 if args.worktiles is None else args.worktiles
    return args


def _target(filename, schema):
    entry = Path(filename).expanduser()
    # Resolve only the directory: a file symlink still belongs to these siblings.
    entry = entry.parent.resolve() / entry.name
    if entry.suffix not in {".yaml", ".yml"} or not entry.is_file():
        raise schema.ToolError(f"{entry}: pass one existing .yaml or .yml file")
    return entry


def _unchanged(files, source_hashes, schema):
    directory = next(iter(files.values())).parent
    if files.keys() != source_hashes.keys():
        raise schema.ToolError(f"{directory}: task files changed after reading; rerun the command")
    current = {path.name for path in directory.iterdir() if path.suffix in {".yaml", ".yml"}}
    if current != {path.name for path in files.values()}:
        raise schema.ToolError(f"{directory}: task files changed after reading; rerun the command")
    contents = {}
    for name, path in files.items():
        contents[name] = path.read_bytes()
        if hashlib.sha256(contents[name]).hexdigest() != source_hashes[name]:
            raise schema.ToolError(f"{path}: source changed after reading; rerun the command")
    return contents


def _stage(path, content, schema):
    if path.is_symlink():
        raise schema.ToolError(f"{path}: normalization needs a regular file; the symlink was preserved")
    information = path.stat()
    if not stat.S_ISREG(information.st_mode):
        raise schema.ToolError(f"{path}: normalization needs a regular file")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".schema-normalize-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content.encode("utf-8") if isinstance(content, str) else content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(stat.S_IMODE(information.st_mode))
        return temporary, information
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_candidates(plan, schema):
    """Stage, replace and strictly reread; undo our writes if publication fails."""
    originals = _unchanged(plan.files, plan.source_hashes, schema)
    staged, written, hashes = {}, [], dict(plan.source_hashes)
    try:
        for name, content in plan.candidates.items():
            staged[name] = _stage(plan.files[name], content, schema)
        for name, (temporary, _) in staged.items():
            _unchanged(plan.files, hashes, schema)
            path = plan.files[name]
            if path.is_symlink():
                raise schema.ToolError(f"{path}: normalization needs a regular file; the symlink was preserved")
            temporary.replace(path)
            written.append(name)
            hashes[name] = plan.documents[name].source_sha256
        _unchanged(plan.files, hashes, schema)
        # Diagnostics and graph identities use actual source bytes/marks, and
        # publication is complete only after these still match the whole plan.
        files, documents = schema.read_directory(next(iter(plan.files.values())).parent)
        expected = {name: document.source_sha256 for name, document in plan.documents.items()}
        if {name: document.source_sha256 for name, document in documents.items()} != expected:
            raise schema.ToolError("task source changed while rereading normalized files; rerun the command")
        _unchanged(files, expected, schema)
        return documents
    except (OSError, schema.SchemaError, schema.ToolError) as error:
        unrestored = []
        for name in reversed(written):
            path = plan.files[name]
            restore = None
            try:
                # Never undo somebody else's newer edit during recovery.
                if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
                    unrestored.append(str(path))
                    continue
                restore, _ = _stage(path, originals[name], schema)
                if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
                    unrestored.append(str(path))
                    continue
                restore.replace(path)
                information = staged[name][1]
                os.utime(path, ns=(information.st_atime_ns, information.st_mtime_ns))
            except (OSError, schema.ToolError):
                unrestored.append(str(path))
            finally:
                if restore is not None:
                    restore.unlink(missing_ok=True)
        if unrestored:
            raise schema.ToolError(f"{error}\nNormalization was interrupted; restored {len(written) - len(unrestored)} "
                                   "file(s); these files could not be restored safely: "
                                   + ", ".join(unrestored) + "; review and rerun the complete task") from error
        if written:
            raise schema.ToolError(f"{error}\nNormalization was interrupted; restored all {len(written)} "
                                   "file(s) written by this command") from error
        raise
    finally:
        for temporary, _ in staged.values():
            temporary.unlink(missing_ok=True)


def _preview_message(message, files, candidates):
    for name in candidates:
        message = message.replace(str(files[name]), str(files[name]) + " (normalization preview; not written)")
    return message


def _report_corrections(plan, documents, *, read_only):
    categories = {"field": "field spelling", "identifier+reference": "declarations and references",
                  "quote": "YAML scalar quoting", "period": "sentence punctuation", "layout": "YAML layout"}
    grouped = {}
    for correction in plan.corrections:
        category, separator, detail = correction.message.partition(": ")
        if not separator or category not in categories:
            category, detail = "normalization", correction.message
        grouped.setdefault(category, {}).setdefault(correction.path, []).append(detail)
    for category, fields in grouped.items():
        print(f"warning: {'planned' if read_only else 'applied'} {categories.get(category, category)} "
              f"in {len(fields)} field(s)", file=sys.stderr)
        for field, details in fields.items():
            name = next((name for name in plan.files if field == name or field.startswith(name + ".")
                         or field.startswith(name + "[")), None)
            location = str(plan.files[name]) if name else field
            position = documents[name].positions.get(field) if name else None
            if position:
                location += f":{position[0]}:{position[1]}"
            if read_only:
                location += " (normalization preview; not written)"
            print(f"  {location} [{field}]: " + "; ".join(dict.fromkeys(details)), file=sys.stderr)
    if plan.candidates:
        print("Review the proposed fields and references." if read_only else
              "Review the normalized fields and references.", file=sys.stderr)


def _normalize(plan, read_only, schema):
    _unchanged(plan.files, plan.source_hashes, schema)
    if read_only:
        _report_corrections(plan, plan.documents, read_only=True)
        print(f"Normalization preview: {len(plan.files)} YAML; {len(plan.candidates)} would change; no files written")
        return plan.documents
    if plan.candidates:
        documents = _write_candidates(plan, schema)
        for name in plan.candidates:
            print(f"Normalized: {plan.files[name]}")
    else:
        documents = plan.documents
    _report_corrections(plan, documents, read_only=False)
    print(f"Normalization completed: {len(plan.files)} YAML; {len(plan.candidates)} changed; versions preserved")
    return documents


def _graph_diagnostic(node, error, resolved, preview):
    from tools.diagnostics import Diagnostic, format_diagnostics

    message = str(error)
    path, separator, detail = message.partition(": ")
    field = node + "." + path if separator and path.startswith("DESIGN.") else node
    rendered = format_diagnostics([Diagnostic(field, detail if field != node else message)],
                                  resolved.source_files, resolved.documents, resolved=resolved)
    return _preview_message(rendered, resolved.source_files, preview)


def _graphs(args, selected, resolved, config, output, preview):
    from tools.pipeline import ExpansionError, GraphError, analyze
    from tools.pipeline.render import RenderError

    completed, status = {}, 0
    if preview:
        print("Graph input: normalization preview; source files were not written")
    for node in selected:
        try:
            graphs = analyze(resolved.effective_task[node], config)
        except ExpansionError as error:
            print("Graph not completed:\n" + _graph_diagnostic(node, error, resolved, preview), file=sys.stderr)
            status = 2
            continue
        except GraphError as error:
            print("Graph counterexample:\n" + _graph_diagnostic(node, error, resolved, preview), file=sys.stderr)
            status = max(status, 1)
            continue
        completed[node] = graphs
        for graph in graphs:
            print(f"Graph checked: {node}/{graph.step}; {len(graph.operations)} actions, "
                  f"{len(graph.edges)} edges, {len(graph.boundaries)} boundary skips; "
                  + ("cycle found" if graph.cycle else "acyclic"))
            if graph.cycle:
                status = max(status, 1)
                operation = graph.operations[graph.cycle[0].rsplit("@", 1)[0]]
                error = GraphError(operation.path + ": event dependency cycle: " + " -> ".join(graph.cycle))
                print("Graph counterexample:\n" + _graph_diagnostic(node, error, resolved, preview), file=sys.stderr)
    print("Graph scope: given sample; one representative worker per step; "
          f"worktiles={config['worktiles']}; parameters=" + json.dumps(config["parameters"], sort_keys=True))
    if len(completed) != len(selected):
        if args.graph == "png":
            print("PNG not updated: requested analyses did not all complete; previous images retained", file=sys.stderr)
        return status
    if args.graph == "png":
        notes = ["Flow checked statically"]
        if args.skip_hardware:
            notes.append("Hardware specification and hardware-dependent checks were skipped explicitly")
        for node, graphs in completed.items():
            try:
                destination = output.write(resolved.source_files[node], graphs,
                                           {graph.step: config for graph in graphs}, notes,
                                           source_sha256=resolved.source_hashes[node],
                                           source_manifest=resolved.source_manifest(node),
                                           effective_identity=resolved.effective_identity(node),
                                           origins=resolved.origins_dict(node))
                print(f"PNG updated: {destination}")
            except RenderError as error:
                print(error, file=sys.stderr)
                status = 2
    return status


def _run(args, source, schema, notices):
    if args.show_effective:
        resolved = schema.resolve_directory(source.parent)
        node = source.stem
        view = {"format": "kernel_schema.effective_view.v1", "validation": "not_run", "node": node,
                "effective_yaml": schema.effective_yaml(resolved, node), "origins": resolved.origins_dict(node),
                "source_files": resolved.source_manifest(node), "effective_identity": resolved.effective_identity(node)}
        print(json.dumps(view, sort_keys=True, ensure_ascii=False))
        return 0
    if args.standalone:
        schema.validate_standalone_file(source, notices=notices, skip_hardware=args.skip_hardware)
        print(f"Standalone checked: {source}; structure, text budgets and relations; read-only")
        return 0

    config, output = None, None
    if args.graph:
        from tools.pipeline import ExpansionError, configuration
        try:
            config = configuration(args.parameters, args.worktiles)
        except ExpansionError as error:
            raise schema.ToolError(str(error)) from error
    plan = schema.normalize_directory(source.parent, skip_hardware=args.skip_hardware)
    files = plan.files
    selected = list(files) if args.all else [source.stem]
    if args.graph == "png":
        from tools.pipeline.output import GraphOutput
        from tools.pipeline.render import RenderError
        output = GraphOutput()
        try:
            output.preflight(files[name] for name in selected)
        except RenderError as error:
            raise schema.ToolError(str(error)) from error
    documents = _normalize(plan, args.read_only, schema)
    source_hashes = (plan.source_hashes if args.read_only else
                     {name: document.source_sha256 for name, document in documents.items()})
    _unchanged(files, source_hashes, schema)
    preview = plan.candidates if args.read_only else {}
    status, resolved, task_failed = int(bool(preview)), None, False
    snapshot_notices = []
    try:
        resolved = schema.validate_documents(files, documents, notices=snapshot_notices, skip_hardware=args.skip_hardware)
        schema.validate_local_repositories(files, documents, skip_hardware=args.skip_hardware)
        print(f"Schema checked: structure, text budgets, from and relations; {len(files)} YAML"
              + ("; normalization preview, not written" if preview else ""))
    except schema.SchemaError as error:
        print(_preview_message(str(error), files, preview), file=sys.stderr)
        status, task_failed = 1, True
    finally:
        notices.extend(_preview_message(notice, files, preview) for notice in snapshot_notices)
    try:
        schema.validate_dispatch(files)
        print(f"Dispatch checked: {source.parent / 'dispatch.md'}")
    except schema.SchemaError as error:
        print(error, file=sys.stderr)
        status, task_failed = 1, True
    if task_failed:
        if args.graph:
            print("Graph not run: task checks failed; PNG not updated", file=sys.stderr)
        return status
    # Unwritten normalization differences do not block finite candidate checks.
    if args.graph:
        _unchanged(files, source_hashes, schema)
        status = max(status, _graphs(args, selected, resolved, config, output, preview))
    return status


def main(argv=None):
    args = _arguments(argv)
    if sys.version_info < (3, 10):
        print("Check not completed: Python 3.10 or newer is required", file=sys.stderr)
        return 2
    try:
        from tools import schema
    except (OSError, ImportError) as error:
        print(f"Check not completed: cannot load checker files: {error}", file=sys.stderr)
        return 2
    notices = []
    try:
        return _run(args, _target(args.yaml, schema), schema, notices)
    except schema.SchemaError as error:
        print(error, file=sys.stderr)
        return 1
    except (schema.ToolError, OSError) as error:
        print(f"Check not completed: {error}", file=sys.stderr)
        if args.graph == "png":
            print("PNG not updated; previous images retained", file=sys.stderr)
        return 2
    finally:
        for notice in notices:
            print(notice, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
