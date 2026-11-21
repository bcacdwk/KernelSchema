"""The sole workflow CLI; extract retains its original CLI and exit semantics."""
import argparse
from pathlib import Path
import sys

from . import api
from .contracts import ProtocolError


def _native_arguments(parser, *, prepare=False):
    parser.add_argument("--problem", help="native Atrex-Bench problem directory")
    parser.add_argument("--repo-root", help="AKA checkout root")
    parser.add_argument("--platform", help="explicit AKA hardware alias, for example L20D")
    parser.add_argument("--framework", default="Triton")
    parser.add_argument("--agent-cli", default="claude")
    parser.add_argument("--max-iters", type=int, default=4, help="native coder iterations per Schema version")
    if prepare:
        parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--owner-account", help="optional existing owner authority path")
    parser.add_argument("--owner-limits", help="JSON file with finite model/evaluator slots for the optional owner adapter")
    parser.add_argument("--doc", action="append", default=[], help="additional schema rationale/document file")
    parser.add_argument("--profile", default="light", help="native AKA composition profile")
    parser.add_argument("--patch", action="append", default=[], help="native AKA composition patch file")
    parser.add_argument("--settings", help="JSON file containing native WorkflowSettings fields")
    parser.add_argument("--runtime", help="JSON file containing native CampaignRuntimeSettings fields")


def _native_options(args):
    from .contracts import loads
    def document(filename):
        return loads(Path(filename).read_bytes()) if filename else None
    return dict(problem=args.problem, repo_root=args.repo_root, platform=args.platform,
        framework=args.framework, agent_cli=args.agent_cli, max_iters=args.max_iters,
        owner_account=args.owner_account, owner_limits=document(args.owner_limits), docs=args.doc,
        profile=args.profile, patches=args.patch, settings=document(args.settings), runtime=document(args.runtime))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "extract":
        from .extract_bridge import extract
        args = argv[1:]
        if args[:1] == ["--"]: args = args[1:]
        return extract(args)
    parser = argparse.ArgumentParser(prog="kernelschema")
    sub = parser.add_subparsers(dest="operation", required=True)
    for operation in ("design", "optimize", "implement"):
        p = sub.add_parser(operation)
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--request")
        group.add_argument("--continue", dest="continue_run")
        if operation in ("design", "optimize"):
            group.add_argument("--task", help="UTF-8 task description; writes Schema with one planner call")
            p.add_argument("--runtime-profile", help="explicit ClaudeProfile JSON; default inspects the existing Claude CLI")
            p.add_argument("--reference", action="append", default=[], help="related Schema, document or wiki file; repeatable")
            p.add_argument("--objective", help="optional current objective file")
            p.add_argument("--environment", help="optional target description file")
            if operation == "optimize":
                p.add_argument("--schema", help="current Schema directory/root.yaml")
                p.add_argument("--feedback", action="append", default=[], help="actual feedback file; repeatable")
        if operation == "implement":
            group.add_argument("--schema", help="Schema file/directory for AKA standard implementation")
            _native_arguments(p, prepare=True)
        p.add_argument("--out")
    p = sub.add_parser("iterate", help="finite Schema versions using write-only design/optimize and native AKA")
    p.add_argument("--task", required=True, help="explicit UTF-8 task-description file")
    p.add_argument("--schema", help="existing Schema; omit to begin with design")
    p.add_argument("--rounds", type=int, default=2, help="maximum Schema versions/native campaigns (default: 2)")
    p.add_argument("--runtime-profile", help="writing ClaudeProfile JSON; defaults to the existing Claude CLI")
    p.add_argument("--reference", action="append", default=[], help="selected writing reference file; repeatable")
    p.add_argument("--objective", help="optional writing objective file")
    p.add_argument("--environment", help="optional writing target description file")
    p.add_argument("--out", required=True, help="new output directory; existing directories are rejected")
    _native_arguments(p)
    sub.add_parser("extract", help="pass original extract arguments after --")
    args = parser.parse_args(argv)
    iteration = args.operation == "iterate"
    standard = args.operation == "implement" and args.schema is not None
    direct_writing = args.operation in ("design", "optimize") and args.task is not None
    if standard or iteration:
        if not args.out or not args.problem or not args.repo_root or not args.platform:
            parser.error("standard implementation requires --problem, --repo-root, --platform and --out")
    elif direct_writing:
        if not args.out: parser.error("--task requires --out")
        if args.operation == "optimize" and not args.schema: parser.error("optimize --task requires --schema")
    elif bool(args.request) != bool(args.out): parser.error("--request requires --out; --continue cannot use --out")
    try:
        if iteration:
            result = api.iterate(task=args.task, schema=args.schema, out=args.out, rounds=args.rounds,
                runtime_profile=args.runtime_profile, references=args.reference, objective=args.objective,
                environment=args.environment, **_native_options(args))
        elif standard:
            result = api.implement(schema=args.schema, out=args.out, prepare_only=args.prepare_only,
                                   **_native_options(args))
        elif direct_writing:
            options = {"schema": args.schema, "feedback": args.feedback} if args.operation == "optimize" else {}
            result = getattr(api, args.operation)(task=args.task, out=args.out, references=args.reference,
                runtime_profile=args.runtime_profile, objective=args.objective, environment=args.environment, **options)
        else:
            result = getattr(api, args.operation)(request=args.request, out=args.out, continue_run=args.continue_run)
    except (ProtocolError, ValueError, OSError) as error:
        print(f"kernelschema: {error}", file=sys.stderr)
        return 2
    print(f"{result.get('operation', args.operation)}: {result['status']} ({result['stop_reason']})")
    directory = result.get("extensions", {}).get("information", {}).get("schema_directory")
    if directory:
        print("Schema: " + str(Path(args.out or args.continue_run).absolute() / directory))
    if iteration:
        print("Stages: " + str(Path(args.out).absolute() / "iterations.json"))
        return 0 if result["status"] == "completed" else 1
    return 0
