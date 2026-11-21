"""The sole workflow CLI; extract retains its original CLI and exit semantics."""
import argparse
from pathlib import Path
import sys

from . import api
from .contracts import ProtocolError


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
            p.add_argument("--problem", help="native Atrex-Bench problem directory")
            p.add_argument("--repo-root", help="AKA checkout root")
            p.add_argument("--platform", help="explicit AKA hardware alias, for example L20D")
            p.add_argument("--framework", default="Triton")
            p.add_argument("--agent-cli", default="claude")
            p.add_argument("--max-iters", type=int, default=4)
            p.add_argument("--prepare-only", action="store_true")
            p.add_argument("--owner-account", help="existing owner authority path")
            p.add_argument("--owner-limits", help="JSON finite model/evaluator slots for the optional owner adapter")
            p.add_argument("--doc", action="append", default=[], help="additional schema rationale/document file")
        p.add_argument("--out")
    sub.add_parser("extract", help="pass original extract arguments after --")
    args = parser.parse_args(argv)
    standard = args.operation == "implement" and args.schema is not None
    direct_writing = args.operation in ("design", "optimize") and args.task is not None
    if standard:
        if not args.out or not args.problem or not args.repo_root or not args.platform:
            parser.error("--schema requires --problem, --repo-root, --platform and --out")
    elif direct_writing:
        if not args.out: parser.error("--task requires --out")
        if args.operation == "optimize" and not args.schema: parser.error("optimize --task requires --schema")
    elif bool(args.request) != bool(args.out): parser.error("--request requires --out; --continue cannot use --out")
    try:
        if standard:
            from .contracts import loads
            result = api.implement(schema=args.schema, problem=args.problem, repo_root=args.repo_root, out=args.out,
                platform=args.platform, framework=args.framework, agent_cli=args.agent_cli, max_iters=args.max_iters,
                prepare_only=args.prepare_only, owner_account=args.owner_account,
                owner_limits=loads(Path(args.owner_limits).read_bytes()) if args.owner_limits else None, docs=args.doc)
        elif direct_writing:
            options = {"schema": args.schema, "feedback": args.feedback} if args.operation == "optimize" else {}
            result = getattr(api, args.operation)(task=args.task, out=args.out, references=args.reference,
                runtime_profile=args.runtime_profile, objective=args.objective, environment=args.environment, **options)
        else:
            result = getattr(api, args.operation)(request=args.request, out=args.out, continue_run=args.continue_run)
    except (ProtocolError, OSError) as error:
        print(f"kernelschema: {error}", file=sys.stderr)
        return 2
    print(f"{result.get('operation', args.operation)}: {result['status']} ({result['stop_reason']})")
    directory = result.get("extensions", {}).get("information", {}).get("schema_directory")
    if directory:
        print("Schema: " + str(Path(args.out or args.continue_run).absolute() / directory))
    return 0
