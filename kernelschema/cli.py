"""The sole workflow CLI; extract retains its original CLI and exit semantics."""
import argparse
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
        p.add_argument("--out")
    sub.add_parser("extract", help="pass original extract arguments after --")
    args = parser.parse_args(argv)
    if bool(args.request) != bool(args.out): parser.error("--request requires --out; --continue cannot use --out")
    try:
        result = getattr(api, args.operation)(request=args.request, out=args.out, continue_run=args.continue_run)
    except (ProtocolError, OSError) as error:
        print(f"kernelschema: {error}", file=sys.stderr)
        return 2
    print(f"{result['operation']}: {result['status']} ({result['stop_reason']})")
    return 0
