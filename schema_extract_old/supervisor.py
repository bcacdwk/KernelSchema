"""Inspect or schedule a read-only supervisor for one extraction run."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema_extract.tools.records import StateError, Store
from schema_extract.tools.scheduling import schedule
from schema_extract.tools.supervisor import status, tick


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--codex", default="codex", help="resolved to an absolute executable when enabling or calling Codex")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("tick", "enable", "disable", "status"):
        command = sub.add_parser(name)
        command.add_argument("run_id")
        if name == "tick":
            command.add_argument("--new-session", action="store_true", help="explicitly use a fresh supervisor thread")
    args = parser.parse_args(argv)
    store = Store(args.repo, args.state_root or args.repo / ".schema_extract")
    try:
        if args.command == "tick":
            result = tick(store, args.run_id, args.codex, new_session=args.new_session)
        elif args.command == "status":
            result = status(store, args.run_id)
        else:
            result = schedule(store, args.run_id, args.codex, enable=args.command == "enable")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("outcome") in {"error", "model_error", "unresolved_invocation"} else 0
    except (StateError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"schema_extract.supervisor: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
