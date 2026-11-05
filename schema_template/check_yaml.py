#!/usr/bin/env python3
"""Check a YAML file and every node in the same task, without changing plans."""

import argparse
import sys

sys.dont_write_bytecode = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml", help="root.yaml or a derived node in a complete task directory")
    parser.add_argument("--skip-hardware", action="store_true",
                        help="explicitly exclude hardware_specification and hardware-dependent checks")
    parser.add_argument("--standalone", action="store_true",
                        help="check one explicit delivery view containing its own Contract, without reading siblings")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        print("Check not completed: Python 3.10 or newer is required.", file=sys.stderr)
        return 2
    try:
        from tools import schema
    except (OSError, ImportError) as error:
        print(f"Check not completed: cannot load checker files: {error}", file=sys.stderr)
        return 2
    notices = []
    try:
        check = schema.validate_standalone_file if args.standalone else schema.validate_file
        directory, task = check(args.yaml, notices=notices, skip_hardware=args.skip_hardware)
    except schema.SchemaError as error:
        print(error, file=sys.stderr)
        return 1
    except (schema.ToolError, OSError, ValueError) as error:
        print(f"Check not completed: {error}", file=sys.stderr)
        return 2
    finally:
        for notice in notices:
            print(notice, file=sys.stderr)
    scope = "standalone view checked" if args.standalone else "whole task checked"
    print(f"Valid: {directory} ({len(task)} YAML node(s); {scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
