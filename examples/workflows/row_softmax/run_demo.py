#!/usr/bin/env python3
"""Prepare, run or recover the finite standalone row-softmax workflow."""
import argparse
from pathlib import Path
import sys

# Source-checkout entry point; API imports otherwise use the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.workflows.row_softmax.driver import Demo, load_runtime
from kernelschema.contracts import ProtocolError, jcs, loads
from kernelschema.evaluation.environment import target_environment
from kernelschema.artifacts import durable_write
from kernelschema.contracts import sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    target = sub.add_parser("target", help="freeze explicit target declarations and original documentation; no qualification")
    target.add_argument("--spec", required=True, help="JSON hardware_specification/runtime/sources; each source has locator/file/license")
    target.add_argument("--out", required=True)
    create = sub.add_parser("prepare", help="freeze five concrete requests; no model/GPU invocation")
    create.add_argument("--out", required=True)
    create.add_argument("--start", choices=("design", "fixed", "partial", "verify"), default="design")
    create.add_argument("--runtime-profile")
    create.add_argument("--environment", help="explicit ks.row-softmax.target.v1 JSON")
    create.add_argument("--objective", help="current objective text; Task hard criteria stay fixed")
    create.add_argument("--schema", help="complete existing raw directory; default is handwritten R0")
    create.add_argument("--receipt", help="optional original extract delivery receipt, consumed without running extract")
    create.add_argument("--remote", help="JSON channel/gpu/python/bwrap; existing tools only")
    for name in ("run", "continue", "inspect", "qualify", "export"):
        command = sub.add_parser(name)
        command.add_argument("--demo", required=True)
        if name == "export": command.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "target":
            spec_path = Path(args.spec).absolute()
            spec = loads(spec_path.read_bytes())
            if set(spec) != {"hardware_specification", "runtime", "sources"}:
                raise ProtocolError("target spec needs hardware_specification/runtime/sources")
            sources = []
            for source in spec["sources"]:
                if set(source) != {"locator", "file", "license"}: raise ProtocolError("source needs locator/file/license")
                data = (spec_path.parent / source["file"]).read_bytes()
                sources.append({"locator": source["locator"], "text": data.decode("utf-8"), "sha256": sha256(data), "license": source["license"]})
            value = target_environment(hardware_specification=spec["hardware_specification"], runtime=spec["runtime"], sources=sources)
            durable_write(Path(args.out).absolute(), jcs(value))
            value = {"target": str(Path(args.out).absolute()), "qualification": "none; explicit target declaration only"}
        elif args.command == "prepare":
            demo = Demo.create(args.out, start=args.start, runtime=load_runtime(args.runtime_profile),
                environment=Path(args.environment).read_bytes() if args.environment else None,
                objective=Path(args.objective).read_bytes() if args.objective else None,
                schema_directory=args.schema, receipt=args.receipt,
                remote=loads(Path(args.remote).read_bytes()) if args.remote else None)
            value = demo.summary()
        else:
            demo = Demo(args.demo)
            value = demo.run() if args.command in ("run", "continue") else demo.prepare_agate() if args.command == "qualify" else {"export": demo.export(args.out)} if args.command == "export" else demo.summary()
        print(jcs(value).decode())
        return 0
    except (ProtocolError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
