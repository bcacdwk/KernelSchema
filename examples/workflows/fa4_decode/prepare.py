"""CPU-only fixed F1 materialization; does not import torch or launch jobs."""
from __future__ import annotations

import argparse
from pathlib import Path

from kernelschema.artifacts import Bundle, Publication, durable_write
from kernelschema.contracts import PROTOCOL, jcs
from kernelschema.evaluation.fa4 import EXAMPLE, freeze_case, freeze_request, stage_original
from kernelschema.resources import resource


def prepare(out: Path, *, checkout: Path):
    out = out.absolute()
    if out.exists():
        raise FileExistsError("preparation requires a new output directory")
    case = freeze_case(Publication(out), checkout=checkout)
    p = Publication(out)
    environment = Bundle(out).document(case["task"], "task")["environment"]
    method = p.put_bytes("fa4/fidelity-obligations.md", resource(EXAMPLE + "fidelity-obligations.md").read_bytes())
    rules = p.put_json("fa4/rules.json", {"protocol": PROTOCOL, "kind": "rules", "files": [method], "tools": []})
    profile = p.put_json("fa4/profile.json", {
        "protocol": PROTOCOL, "kind": "backend_profile", "name": "aka", "version": "fa4-preparation-only-v1",
        "runtime": "unqualified", "model": "unselected", "parameters": {}, "native_limits": {},
        "environment": environment,
        "capabilities": {"protocols": [], "operations": [], "modes": [], "source_formats": [], "empty_start": False,
                         "recovery": [], "token_enforcement": "unsupported", "extensions": []},
        "extensions": {"information": {"status": "preparation only; install/qualify FA4 AKA and K3 seams before a new request"}}})
    p.commit([rules, profile])
    request = freeze_request(Publication(out), case, rules=rules, profile=profile,
        limits={"calls": {"planner": 0, "coder": 0, "reviewer": 0, "extract": 0}, "evaluator_jobs": 0, "active_ms": 0, "tokens": None})
    staged = stage_original(Bundle(out), case, out / "stage")
    result = {"case": case, "request": request, "profile": profile, "rules": rules, "stage": staged,
              "K8_engineering": "prepared", "K8_real": "not_qualified", "model_calls": 0, "gpu_jobs": 0}
    durable_write(out / "prepared.json", jcs(result))
    durable_write(out / "request.json", Bundle(out).read(request))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkout", type=Path, default=resource("AGENTS.md").parent)
    args = parser.parse_args()
    result = prepare(args.out, checkout=args.checkout)
    print(jcs({"prepared": str(args.out.absolute() / "prepared.json"),
        "request": result["request"], "stage_sha256": result["stage"]["sha256"],
        "execution": "not_run", "qualification": "none"}).decode())


if __name__ == "__main__":
    main()
