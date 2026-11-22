"""CPU-only fixed F1 materialization; does not import torch or launch jobs."""
from __future__ import annotations

import argparse
from pathlib import Path

from kernelschema.artifacts import Bundle, Publication, durable_write, import_bundle
from kernelschema.contracts import PROTOCOL, jcs, loads, ProtocolError
from kernelschema.evaluation.fa4 import EXAMPLE, freeze_case, freeze_request, stage_original
from kernelschema.resources import resource
from kernelschema import fidelity
from kernelschema.runtime import ClaudeProfile
from kernelschema.implement.reference import ReferenceBackend


def prepare(out: Path, *, checkout: Path, claude=None, evaluation=None, limits=None,
            intent="original_verify", seed=None, seed_bundle=None):
    out = out.absolute()
    if out.exists():
        raise FileExistsError("preparation requires a new output directory")
    case = freeze_case(Publication(out), checkout=checkout)
    if seed is not None:
        if seed_bundle is None:
            raise ProtocolError("an explicit seed needs its source bundle")
        import_bundle(Bundle(seed_bundle), out, [seed])
    p = Publication(out)
    environment = Bundle(out).document(case["task"], "task")["environment"]
    method = p.put_bytes("fa4/fidelity-obligations.md", resource(EXAMPLE + "fidelity-obligations.md").read_bytes())
    finite = fidelity.freeze_method(p)
    rules = p.put_json("fa4/rules.json", {"protocol": PROTOCOL, "kind": "rules", "files": sorted([method, finite["file"]], key=lambda r: r["path"]), "tools": [finite["tool"]]})
    profile_doc = {
        "protocol": PROTOCOL, "kind": "backend_profile", "name": "aka", "version": "fa4-preparation-only-v1",
        "runtime": "unqualified", "model": "unselected", "parameters": {}, "native_limits": {},
        "environment": environment,
        "capabilities": {"protocols": [], "operations": [], "modes": [], "source_formats": [], "empty_start": False,
                         "recovery": [], "token_enforcement": "unsupported", "extensions": []},
        "extensions": {"information": {"status": "preparation only; select explicit reference runtime/evaluation profile for execution"}}}
    if claude is not None:
        settings = {**claude, "reported_models": tuple(claude["reported_models"]), "builtin_plugins": tuple(claude.get("builtin_plugins", []))}
        profile_doc = ClaudeProfile(**settings).backend_profile(environment, name="reference")
        profile_doc["capabilities"] = ReferenceBackend().capabilities(profile_doc)
        if evaluation is not None:
            profile_doc["parameters"]["evaluation"] = evaluation
    elif evaluation is not None:
        raise ProtocolError("evaluation selection needs an explicit Claude profile")
    profile = p.put_json("fa4/profile.json", profile_doc)
    p.commit([rules, profile])
    request = freeze_request(Publication(out), case, rules=rules, profile=profile,
        limits=limits or {"calls": {"planner": 0, "coder": 0, "reviewer": 0, "extract": 0}, "evaluator_jobs": 0, "active_ms": 0, "tokens": None},
        intent=intent, source=seed)
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
    parser.add_argument("--claude-profile", type=Path, help="explicit ClaudeProfile.as_dict JSON; no probe or model call")
    parser.add_argument("--evaluation", type=Path, help="explicit fa4_channel.selection JSON")
    parser.add_argument("--limits", type=Path, help="ordinary workflow limits JSON")
    parser.add_argument("--intent", choices=("original_verify", "seed_continuation", "schema_rebuild"), default="original_verify")
    parser.add_argument("--seed-bundle", type=Path)
    parser.add_argument("--seed-ref", type=Path, help="JSON Source Ref, read only within --seed-bundle")
    args = parser.parse_args()
    read = lambda path: loads(path.read_bytes()) if path else None
    result = prepare(args.out, checkout=args.checkout, claude=read(args.claude_profile), evaluation=read(args.evaluation),
                     limits=read(args.limits), intent=args.intent, seed=read(args.seed_ref), seed_bundle=args.seed_bundle)
    print(jcs({"prepared": str(args.out.absolute() / "prepared.json"),
        "request": result["request"], "stage_sha256": result["stage"]["sha256"],
        "execution": "not_run", "qualification": "none"}).decode())


if __name__ == "__main__":
    main()
