#!/usr/bin/env python3
"""Executable CLI substitute: edits files, resumes UUIDs and returns real hashes."""

import hashlib
import json
from pathlib import Path
import re
import sys
from uuid import uuid4

import yaml


if "--version" in sys.argv:
    print("codex-cli controlled-test")
    raise SystemExit(0)

base = Path(__file__).parent
scenario = json.loads((base / "scenario.json").read_text())
prompt = sys.stdin.read()
context = json.loads(re.search(r"<invocation>\n(.*?)\n</invocation>", prompt, re.S).group(1))
identity = context["identity"]
stage, role = identity["stage"], identity["role"]
log_path = base / "calls.jsonl"
previous = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
number = 1 + sum(call["stage"] == stage and call["role"] == role for call in previous)
action = scenario.get(f"{stage}.{role}.{number}", "normal")
resumed = sys.argv[sys.argv.index("resume") + 1] if "resume" in sys.argv else None
session_id = resumed or str(uuid4())
if action == "wrong_session":
    session_id = str(uuid4())
if action == "reuse_worker":
    session_id = next(call["session_id"] for call in previous if call["role"] == "worker")
print(json.dumps({"type": "thread.started", "thread_id": session_id}), flush=True)
record = {"stage": stage, "role": role, "session_id": session_id, "resumed": resumed,
           "argv": sys.argv[1:], "prompt": prompt, "identity": identity,
           "mode": identity.get("mode", "normal"), "node": identity["node"]}
with log_path.open("a") as stream:
    stream.write(json.dumps(record) + "\n")

workspace = Path.cwd()
def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace.iterdir() if p.is_file()}
before = hashes()
root = yaml.safe_load((workspace / "root.yaml").read_text())
if role == "worker":
    if stage == "P1":
        existing = root["meta_data"]["name"] is not None
        root["meta_data"].update({"name": "Increment", "describe": "Increment each input element.",
                                  "source": context["implementation_source"], "version": 2 if existing else 1})
        root["contract"]["algorithm_requirement"] = {
            "input": {"X": {"meaning": "input", "shape": [{"element": "N"}], "dtype": "float32"}},
            "output": {"Y": {"meaning": "output", "shape": [{"element": "N"}], "dtype": "float32"}},
            "compute": [{"name": "increment", "math": "Y[n] = X[n] + 1"}],
            "golden_reference": {"source": context["reference_source"], "tolerance": {"Y": "Absolute error <= 0.001."}}}
    elif stage == "P2":
        root["meta_data"]["version"] += 1
        root["meta_data"]["dispatch"] = {"when": "Select for every input in the contract domain."}
        (workspace / "dispatch.md").write_text("# 路径草案\n\n[root](root.yaml) 覆盖当前 workload。\n\n其他路径尚无实质设计差异。\n")
    elif stage == "P3":
        root["design"] = {
            "math_reshape": {"thesis": "Compute each increment directly without intermediates.",
                "rewrite": [{"step": "increment", "name": "increment", "math": "Y follows Contract compute increment."}]},
            "work_schedule": {"thesis": "One worker produces the complete vector.",
                "flow": {"launch": [{"phase": ["increment"]}]},
                "phase": {"increment": {"describe": "Produce all Y values.",
                    "decomposition": {"partition": {}, "target": {"single_tile": {"Y": {"shape": [{"element": "N"}], "dtype": "float32"}}}},
                    "mapping": {"worker": {"unit": "CTA", "unit_count": 1}, "worker_count": 1},
                    "schedule": {"worker_assignment": {"preassigned": {"start": 0, "count": 1}, "refill": "none", "exit": "assigned_done"}}}}},
            "tile_pipeline": {"thesis": "Synchronous production needs no cross-role handoff.",
                "phase": {"increment": {"role": {"Main": {"info": {"execution_mode": "serial", "member": "warp[0:1]"},
                    "pipe": {"steady": ["produce:Y"]}}}}}}}
        root["meta_data"]["version"] += 1
    elif stage == "P4":
        (workspace / "dispatch.md").write_text("# 路径索引\n\n[root](root.yaml) 完成全部向量元素的独立加一计算。\n")
    if action == "protected_contract":
        root["contract"]["algorithm_requirement"]["compute"][0]["math"] = "Y = 42"
    if action == "design_write":
        root["design"]["math_reshape"]["thesis"] = "This is outside P1."
    (workspace / "root.yaml").write_text(yaml.safe_dump(root, sort_keys=False, allow_unicode=True))
    if action == "extra_file":
        (workspace / "unexpected.yaml").write_text("invalid: true\n")
elif action == "reviewer_write":
    (workspace / "root.yaml").write_text((workspace / "root.yaml").read_text() + "# reviewer changed it\n")
if action == "state_write":
    state_path = workspace.parents[2] / "state.json"
    state_path.write_text('{"format":"forged"}')
if action == "source_write":
    target = Path(context["source_root"]) / context["inputs"]["entrypoint"]["path"]
    target.chmod(0o644)
    target.write_text("def kernel(x): return 42\n")

after = hashes()
def anchor(path, relative):
    lines = path.read_text().splitlines()
    line = next(i for i, value in enumerate(lines) if value.strip())
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "line_start": line + 1, "line_end": line + 1, "quote": lines[line][:200], "note": "Supports the recorded conclusion."}
categories = context["required_evidence"]
source_evidence = []
for category in categories:
    binding = context["inputs"]["entrypoint" if category == "implementation" else category]
    locator = binding[0] if isinstance(binding, list) else binding
    source_evidence.append({"category": category, "source_id": context["source_id"],
                            **anchor(Path(context["source_root"]) / locator["path"], locator["path"])})
decision = ("continue" if action == "continue" else "complete") if role == "worker" else "approve"
if action == "role_forge":
    decision = "approve"
receipt = {"format": "schema_extract.receipt.v2", "identity": identity, "decision": decision,
           "summary": f"Controlled {stage} {role} result.",
           "changed_files": sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name)),
           "artifacts": [{"path": name, "sha256": value} for name, value in sorted(after.items())],
           "source_evidence": source_evidence,
           "artifact_evidence": [anchor(workspace / name, name) for name in sorted(after)] if role == "reviewer" else [],
           "conclusions": [{"subject": name, "text": "The fixed evidence establishes this finding.", "evidence": [0]}
                           for name in context["required_conclusions"]],
           "checks": [], "checks_note": "P1/P2 allow incomplete Design; full checker not run.",
           "issues": [], "framework_candidates": [],
           "handoff": "已批准当前阶段。\nroot 覆盖固定调用范围。\n从 host 入口继续核对设计与路径。" if stage in {"P2", "P3", "P4"} and role == "reviewer" else ""}
if stage == "P4":
    receipt["plan"] = [{"id": "root", "file": "root.yaml", "parent": "", "workload": "All N-element inputs.",
                        "difference": "Direct increment of every vector element.", "selection": "The fixed kernel is unconditional.", "evidence": [0]}]
if stage == "P6":
    receipt["revision_scope"], receipt["affected_nodes"] = [], []
if stage != "P1" and identity.get("mode", "normal") == "normal":
    receipt["reopen"] = None
if action == "bad_quote":
    receipt["source_evidence"][0]["quote"] = "THIS QUOTE DOES NOT EXIST IN THE SOURCE"
if action == "bad_artifact_sha":
    receipt["artifacts"][0]["sha256"] = "0" * 64
if action == "bad_fingerprint":
    receipt["identity"]["input_fingerprint"] = "0" * 64
if action == "replay":
    receipt = json.loads((base / "first_receipt.json").read_text())
if not (base / "first_receipt.json").exists():
    (base / "first_receipt.json").write_text(json.dumps(receipt))
Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text(json.dumps(receipt))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}), flush=True)
