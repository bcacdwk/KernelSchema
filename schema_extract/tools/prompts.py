"""Frozen Markdown instructions and exact materials for one invocation."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys

from .artifacts import _same, design_context, load_root, load_yaml
from .budget import call_limit
from .common import ExtractError, digest, manifest, read_json, regular_file, rules_root, seal, sha256, source_root, write_json
from .receipts import REOPEN_TARGETS, call_identity, required_artifact_evidence
from .snapshots import frozen_reader
from .stages import STAGES, effective_nodes, owner_for, read_specs, required_conclusions, scope_for


PROMPT_ROOT = "schema_extract/prompt"
HELPER_FILES = ("__init__.py", "common.py", "inputs.py", "artifacts.py", "reading.py")


def operation_text(name: str, *, rules_lock: dict | None = None) -> str:
    if not re.fullmatch(r"[a-z][a-z_]*", name):
        raise ExtractError("invalid prompt operation name")
    return regular_file(rules_root(rules_lock), f"{PROMPT_ROOT}/operations/{name}.md").read_text(encoding="utf-8")


def render_operation(name: str, context: dict | None = None, *, rules_lock: dict | None = None) -> str:
    text = operation_text(name, rules_lock=rules_lock).rstrip()
    if context is not None:
        text += "\n\n<operation-context>\n" + json.dumps(context, ensure_ascii=False, indent=2) + "\n</operation-context>"
    return text + "\n"


def _documents(attempt: dict) -> list[str]:
    mode = attempt.get("mode", "normal")
    operation = "maintenance" if attempt.get("purpose") == "maintenance" else (
        "normal_" + attempt["role"] if mode == "normal" else mode)
    result = [f"{PROMPT_ROOT}/common.md", f"{PROMPT_ROOT}/roles/{attempt['role']}.md",
              f"{PROMPT_ROOT}/operations/response.md", f"{PROMPT_ROOT}/operations/check.md",
              f"{PROMPT_ROOT}/operations/{operation}.md", f"{PROMPT_ROOT}/phases/{STAGES[attempt['stage']]['page']}"]
    if attempt["stage"] in REOPEN_TARGETS and mode == "normal" and attempt.get("purpose") != "maintenance":
        result.append(f"{PROMPT_ROOT}/operations/reopen.md")
    if attempt.get("recovery_handoff"):
        result.append(f"{PROMPT_ROOT}/operations/recovery.md")
    return result


def _read_specs(attempt: dict) -> list[str]:
    return list(dict.fromkeys([*read_specs(attempt), f"{PROMPT_ROOT}/roles/reviewer.md"]))


def _linked_rules(lock: dict, names) -> set[str]:
    """Keep local references of selected frozen reading documents available."""
    available, root = lock["rules"]["files"], rules_root(lock)
    pending, selected = list(names), set()
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        if name not in available:
            raise ExtractError("required reading is not frozen: " + name)
        selected.add(name)
        if not name.endswith(".md"):
            continue
        for target in re.findall(r"\]\(([^)]+)\)", (root / name).read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or ":" in target or target.startswith("/"):
                continue
            try:
                relative = ((root / name).parent / target).resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            if relative in available and (relative.startswith(("schema_docs/", PROMPT_ROOT + "/")) or
                                          relative in {"EXTRACT_SCHEMA.md", "schema_template/README.md", "schema_template/starter.yaml"}):
                pending.append(relative)
    return selected


def _first_draft(state: dict, attempt: dict, rules: Path) -> bool:
    if attempt["stage"] == "P5":
        return (not state["nodes"][attempt["node"]].get("authored", False) and
                load_yaml(Path(attempt["workspace"]), attempt["node"] + ".yaml", parser_root=rules)["META_DATA"]["Version"] == 1)
    if attempt["stage"] == "P1":
        return _same(load_root(Path(attempt["workspace"]), parser_root=rules),
                     load_root(rules / "schema_template", "starter.yaml", parser_root=rules))
    return False


def prepare_attempt_materials(state: dict, lock: dict, attempt: dict, *, rules_lock: dict | None = None,
                              executor_lock: dict | None = None) -> dict:
    """Copy evidence and baselines; expose no run directory or authoritative state."""
    semantic, executor = rules_lock or lock, executor_lock or rules_lock or lock
    directory, workspace = Path(attempt["directory"]), Path(attempt["workspace"])
    run_dir, destination = directory.parent.parent, directory / "materials"
    destination.mkdir()
    files, records, candidates = {}, [], []

    def copy_file(source: Path, name: str, *, expected=None) -> str:
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ExtractError("invalid attempt material path")
        data = regular_file(source.parent, source.name).read_bytes()
        checksum = sha256(data)
        if expected is not None and checksum != expected:
            raise ExtractError("material differs from its recorded source: " + str(source))
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[name] = {"source": str(source), "sha256": checksum, "path": str(target)}
        return str(target)

    needed_stages = {"P1": [], "P2": ["P1"], "P3": ["P2"], "P4": ["P2", "P3"],
                     "P5": ["P2", "P4"], "P6": ["P2", "P4"]}[attempt["stage"]]
    approvals = [(stage, "root", state["stages"][stage].get("approval")) for stage in needed_stages]
    if attempt["stage"] == "P6":
        approvals.extend(("P5", node, entry.get("approval")) for node, entry in state["nodes"].items() if node != "root")
    seen_receipts = set()

    def receipt_record(attempt_id: str, reason: str, *, expected_digest=None):
        if attempt_id in seen_receipts:
            return
        path = run_dir / "attempts" / attempt_id / "receipt.json"
        value = read_json(path)
        expected_digest = expected_digest or state.get("accepted_receipts", {}).get(attempt_id)
        if expected_digest is None or digest(value) != expected_digest:
            raise ExtractError("material receipt differs from its accepted identity: " + attempt_id)
        identity = value["identity"]
        invocation = read_json(path.parent / "attempt.json")
        if (identity != call_identity(invocation) or identity["run_id"] != state["run_id"] or
                identity["input_fingerprint"] != state["input_fingerprint"]):
            raise ExtractError("material receipt differs from this run's fixed source invocation")
        selected = value.get("plan") if identity["stage"] == "P2" else invocation["selected_designs"]
        instances = [entry for entry in selected or []
                     if identity["stage"] in {"P2", "P4"} or entry["id"] == identity["node"]]
        current = {entry["id"]: entry["instance"] for entry in state.get("selected_designs") or []}
        relation = {entry["id"]: ("removed" if entry["id"] not in current else
                                   "current_instance" if entry["instance"] == current[entry["id"]] else "changed_instance")
                    for entry in instances}
        copied = copy_file(path, "records/" + attempt_id + ".json")
        records.append({"kind": "receipt", "stage": identity["stage"], "node": identity["node"],
                        "role": identity["role"], "attempt_id": attempt_id, "path": copied,
                        "receipt_sha256": expected_digest, "source_id": lock["source"]["id"],
                        "generation": identity.get("generation"), "instances": instances, "instance_relation": relation,
                        "purpose": reason, "read_after": "current_artifacts"})
        seen_receipts.add(attempt_id)

    for stage, node, approval in approvals:
        if not approval:
            continue
        receipt_record(approval["attempt_id"], "current_decision", expected_digest=approval["receipt_sha256"])
        path = run_dir / "stages" / stage
        if stage == "P5":
            path /= node
        path /= "handoff.md"
        if path.is_file():
            copied = copy_file(path, f"handoffs/{stage}/{node}.md")
            records.append({"kind": "handoff", "stage": stage, "node": node, "path": copied,
                            "purpose": "knowledge_destination", "read_after": "current_artifacts"})
    latest_scope = False
    for attempt_id in reversed(state["attempts"]):
        if attempt_id not in state["accepted_receipts"]:
            continue
        value = read_json(run_dir / "attempts" / attempt_id / "receipt.json")
        identity = value["identity"]
        same_scope = (identity["stage"], identity["node"]) == (attempt["stage"], attempt["node"])
        if same_scope and not latest_scope:
            receipt_record(attempt_id, "current_feedback")
            latest_scope = True
        if same_scope and value.get("framework_candidates"):
            receipt_record(attempt_id, "framework_candidates")
        transfer = identity["stage"] in needed_stages or (
            attempt["stage"] == "P6" and identity["stage"] in {"P2", "P3", "P4", "P5"})
        if (same_scope or transfer) and (value.get("handoff", "").strip() or value.get("issues")):
            receipt_record(attempt_id, "knowledge_destination")

    scope = scope_for(state, attempt)
    if attempt.get("mode") == "adjudicate":
        for entry in attempt["adjudication_candidates"]:
            checkpoint = entry["checkpoint"]
            source = run_dir / "checkpoints" / checkpoint / "artifacts"
            for name, checksum in manifest(source).items():
                copy_file(source / name, f"candidates/{checkpoint}/{name}", expected=checksum)
            candidates.append({**entry, "artifacts": str(destination / "candidates" / checkpoint)})
    if attempt["role"] == "worker":
        for name, checksum in attempt["dependency_digests"].items():
            copy_file(workspace / name, "baseline/" + name, expected=checksum)
    historical = attempt.get("baseline_context_checkpoint")
    if historical:
        source = run_dir / "checkpoints" / historical / "artifacts"
        for name, checksum in manifest(source).items():
            copy_file(source / name, "historical/" + name, expected=checksum)
    selected = state.get("plan_draft") if attempt["stage"] == "P2" and attempt["role"] == "reviewer" else state.get("selected_designs")
    node_entry = next((entry for entry in state.get("selected_designs") or [] if entry["id"] == attempt["node"]), None)
    check = {"format": "schema_extract.stage_check.v1", "attempt_id": attempt["attempt_id"],
             "stage": attempt["stage"], "node": attempt["node"], "role": attempt["role"],
             "workspace": str(workspace), "rules_root": str(rules_root(semantic)), "baseline": str(destination / "baseline"),
             "baseline_materials": str(destination / "historical") if historical else None,
             "source_root": str(source_root(lock)), "source": {key: lock["source"][key] for key in
                 ("id", "repository_id", "ref", "files", "bindings")},
             "selected_designs": selected, "node_entry": node_entry if attempt["stage"] == "P5" else None,
             "artifact_files": sorted(attempt["dependency_digests"]),
             "revision_scope": attempt.get("revision_scope", state.get("revision_scope", [])),
             "first_draft": _first_draft(state, attempt, rules_root(semantic)),
             "checked_nodes": sorted(effective_nodes(state)), "maintenance_fields": None}
    if attempt.get("purpose") == "maintenance":
        check["maintenance_fields"] = owner_for(state, attempt).get("fields", [])
    write_json(destination / "check.json", check)
    files["check.json"] = {"source": "registered_attempt", "sha256": sha256((destination / "check.json").read_bytes()),
                           "path": str(destination / "check.json")}
    selected_rules = _linked_rules(semantic, [*_documents(attempt), *_read_specs(attempt), "EXTRACT_SCHEMA.md", "schema_template/starter.yaml",
                                            f"{PROMPT_ROOT}/operations/recovery.md"])
    selected_rules.update(name for name in semantic["rules"]["files"] if name.startswith("schema_template/tools/") and name.endswith(".py"))
    selected_rules.add("schema_template/check_yaml.py")
    frozen_reader(executor)
    rule_materials = []
    for name in sorted(selected_rules):
        path = regular_file(rules_root(semantic), name)
        rule_materials.append({"path": str(path), "sha256": semantic["rules"]["files"][name]})
    for name in HELPER_FILES:
        relative = "schema_extract/tools/" + name
        path = regular_file(rules_root(executor), relative)
        rule_materials.append({"path": str(path), "sha256": executor["rules"]["files"][relative]})
    material = {"format": "schema_extract.attempt_materials.v1", "directory": str(destination),
                "source_root": str(source_root(lock)), "rules_root": str(rules_root(semantic)), "executor_root": str(rules_root(executor)),
                "source": {"id": lock["source"]["id"], "files": lock["source"]["files"]},
                "rules": rule_materials, "files": files, "records": records, "candidates": candidates,
                "check": str(destination / "check.json"), "first_draft": check["first_draft"]}
    command = destination / "check-helper.py"
    command.write_text("#!" + sys.executable + "\nimport sys\nif len(sys.argv) != 1: raise SystemExit(125)\n" +
        "sys.path.insert(0, " + repr(str(rules_root(executor))) + ")\n" +
        "from schema_extract.tools.reading import main\nraise SystemExit(main(" +
        repr(["--check-materials", str(destination / "check.json"), "--read-only", "--skip-hardware"]) + "))\n",
        encoding="utf-8")
    files[command.name] = {"source": "registered_attempt", "sha256": sha256(command.read_bytes()), "path": str(command)}
    write_json(destination / "manifest.json", material)
    seal(destination)
    command.chmod(0o500)
    return {"materials": material, "check_command": str(command),
            "read_directories": list(dict.fromkeys([str(source_root(lock)), str(destination),
                *(str(Path(item["path"]).parent) for item in rule_materials)]))}


def build_prompt(state: dict, lock: dict, attempt: dict, *, rules_lock: dict | None = None,
                 executor_lock: dict | None = None) -> str:
    semantic, executor = rules_lock or lock, executor_lock or rules_lock or lock
    rules, definition, materials = rules_root(semantic), STAGES[attempt["stage"]], attempt["materials"]
    documents, bindings = _documents(attempt), lock["source"]["bindings"]

    def source_reference(key):
        entry = bindings[key]
        symbol = "::" + entry["symbol"] if entry["symbol"] else ""
        return f"{lock['source']['repository_id']}@{lock['source']['ref']}:{entry['path']}{symbol}"

    resolution = (design_context(state, attempt, Path(attempt["workspace"]), parser_root=rules)
                  if attempt["stage"] in {"P5", "P6"} else {})
    scope, owner = scope_for(state, attempt), owner_for(state, attempt)
    selected = state.get("selected_designs")
    node_entry = next((entry for entry in selected or [] if entry["id"] == attempt["node"]), None)
    phase_name = f"{PROMPT_ROOT}/phases/{definition['page']}"
    locating = {key: attempt[key] for key in ("run_id", "attempt_id", "stage", "role", "node", "base_checkpoint", "dependency_digests")}
    locating.update(mode=attempt.get("mode", "normal"), purpose=attempt.get("purpose", "normal"),
        backend=attempt["backend"], model=attempt["model"], effort=attempt["effort"], session_scope=attempt["session_scope"],
        workspace=attempt["workspace"], writable_files=attempt["writable_files"],
        source_id=lock["source"]["id"], source_ref=lock["source"]["ref"], source_root=str(source_root(lock)),
        rules_id=semantic["rules"]["id"], rules_root=str(rules), executor_rules_id=executor["rules"]["id"], executor_root=str(rules_root(executor)),
        workflow_page=str(rules / "EXTRACT_SCHEMA.md"), phase_page=str(rules / phase_name),
        inputs={key: bindings[key] for key in ("entrypoint", "host", "io", "input", "reference", "acceptance", "scope")},
        implementation_source=source_reference("entrypoint"), reference_source=source_reference("reference"),
        revision_scope=attempt.get("revision_scope", state.get("revision_scope", [])),
        provided_documents=[str(rules / name) for name in documents],
        read_specs=[str(rules / name) for name in _read_specs(attempt) if name not in documents],
        feedback=owner.get("feedback", []), process_records=materials["records"],
        required_conclusions=required_conclusions(attempt), required_evidence=definition["evidence"],
        required_artifact_evidence=sorted(required_artifact_evidence(state, attempt, attempt["dependency_digests"], resolution=resolution.get("design_resolution"))),
        selected_designs=selected, writing_plan=state.get("writing_plan"), node_entry=node_entry,
        plan_draft=state.get("plan_draft") if attempt["stage"] in {"P2", "P4"} else None,
        first_draft=materials["first_draft"], pending_rechecks=state.get("pending_rechecks", []),
        source_identities=attempt.get("source_identities", {}), recovery_handoff=attempt.get("recovery_handoff"),
        preparation=attempt.get("preparation"),
        historical_definitions=({"checkpoint": attempt["baseline_context_checkpoint"], "path": str(Path(materials["directory"]) / "historical"),
                                "purpose": "Read-only historical baseline; never an eligible current from source"}
                               if attempt.get("baseline_context_checkpoint") else None),
        reading_order=["current_specs", "current_artifacts", "fixed_source", "process_records"], check_command=shlex.quote(attempt["check_command"]),
        budget={**{key: scope.get(key) for key in ("revisions", "continues", "receipt_repairs", "cli_failures", "calls", "deadline")},
                "call_seconds": attempt.get("timeout_seconds"), "max_calls": call_limit(state), "revision_limit": state["config"]["revision_limit"]})
    locating.update(resolution)
    if attempt["stage"] in REOPEN_TARGETS and attempt.get("mode", "normal") == "normal" and attempt.get("purpose") != "maintenance":
        eligible = {"P5:" + node for node in attempt.get("source_identities", {}) if node not in {"root", attempt["node"]}}
        locating["reopen_targets"] = sorted(set(REOPEN_TARGETS[attempt["stage"]]) | set(resolution.get("provider_reopen_targets", {})) | eligible | {"external"})
    if attempt.get("purpose") == "maintenance":
        locating["maintenance"] = {key: owner.get(key) for key in ("task_id", "action", "step", "reason", "issues", "fields")}
        locating["revision_scope"] = owner.get("fields", [])
    if attempt.get("mode") == "adjudicate":
        locating["candidates"] = materials["candidates"]
    locating["reading"] = {"file_tools": ["Read", "Glob", "Grep"], "source_root": str(source_root(lock)), "artifact_root": attempt["workspace"],
                           "evidence_reader": str(rules_root(executor) / "schema_extract/tools/reading.py")}
    parts = [f'<document path={json.dumps(str(rules / name))}>\n{(rules / name).read_text(encoding="utf-8").rstrip()}\n</document>' for name in documents]
    parts.append("<invocation>\n" + json.dumps(locating, ensure_ascii=False, indent=2) + "\n</invocation>")
    return "\n\n".join(parts) + "\n"
