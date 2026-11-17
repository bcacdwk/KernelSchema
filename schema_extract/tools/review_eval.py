"""Bounded, independent Reviewer calls over document-integrity regression cases."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4

from .agent import AgentCLI
from .backends import DEFAULT_BACKEND, DEFAULT_EFFORT, DEFAULT_MODEL, backend_for
from .artifacts import check_complete, load_yaml, schema_for, validate_artifact_sources, write_yaml
from .common import ExtractError, digest, manifest, read_json, rules_root, source_root, write_json
from .inputs import SourceTree
from .receipts import TEXT, obj, validate_shape
from .snapshots import freeze_inputs
from .prompts import PROMPT_ROOT, _linked_rules, render_operation
from .stages import DOCUMENT_CONCLUSIONS


def cases_from_suite(suite_path: Path) -> tuple[dict, dict[str, dict]]:
    suite = read_json(suite_path)
    if suite.get("format") != "schema_extract.review_suite.v1":
        raise ExtractError("unsupported review suite")
    entries = {case["id"]: case for case in suite["cases"]}
    if len(entries) != len(suite["cases"]) or not entries:
        raise ExtractError("review case IDs must be unique and nonempty")
    base = load_yaml(suite_path.parent, suite["base"])
    expanded, visiting = {}, set()
    def expand(identity):
        if identity in expanded:
            return deepcopy(expanded[identity])
        if identity not in entries or identity in visiting:
            raise ExtractError("missing or cyclic review case parent: " + identity)
        visiting.add(identity)
        case = entries[identity]
        value = expand(case["extends"]) if case.get("extends") else deepcopy(base)
        for patch in case["patches"]:
            if not patch["path"]:
                raise ExtractError("review patch needs a field path")
            target = value
            for part in patch["path"][:-1]:
                target = target[part]
            target[patch["path"][-1]] = deepcopy(patch["value"])
        visiting.remove(identity)
        expanded[identity] = value
        return deepcopy(value)
    for identity in entries:
        expand(identity)
    return suite, expanded


def response_schema(attempt_id: str) -> dict:
    return obj({
        "attempt_id": {"type": "string", "enum": [attempt_id]},
        "decision": {"type": "string", "enum": ["approve", "revise"]},
        "conclusions": {"type": "array", "items": obj({
            "subject": {"type": "string", "enum": list(DOCUMENT_CONCLUSIONS)}, "text": TEXT})},
        "issues": {"type": "array", "items": obj({
            "criterion": {"type": "string", "enum": list(DOCUMENT_CONCLUSIONS)},
            "path": TEXT, "detail": TEXT})},
    })


def grade_response(response: dict, expected: dict, *, attempt_id: str) -> dict:
    try:
        validate_shape(response, response_schema(attempt_id), "review")
        if {item["subject"] for item in response["conclusions"] if item["text"].strip()} != set(DOCUMENT_CONCLUSIONS):
            raise ExtractError("all four document conclusions are required")
    except (ExtractError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": "invalid_response", "detail": str(exc)}
    if response["decision"] != expected["decision"]:
        return {"passed": False, "reason": "false_rejection" if expected["decision"] == "approve" else "missed_defect"}
    if expected["decision"] == "approve":
        return {"passed": not response["issues"], "reason": "correct" if not response["issues"] else "approval_with_issues"}
    for issue in response["issues"]:
        path = issue["path"]
        expected_path = expected["path"]
        related = path == expected_path or any(left.startswith(right + separator)
            for left, right in ((path, expected_path), (expected_path, path)) for separator in (".", "["))
        if (issue["criterion"] in expected["criteria"] and issue["detail"].strip()
                and related and path.partition(".")[0] in {"META_DATA", "CONTRACT", "DESIGN", "REALIZATION"}):
            return {"passed": True, "reason": "correct"}
    return {"passed": False, "reason": "wrong_defect_or_location"}


EVALUATION_READS = ["schema_docs/0.common.md", "schema_docs/1.contract.algorithm_requirement.md",
                    "schema_docs/3.design.math_reshape.md", "schema_docs/4.design.work_schedule.md",
                    "schema_docs/5.design.tile_pipeline.md", "schema_docs/6.realization.md"]


def evaluation_prompt(lock: dict, attempt: dict) -> str:
    method = f"{PROMPT_ROOT}/roles/reviewer.md"
    operation = f"{PROMPT_ROOT}/operations/review_eval.md"
    source = lock["source"]
    context = {"attempt_id": attempt["attempt_id"], "role": "reviewer", "workspace": attempt["workspace"],
               "writable_files": [], "source_root": str(source_root(lock)), "source_id": source["id"],
               "repository_id": source["repository_id"], "source_ref": source["ref"],
               "rules_root": str(rules_root(lock)), "rules_id": lock["rules"]["id"],
               "provided_documents": [str(rules_root(lock) / name) for name in (operation, method)],
               "read_specs": [str(rules_root(lock) / name) for name in EVALUATION_READS],
               "required_conclusions": list(DOCUMENT_CONCLUSIONS)}
    return (render_operation("review_eval", context, rules_lock=lock) +
            f'\n<document path="{rules_root(lock) / method}">\n' +
            (rules_root(lock) / method).read_text(encoding="utf-8") + "\n</document>\n")


def _materials(lock: dict, attempt: dict) -> dict:
    names = _linked_rules(lock, [*EVALUATION_READS, f"{PROMPT_ROOT}/roles/reviewer.md",
                                 f"{PROMPT_ROOT}/operations/review_eval.md"])
    evidence = [{"path": str(source_root(lock)), "files": lock["source"]["files"]}]
    for name in sorted(names):
        evidence.append({"path": str(rules_root(lock) / name), "sha256": lock["rules"]["files"][name]})
    write_json(Path(attempt["directory"]) / "materials.json", evidence)
    return [str(source_root(lock)), *sorted({str(Path(item["path"]).parent) for item in evidence[1:]})]


def evaluate(suite_path: Path, output: Path, repo: Path, *, backend=DEFAULT_BACKEND,
             agent_binary=None, model=None, effort=None, call_seconds=240, selected=None) -> dict:
    if backend == "codex" and (model is None or effort is None):
        raise ExtractError("Codex review evaluation requires an explicit model and effort")
    model = model if model is not None else DEFAULT_MODEL
    effort = effort if effort is not None else DEFAULT_EFFORT
    if not isinstance(model, str) or not model.strip() or not 1 <= call_seconds <= 1800:
        raise ExtractError("evaluation requires a model and a call_seconds limit in 1..1800")
    executable = shutil.which(agent_binary or backend)
    if executable is None:
        raise ExtractError("evaluation backend executable is missing")
    executable = str(Path(executable).resolve())
    backend_for(backend).command(executable, model=model, effort=effort, role="reviewer", session_id=None,
                                 schema=response_schema(str(uuid4())), cwd=output, workspace=output,
                                 schema_path=output / "response.schema.json", output_path=output / "response.json")
    suite, expanded = cases_from_suite(suite_path)
    cases = [case for case in suite["cases"] if selected is None or case["id"] in selected]
    if not cases or selected is not None and set(selected) != {case["id"] for case in cases}:
        raise ExtractError("requested review case does not exist")
    if output.exists():
        raise ExtractError("review evaluation output must be a new directory")
    output.mkdir(parents=True)
    seed = output / "source"
    seed.mkdir()
    for name in ("reference.py", "source.md"):
        shutil.copyfile(suite_path.parent / name, seed / name)
    one = lambda symbol: {"path": "reference.py", "symbol": symbol}
    bindings = {"id": "review_fixture", "directory": ".", "entrypoint": one("kernel"),
                "host": one("kernel"), "input": one("inputs"), "reference": one("reference"),
                **{kind: [{"path": "source.md", "symbol": ""}] for kind in ("io", "acceptance", "scope")}}
    config = {"suite": digest(suite), "case_count": len(cases), "backend": backend,
              "agent_binary": executable, "model": model, "effort": effort,
              "call_seconds": call_seconds, "max_calls_per_case": 1}
    lock = freeze_inputs(output / "runtime", SourceTree(seed), bindings,
                         ["reference.py", "source.md"], config, rules_repo=repo)
    rules, checker = rules_root(lock), schema_for(rules_root(lock))
    run_id = str(uuid4())
    report = {"format": "schema_extract.review_eval.v1", "run_id": run_id,
              "created_at": time.time(), "config": config, "input_fingerprint": lock["fingerprint"],
              "rules_id": lock["rules"]["id"], "cases": [], "passed": False}
    write_json(output / "report.json", report)
    cli = AgentCLI(backend=backend, executable=executable, model=model)
    for case in cases:
        attempt_id = str(uuid4())
        directory = output / "attempts" / attempt_id
        workspace = directory / "workspace"
        workspace.mkdir(parents=True)
        value = deepcopy(expanded[case["id"]])
        prefix = f"{lock['source']['repository_id']}@{lock['source']['ref']}:reference.py"
        value["META_DATA"]["Source"] = prefix + "::kernel"
        value["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]["Source"] = prefix + "::reference"
        write_yaml(workspace / "root.yaml", value, parser_root=rules)
        (workspace / "dispatch.md").write_text(
            "# 标量复制\n\n## 范围\n\n合成的文档审阅材料。\n\n## 完整设计\n\n"
            "| 设计 | 具体机制与保留理由 | 关系 |\n| --- | --- | --- |\n"
            "| [root](root.yaml) | 完整标量复制设计 | 唯一方案 |\n", encoding="utf-8")
        normalized = checker.normalize_directory(workspace, skip_hardware=True)
        for node, text in normalized.candidates.items():
            (workspace / (node + ".yaml")).write_text(text, encoding="utf-8")
        check = check_complete(workspace, rules, selected_designs=[{"id": "root", "instance": "reference.py::kernel"}])
        if check["exit_code"]:
            raise ExtractError("review case is not structurally valid: " + "; ".join(check["diagnostics"]))
        validate_artifact_sources(workspace, lock, decision="complete", parser_root=rules)
        before = manifest(workspace)
        scope = {"stage": "document_review", "node": case["id"], "role": "reviewer", "operation": "review_eval"}
        attempt = {"run_id": run_id, "attempt_id": attempt_id, **{key: scope[key] for key in ("stage", "node", "role")},
                   "backend": backend, "model": model, "effort": effort, "executable": executable,
                   "session_scope": scope, "session_directory": str(output / "sessions" / digest(scope)),
                   "session_id": None, "writable_files": [], "directory": str(directory), "workspace": str(workspace)}
        attempt["read_directories"] = _materials(lock, attempt)
        write_json(directory / "attempt.json", attempt)
        result = cli.run(attempt=attempt, prompt=evaluation_prompt(lock, attempt),
                         schema=response_schema(attempt_id), timeout=call_seconds)
        grade = (grade_response(result["response"], case["expected"], attempt_id=attempt_id)
                 if result["outcome"] == "completed" else {"passed": False, "reason": result["outcome"]})
        if manifest(workspace) != before:
            grade = {"passed": False, "reason": "reviewer_changed_input"}
        response = result.get("response")
        row = {"id": case["id"], "attempt_id": attempt_id, "expected": case["expected"],
               "grade": grade, "input_files": before,
               **{key: result.get(key) for key in ("backend", "outcome", "session_id", "session_status", "native_status",
                   "requested_model", "requested_effort", "reported_model", "reported_effort", "cli_version",
                   "prompt_sha256", "elapsed_seconds", "usage", "raw_logs")},
               "decision": response.get("decision") if isinstance(response, dict) else None}
        report["cases"].append(row)
        write_json(output / "report.json", report)
        print(json.dumps({"case": case["id"], "decision": row["decision"], **grade}, ensure_ascii=False), flush=True)
        if not result["termination"]["confirmed"]:
            break
    report["completed_at"] = time.time()
    report["passed"] = len(report["cases"]) == len(cases) and all(row["grade"]["passed"] for row in report["cases"])
    write_json(output / "report.json", report)
    return report
