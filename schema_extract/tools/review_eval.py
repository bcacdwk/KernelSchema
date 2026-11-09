"""Bounded, independent Reviewer calls over document-integrity regression cases."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

from .agent import CodexCLI
from .artifacts import load_yaml, parser_for, write_yaml
from .common import ExtractError, digest, manifest, read_json, rules_root, source_root, write_json
from .inputs import SourceTree
from .receipts import TEXT, obj, validate_shape
from .snapshots import freeze_inputs
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


def response_schema() -> dict:
    return obj({
        "decision": {"type": "string", "enum": ["approve", "revise"]},
        "conclusions": {"type": "array", "items": obj({
            "subject": {"type": "string", "enum": list(DOCUMENT_CONCLUSIONS)}, "text": TEXT})},
        "issues": {"type": "array", "items": obj({
            "criterion": {"type": "string", "enum": list(DOCUMENT_CONCLUSIONS)},
            "path": TEXT, "detail": TEXT})},
    })


def grade_response(response: dict, expected: dict) -> dict:
    try:
        validate_shape(response, response_schema(), "review")
        if {item["subject"] for item in response["conclusions"] if item["text"].strip()} != set(DOCUMENT_CONCLUSIONS):
            raise ExtractError("all four document conclusions are required")
    except (ExtractError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": "invalid_response", "detail": str(exc)}
    if response["decision"] != expected["decision"]:
        return {"passed": False, "reason": "false_rejection" if expected["decision"] == "approve" else "missed_defect"}
    if expected["decision"] == "approve":
        return {"passed": not response["issues"], "reason": "correct" if not response["issues"] else "approval_with_issues"}
    for issue in response["issues"]:
        path = issue["path"].removeprefix("root.yaml::").removeprefix("root.yaml:")
        if (issue["criterion"] in expected["criteria"] and issue["detail"].strip()
                and (path.startswith(expected["path"]) or expected["path"].startswith(path))
                and path.startswith(("meta_data", "contract", "design", "realization"))):
            return {"passed": True, "reason": "correct"}
    return {"passed": False, "reason": "wrong_defect_or_location"}


def evaluation_prompt(lock: dict, workspace: Path) -> str:
    rules, source = rules_root(lock), source_root(lock)
    method = (rules / "schema_extract/REVIEW.md").read_text(encoding="utf-8")
    specs = [rules / "schema_docs" / name for name in (
        "0.common.md", "1.contract.algorithm_requirement.md", "3.design.math_reshape.md",
        "4.design.work_schedule.md", "5.design.tile_pipeline.md", "6.realization.md")]
    return f"""你是独立 Reviewer，本轮只验收当前文档的四项性质，不执行整个提取流程。
当前任务是合成的标量复制设计；来源和 reference 固定在下述 source_root。只核对文档表达，不判断真实 GPU 实现或性能。
先阅读下列规范与当前 root.yaml、dispatch.md，再按需读取固定来源。没有历史批准、旧稿或交接可以补充文件含义。
当前内容可能正确，也可能有缺陷。合法的数学文本、自定义动作、已执行的 no-op、规范允许的来源缺口应按其含义判断，不能为了找问题而拒绝。
硬件区明确跳过。无需重新运行 check_yaml，它的结构和已实现关联检查已通过；该结果不代替语义验收。
仅可读取当前 workspace、列明的 rules_root 和 source_root。不要读取宿主仓库、其他调用或运行目录中的文件。
在一次答复中完成当前文件的完整文档审阅，返回 output-schema JSON：四项 conclusions 均给出简短具体判断；
没有问题时 approve 且 issues=[]，有问题时 revise，issues 指明 criterion、原 YAML 字段路径和具体缺陷。path 从 meta_data/contract/design/realization 开始，不加文件名；issues 只列文档缺陷，不将允许保留的来源缺口另列为缺陷。不要建议新增字段来迁就现有内容。
workspace: {workspace}
rules_root: {rules}
source_root: {source}
schema_docs: {json.dumps([str(path) for path in specs], ensure_ascii=False)}

{method}
"""


def evaluate(suite_path: Path, output: Path, repo: Path, *, codex_bin="codex",
             model=None, effort="xhigh", call_seconds=240, selected=None) -> dict:
    suite, expanded = cases_from_suite(suite_path)
    cases = [case for case in suite["cases"] if selected is None or case["id"] in selected]
    if not cases or selected is not None and set(selected) != {case["id"] for case in cases}:
        raise ExtractError("requested review case does not exist")
    if output.exists():
        raise ExtractError("review evaluation output must be a new directory")
    output.mkdir(parents=True)
    seed = output / "seed"
    seed.mkdir()
    for name in ("reference.py", "source.md"):
        shutil.copy2(suite_path.parent / name, seed / name)
    one = lambda symbol: {"path": "reference.py", "symbol": symbol}
    bindings = {"id": "review_fixture", "directory": ".", "entrypoint": one("kernel"),
                "host": one("kernel"), "input": one("inputs"), "reference": one("reference"),
                **{kind: [{"path": "source.md", "symbol": ""}] for kind in ("io", "acceptance", "scope")}}
    config = {"suite": digest(suite), "case_count": len(cases), "model": model,
              "effort": effort, "call_seconds": call_seconds, "max_calls_per_case": 1}
    lock = freeze_inputs(output / "runtime", SourceTree(seed), bindings,
                         ["reference.py", "source.md"], config, rules_repo=repo)
    parser = parser_for(str(rules_root(lock)))
    checker = importlib.import_module(parser.__package__ + ".schema")
    run_id = str(uuid4())
    report = {"format": "schema_extract.review_eval.v1", "run_id": run_id,
              "created_at": time.time(), "config": config, "input_fingerprint": lock["fingerprint"],
              "rules_id": lock["rules"]["id"], "cases": [], "passed": False}
    write_json(output / "report.json", report)
    cli = CodexCLI(codex_bin, model)
    for case in cases:
        attempt_id = str(uuid4())
        directory = output / "attempts" / attempt_id
        workspace = directory / "workspace"
        workspace.mkdir(parents=True)
        value = deepcopy(expanded[case["id"]])
        prefix = f"{lock['source']['repository_id']}@{lock['source']['ref']}:reference.py"
        value["meta_data"]["source"] = prefix + "::kernel"
        value["contract"]["algorithm_requirement"]["golden_reference"]["source"] = prefix + "::reference"
        write_yaml(workspace / "root.yaml", value, parser_root=rules_root(lock))
        (workspace / "dispatch.md").write_text(
            "# 路径索引\n\n| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| [root](root.yaml) | 无 | 当前 Contract 的全部输入 | 完整标量复制设计 | 全域选择 |\n\n"
            "## 覆盖缺口\n\n没有已知的输入域覆盖缺口。\n", encoding="utf-8")
        checker.validate_file(workspace / "root.yaml", skip_hardware=True)
        before = manifest(workspace)
        attempt = {"run_id": run_id, "attempt_id": attempt_id, "stage": "document_review",
                   "node": "root", "role": "reviewer", "session_group": "document_review",
                   "session_id": None, "directory": str(directory), "workspace": str(workspace)}
        result = cli.run(attempt=attempt, prompt=evaluation_prompt(lock, workspace),
                         schema=response_schema(), effort=effort, timeout=call_seconds)
        grade = (grade_response(result["response"], case["expected"])
                 if result["outcome"] == "completed" else {"passed": False, "reason": result["outcome"]})
        if manifest(workspace) != before:
            grade = {"passed": False, "reason": "reviewer_changed_input"}
        response = result.get("response")
        row = {"id": case["id"], "attempt_id": attempt_id, "expected": case["expected"],
               "outcome": result["outcome"], "grade": grade, "input_files": before,
               "prompt_sha256": result["prompt_sha256"], "elapsed_seconds": result["elapsed_seconds"],
               "usage": result["usage"], "decision": response.get("decision") if isinstance(response, dict) else None}
        report["cases"].append(row)
        write_json(output / "report.json", report)
        print(json.dumps({"case": case["id"], "decision": row["decision"], **grade}, ensure_ascii=False), flush=True)
        if not result["termination"]["confirmed"]:
            break
    report["completed_at"] = time.time()
    report["passed"] = len(report["cases"]) == len(cases) and all(row["grade"]["passed"] for row in report["cases"])
    write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--suite", type=Path, default=repo / "schema_extract/archive/six-operators/20261104-review-cases/suite.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="xhigh")
    parser.add_argument("--call-seconds", type=int, default=240)
    parser.add_argument("--case", action="append", dest="cases")
    args = parser.parse_args(argv)
    if not 1 <= args.call_seconds <= 1800:
        parser.error("--call-seconds must be between 1 and 1800")
    try:
        report = evaluate(args.suite.resolve(), args.output.resolve(), repo, codex_bin=args.codex_bin,
                          model=args.model, effort=args.effort, call_seconds=args.call_seconds,
                          selected=args.cases)
    except (ExtractError, OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
