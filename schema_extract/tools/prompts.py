"""Current frozen phase page, fixed handoffs, and one invocation's locating context."""

from __future__ import annotations

import json
from pathlib import Path
import shlex

from .artifacts import _same, load_root
from .budget import call_limit
from .receipts import REOPEN_TARGETS, required_artifact_evidence
from .stages import STAGES, owner_for, read_specs, required_conclusions, scope_for
from .common import ExtractError, read_json, rules_root, source_root
from .snapshots import frozen_reader


def build_prompt(state: dict, lock: dict, attempt: dict, *, rules_lock: dict | None = None,
                 executor_lock: dict | None = None) -> str:
    definition = STAGES[attempt["stage"]]
    if attempt.get("semantic_revision") and rules_lock is None:
        raise ExtractError("semantic prompt requires its explicit rules lock")
    semantic = rules_lock or lock
    executor = executor_lock or semantic
    mode = attempt.get("mode", "normal")
    maintaining = attempt.get("purpose") == "maintenance"
    rules, source = rules_root(semantic), source_root(lock)
    page_path = rules / "schema_extract/phases" / definition["page"]
    bindings = lock["source"]["bindings"]
    def source_reference(key):
        entry = bindings[key]
        symbol = "::" + entry["symbol"] if entry["symbol"] else ""
        return f"{lock['source']['repository_id']}@{lock['source']['ref']}:{entry['path']}{symbol}"
    node_entry = next((entry for entry in state.get("plan") or [] if entry["id"] == attempt["node"]), None)
    plan = state.get("plan") or state.get("plan_draft")
    if attempt["stage"] == "P4" and attempt["role"] == "reviewer" and state.get("plan_draft") is not None:
        # A replan reviewer must assess the current worker proposal, including
        # an empty incomplete draft, rather than the previously approved tree.
        plan = state["plan_draft"]
    run_dir = Path(attempt["directory"]).parent.parent
    prior_stage = {"P3": "P2", "P4": "P3", "P5": "P4"}.get(attempt["stage"])
    handoff_path = run_dir / "stages" / (prior_stage or "") / "handoff.md"
    handoff = (handoff_path.read_text(encoding="utf-8")
               if attempt["role"] == "worker" and prior_stage and handoff_path.is_file() else "")
    scope = scope_for(state, attempt)
    plan_receipt, node_evidence = None, []
    relevant_nodes = set(state["nodes"])
    ancestors = []
    if attempt["stage"] == "P5":
        if attempt["role"] == "worker":
            approved_plan = state["stages"]["P4"]["approval"]
            plan_receipt = run_dir / "attempts" / approved_plan["attempt_id"] / "receipt.json"
            approved_evidence = read_json(plan_receipt)["source_evidence"]
            node_evidence = [approved_evidence[index] for index in node_entry["evidence"]]
        relevant_nodes = {attempt["node"]}
        parent = node_entry["parent"]
        while parent:
            relevant_nodes.add(parent)
            ancestors.append(next(entry for entry in state["plan"] if entry["id"] == parent))
            parent = state["nodes"][parent]["parent"]
    def approval_reference(value):
        return {"checkpoint": value["checkpoint"], "receipt_sha256": value["receipt_sha256"],
                "receipt": str(run_dir / "attempts" / value["attempt_id"] / "receipt.json")}
    first_draft = attempt["stage"] == "P5" and not state["nodes"][attempt["node"]].get("authored", False)
    if attempt["stage"] == "P1":
        baseline = run_dir / "checkpoints" / attempt["base_checkpoint"] / "artifacts"
        first_draft = _same(load_root(baseline, parser_root=rules),
                            load_root(rules / "schema_template", "starter.yaml", parser_root=rules))
    locating = {"attempt_id": attempt["attempt_id"], "stage": attempt["stage"], "role": attempt["role"],
                "node": attempt["node"], "mode": mode, "purpose": attempt.get("purpose", "normal"),
                "base_checkpoint": attempt["base_checkpoint"],
                "dependency_digests": attempt["dependency_digests"],
                "source_id": lock["source"]["id"], "source_ref": lock["source"]["ref"],
                "source_root": str(source), "rules_id": semantic["rules"]["id"],
                "semantic_revision": attempt.get("semantic_revision"),
                "executor_rules_id": executor["rules"]["id"], "executor_root": str(rules_root(executor)),
                "execution_binding": attempt.get("execution_binding"),
                "rules_root": str(rules), "phase_page": str(page_path), "workspace": attempt["workspace"],
                "inputs": {key: bindings[key] for key in ("entrypoint", "host", "io", "input", "reference", "acceptance", "scope")},
                "implementation_source": source_reference("entrypoint"), "reference_source": source_reference("reference"),
                "writable_files": attempt["writable_files"], "revision_scope": state["revision_scope"],
                "read_specs": read_specs(attempt), "feedback": owner_for(state, attempt)["feedback"], "handoff": handoff,
                "required_conclusions": required_conclusions(attempt), "required_evidence": definition["evidence"],
                "required_artifact_evidence": sorted(required_artifact_evidence(state, attempt, attempt["dependency_digests"])),
                "node_entry": node_entry, "plan": plan,
                "plan_receipt": str(plan_receipt) if plan_receipt else None, "node_evidence": node_evidence,
                "reopening": state.get("reopening"),
                "recovery_handoff": attempt.get("recovery_handoff"),
                "imported_draft": state.get("imported_draft"),
                "first_draft": first_draft,
                "pending_rechecks": state["pending_rechecks"],
                "prior_approvals": {key: approval_reference(value["approval"]) for key, value in state["stages"].items()
                                    if value["approval"] and (attempt["stage"] != "P5" or key == "P4")},
                "node_approvals": {key: approval_reference(value["approval"]) for key, value in state["nodes"].items()
                                   if key in relevant_nodes and value.get("approval")},
                "reviewed_artifacts": {key: str(run_dir / "checkpoints" / value["approval"]["checkpoint"] / "artifacts" / value["file"])
                                       for key, value in state["nodes"].items() if key in relevant_nodes and value.get("approval")},
                "budget": {**{key: scope[key] for key in ("revisions", "continues", "receipt_repairs", "cli_failures", "calls", "deadline")},
                           "call_seconds": attempt.get("timeout_seconds"), "max_calls": call_limit(state),
                           "revision_limit": state["config"]["revision_limit"]}}
    if maintaining:
        task = state["maintenance"]["tasks"][attempt["maintenance_task_id"]]
        locating["maintenance"] = {key: task.get(key) for key in
                                   ("task_id", "action", "step", "plan_reason", "reason", "issues", "fields")}
        locating["revision_scope"] = task.get("fields", [])
        locating["feedback"] = task.get("feedback", [])
        locating.pop("reopening")
    if attempt["stage"] == "P5":
        locating.pop("plan")
        locating["ancestor_entries"] = ancestors
        locating["revision_scope"] = [item for item in locating["revision_scope"] if item["path"] == node_entry["file"]]
        locating["pending_rechecks"] = [node for node in state["pending_rechecks"] if node in relevant_nodes]
        # P4 has already handed off the relevant source/plan decision. Its old
        # whole-tree history and an imported legacy tree are not branch input.
        locating.pop("reopening", None)
        locating.pop("imported_draft")
    continuation = state.get("continuation")
    if continuation and attempt["role"] == "worker" and attempt["stage"] != "P6":
        old = Path(continuation["evidence"]) / "artifacts"
        names = ({state["nodes"][node]["file"] for node in relevant_nodes} if attempt["stage"] == "P5" else
                 {"root.yaml"} if attempt["stage"] == "P1" else {"root.yaml", "dispatch.md"})
        locating["continued_draft"] = {"predecessor": continuation["predecessor"],
            "artifacts": {name: str(old / name) for name in sorted(names) if (old / name).is_file()}}
    if mode == "adjudicate" and not maintaining:
        locating["candidates"] = [{**entry, "artifacts": str(run_dir / "checkpoints" / entry["checkpoint"] / "artifacts")}
                                  for entry in scope["candidates"]]
    if attempt["stage"] == "P4":
        locating["current_node_artifacts"] = dict(locating["reviewed_artifacts"])
    review_instructions = ""
    if attempt["role"] == "reviewer":
        for key in ("feedback", "handoff", "plan_receipt", "node_evidence", "prior_approvals",
                    "node_approvals", "reviewed_artifacts", "imported_draft",
                    "continued_draft", "recovery_handoff", "reopening"):
            locating.pop(key, None)
        locating["review_scope"] = {
            "files": locating["required_artifact_evidence"],
            "method": str(rules / "schema_extract/REVIEW.md"),
            "order": ["current_specs", "current_artifacts", "fixed_source"],
        }
        review_instructions = (rules / "schema_extract/REVIEW.md").read_text(encoding="utf-8")
    reader = frozen_reader(executor)
    example = f"python3 -B {shlex.quote(str(reader))} --root {shlex.quote(str(source))} --path {shlex.quote(bindings['host']['path'])} --start-line 1 --end-line 80"
    checker = rules / "schema_template/check_yaml.py"
    target = node_entry["file"] if attempt["stage"] == "P5" else "root.yaml"
    check_command = f"python3 -B {shlex.quote(str(checker))} --skip-hardware {shlex.quote(str(Path(attempt['workspace']) / target))}"
    if attempt["role"] == "worker":
        role_instructions = """只修改writable_files与当前阶段、本轮字段授权的交集；已有稿件优先，不重置内容或版本。
提交前按rules_root/schema_extract/REVIEW.md的共同方法自查独立理解、分层、一致与必要性；沿该文处理来源缺项、源实现缺陷、规范候选和issue证据。每段先查当前schema_docs，采用已定义的合法写法。
decision=continue只保存本轮草稿并接续原逻辑步骤，不执行CLI continue的语义修订准备。"""
    else:
        role_instructions = "只读当前workspace；按下方共用方法独立完成当前阶段的全文审阅，检查器退出码不代替语义判断。"
    if maintaining:
        mode_instructions = """本轮仅执行maintenance.step；阶段页的普通推进不适用。维护不消耗或重新授予领域裁决/签收，不使用reopen决策；output-schema要求reopen字段时填null。
review/verify的reviewer返回approve/revise/blocked，完整审阅本阶段及当前前提，fields不缩小审阅范围。review有据退修时，revision_scope逐项给出带当前artifact_evidence的{path,fields}，受已有fields上限约束；无法可靠定位字段则填[]并blocked。review最多进入一次patch再独立verify；verify仍有问题不再开启修补循环。
patch的worker返回complete/continue/blocked，原样回传maintenance.fields作为revision_scope。continue保存当前草稿并继续同一patch逻辑步骤，沿用调用/时间预算；complete才进入verify，blocked保存草稿与恢复条件。
需越界或修改上游时说明具体问题、字段和恢复条件，交Supervisor追加有依据的计划；不得自行扩大范围。"""
    elif mode == "adjudicate":
        mode_instructions = """本轮为原reviewer的一次裁决，只返回approve或blocked，不直接改workspace。
adjudication.action=select只能选列出的候选；patch基于当前checkpoint，patches为当前阶段/授权字段内的{path,old,new}，old非空且唯一匹配。
Runner在隔离副本应用、核对版本并检查；关键事实缺失、补丁或检查失败即blocked，不再裁决或重试。证据仍指实际读取的原workspace，补丁后的文件SHA-256由Runner另存。"""
        if attempt["stage"] == "P6":
            mode_instructions += "\nP6亲自patch最终稿后，由全新独立reviewer签收一次；选择已有稿不追加签收。"
    elif mode == "signoff":
        mode_instructions = "本轮为P6裁决补丁后的全新独立reviewer签收。只返回approve或blocked，完整审阅最终稿并覆盖全部pending_rechecks；不再返工、裁决或重试。"
    else:
        mode_instructions = ("返回complete/continue/blocked。continue保存草稿并续做同一普通撰写步骤，沿用调用/时间预算；blocked将合法草稿和缺项交独立reviewer判断。"
                             if attempt["role"] == "worker" else "返回approve/revise/blocked，按共用方法给出当前阶段结论。")
        mode_instructions += "\n普通退修沿用原角色；达到budget.revision_limit后由原reviewer仅裁决一次。格式修复不算实质退修，恢复沿用原预算。"
    if attempt["stage"] in REOPEN_TARGETS and mode == "normal" and not maintaining:
        locating["reopen_targets"] = sorted(REOPEN_TARGETS[attempt["stage"]]) + ["external"]
        mode_instructions += """
固定来源推翻上游时可reopen，填写稳定issue_id、target、reason和本次来源证据索引；普通提案reopen=null。
树/选父回P4，代表路径回P2，现有root Design/when回P3，共享Contract回P1；仅能使用reopen_targets。
外部任务要求改变用external，等待任务方明确确认。已读源行的转录错误用issues.kind=transcription；当前产物的规范表达错误用schema_conformance。两者均由scope定位错误产物，以artifact_evidence引用该字段；规范错误还须在detail指出对应条款。其他重开需新源码发现。同一issue_id与固定源码范围只重开一次，换note/引文不重计。"""
    checking = ("本阶段使用阶段产物门禁；未完成或未生成的Design不要求完整检查。"
                if attempt["stage"] in {"P1", "P2", "P4"} else
                f"Runner接收complete/approve/revise时自动运行冻结检查器；complete/approve须退出0，revise可附退出1的产物诊断交回作者，退出2报工具/环境问题。\n需要提前自检或定位诊断时可运行：{check_command}；不要求worker/reviewer各自重复执行。")
    context_instructions = []
    if locating.get("recovery_handoff"):
        context_instructions.append("有recovery_handoff：旧会话已确认中断并退休，以本页、当前checkpoint和固定来源重建上下文。")
    if locating.get("imported_draft") or locating.get("continued_draft"):
        context_instructions.append("imported_draft/continued_draft仅是列明的只读旧稿线索，旧state/批准不授予写权限或放行；证据仍指当前source_root/workspace。")
    handoff_instructions = ("批准时handoff只写已定事项、理由、关键入口和缺口，最多20个非空行/1200字符。"
                            if attempt["role"] == "reviewer" and attempt["stage"] in {"P2", "P3", "P4"} else
                            "本轮无必需短交接，handoff可为空串。")
    return f"""你是独立的 {attempt['role']}，只执行本次 {attempt['stage']} / {attempt['node']} / {locating['purpose']} / {mode}。
目标是按当前冻结schema_docs把固定kernel提取为忠实、简洁的流水线设计。source_root绑定原固定源码，rules_root绑定本次语义规则；executor_root只提供绑定的执行工具，不能替代rules_root。不得从原工作树或旧状态切换来源、规则或权限。
{role_instructions}
Runner唯一写运行状态。worker/reviewer不开子代理，不改kernel，不做GPU测试；有据的源低层缺陷不要求先修kernel，按共用方法保留事实与必要设计。
按嵌入阶段页及本轮模式工作，只读列明的冻结来源/规则/执行工具、当前workspace、回执、已审产物、候选与旧稿线索；沿当前问题读取规范和页内方法。
先定位当前问题需要的源码范围；不因文件较长而通读。有next_cursor表示所选范围尚未读完，需要续读该范围时保留起止行并用--cursor：
{example}
最后按output-schema仅返回schema_extract.proposal.v1 JSON，原样回传本次attempt_id nonce；缺失或陈旧nonce硬拒绝。summary由你填写本轮简短、非空的语义结论。identity、文件SHA-256、文件清单、实际差异由Runner从已登记调用与真实字节生成，不手填、不另写回执文件。
证据由你亲读并选择：相对path、1起始line_start/line_end（最多12行）、quote_offset（范围原文中从0开始的字符偏移，通常0）和说明语义的note；来源另选category。用reader --evidence --end-line N [--quote-offset N]预览该范围生成的最多320字符精确引文，不会替你寻找依据。不要回传sha256/source_id/quote。
complete/approve覆盖required_evidence、required_conclusions；每类至少选对应绑定入口，调用链可补充。conclusions/issues等的evidence从0索引本次source_evidence。reviewer各种决策均须提供实际源码证据，artifact_evidence覆盖required_artifact_evidence每个文件。文档判断、问题分流按共用方法；P6的tree证据例外见当前阶段页。
{mode_instructions}
{chr(10).join(context_instructions)}
checks仅记亲自执行的命令/退出码；交Runner自动检查时填[]，checks_note说明由Runner执行。
{handoff_instructions}
{checking}

<document-review-method>
{review_instructions}
</document-review-method>

<invocation>
{json.dumps(locating, ensure_ascii=False, indent=2)}
</invocation>

<current-phase-page path={json.dumps(str(page_path))}>
{page_path.read_text(encoding='utf-8')}
</current-phase-page>
"""
